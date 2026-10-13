"""RAPPO algorithm specification (PPO with a distributional critic)."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState

from .ppo import make_eval_step_fn
from .base import AlgorithmSpec, default_build_eval_state
from algorithms.common import gae_standard
from net_utils import create_optimizer, make_linear_schedule
from buffers import Transition, RunnerState, UpdateState
from networks import ActorDCritic
from envs.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    DomainRandomizationWrapper,
    ClipAction,
    ResetEnvWrapper,
    SensitivityWrapper,
    ConstraintEvalWrapper,
    ConstraintPenaltyWrapper,
)


def wrap_env(env, config):
    """Apply the RAPPO wrapper stack to an environment.

    Args:
        env (environment.Environment): Base environment (e.g. ``TilthexV1``).
        config (dict): Training configuration; reads ``CONSTRAINTS``, ``SENSITIVITY``,
            ``DOMAIN_RANDOMIZATION``, ``NORMALIZE_ENV`` and ``GAMMA``.

    Returns:
        GymnaxWrapper: The wrapped, vectorised environment.
    """
    env = GymnaxWrapper(env)
    env = ClipAction(env)

    # Optional constraint penalty wrapper
    if config.get("CONSTRAINTS", False):
        env = ConstraintEvalWrapper(env)
        penalty = config.get("CONSTRAINT_PENALTY", 1.0)
        env = ConstraintPenaltyWrapper(env, penalty=penalty)

    # Optional sensitivity wrapper (for online sensitivity tracking, no policy modification)
    if config.get("SENSITIVITY", False):
        env = SensitivityWrapper(env)
    # if config.get("DOMAIN_RANDOMIZATION", False):
    env = DomainRandomizationWrapper(env, env.default_params)
    # else:
    #    env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    return env


def calculate_huber_loss(predictions, targets, delta=1.0):
    """Element-wise Huber loss between predicted and target quantiles.

    Args:
        predictions (jax.Array): Predicted quantiles.
        targets (jax.Array): Target values.
        delta (float): Threshold between the quadratic and the linear region.

    Returns:
        jax.Array: Huber loss, same shape as the broadcast of the inputs.
    """
    error = predictions - targets
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber_loss = 0.5 * quadratic**2 + delta * linear
    return huber_loss


def calculate_quantile_huber_loss(huber_loss, predictions, targets, taus):
    """Quantile Huber loss used to train the distributional critic.

    Args:
        huber_loss (jax.Array): Output of ``calculate_huber_loss``.
        predictions (jax.Array): Predicted quantiles, shape ``(B, N_TAUS, 1)``.
        targets (jax.Array): Target values.
        taus (jax.Array): Quantile fractions.

    Returns:
        jax.Array: Weighted loss, same shape as ``huber_loss``.
    """
    return (
        jnp.abs(
            taus.reshape(-1, predictions.shape[1], 1)
            - (targets - predictions < 0).astype(jnp.float32)
        )
        * huber_loss
    )


def make_loss_fn(config, graphdef):
    """Create the RAPPO loss function of the policy network.

    Args:
        config (dict): Training configuration (clipping, loss coefficients).
        graphdef (nnx.GraphDef): Graph definition of the ActorDCritic network.

    Returns:
        Callable: ``loss_fn(params, traj_batch, advantages, targets, rng) -> (total_loss, aux)``, where ``aux``
        holds the value loss, actor loss, entropy and the KL and clipping statistics.
    """

    def _loss_fn(params, traj_batch, advantages, targets, rng):
        # RERUN NETWORK
        model = nnx.merge(graphdef, params)
        pi, (quantile_values, taus) = model(traj_batch.obs, rng)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        # Huber loss
        element_wise_huber_loss = calculate_huber_loss(
            quantile_values.reshape(-1, config.get("N_TAUS", 64), 1),
            targets.reshape(-1, config.get("N_TAUS", 64), 1).transpose(0, 2, 1),
        )
        quantile_huber_loss = calculate_quantile_huber_loss(
            element_wise_huber_loss,
            quantile_values.reshape(-1, config.get("N_TAUS", 64), 1),
            targets.reshape(-1, config.get("N_TAUS", 64), 1).transpose(0, 2, 1),
            taus,
        )  # Shape: (B, T, N_TAUS, 1)
        value_loss = quantile_huber_loss.mean()

        # CALCULATE ACTOR LOSS
        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfracs = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])

        advantages = advantages.reshape(-1)
        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        )
        return total_loss, (
            value_loss,
            loss_actor,
            entropy,
            old_approx_kl,
            approx_kl,
            clipfracs,
        )

    return _loss_fn


def make_collect_fn(config, env, env_params, networks):
    """Create the step function that collects one transition per environment.

    Args:
        config (dict): Training configuration.
        env (GymnaxWrapper): Wrapped environment returned by ``wrap_env``.
        env_params (EnvParams): Environment parameters.
        networks (dict): Output of ``init_networks``.

    Returns:
        Callable: ``env_step(runner_state, unused) -> (runner_state, transition)``,
        for use with ``jax.lax.scan``.
    """
    graphdef = networks["graphdef"]

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        model = nnx.merge(graphdef, runner_state.train_state.params)
        pi, (quantile_values, _) = model(runner_state.last_obs, rng)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )
        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        transition = Transition(
            done.reshape(-1, 1, 1).repeat(config.get("N_TAUS", 64), 1),
            action,
            quantile_values,
            reward.reshape(-1, 1, 1).repeat(config.get("N_TAUS", 64), 1),
            log_prob,
            runner_state.last_obs,
            info,
        )
        new_runner_state = RunnerState(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        return new_runner_state, transition

    return _env_step


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create the function that runs one RAPPO update epoch over the minibatches.

    Args:
        config (dict): Training configuration (``NUM_MINIBATCHES``, ``UPDATE_EPOCHS``).
        loss_fn (Callable): Loss function returned by ``make_loss_fn``.
        shuffle_batch_fn (Callable): Function that shuffles a batch and splits it into minibatches.
        networks (dict): Output of ``init_networks``.

    Returns:
        Callable: ``update_epoch(update_state, unused) -> (update_state, losses)``,
        for use with ``jax.lax.scan``.
    """

    def _update_epoch(update_state: UpdateState, unused):
        def _update_minbatch(curry, batch_info):
            train_state, rng = curry
            rng, _rng = jax.random.split(rng)

            traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets, _rng
            )
            train_state = train_state.apply_gradients(grads=grads)
            return (train_state, rng), total_loss

        rng, _rng = jax.random.split(update_state.rng)
        minibatches = shuffle_batch_fn(
            _rng,
            (update_state.traj_batch, update_state.advantages, update_state.targets),
            config,
        )
        (train_state, rng), total_loss = jax.lax.scan(
            _update_minbatch, (update_state.train_state, rng), minibatches
        )
        new_update_state = UpdateState(
            train_state=train_state,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            targets=update_state.targets,
            rng=rng,
            extras=update_state.extras,
        )
        return new_update_state, total_loss

    return _update_epoch


def calculate_gae(config, traj_batch, networks, collect_state):
    """Compute the generalised advantage estimates of a rollout.

    Args:
        config (dict): Training configuration (``GAMMA``, ``GAE_LAMBDA``).
        traj_batch (Transition): Collected transitions, shape ``(NUM_STEPS, NUM_ENVS, ...)``.
        networks (dict): Output of ``init_networks``.
        collect_state (RunnerState): Runner state at the end of the rollout, used to bootstrap the last value.

    Returns:
        tuple[dict, RunnerState]: The dict with ``advantages`` and ``targets``, and the runner state.
    """
    rng, _rng = jax.random.split(collect_state.rng)
    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    _, (last_val, _) = model(collect_state.last_obs, _rng)
    # Standard GAE but giving B x T
    advantages, targets = gae_standard(config, traj_batch, last_val)
    advantages = jnp.mean(advantages, axis=-2)
    return {
        "advantages": advantages,
        "targets": targets,
    }, RunnerState(
        train_state=collect_state.train_state,
        env_state=collect_state.env_state,
        last_obs=collect_state.last_obs,
        rng=rng,
    )


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Create the networks and their training states.

    Args:
        config (dict): Training configuration (layer sizes, activation, learning rate).
        env (GymnaxWrapper): Wrapped environment, used for the observation and action sizes.
        env_params (EnvParams): Environment parameters.
        rng (jax.Array): PRNG key for the initialisation.
        load_params_fn (Callable, optional): Function that loads pre-trained actor weights.

    Returns:
        dict: ``graphdef`` and ``train_state``.
    """
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).shape[0]

    network = ActorDCritic(
        obs_dim,
        action_dim,
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        hidden_size=config.get("CRITIC_HIDDEN_SIZE", 256),
        n_cosines=config.get("N_COSINES", 64),
        n_taus=config.get("N_TAUS", 64),
        lb=config.get("SAMPLE_LB", 0.0),
        ub=config.get("ALPHA", 1.0),
        rngs=nnx.Rngs(int(rng[0])),
    )

    graphdef, state = nnx.split(network)

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=None,
        params=state,
        tx=tx,
    )

    return {
        "graphdef": graphdef,
        "train_state": train_state,
    }


SPEC = AlgorithmSpec(
    algo_name="RAPPO",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
    make_eval_step_fn=make_eval_step_fn,
    build_eval_state=default_build_eval_state,
)
