"""SPPO with augmented sensitivity estimation (GRU-based value function)."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState

from .ppo import make_eval_step_fn
from .base import AlgorithmSpec, default_build_eval_state
from buffers import Transition, RunnerState, UpdateState
from algorithms.common import gae_sensitivity
from net_utils import create_optimizer, make_linear_schedule
from networks import ActorCritic, CriticGRU
from utils import init_network_params
from envs.wrappers import (
    GymnaxWrapper,
    ClipAction,
    SensitivityWrapper,
    ComputeSensitivityReward,
    DomainRandomizationWrapper,
    ResetEnvWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecRewardSensitivity,
    NormalizeVecReward,
    ConstraintEvalWrapper,
    ConstraintPenaltyWrapper,
)


def wrap_env(env, config):
    """Apply the SPPO-aug wrapper stack to an environment.

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

    env = SensitivityWrapper(env)
    env = ComputeSensitivityReward(env)
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecRewardSensitivity(env, config["GAMMA"])

    return env


def make_loss_fn(config, graphdef):
    """Create the SPPO-aug loss function of the policy network.

    Args:
        config (dict): Training configuration (clipping, loss coefficients).
        graphdef (nnx.GraphDef): Graph definition of the ActorCritic network.

    Returns:
        Callable: ``loss_fn(params, traj_batch, advantages, advantages_s, targets) -> (total_loss, aux)``, where ``aux``
        holds the value loss, actor loss, entropy and the KL and clipping statistics.
    """

    def _loss_fn(params, traj_batch, advantages, advantages_s, targets):
        # RERUN NETWORK
        model = nnx.merge(graphdef, params)
        pi, value = model(traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfracs = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])

        # Normalize and combine advantages
        gae_r = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        gae_s = (advantages_s - advantages_s.mean()) / (advantages_s.std() + 1e-8)
        # Conflict: the task is failing (gae_r negative) but is less sensitive than expected (gae_s positive)
        # conflict_mask = (advantages < 0) & (advantages_s > 0)
        conflict_mask = (gae_r < 0) & (gae_s > 0)

        weights_s = jnp.where(
            conflict_mask,
            -2.0 * config["SENSITIVITY_REGULARIZATION_WEIGHT"],
            config["SENSITIVITY_REGULARIZATION_WEIGHT"],
        )
        # weights_s = config["SENSITIVITY_REGULARIZATION_WEIGHT"]  # Penalize positive and negative sensitivity, but stronger for negative (less sensitive than expected)

        # gae = jnp.where(gae_r > 0, gae_r + config["SENSITIVITY_REGULARIZATION_WEIGHT"] * gae_s, gae_r + 2.0 * config["SENSITIVITY_REGULARIZATION_WEIGHT"] * gae_s)
        # gae = jnp.where(gae_r < 0, gae_r + 1.0 * config["SENSITIVITY_REGULARIZATION_WEIGHT"] * gae_s, gae)
        # gae = jnp.where((gae_r < 0) & (gae_s < 0), - gae_r * config["SENSITIVITY_REGULARIZATION_WEIGHT"] * gae_s, gae)

        gae = gae_r + (weights_s * gae_s)
        # re-normalize combined advantage
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

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


def make_loss_fn_vs(config, graphdef_vs):
    """Create the loss function of the recurrent sensitivity critic.

    Args:
        config (dict): Training configuration (``CLIP_EPS``, ``VF_COEF``).
        graphdef_vs (nnx.GraphDef): Graph definition of the ``CriticGRU`` network.

    Returns:
        Callable: ``loss_fn_vs(params_vs, traj_batch, targets_s) -> value_loss_s``.
    """

    def _loss_fn_vs(params_vs, traj_batch, targets_s):
        # Process batch through GRU
        model_vs = nnx.merge(graphdef_vs, params_vs)
        value_s, _ = model_vs(
            traj_batch.obs,
            traj_batch.extras["gru_hidden"],
        )

        # CALCULATE VALUE SENSITIVITY LOSS
        value_s_pred_clipped = traj_batch.extras["value_s"] + (
            value_s - traj_batch.extras["value_s"]
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_s_losses = jnp.square(value_s - targets_s)
        value_s_losses_clipped = jnp.square(value_s_pred_clipped - targets_s)
        value_loss_s = config["VF_COEF"] * (
            0.5 * jnp.maximum(value_s_losses, value_s_losses_clipped).mean()
        )
        return value_loss_s

    return _loss_fn_vs


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
    graphdef_vs = networks["graphdef_vs"]

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        model = nnx.merge(graphdef, runner_state.train_state.params)
        pi, value = model(runner_state.last_obs)
        model_vs = nnx.merge(graphdef_vs, runner_state.extras["train_state_vs"].params)
        value_s, new_gru_hidden = model_vs(
            runner_state.last_obs, runner_state.extras["gru_hidden"]
        )
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )
        reward_s = info.get("sensitivity_state_rw", jnp.zeros_like(reward))

        # Reset GRU hidden on episode end
        new_gru_hidden = (1 - done.reshape(-1, 1)) * new_gru_hidden + done.reshape(
            -1, 1
        ) * jnp.zeros_like(new_gru_hidden)

        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        new_gru_hidden = jax.lax.stop_gradient(new_gru_hidden)

        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            runner_state.last_obs,
            info,
            extras={
                "value_s": value_s,
                "reward_s": reward_s,
                "gru_hidden": runner_state.extras["gru_hidden"],
            },
        )
        new_runner_state = RunnerState(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            extras={**runner_state.extras, "gru_hidden": new_gru_hidden},
        )
        return new_runner_state, transition

    return _env_step


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create the function that runs one SPPO-aug update epoch over the minibatches.

    Args:
        config (dict): Training configuration (``NUM_MINIBATCHES``, ``UPDATE_EPOCHS``).
        loss_fn (Callable): Loss function returned by ``make_loss_fn``.
        shuffle_batch_fn (Callable): Function that shuffles a batch and splits it into minibatches.
        networks (dict): Output of ``init_networks``.

    Returns:
        Callable: ``update_epoch(update_state, unused) -> (update_state, losses)``,
        for use with ``jax.lax.scan``.
    """
    loss_fn_vs = networks["loss_fn_vs"]

    def _update_epoch(update_state: UpdateState, unused):
        def _update_minbatch(ts, batch_info):
            train_state, train_state_vs = ts
            traj_batch, advantages, advantages_s, targets, targets_s = batch_info

            # Update main network
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, advantages_s, targets
            )
            train_state = train_state.apply_gradients(grads=grads)

            # Update sensitivity value network
            grad_fn_vs = jax.value_and_grad(loss_fn_vs, has_aux=False)
            total_loss_vs, grads_vs = grad_fn_vs(
                train_state_vs.params, traj_batch, targets_s
            )
            train_state_vs = train_state_vs.apply_gradients(grads=grads_vs)

            return (train_state, train_state_vs), total_loss + (total_loss_vs,)

        rng, _rng = jax.random.split(update_state.rng)
        minibatches = shuffle_batch_fn(
            _rng,
            (
                update_state.traj_batch,
                update_state.advantages,
                update_state.extras["advantages_s"],
                update_state.targets,
                update_state.extras["targets_s"],
            ),
            config,
        )
        (train_state, train_state_vs), total_loss = jax.lax.scan(
            _update_minbatch,
            (update_state.train_state, update_state.extras["train_state_vs"]),
            minibatches,
        )
        new_update_state = UpdateState(
            train_state=train_state,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            targets=update_state.targets,
            rng=rng,
            extras={**update_state.extras, "train_state_vs": train_state_vs},
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
        tuple[dict, RunnerState]: The dict with ``advantages``, ``advantages_s``, ``targets`` and ``targets_s``, and the runner state.
    """

    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    _, last_val = model(collect_state.last_obs)
    model_vs = nnx.merge(
        networks["graphdef_vs"], collect_state.extras["train_state_vs"].params
    )
    last_val_s, _ = model_vs(
        collect_state.last_obs,
        collect_state.extras["gru_hidden"],
    )
    advantages, advantages_s, targets, targets_s = gae_sensitivity(
        config, traj_batch, last_val, last_val_s
    )
    return {
        "advantages": advantages,
        "advantages_s": advantages_s,
        "targets": targets,
        "targets_s": targets_s,
    }, collect_state


def extract_losses(loss_info):
    """Turn the loss information of an update into scalar metrics for logging.

    Args:
        loss_info (tuple): Loss values returned by the update epochs.

    Returns:
        tuple[dict, dict | None]: The standard PPO metrics and the algorithm-specific ones.
    """
    (
        total_losses,
        (
            value_losses,
            actor_losses,
            entropies,
            old_approx_kl,
            approx_kl,
            clipfracs,
        ),
        vs_losses,
    ) = loss_info
    losses = {
        "total": total_losses.mean(),
        "value": value_losses.mean(),
        "actor": actor_losses.mean(),
        "entropy": entropies.mean(),
        "old_approx_kl": old_approx_kl.mean(),
        "approx_kl": approx_kl.mean(),
        "clipfrac": jnp.concatenate(clipfracs).mean(),
    }
    extra_losses = {"vs_loss": vs_losses.mean()}
    return losses, extra_losses


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Create the networks and their training states.

    Args:
        config (dict): Training configuration (layer sizes, activation, learning rate).
        env (GymnaxWrapper): Wrapped environment, used for the observation and action sizes.
        env_params (EnvParams): Environment parameters.
        rng (jax.Array): PRNG key for the initialisation.
        load_params_fn (Callable, optional): Function that loads pre-trained actor weights.

    Returns:
        dict: ``graphdef``, ``train_state``, ``graphdef_vs``, ``train_state_vs``, ``loss_fn_vs`` and ``gru_hidden_size``.
    """
    gru_hidden_size = config.get("GRU_HIDDEN_SIZE", 64)
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).shape[0]

    # Main policy network
    network = ActorCritic(
        obs_dim,
        action_dim,
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        rngs=nnx.Rngs(int(rng[0])),
    )

    # Optionally load pre-trained actor weights (keeps fresh critic)
    network = init_network_params(network, config)

    graphdef, state = nnx.split(network)

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=None,
        params=state,
        tx=tx,
    )

    # Secondary sensitivity value network (GRU-based)
    rng, vs_rng = jax.random.split(rng)
    network_vs = CriticGRU(
        obs_dim,
        activation=config["ACTIVATION"],
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        gru_hidden_size=gru_hidden_size,
        rngs=nnx.Rngs(int(vs_rng[0])),
    )

    graphdef_vs, state_vs = nnx.split(network_vs)

    tvx = create_optimizer(config, linear_schedule)
    train_state_vs = TrainState.create(
        apply_fn=None,
        params=state_vs,
        tx=tvx,
    )

    loss_fn_vs = make_loss_fn_vs(config, graphdef_vs)

    return {
        "graphdef": graphdef,
        "train_state": train_state,
        "graphdef_vs": graphdef_vs,
        "train_state_vs": train_state_vs,
        "loss_fn_vs": loss_fn_vs,
        "gru_hidden_size": gru_hidden_size,
    }


def init_runner_state(config, env, env_params, networks, rng):
    """Reset the environments and build the initial runner state.

    Args:
        config (dict): Training configuration (``NUM_ENVS``).
        env (GymnaxWrapper): Wrapped environment.
        env_params (EnvParams): Environment parameters.
        networks (dict): Output of ``init_networks``.
        rng (jax.Array): PRNG key.

    Returns:
        RunnerState: Initial runner state, with the algorithm-specific entries in ``extras``.
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    gru_hidden_size = networks["gru_hidden_size"]
    init_gru_hidden = jnp.zeros((config["NUM_ENVS"], gru_hidden_size))

    # Extra split before scan
    rng, _rng = jax.random.split(rng)

    # Dummy GAE extras for pytree consistency (overwritten each iteration)
    dummy_gae = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]))

    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,
        extras={
            "train_state_vs": networks["train_state_vs"],
            "gru_hidden": init_gru_hidden,
            "advantages_s": dummy_gae,
            "targets_s": dummy_gae,
        },
    )


SPEC = AlgorithmSpec(
    algo_name="SPPO_AUG",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
    extract_losses=extract_losses,
    init_runner_state=init_runner_state,
    make_eval_step_fn=make_eval_step_fn,
    build_eval_state=default_build_eval_state,
    uses_sensitivity=True,
)
