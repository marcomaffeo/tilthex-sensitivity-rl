"""CPPO algorithm with CVaR."""

import jax
import jax.numpy as jnp
from flax import nnx

from .ppo import make_eval_step_fn
from .base import AlgorithmSpec, default_build_eval_state
from algorithms.common import gae_standard
from buffers import Transition, RunnerState, UpdateState
from .ppo import init_networks  # Same network setup as PPO
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
    """Apply the CPPO wrapper stack to an environment.

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
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    return env


def make_loss_fn(config, graphdef):
    """Create the CPPO loss function of the policy network.

    Args:
        config (dict): Training configuration (clipping, loss coefficients).
        graphdef (nnx.GraphDef): Graph definition of the ActorCritic network.

    Returns:
        Callable: ``loss_fn(params, traj_batch, advantages, targets) -> (total_loss, aux)``, where ``aux``
        holds the value loss, actor loss, entropy and the KL and clipping statistics.
    """

    def _loss_fn(params, traj_batch, advantages, targets):
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


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create the function that runs one CPPO update epoch and updates ``nu`` and ``lambda``.

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
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        rng, _rng = jax.random.split(update_state.rng)

        # Standard PPO Update Loop
        minibatches = shuffle_batch_fn(
            _rng,
            (update_state.traj_batch, update_state.advantages, update_state.targets),
            config,
        )
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, update_state.train_state, minibatches
        )

        # CPPO variable updates (nu and lambda)
        # 1. Update Nu (VaR threshold)
        batch_avg_return = jnp.mean(update_state.targets)
        nu_lr = config.get("CVAR_NU_LR", 1e-3)
        nu_delay_factor = config.get("CVAR_NU_DELAY", 1.0)

        # Moving average update towards the batch mean return
        new_nu = (
            update_state.extras["nu"]
            + nu_lr * (batch_avg_return - update_state.extras["nu"]) * nu_delay_factor
        )

        # 2. Update Lambda (Penalty Weight)
        # Lambda increases if we violate the constraint (Risk > Beta)
        lam_lr = config.get("CVAR_LAM_LR", 1e-3)
        beta = config.get("CVAR_BETA", 0.0)  # Target safety threshold

        # If Nu (safe level) < Beta (target), increase penalty
        lam_delta = beta - new_nu
        new_cvarlam = update_state.extras["cvarlam"] + lam_lr * lam_delta

        # Ensure lambda stays positive and bounded
        new_cvarlam = jnp.clip(
            new_cvarlam, a_min=0.0, a_max=config.get("CVAR_LAM_MAX", 100.0)
        )

        new_update_state = UpdateState(
            train_state=train_state,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            targets=update_state.targets,
            rng=rng,
            extras={
                **update_state.extras,
                "nu": new_nu,
                "cvarlam": new_cvarlam,
            },
        )

        new_total_loss = (
            total_loss[0],
            total_loss[1]
            + (
                new_nu,
                new_cvarlam,
            ),
        )  # Augment loss for logging

        return new_update_state, new_total_loss

    return _update_epoch


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
        pi, value = model(runner_state.last_obs)
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
            done, action, value, reward, log_prob, runner_state.last_obs, info
        )

        new_runner_state = RunnerState(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            extras={
                "nu": runner_state.extras["nu"],
                "cvarlam": runner_state.extras["cvarlam"],
            },
        )
        return new_runner_state, transition

    return _env_step


def calculate_gae(config, traj_batch, networks, collect_state):
    """Compute the generalised advantage estimates of a rollout.

    Args:
        config (dict): Training configuration (``GAMMA``, ``GAE_LAMBDA``).
        traj_batch (Transition): Collected transitions, shape ``(NUM_STEPS, NUM_ENVS, ...)``.
        networks (dict): Output of ``init_networks``.
        collect_state (RunnerState): Runner state at the end of the rollout, used to bootstrap the last value.

    The advantages are reduced by the CVaR penalty
    ``lambda / (1 - alpha) * max(nu - return, 0)`` on the transitions whose return
    is below the VaR threshold ``nu``.

    Returns:
        tuple[dict, RunnerState]: The dict with the penalised ``advantages`` and the ``targets``, and the runner state.
    """
    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    _, last_val = model(collect_state.last_obs)

    # 1. Standard GAE Calculation
    advantages, targets = gae_standard(config, traj_batch, last_val)

    # 2. CPPO Logic: Calculate Risk Penalty
    # We use 'targets' (rewards-to-go) as the proxy for trajectory return D(xi)
    estimated_return = targets

    # Retrieve current CPPO params
    nu = collect_state.extras["nu"]
    cvarlam = collect_state.extras["cvarlam"]
    alpha = config.get("CVAR_ALPHA", 0.99)
    delay = config.get("CVAR_DELAY", 1.0)

    # Identify "risky" steps where return is less than VaR threshold (nu)
    # The penalty applies only when estimated_return < nu
    risk_mask = (estimated_return < nu).astype(jnp.float32)

    # Calculate penalty: lambda / (1 - alpha) * (nu - return)
    penalty_raw = (cvarlam / (1.0 - alpha)) * (nu - estimated_return)

    # Optional: Clip penalty magnitude relative to Value to prevent explosion
    cvar_clip_ratio = config.get("CVAR_CLIP_RATIO", 0.05)
    max_penalty = jnp.abs(traj_batch.value) * cvar_clip_ratio
    penalty = jnp.minimum(penalty_raw, max_penalty) * delay

    # Apply penalty only to risky transitions
    penalty_term = penalty * risk_mask

    # 3. Modify Advantages
    # The CPPO algorithm maximizes L - penalty, which is equivalent
    # to passing (Advantage - Penalty) to the standard PPO loss.
    cppo_advantages = advantages - penalty_term

    return {
        "advantages": cppo_advantages,
        "targets": targets,
    }, collect_state


def extract_losses(loss_info):
    """Turn the loss information of an update into scalar metrics for logging.

    Args:
        loss_info (tuple): Loss values returned by the update epochs.

    Returns:
        tuple[dict, dict]: The standard PPO metrics, and the mean ``nu`` and ``cvarlam``.
    """
    total_losses, (
        value_losses,
        actor_losses,
        entropies,
        old_approx_kl,
        approx_kl,
        clipfracs,
        nu,
        cvarlam,
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
    return losses, {"nu": nu.mean(), "cvarlam": cvarlam.mean()}  # losses, extra_losses


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

    rng, _rng = jax.random.split(rng)

    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,
        extras={
            "nu": config.get("CVAR_NU_START", 0.0),  # Initial VaR guess
            "cvarlam": config.get("CVAR_LAM_START", 0.5),  # Initial Penalty weight
        },
    )


SPEC = AlgorithmSpec(
    algo_name="CPPO",
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
)
