"""SPPO-grad: SPPO with gradient-based sensitivity regularisation.

Extends SPPO by adding a differentiable sensitivity loss to the PPO objective.
The environment wrapper already computes the sensitivity norm and folds it into
the reward, but that gradient signal only reaches the policy indirectly through
advantage estimation.  Here we add a direct loss term that recomputes the
sensitivity update using the *current* policy Jacobian dμ/dobs (differentiable)
while keeping the environment Jacobians (df/dx, df/du, df/dp, dobs/dx) as
stop-gradiented constants collected during rollouts.

Shared infrastructure (GAE, network init, runner state) is reused from
``sppo``; only ``wrap_env`` (with ``expose_jacobians=True``), the loss
function, and the collect function are overridden.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .base import AlgorithmSpec
from .ppo import make_update_fn
from .sppo import (
    calculate_gae,
    init_networks,
    init_runner_state,
    build_eval_state,
    make_eval_step_fn,
)
from buffers import Transition, RunnerState
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
    ConstraintEvalWrapper,
    ConstraintPenaltyWrapper,
)


# ---------------------------------------------------------------------------
# Environment wrapping  (expose_jacobians=True for the grad loss)
# ---------------------------------------------------------------------------

_JAC_KEYS = ("sens_bias", "df_du", "dobs_dx_sens")


def wrap_env(env, config):
    """Same wrapper stack as SPPO but with Jacobians exposed in info."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)

    # Optional constraint penalty wrapper
    if config.get("CONSTRAINTS", False):
        env = ConstraintEvalWrapper(env)
        penalty = config.get("CONSTRAINT_PENALTY", 1.0)
        env = ConstraintPenaltyWrapper(env, penalty=penalty)

    env = SensitivityWrapper(env, expose_jacobians=True)

    # Store W in config so it can be used in the loss function
    config["SENS_W"] = env.W

    env = ComputeSensitivityReward(env)
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecRewardSensitivity(env, config["GAMMA"])

    return env


# ---------------------------------------------------------------------------
# Differentiable sensitivity loss
# ---------------------------------------------------------------------------


def _sensitivity_loss(graphdef, params, obs, sens_bias, df_du, dobs_dx_S, W):
    """Recompute the sensitivity norm differentiably through the policy.

    All environment Jacobians are treated as constants; only the policy
    Jacobian dμ/dobs carries gradients.

    The sensitivity update is:
        M = sens_bias  +  df/du · (dμ/dobs) · (dobs/dx·S)
    where sens_bias = df/dx·S + df/dp_norm (precomputed, no policy dependence).

    Returns -log(||M||_F + ε) per sample.
    """
    sens_bias = jax.lax.stop_gradient(sens_bias)
    df_du = jax.lax.stop_gradient(df_du)
    dobs_dx_S = jax.lax.stop_gradient(dobs_dx_S)
    W = jax.lax.stop_gradient(W)

    # Policy Jacobian per sample: (B, action_dim, obs_dim)
    def _actor_mean(o):
        return nnx.merge(graphdef, params).actor(o).loc

    J = jax.jacfwd(_actor_mean)(obs)

    # Full sensitivity matrix: (B, state_dim, param_dim)
    M = sens_bias + df_du @ J @ dobs_dx_S

    # Apply weight matrix
    M_weighted = W @ M

    return -jnp.log(jnp.linalg.matrix_norm(M_weighted, ord="fro") + 1e-6)


sensitivity_loss = jax.vmap(_sensitivity_loss, in_axes=(None, None, 0, 0, 0, 0, None))

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def make_loss_fn(config, graphdef):
    """PPO clipped loss + direct sensitivity gradient term."""

    def _loss_fn(params, traj_batch, advantages, targets):
        model = nnx.merge(graphdef, params)
        pi, value, _ = model(traj_batch.obs, traj_batch.extras["gru_hidden"])
        log_prob = pi.log_prob(traj_batch.action)

        # ---- value loss (clipped) ----
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_loss = (
            0.5
            * jnp.maximum(
                jnp.square(value - targets),
                jnp.square(value_pred_clipped - targets),
            ).mean()
        )

        # ---- actor loss (clipped PPO) ----
        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)
        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        sens_new = sensitivity_loss(
            graphdef,
            params,
            traj_batch.obs,
            traj_batch.extras["sens_bias"],
            traj_batch.extras["df_du"],
            traj_batch.extras["dobs_dx_S"],
            config.get("SENS_W", jnp.eye(traj_batch.extras["sens_bias"].shape[-2])),
        )
        sens_new = sens_new / traj_batch.extras["sens_norm_factor"]
        sens_old = traj_batch.extras["sens_old"]
        sens_per_sample = sens_new - sens_old

        loss_actor = -jnp.minimum(
            ratio
            * (gae + config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sens_per_sample),
            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
            * (gae + config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sens_per_sample),
        ).mean()
        entropy = pi.entropy().mean()

        # ---- differentiable sensitivity loss ----
        # ratio is stop-gradiented so only the sensitivity term carries grads
        total_loss = (
            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        )

        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfracs = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])

        return total_loss, (
            value_loss,
            loss_actor,
            entropy,
            old_approx_kl,
            approx_kl,
            clipfracs,
        )

    return _loss_fn


# ---------------------------------------------------------------------------
# Collect function (stores environment Jacobians for the sensitivity loss)
# ---------------------------------------------------------------------------


def make_collect_fn(config, env, env_params, networks):
    """SPPO collect step that also stores the environment Jacobians in extras."""
    graphdef = networks["graphdef"]

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)

        model = nnx.merge(graphdef, runner_state.train_state.params)
        pi, value, new_gru_hidden = model(
            runner_state.last_obs,
            runner_state.extras["gru_hidden"],
        )
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )

        sensitivity_reward = info.get("sensitivity_state_rw", jnp.zeros_like(reward))
        reward += config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sensitivity_reward

        new_gru_hidden = jnp.where(done[:, None], 0.0, new_gru_hidden)

        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        new_gru_hidden = jax.lax.stop_gradient(new_gru_hidden)

        # Move the Jacobians to extras for the loss and drop them from info
        info_clean = {k: v for k, v in info.items() if k not in _JAC_KEYS}

        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            runner_state.last_obs,
            info_clean,
            extras={
                "gru_hidden": runner_state.extras["gru_hidden"],
                "sens_bias": info[
                    "sens_bias"
                ],  # df/dx@S + df/dp_norm (precomputed in wrapper)
                "df_du": info["df_du"],  # df/du
                "dobs_dx_S": info["dobs_dx_sens"],  # dobs/dx @ S
                "sens_old": sensitivity_reward,  # -log(||M_old||_F + eps) from rollout
                "sens_norm_factor": info.get("sens_norm_factor", jnp.ones_like(reward)),
            },
        )
        new_runner_state = RunnerState(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            extras={"gru_hidden": new_gru_hidden},
        )
        return new_runner_state, transition

    return _env_step


# ---------------------------------------------------------------------------
# Algorithm specification
# ---------------------------------------------------------------------------

SPEC = AlgorithmSpec(
    algo_name="SPPO_GRAD",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
    init_runner_state=init_runner_state,
    build_eval_state=build_eval_state,
    make_eval_step_fn=make_eval_step_fn,
    uses_sensitivity=True,
)
