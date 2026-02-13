"""CPPO algorithm with CVaR."""

import jax
import jax.numpy as jnp

from .base import AlgorithmSpec
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
    MujocoSensitivityWrapper,
)


def wrap_env(env, config):
    """Apply PPO-specific environment wrappers."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    
    # Optional sensitivity wrapper (for online sensitivity tracking, no policy modification)
    if config.get("SENSITIVITY", False):
        if getattr(env, "type", "") == "mujoco":
            env = MujocoSensitivityWrapper(env)
        else:
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


def make_loss_fn(config, network):
    """Create PPO loss function.
    
    Returns a function with signature:
        loss_fn(params, traj_batch, advantages, targets) -> (total_loss, aux_metrics)
    """
    def _loss_fn(params, traj_batch, advantages, targets):
        # RERUN NETWORK
        pi, value = network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

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
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
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
    """Create update epoch function for CPPO.
    
    Updates Policy/Value networks AND CPPO variables (nu, cvarlam).
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
        minibatches = shuffle_batch_fn(_rng, (update_state.traj_batch, update_state.advantages, update_state.targets), config)
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, update_state.train_state, minibatches
        )

        # CPPO Variable Updates (Nu & Lambda) [cite: 437-438]
        # 1. Update Nu (VaR threshold)
        batch_avg_return = jnp.mean(update_state.targets) 
        nu_lr = config.get("CVAR_NU_LR", 1e-3)
        nu_delay_factor = config.get("CVAR_NU_DELAY", 1.0)
        
        # Moving average update towards the batch mean return
        new_nu = update_state.extras["nu"] + nu_lr * (batch_avg_return - update_state.extras["nu"]) * nu_delay_factor

        # 2. Update Lambda (Penalty Weight)
        # Lambda increases if we violate the constraint (Risk > Beta)
        lam_lr = config.get("CVAR_LAM_LR", 1e-3)
        beta = config.get("CVAR_BETA", 0.0) # Target safety threshold
        
        # If Nu (safe level) < Beta (target), increase penalty
        lam_delta = beta - new_nu
        new_cvarlam = update_state.extras["cvarlam"] + lam_lr * lam_delta
        
        # Ensure lambda stays positive and bounded
        new_cvarlam = jnp.clip(new_cvarlam, a_min=0.0, a_max=config.get("CVAR_LAM_MAX", 100.0))
        
        new_update_state = UpdateState(
            train_state=train_state,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            targets=update_state.targets,
            rng=rng,
            extras={"nu": new_nu, "cvarlam": new_cvarlam},
        )
        
        new_total_loss = (total_loss[0], total_loss[1] + (new_nu, new_cvarlam,) )  # Augment loss for logging

        return new_update_state, new_total_loss
    
    return _update_epoch


def make_collect_fn(config, env, env_params, networks):
    """Create trajectory collection function for CPPO."""
    network = networks["network"]
    
    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        pi, value = network.apply(runner_state.train_state.params, runner_state.last_obs)
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
            extras={"nu": runner_state.extras["nu"], "cvarlam": runner_state.extras["cvarlam"]},
        )
        return new_runner_state, transition
    
    return _env_step


def calculate_gae(config, traj_batch, networks, collect_state):
    """Calculate GAE and apply CPPO Risk Penalty to Advantages.
    
    This implements the CPPO Logic: Maximize (Advantage - Penalty)
    Penalty = lambda / (1 - alpha) * (nu - Return)+
    """
    _, last_val = networks["network"].apply(collect_state.train_state.params, collect_state.last_obs)
    
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
    # The penalty applies only when estimated_return < nu [cite: 167]
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
    """Extract loss metrics from loss_info."""
    total_losses, (
        value_losses, actor_losses, entropies,
        old_approx_kl, approx_kl, clipfracs, nu, cvarlam
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
    """Create initial runner state for CPPO with initial Nu and Lambda."""
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
            "nu": config.get("CVAR_NU_START", 0.0),       # Initial VaR guess
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
)
