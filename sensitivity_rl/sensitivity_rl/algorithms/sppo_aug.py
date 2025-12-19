"""SPPO with augmented sensitivity estimation (GRU-based value function)."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .base import AlgorithmSpec
from buffers import RolloutSensAugTransition, RunnerStateDual, UpdateStateDual
from algorithms.common import gae_sensitivity
from net_utils import create_optimizer, make_linear_schedule
from networks import ActorCritic, ActorCriticSensAug
from envs.wrappers import (
    GymnaxWrapper,
    ClipAction,
    SensitivityWrapper,
    ComputeSensitivityReward,
    MujocoSensitivityWrapper,
    ResetEnvWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecRewardSensitivity,
    NormalizeVecReward,
)


def wrap_env(env, config):
    """Apply SPPO-aug specific environment wrappers."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    
    if env.type == "mujoco":
        env = MujocoSensitivityWrapper(env)
    else:
        env = SensitivityWrapper(env)
    env = ComputeSensitivityReward(env)
    env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecRewardSensitivity(env, config["GAMMA"])

    return env


def make_loss_fn(config, network):
    """Create SPPO-aug loss function for the main policy network.
    
    Returns a function with signature:
        loss_fn(params, traj_batch, advantages, advantages_s, targets) -> (total_loss, aux_metrics)
    """
    def _loss_fn(params, traj_batch, advantages, advantages_s, targets):
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

        # Normalize and combine advantages
        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        gae_s = (advantages_s - advantages_s.mean()) / (advantages_s.std() + 1e-8)
        gae = gae + config["SENSITIVITY_REGULARIZATION_WEIGHT"] * gae_s
        
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


def make_loss_fn_vs(config, network_vs):
    """Create loss function for the sensitivity value network (GRU-based).
    
    Returns a function with signature:
        loss_fn_vs(params_vs, traj_batch, targets_s) -> value_loss_s
    """
    def _loss_fn_vs(params_vs, traj_batch, targets_s):
        # Process batch through GRU
        value_s, _ = network_vs.apply(
            params_vs,
            traj_batch.obs,
            traj_batch.gru_hidden,
        )

        # CALCULATE VALUE SENSITIVITY LOSS
        value_s_pred_clipped = traj_batch.value_s + (
            value_s - traj_batch.value_s
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_s_losses = jnp.square(value_s - targets_s)
        value_s_losses_clipped = jnp.square(value_s_pred_clipped - targets_s)
        value_loss_s = config["VF_COEF"] * (
            0.5 * jnp.maximum(value_s_losses, value_s_losses_clipped).mean()
        )
        return value_loss_s
    
    return _loss_fn_vs


def make_collect_fn(config, env, env_params, networks):
    """Create trajectory collection function for SPPO-aug.
    
    Args:
        config: Training configuration
        env: Wrapped environment
        env_params: Environment parameters
        networks: Dict returned by init_networks containing 'network', 'network_vs', etc.
    
    Returns a function for use with jax.lax.scan that collects transitions.
    """
    network = networks["network"]
    network_vs = networks["network_vs"]
    
    def _env_step(runner_state: RunnerStateDual, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        pi, value = network.apply(runner_state.train_state.params, runner_state.last_obs)
        value_s, new_gru_hidden = network_vs.apply(
            runner_state.train_state_vs.params, runner_state.last_obs, runner_state.gru_hidden
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
        new_gru_hidden = (1 - done.reshape(-1, 1)) * new_gru_hidden + done.reshape(-1, 1) * jnp.zeros_like(new_gru_hidden)
        
        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        new_gru_hidden = jax.lax.stop_gradient(new_gru_hidden)

        transition = RolloutSensAugTransition(
            done, action, value, value_s, reward, reward_s,
            log_prob, runner_state.last_obs, runner_state.gru_hidden, info
        )
        new_runner_state = RunnerStateDual(
            train_state=runner_state.train_state,
            train_state_vs=runner_state.train_state_vs,
            env_state=env_state,
            last_obs=obsv,
            gru_hidden=new_gru_hidden,
            rng=rng,
        )
        return new_runner_state, transition
    
    return _env_step


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create update epoch function for SPPO-aug.
    
    Args:
        config: Training configuration
        loss_fn: Loss function created by make_loss_fn
        shuffle_batch_fn: Function to shuffle and batch trajectories
        networks: Dict returned by init_networks containing 'loss_fn_vs', etc.
    
    Returns a function for use with jax.lax.scan that performs one epoch of updates.
    """
    loss_fn_vs = networks["loss_fn_vs"]
    def _update_epoch(update_state: UpdateStateDual, unused):
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
            (update_state.traj_batch, update_state.advantages, update_state.advantages_s, update_state.targets, update_state.targets_s),
            config,
        )
        (train_state, train_state_vs), total_loss = jax.lax.scan(
            _update_minbatch, (update_state.train_state, update_state.train_state_vs), minibatches
        )
        new_update_state = UpdateStateDual(
            train_state=train_state,
            train_state_vs=train_state_vs,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            advantages_s=update_state.advantages_s,
            targets=update_state.targets,
            targets_s=update_state.targets_s,
            rng=rng,
        )
        return new_update_state, total_loss
    
    return _update_epoch


def calculate_gae(config, traj_batch, last_val, networks, collect_state):
    """Calculate GAE for SPPO-aug (both standard and sensitivity)."""
    network_vs = networks["network_vs"]
    last_val_s, _ = network_vs.apply(
        collect_state.train_state_vs.params,
        collect_state.last_obs,
        collect_state.gru_hidden,
    )
    advantages, advantages_s, targets, targets_s = gae_sensitivity(
        config, traj_batch, last_val, last_val_s
    )
    return {
        "advantages": advantages,
        "advantages_s": advantages_s,
        "targets": targets,
        "targets_s": targets_s,
    }


def pack_update_state(collect_state, traj_batch, gae_results):
    """Pack state for update epochs."""
    return UpdateStateDual(
        train_state=collect_state.train_state,
        train_state_vs=collect_state.train_state_vs,
        traj_batch=traj_batch,
        advantages=gae_results["advantages"],
        advantages_s=gae_results["advantages_s"],
        targets=gae_results["targets"],
        targets_s=gae_results["targets_s"],
        rng=collect_state.rng,
    )


def rebuild_runner_state(update_state: UpdateStateDual, collect_state: RunnerStateDual):
    """Rebuild runner state after update epochs."""
    return RunnerStateDual(
        train_state=update_state.train_state,
        train_state_vs=update_state.train_state_vs,
        env_state=collect_state.env_state,
        last_obs=collect_state.last_obs,
        gru_hidden=collect_state.gru_hidden,
        rng=update_state.rng,
    )


def extract_losses(loss_info):
    """Extract loss metrics from loss_info."""
    total_losses, (
        value_losses, actor_losses, entropies,
        old_approx_kl, approx_kl, clipfracs,
    ), vs_losses = loss_info
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
    """Initialize networks and train states for SPPO-aug.
    
    Returns:
        dict with keys: network, train_state, network_vs, train_state_vs, loss_fn_vs, gru_hidden_size
    """
    gru_hidden_size = config.get("GRU_HIDDEN_SIZE", 64)
    
    # Main policy network
    network = ActorCritic(
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
    )
    
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    
    if load_params_fn and config.get("LOAD_ACTOR_PARAMS_PATH"):
        network_params = load_params_fn(rng, network, init_x, config)
    else:
        network_params = network.init(rng, init_x)
    
    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    # Secondary sensitivity value network (GRU-based)
    network_vs = ActorCriticSensAug(
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        gru_hidden_size=gru_hidden_size,
    )
    init_gru_hidden = jnp.zeros((1, gru_hidden_size))
    network_vs_params = network_vs.init(rng, init_x, init_gru_hidden)
    
    tvx = create_optimizer(config, linear_schedule)
    train_state_vs = TrainState.create(
        apply_fn=network_vs.apply,
        params=network_vs_params,
        tx=tvx,
    )
    
    loss_fn_vs = make_loss_fn_vs(config, network_vs)
    
    return {
        "network": network,
        "train_state": train_state,
        "network_vs": network_vs,
        "train_state_vs": train_state_vs,
        "loss_fn_vs": loss_fn_vs,
        "gru_hidden_size": gru_hidden_size,
    }


def init_runner_state(config, env, env_params, networks, rng):
    """Create initial runner state for SPPO-aug.
    
    Returns:
        RunnerStateDual NamedTuple
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)
    
    gru_hidden_size = networks["gru_hidden_size"]
    init_gru_hidden = jnp.zeros((config["NUM_ENVS"], gru_hidden_size))
    
    # Extra split before scan (matching original sppo_aug.py)
    rng, _rng = jax.random.split(rng)
    
    return RunnerStateDual(
        train_state=networks["train_state"],
        train_state_vs=networks["train_state_vs"],
        env_state=env_state,
        last_obs=obsv,
        gru_hidden=init_gru_hidden,
        rng=_rng,  # Use _rng, not rng, to match original
    )




SPEC = AlgorithmSpec(
    name="sppo_aug",
    transition_cls=RolloutSensAugTransition,
    make_loss_fn=make_loss_fn,
    wrap_env=wrap_env,
    network_cls=ActorCritic,  # Main network
    uses_sensitivity=True,
    gae_fn=gae_sensitivity,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    pack_update_state=pack_update_state,
    rebuild_runner_state=rebuild_runner_state,
    extract_losses=extract_losses,
    init_networks=init_networks,
    init_runner_state=init_runner_state,
)
