"""SPPO with GRU-based ActorCritic (single network with hidden state for critic)."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .base import AlgorithmSpec
from .ppo import extract_losses, make_update_fn, pack_update_state  # Reuse from PPO
from algorithms.common import gae_standard
from buffers import RolloutGRUTransition, RunnerStateGRU, UpdateState
from net_utils import create_optimizer, make_linear_schedule
from networks import ActorCriticGRU
from envs.wrappers import (
    GymnaxWrapper,
    ClipAction,
    SensitivityWrapper,
    AddSensitivityToReward,
    ComputeSensitivityReward,
    DomainRandomizationWrapper,
    MujocoSensitivityWrapper,
    ResetEnvWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecRewardSensitivity,
)


def wrap_env(env, config):
    """Apply SPPO-specific environment wrappers."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    
    if env.type == "mujoco":
        env = MujocoSensitivityWrapper(env)
    else:
        env = SensitivityWrapper(env)
    # env = AddSensitivityToReward(env, config["SENSITIVITY_REGULARIZATION_WEIGHT"])  # Adds sensitivity penalty directly to reward
    env = ComputeSensitivityReward(env)  # Computes sensitivity norm but does not modify reward
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecRewardSensitivity(env, config["GAMMA"])

    return env


def make_loss_fn(config, network):
    """Create SPPO loss function with GRU-based ActorCritic.
    
    Args:
        network: ActorCriticGRU network instance
    
    Returns a function with signature:
        loss_fn(params, traj_batch, advantages, targets) -> (total_loss, aux_metrics)
    """
    def _loss_fn(params, traj_batch, advantages, targets):
        # Full forward pass (need hidden state for critic value)
        pi, value, _ = network.apply(params, traj_batch.obs, traj_batch.gru_hidden)
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

        # Normalize advantages
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



def make_collect_fn(config, env, env_params, networks):
    """Create trajectory collection function for SPPO.
    
    Args:
        config: Training configuration
        env: Wrapped environment
        env_params: Environment parameters
        networks: Dict returned by init_networks containing 'network', etc.
    
    Returns a function for use with jax.lax.scan that collects transitions.
    """
    network = networks["network"]
    
    def _env_step(runner_state: RunnerStateGRU, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        
        # Full forward pass (need hidden state for critic and to update it)
        pi, value, new_gru_hidden = network.apply(
            runner_state.train_state.params, 
            runner_state.last_obs, 
            runner_state.gru_hidden
        )
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )
        
        # Add sensitivity regularization to reward at the end so that it is normalized
        sensitivity_reward = info.get("sensitivity_state_rw", jnp.zeros_like(reward))
        reward += config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sensitivity_reward

        # Reset GRU hidden on episode end
        new_gru_hidden = (1 - done.reshape(-1, 1)) * new_gru_hidden + done.reshape(-1, 1) * jnp.zeros_like(new_gru_hidden)
        
        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        new_gru_hidden = jax.lax.stop_gradient(new_gru_hidden)

        transition = RolloutGRUTransition(
            done, action, value, reward,
            log_prob, runner_state.last_obs, runner_state.gru_hidden, info
        )
        new_runner_state = RunnerStateGRU(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            gru_hidden=new_gru_hidden,
            rng=rng,
        )
        return new_runner_state, transition
    
    return _env_step

def calculate_gae(config, traj_batch, networks, collect_state):
    """Calculate GAE for SPPO-aug (both standard and sensitivity)."""

    last_val, _ = networks["network"].apply(collect_state.train_state.params, collect_state.last_obs, collect_state.gru_hidden, method=networks["network"].critic)
    
    advantages, targets = gae_standard(
        config, traj_batch, last_val,
    )
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state

def rebuild_runner_state(update_state: UpdateState, collect_state: RunnerStateGRU):
    """Rebuild runner state after update epochs."""
    return RunnerStateGRU(
        train_state=update_state.train_state,
        env_state=collect_state.env_state,
        last_obs=collect_state.last_obs,
        gru_hidden=collect_state.gru_hidden,
        rng=update_state.rng,
    )


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Initialize GRU-based ActorCritic network for SPPO.
    
    Returns:
        dict with keys: network, train_state, gru_hidden_size
    """
    gru_hidden_size = config.get("GRU_HIDDEN_SIZE", 64)
    
    network = ActorCriticGRU(
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        gru_hidden_size=gru_hidden_size,
    )
    
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    init_gru_hidden = jnp.zeros((1, gru_hidden_size))
    
    if load_params_fn and config.get("LOAD_ACTOR_PARAMS_PATH"):
        # For GRU network, we need custom loading logic
        network_params = network.init(rng, init_x, init_gru_hidden)
    else:
        network_params = network.init(rng, init_x, init_gru_hidden)
    
    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    return {
        "network": network,
        "train_state": train_state,
        "gru_hidden_size": gru_hidden_size,
    }


def init_runner_state(config, env, env_params, networks, rng):
    """Create initial runner state for SPPO.
    
    Returns:
        RunnerStateGRU NamedTuple
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)
    
    gru_hidden_size = networks["gru_hidden_size"]
    init_gru_hidden = jnp.zeros((config["NUM_ENVS"], gru_hidden_size))
    
    # Extra split before scan
    rng, _rng = jax.random.split(rng)
    
    return RunnerStateGRU(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        gru_hidden=init_gru_hidden,
        rng=_rng,
    )

def run_dir(config, runs_root, ts):
    algo_name = "SPPO_AUG"
    if config.get("DOMAIN_RANDOMIZATION", False):
        run_dir = runs_root / config["ENV_NAME"] / f"{algo_name}_DR_{ts}"
    else:
        run_dir = runs_root / config["ENV_NAME"] / f"{algo_name}_{ts}"
    
    return run_dir

SPEC = AlgorithmSpec(
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    pack_update_state=pack_update_state,
    rebuild_runner_state=rebuild_runner_state,
    extract_losses=extract_losses,
    init_networks=init_networks,
    init_runner_state=init_runner_state,
    run_dir=run_dir,
    uses_sensitivity=True,
)
