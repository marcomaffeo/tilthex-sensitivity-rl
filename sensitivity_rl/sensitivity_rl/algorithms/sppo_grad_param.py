"""SPPO with GRU-based ActorCritic (single network with hidden state for critic)."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from .base import AlgorithmSpec
from algorithms.common import gae_standard
from flax import traverse_util
from buffers import Transition, RunnerState, UpdateState
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
    env = ComputeSensitivityReward(
        env
    )  # Computes sensitivity norm but does not modify reward
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
        pi, value, _ = network.apply(
            params, traj_batch.obs, traj_batch.extras["gru_hidden"]
        )
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

        ratio_stopped = jax.lax.stop_gradient(ratio)

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

        # Sensitivity gradient loss via JVPs (avoids jacobian-of-jacobian).
        # df_du and dobs_dx_sens are stored constants from the rollout;
        # only the policy Jacobian (action_gains) is recomputed with current
        # params so that grad(loss) flows through the actor network.
        def _single_sensitivity_loss(obs_i, df_dx_i, df_dp_i, df_du_i, dobs_dx_sens_i):
            df_du_i = jax.lax.stop_gradient(df_du_i)
            dobs_dx_sens_i = jax.lax.stop_gradient(dobs_dx_sens_i)
            df_dp_i = jax.lax.stop_gradient(df_dp_i)
            df_dx_i = jax.lax.stop_gradient(df_dx_i)

            def mu_fn(o):
                pi = network.apply(params, o, method=network.actor)
                return pi.loc  # mean action

            def compute_col(tangent):
                _, jvp_out = jax.jvp(mu_fn, (obs_i,), (tangent,))
                return df_du_i @ jvp_out

            # Each column of dobs_dx_sens_i is a tangent direction
            M = jax.vmap(compute_col)(dobs_dx_sens_i.T)  # (param_dim, state_dim)
            M = df_dx_i + M.T + df_dp_i
            # Log for monitoring, but don't backprop through log
            # Use squared Frobenius norm to avoid NaN gradient when M ≈ 0
            # (grad of sqrt(x) at x=0 is undefined; grad of x is always 1)
            return jnp.log(jnp.sum(M**2) + 1e-6)

        sensitivity_loss = (
            ratio_stopped
            * jax.vmap(_single_sensitivity_loss)(
                traj_batch.obs,
                traj_batch.extras["df_dx"],
                traj_batch.extras["df_dp"],
                traj_batch.extras["df_du"],
                traj_batch.extras["dobs_dx_sens"],
            )
        ).mean()

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
            + config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sensitivity_loss
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

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)

        # Full forward pass (need hidden state for critic and to update it)
        pi, value, new_gru_hidden = network.apply(
            runner_state.train_state.params,
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

        # Add sensitivity regularization to reward at the end so that it is normalized
        sensitivity_reward = info.get("sensitivity_state_rw", jnp.zeros_like(reward))
        reward += config["SENSITIVITY_REGULARIZATION_WEIGHT"] * sensitivity_reward

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
                "gru_hidden": runner_state.extras["gru_hidden"],
                "sensitivity_reward": sensitivity_reward,
                "df_dx": info["df_dx"],
                "df_dp": info["df_dp"],
                "df_du": info["df_du"],
                "dobs_dx_sens": info["dobs_dx_sens"],
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


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create update epoch function for PPO.

    Args:
        config: Training configuration
        loss_fn: Loss function created by make_loss_fn
        shuffle_batch_fn: Function to shuffle and batch trajectories
        networks: Dict returned by init_networks (unused in PPO but needed for interface)

    Returns a function for use with jax.lax.scan that performs one epoch of updates.
    """

    def _update_epoch(update_state: UpdateState, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state_new = train_state.apply_gradients(grads=grads)

            # 2. Flatten the parameter dictionaries to easily filter by your layer names
            flat_new_params = traverse_util.flatten_dict(train_state_new.params)
            flat_old_params = traverse_util.flatten_dict(train_state.params)

            tau = config.get("TAU", 0.05)
            flat_blended_params = {}

            # 3. Iterate through the parameters and selectively apply the Polyak update
            for key in flat_new_params.keys():
                # 'key' is a tuple of strings representing the path, e.g., ('actor_dense1', 'kernel')
                is_actor_param = any("actor" in node for node in key)

                if is_actor_param:
                    # Soft update for the actor
                    flat_blended_params[key] = (
                        tau * flat_new_params[key] + (1.0 - tau) * flat_old_params[key]
                    )
                else:
                    # Hard update for the critic (use the completely new weights)
                    flat_blended_params[key] = flat_new_params[key]

            # 4. Unflatten back into the standard Flax nested dictionary format
            blended_params = traverse_util.unflatten_dict(flat_blended_params)

            # 5. Replace the parameters in the state
            final_train_state = train_state_new.replace(params=blended_params)
            return final_train_state, total_loss

        rng, _rng = jax.random.split(update_state.rng)
        minibatches = shuffle_batch_fn(
            _rng,
            (update_state.traj_batch, update_state.advantages, update_state.targets),
            config,
        )
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, update_state.train_state, minibatches
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
    """Calculate GAE for SPPO-aug (both standard and sensitivity)."""

    last_val, _ = networks["network"].apply(
        collect_state.train_state.params,
        collect_state.last_obs,
        collect_state.extras["gru_hidden"],
        method=networks["network"].critic,
    )

    advantages, targets = gae_standard(
        config,
        traj_batch,
        last_val,
    )
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


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
        RunnerState NamedTuple
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    gru_hidden_size = networks["gru_hidden_size"]
    init_gru_hidden = jnp.zeros((config["NUM_ENVS"], gru_hidden_size))

    # Extra split before scan
    rng, _rng = jax.random.split(rng)

    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,
        extras={"gru_hidden": init_gru_hidden},
    )


SPEC = AlgorithmSpec(
    algo_name="SPPO_GRAD_PARAM",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
    init_runner_state=init_runner_state,
    uses_sensitivity=True,
)
