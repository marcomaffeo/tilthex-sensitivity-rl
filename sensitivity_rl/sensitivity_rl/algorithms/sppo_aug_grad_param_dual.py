"""SPPO with augmented sensitivity estimation (GRU-based value function)."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState

from .ppo import make_eval_step_fn
from .base import AlgorithmSpec, default_build_eval_state
from buffers import Transition, RunnerState, UpdateState
from algorithms.common import gae_sensitivity_dual
from net_utils import create_optimizer, make_linear_schedule
from networks import ActorCritic, CriticGRU
from envs.wrappers import (
    GymnaxWrapper,
    ClipAction,
    SensitivityWrapper,
    ComputeSensitivityReward,
    DomainRandomizationWrapper,
    MujocoSensitivityWrapper,
    ResetEnvWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecRewardSensitivity,
    NormalizeVecReward,
    ConstraintEvalWrapper,
    ConstraintPenaltyWrapper,
)


def wrap_env(env, config):
    """Apply SPPO-aug specific environment wrappers."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)

    # Optional constraint penalty wrapper
    if config.get("CONSTRAINTS", False):
        env = ConstraintEvalWrapper(env)
        penalty = config.get("CONSTRAINT_PENALTY", 1.0)
        env = ConstraintPenaltyWrapper(env, penalty=penalty)

    if env.type == "mujoco":
        env = MujocoSensitivityWrapper(env, config=config, expose_jacobians=True)
    else:
        env = SensitivityWrapper(env, config=config, expose_jacobians=True)
    
    # Store W in config so it can be used in the loss function
    config["SENS_W"] = env.W

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


def _actor_param_leaves(params):
    """Return only actor-related parameter leaves in a stable traversal order."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params)
    return tuple(
        leaf
        for path, leaf in leaves_with_paths
        if any("actor" in str(path_entry) for path_entry in path)
    )


def _actor_l2_distance(params, old_actor_params):
    """Compute L2 distance using actor params only."""
    actor_params = _actor_param_leaves(params)
    sq_sums = []
    for new_param, old_param in zip(actor_params, old_actor_params):
        # old_param can carry extra leading rollout/minibatch dims; drop them.
        while old_param.ndim > new_param.ndim:
            old_param = old_param[0]
        sq_sums.append(jnp.sum(jnp.square(new_param - old_param)))
    return jnp.sqrt(jnp.sum(jnp.stack(sq_sums)) + 1e-12)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def make_loss_fn(config, graphdef):
    """Create SPPO-aug loss function for the main policy network.

    Returns a function with signature:
        loss_fn(params, traj_batch, advantages, advantages_s, targets) -> (total_loss, aux_metrics)
    """

    def _loss_fn(
        params, traj_batch, advantages, advantages_s, targets, old_actor_params
    ):
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
        sens_old = traj_batch.extras["reward_s"]
        sens_per_sample = sens_new - sens_old

        # Normalize and combine advantages
        weights_s = config["SENSITIVITY_REGULARIZATION_WEIGHT"]
        xi = config.get("XI", 0.08)
        rho = config.get("RHO", 0.3)
        delta_w = advantages.var() + xi * pi.entropy().mean()
        w = jnp.exp(
            -rho * jnp.maximum(jnp.array([0.0]), delta_w)
        )  # Higher sensitivity variance or lower entropy -> more weight on sensitivity

        gae = advantages + (weights_s * w * advantages_s)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        gae = gae + (weights_s * w * sens_per_sample)

        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )

        loss_actor = -jnp.minimum(loss_actor1, loss_actor2) + config[
            "TAU"
        ] * _actor_l2_distance(params, old_actor_params)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        )

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
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
            w,
        )

    return _loss_fn


def make_loss_fn_vs(config, graphdef_vs):
    """Create loss function for the sensitivity value network (GRU-based).

    Returns a function with signature:
        loss_fn_vs(params_vs, traj_batch, targets_s) -> value_loss_s
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
    """Create trajectory collection function for SPPO-aug.

    Args:
        config: Training configuration
        env: Wrapped environment
        env_params: Environment parameters
        networks: Dict returned by init_networks containing 'network', 'network_vs', etc.

    Returns a function for use with jax.lax.scan that collects transitions.
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

        # Reset GRU hidden on episode end using sensitivity done signal
        done_sens = info.get("done_sens", done)
        new_gru_hidden = (1 - done_sens.reshape(-1, 1)) * new_gru_hidden + done_sens.reshape(
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
                "sens_bias": info[
                    "sens_bias"
                ],  # df/dx@S + df/dp_norm (precomputed in wrapper)
                "df_du": info["df_du"],  # df/du
                "dobs_dx_S": info["dobs_dx_sens"],  # dobs/dx @ S
                "sens_norm_factor": info.get("sens_norm_factor", jnp.ones_like(reward)),
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
    """Create update epoch function for SPPO-aug.

    Args:
        config: Training configuration
        loss_fn: Loss function created by make_loss_fn
        shuffle_batch_fn: Function to shuffle and batch trajectories
        networks: Dict returned by init_networks containing 'loss_fn_vs', etc.

    Returns a function for use with jax.lax.scan that performs one epoch of updates.
    """
    loss_fn_vs = networks["loss_fn_vs"]

    def _update_epoch(update_state: UpdateState, unused):
        # Keep a fixed actor snapshot for the whole PPO update block.
        old_actor_params = update_state.extras.get("old_actor_params")
        if old_actor_params is None:
            old_actor_params = _actor_param_leaves(update_state.train_state.params)

        def _update_minbatch(ts, batch_info):
            train_state, train_state_vs = ts
            traj_batch, advantages, advantages_s, targets, targets_s = batch_info

            # Update main network
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params,
                traj_batch,
                advantages,
                advantages_s,
                targets,
                old_actor_params,
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
    """Calculate GAE for SPPO-aug (both standard and sensitivity)."""

    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    _, last_val = model(collect_state.last_obs)
    model_vs = nnx.merge(
        networks["graphdef_vs"], collect_state.extras["train_state_vs"].params
    )
    last_val_s, _ = model_vs(
        collect_state.last_obs,
        collect_state.extras["gru_hidden"],
    )
    advantages, advantages_s, targets, targets_s = gae_sensitivity_dual(
        config, traj_batch, last_val, last_val_s
    )
    return {
        "advantages": advantages,
        "advantages_s": advantages_s,
        "targets": targets,
        "targets_s": targets_s,
    }, collect_state


def extract_losses(loss_info):
    """Extract loss metrics from loss_info."""
    (
        total_losses,
        (
            value_losses,
            actor_losses,
            entropies,
            old_approx_kl,
            approx_kl,
            clipfracs,
            omega,
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
    extra_losses = {
        "vs_loss": vs_losses.mean(),
        "omega": omega.mean(),
    }
    return losses, extra_losses


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Initialize networks and train states for SPPO-aug.

    Returns:
        dict with keys: graphdef, train_state, graphdef_vs, train_state_vs, loss_fn_vs, gru_hidden_size
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
    """Create initial runner state for SPPO-aug.

    Returns:
        RunnerState NamedTuple
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    gru_hidden_size = networks["gru_hidden_size"]
    init_gru_hidden = jnp.zeros((config["NUM_ENVS"], gru_hidden_size))

    # Extra split before scan (matching original sppo_aug.py)
    rng, _rng = jax.random.split(rng)

    # Dummy GAE extras for pytree consistency (overwritten each iteration)
    dummy_gae = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]))

    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,  # Use _rng, not rng, to match original
        extras={
            "train_state_vs": networks["train_state_vs"],
            "gru_hidden": init_gru_hidden,
            "advantages_s": dummy_gae,
            "targets_s": dummy_gae,
        },
    )


SPEC = AlgorithmSpec(
    algo_name="SPPO_AUG_GRAD_PARAM_DUAL",
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
