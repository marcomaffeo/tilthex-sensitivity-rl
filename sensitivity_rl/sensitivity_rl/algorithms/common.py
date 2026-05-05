import jax
import jax.numpy as jnp

__all__ = ["gae_standard", "gae_sensitivity", "gae_sensitivity_dual"]


def gae_standard(config, traj_batch, last_val):
    """Standard GAE for PPO-style returns.
    Returns (advantages, targets).
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            transition.reward
            + config["GAMMA"] * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


def gae_sensitivity(config, traj_batch, last_val, last_val_s):
    """GAE for sensitivity-augmented trajectories.
    Returns (advantages, advantages_s, targets, targets_s).
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, gae_s, next_value, next_value_s = gae_and_next_value
        delta = (
            transition.reward
            + config["GAMMA"] * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae

        delta_s = (
            transition.extras["reward_s"]
            + config["GAMMA"] * next_value_s * (1 - transition.done)
            - transition.extras["value_s"]
        )
        gae_s = delta_s + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae_s
        return (gae, gae_s, transition.value, transition.extras["value_s"]), (gae, gae_s)

    _, (advantages, advantages_s) = jax.lax.scan(
        _get_advantages,
        (
            jnp.zeros_like(last_val),
            jnp.zeros_like(last_val),
            last_val,
            last_val_s,
        ),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    targets_s = advantages_s + traj_batch.extras["value_s"]
    return advantages, advantages_s, targets, targets_s

def gae_sensitivity_dual(config, traj_batch, last_val, last_val_s):
    """GAE for sensitivity-augmented trajectories with a dual done signal.
    Uses transition.info['done_sens'] for sensitivity value advantage calculation.
    Returns (advantages, advantages_s, targets, targets_s).
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, gae_s, next_value, next_value_s = gae_and_next_value
        delta = (
            transition.reward
            + config["GAMMA"] * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae

        done_sens = transition.info.get("done_sens", transition.done)

        delta_s = (
            transition.extras["reward_s"]
            + config["GAMMA"] * next_value_s * (1 - done_sens)
            - transition.extras["value_s"]
        )
        gae_s = delta_s + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done_sens) * gae_s
        return (gae, gae_s, transition.value, transition.extras["value_s"]), (gae, gae_s)

    _, (advantages, advantages_s) = jax.lax.scan(
        _get_advantages,
        (
            jnp.zeros_like(last_val),
            jnp.zeros_like(last_val),
            last_val,
            last_val_s,
        ),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    targets_s = advantages_s + traj_batch.extras["value_s"]
    return advantages, advantages_s, targets, targets_s
