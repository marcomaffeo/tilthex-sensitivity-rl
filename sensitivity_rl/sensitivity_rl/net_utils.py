import jax
import optax

__all__ = ["create_optimizer", "make_linear_schedule"]


def create_optimizer(config, linear_schedule_fn):
    """Create optimizer with optional learning rate annealing."""
    max_grad_norm = config["MAX_GRAD_NORM"]
    policy_max_grad_norm = config.get("POLICY_MAX_GRAD_NORM")

    learning_rate = linear_schedule_fn if config["ANNEAL_LR"] else config["LR"]

    # Single global clipping for all parameters.
    if policy_max_grad_norm is None:
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )

    policy_max_grad_norm = (
        max_grad_norm if policy_max_grad_norm is None else policy_max_grad_norm
    )

    def _label_by_path(path, _):
        path_text = str(path).lower()
        if "actor" in path_text:
            return "policy"
        if "critic" in path_text:
            return "value"
        return "default"

    return optax.multi_transform(
        {
            "policy": optax.chain(
                optax.clip_by_global_norm(policy_max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            ),
            "value": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            ),
            "default": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=learning_rate, eps=1e-5),
            ),
        },
        lambda params: jax.tree_util.tree_map_with_path(_label_by_path, params),
    )


def make_linear_schedule(config):
    """Create linear learning rate schedule."""

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    return linear_schedule
