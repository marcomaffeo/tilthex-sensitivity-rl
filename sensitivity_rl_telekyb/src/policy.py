
import jax
import jax.numpy as jnp
from flax import serialization
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

from pathlib import Path
from typing import Any, Dict, Tuple
import json

import functools




class Actor(nn.Module):
    """Standalone actor network for rollouts/evaluation.
    
    Uses the same layer names as ActorCritic/ActorCriticGRU so parameters
    can be extracted from trained policies.
    """
    action_dim: int
    layer_sizes: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor = x
        for idx, layer_dim in enumerate(self.layer_sizes):
            actor = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"actor_dense{idx + 1}",
            )(actor)
            actor = activation(actor)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(actor)
        actor_logtstd = self.param(
            "actor_log_std", nn.initializers.zeros, (self.action_dim,)
        )
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))



class Policy:
    def __init__(self, path: Path):

        config, params = self.load(path)
        
        # For now these values are hardcoded, but they could be made configurable later
        self.ACTION_DIM = config["env"]["action_dim"]

        self.MIN_INPUT = jnp.array(config["env"].get("MIN_INPUT"))
        self.MAX_INPUT = jnp.array(config["env"].get("MAX_INPUT"))
        self.MIN_ACTION = jnp.array(config["env"].get("MIN_ACTION"))
        self.MAX_ACTION = jnp.array(config["env"].get("MAX_ACTION"))

        self.model, self.policy_params = self.init_model_and_params(config, params)


    def rescale_action(self, action: jax.Array) -> jax.Array:
        action = jnp.clip(action, self.MIN_ACTION, self.MAX_ACTION)
        action_span = self.MAX_ACTION - self.MIN_ACTION
        input_span = self.MAX_INPUT - self.MIN_INPUT
        normalized = (action - self.MIN_ACTION) / action_span
        return self.MIN_INPUT + normalized * input_span
        

    @functools.partial(jax.jit, static_argnames=['self'])
    def get_action(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = self.model.apply(self.policy_params, observation)
        return self.rescale_action(action.mean())

    def load(self, run_dir: Path) -> Tuple[Dict[str, Any], bytes]:
        cfg_path = run_dir / "config.json"
        params_path = run_dir / "policy_params.msgpack"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {run_dir}")
        if not params_path.exists():
            raise FileNotFoundError(f"Missing policy_params.msgpack in {run_dir}")
        with cfg_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        params_bytes = params_path.read_bytes()
        return config, params_bytes

    def init_model_and_params(self,
        config: Dict[str, Any], params_bytes: bytes
    ):
        full_params = serialization.msgpack_restore(params_bytes)

        flat_params = full_params.get("params", full_params)
        actor_params = {}
        for key, value in flat_params.items():
            if key.startswith("actor_") or key == "actor_log_std":
                actor_params[key] = value
        actor_params = {"params": actor_params}

        actor_layer_sizes = tuple(config["ACTOR_LAYER_SIZES"])

        model = Actor(
            action_dim=self.ACTION_DIM,
            activation=config.get("ACTIVATION", "tanh"),
            layer_sizes=actor_layer_sizes,
        )
        return model, actor_params


if __name__ == "__main__":
    import time

    policy = Policy(Path("configs/policy/default"))
    dummy_obs = jnp.array([
                        0.0, 0.0, 0.0, # position difference
                        0.0, 0.0, 0.0, # velocity
                        1.0, 0.0, 0.0, 0.0, # attitude quaternion
                        0.0, 0.0, 0.0, # angular velocity
                        0.0 # yaw error
                        ])
    for i in range(100):
        start = time.perf_counter()
        dummy_obs = dummy_obs + 0.001  # change input to avoid caching effects
        action = policy.get_action(dummy_obs).block_until_ready()
        end = time.perf_counter()

        print("Action:", action)
        print(f"Inference time: {(end - start) * 1000:.2f} ms")