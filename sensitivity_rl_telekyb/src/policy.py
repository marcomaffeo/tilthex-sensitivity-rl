import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization
import functools
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from networks import ActorCritic, ActorCriticGRU, ActorDCritic

class Policy:
    def __init__(self, path: Path):
        config, params_bytes = self.load(path)
        
        self.ACTION_DIM = config["env"]["action_dim"]
        self.MIN_INPUT = jnp.array(config["env"].get("MIN_INPUT"))
        self.MAX_INPUT = jnp.array(config["env"].get("MAX_INPUT"))
        self.MIN_ACTION = jnp.array(config["env"].get("MIN_ACTION"))
        self.MAX_ACTION = jnp.array(config["env"].get("MAX_ACTION"))

        self.graphdef, self.restored_state = self.init_model_and_params(config, params_bytes)

    def rescale_action(self, action: jax.Array) -> jax.Array:
        action = jnp.clip(action, self.MIN_ACTION, self.MAX_ACTION)
        action_span = self.MAX_ACTION - self.MIN_ACTION
        input_span = self.MAX_INPUT - self.MIN_INPUT
        normalized = (action - self.MIN_ACTION) / action_span
        return self.MIN_INPUT + normalized * input_span

    @functools.partial(jax.jit, static_argnames=['self'])
    def _get_action_jit(self, observation: jnp.ndarray, state: nnx.State) -> jnp.ndarray:
        model = nnx.merge(self.graphdef, state)
        action_mean = model.actor(observation).mean()
        return self.rescale_action(action_mean)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim == 1:
            observation = jnp.expand_dims(observation, axis=0)
            
        action_jax = self._get_action_jit(observation, self.restored_state)
        return np.asarray(action_jax[0], dtype=np.float64)

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

    def init_model_and_params(self, config: Dict[str, Any], params_bytes: bytes):
        full_pure_dict = serialization.msgpack_restore(params_bytes)

        actor_layer_sizes = tuple(config["ACTOR_LAYER_SIZES"])
        critic_layer_sizes = tuple(config.get("CRITIC_LAYER_SIZES", config["ACTOR_LAYER_SIZES"]))
        action_dim = config["env"]["action_dim"]
        obs_dim = config["env"]["obs_dim"]
        activation = config.get("ACTIVATION", "tanh")
        rngs = nnx.Rngs(0)
        
        algorithm = config.get("ALGORITHM", "ppo").lower()
        if algorithm in ("sppo", "sppo_grad"):
            gru_hidden_size = config.get("GRU_HIDDEN_SIZE", 64)
            model_template = ActorCriticGRU(
                obs_dim, action_dim,
                activation=activation,
                actor_layer_sizes=actor_layer_sizes,
                critic_layer_sizes=critic_layer_sizes,
                gru_hidden_size=gru_hidden_size,
                rngs=rngs,
            )
        else:
            model_template = ActorCritic(
                obs_dim, action_dim,
                activation=activation,
                actor_layer_sizes=actor_layer_sizes,
                critic_layer_sizes=critic_layer_sizes,
                rngs=rngs,
            )

        graphdef, template_state = nnx.split(model_template)
        treedef = jax.tree.structure(template_state)
        flat_leaves = jax.tree.leaves(full_pure_dict)
        restored_state = jax.tree.unflatten(treedef, flat_leaves)

        return graphdef, restored_state