from flax import nnx
from flax.linen.initializers import constant, orthogonal
import distrax
import jax
import jax.numpy as jnp


class ActorCritic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        critic_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation_name = activation

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(action_dim))

        # Build critic layers
        _critic_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x):
        """Full forward pass with actor and critic."""
        pi = self.actor(x)
        value = self.critic(x)
        return pi, value

    def actor(self, x):
        """Actor forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.actor_layers:
            x = activation(layer(x))
        actor_mean = self.actor_output(x)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.actor_log_std.value)
        )

    def critic(self, x):
        """Critic forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.critic_layers:
            x = activation(layer(x))
        return jnp.squeeze(self.critic_output_layer(x), axis=-1)


class ActorDCritic(nnx.Module):
    """Actor-Critic with distributional critic using cosine embeddings for quantile regression.

    Implementation of the paper GMAC https://proceedings.mlr.press/v139/nam21a.html
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        hidden_size: int = 32,
        n_cosines: int = 64,
        n_taus: int = 64,
        lb: float = 0.0,
        ub: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation_name = activation
        self.hidden_size = hidden_size
        self.n_cosines = n_cosines
        self.n_taus = n_taus
        self.lb = lb
        self.ub = ub

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(action_dim))

        # Build critic layers
        self.cosine_embedding = nnx.Linear(
            n_cosines,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.state_embedding = nnx.Linear(
            obs_dim,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            hidden_size,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.fc3 = nnx.Linear(
            hidden_size,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, state, rng_key=None):
        """Full forward pass with actor and distributional critic."""
        pi = self.actor(state)
        quantile_values, taus = self.critic(state, rng_key)
        return pi, (quantile_values, taus)

    def actor(self, x):
        """Actor forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.actor_layers:
            x = activation(layer(x))
        actor_mean = self.actor_output(x)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.actor_log_std.value)
        )

    def critic(self, state, rng_key=None):
        """Distributional critic forward pass."""
        if rng_key is None:
            raise ValueError("rng_key is required for sampling taus in DActorCritic")

        bs = state.shape[0]
        cos, taus = self.calc_cos(bs, rng_key)

        cos_embd = self.cosine_embedding(cos)
        state_embd = self.state_embedding(state)
        state_embd = state_embd.reshape(-1, 1, state_embd.shape[-1])

        comb_embd = (state_embd * cos_embd).reshape(bs * self.n_taus, -1)

        x = self.fc2(comb_embd)
        x = nnx.relu(x)
        out = self.fc3(x)

        return out.reshape(bs, self.n_taus, -1), jnp.squeeze(taus, -1)

    def calc_cos(self, batch_size, rng_key):
        """Calculate cosine values for quantile embedding."""
        pis = jnp.array([jnp.pi * i for i in range(1, self.n_cosines + 1)])
        pis = pis.reshape(1, 1, self.n_cosines)

        taus = (
            jax.random.uniform(
                rng_key,
                (batch_size, self.n_taus),
                minval=self.lb,
                maxval=1.0,
            )
            * self.ub
        )
        taus = jnp.expand_dims(taus, -1)

        cos = jnp.cos(taus * pis)
        assert cos.shape == (batch_size, self.n_taus, self.n_cosines), "cos shape is incorrect"
        return cos, taus


class SoftQNetwork(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        layer_sizes: tuple = (256, 256),
        activation: str = "relu",
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation
        in_dim = state_dim + action_dim
        _layers = []
        for layer_dim in layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, state, action):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = jnp.concatenate([state, action], axis=-1)
        for layer in self.layers:
            x = activation(layer(x))
        return self.output_layer(x)


class SensQNetwork(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sens_dim: int,
        *,
        layer_sizes: tuple = (256, 256),
        activation: str = "relu",
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation
        in_dim = state_dim + action_dim + sens_dim
        _layers = []
        for layer_dim in layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, state, action, sens):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = jnp.concatenate([state, action, sens], axis=-1)
        for layer in self.layers:
            x = activation(layer(x))
        return self.output_layer(x)


class Actor(nnx.Module):
    """Standalone actor network for rollouts/evaluation.

    Uses the same architecture as the actor in ActorCritic/ActorCriticGRU.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        layer_sizes: tuple = (256, 256),
        activation: str = "tanh",
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation_name = activation

        _layers = []
        in_dim = obs_dim
        for layer_dim in layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.layers:
            x = activation(layer(x))
        actor_mean = self.actor_output(x)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.actor_log_std.value)
        )


def get_sac_action_dist(pi, scale_action, bias_action, _rng, action, sample: bool):
    """
    Sample from a Gaussian policy pi, apply tanh squashing, and rescale to the
    environment action space with (scale_action, bias_action).
    """
    def sample_fn(action):
        return pi.sample(seed=_rng)

    def use_action_fn(action):
        pi.sample(seed=_rng)
        return action

    x_t = jax.lax.cond(sample, sample_fn, use_action_fn, action)

    y_t = jnp.tanh(x_t)
    log_prob = pi.log_prob(x_t) - jnp.log(
        scale_action * (1 - y_t**2) + 1e-6
    ).sum(-1)
    mean_tanh = jnp.tanh(pi.mean())
    action_out = y_t * scale_action + bias_action
    mean = mean_tanh * scale_action + bias_action
    log_prob = jnp.expand_dims(log_prob, -1)
    return action_out, log_prob, mean, x_t


def get_ddpg_action_dist(pi, scale_action, bias_action, _rng):
    mean_tanh = jnp.tanh(pi.mean())
    mean = mean_tanh * scale_action + bias_action
    return mean, None, mean


class ActorCriticDual(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        critic_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation_name = activation

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(action_dim))

        # Build main critic layers
        _critic_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

        # Build sensitivity critic layers
        _critic_s_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_s_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_s_layers = nnx.List(_critic_s_layers)
        self.critic_s_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x):
        """Full forward pass with actor and dual critics."""
        pi = self.actor(x)
        value, value_s = self.critic(x)
        return pi, (value, value_s)

    def actor(self, x):
        """Actor forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.actor_layers:
            x = activation(layer(x))
        actor_mean = self.actor_output(x)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.actor_log_std.value)
        )

    def critic(self, x):
        """Dual critic forward pass (value and sensitivity value)."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh

        c = x
        for layer in self.critic_layers:
            c = activation(layer(c))
        c = jnp.squeeze(self.critic_output_layer(c), axis=-1)

        cs = x
        for layer in self.critic_s_layers:
            cs = activation(layer(cs))
        cs = jnp.squeeze(self.critic_s_output_layer(cs), axis=-1)

        return c, cs


class CriticGRU(nnx.Module):
    """GRU-based critic for value estimation with temporal dependencies."""

    def __init__(
        self,
        obs_dim: int,
        *,
        activation: str = "tanh",
        critic_layer_sizes: tuple = (256, 256),
        gru_hidden_size: int = 64,
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation
        self.gru_hidden_size = gru_hidden_size

        self.gru_cell = nnx.GRUCell(
            in_features=obs_dim,
            hidden_features=gru_hidden_size,
            rngs=rngs,
        )

        _critic_s_layers = []
        in_dim = gru_hidden_size
        for layer_dim in critic_layer_sizes:
            _critic_s_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_s_layers = nnx.List(_critic_s_layers)
        self.critic_s_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, obs, hidden_state):
        """
        Args:
            obs: observation array [batch, obs_dim]
            hidden_state: GRU hidden state [batch, gru_hidden_size]
        Returns:
            value_s: sensitivity value estimate [batch]
            new_hidden_state: updated GRU hidden state [batch, gru_hidden_size]
        """
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh

        new_hidden_state, _ = self.gru_cell(hidden_state, obs)

        critic_s = new_hidden_state
        for layer in self.critic_s_layers:
            critic_s = activation(layer(critic_s))
        critic_s = jnp.squeeze(self.critic_s_output_layer(critic_s), axis=-1)

        return critic_s, new_hidden_state


class ActorCriticGRU(nnx.Module):
    """ActorCritic with GRU-based critic for temporal dependencies.

    The actor is a standard MLP that doesn't depend on hidden state.
    The critic uses a GRU to capture temporal information.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        critic_layer_sizes: tuple = (256, 256),
        gru_hidden_size: int = 64,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation_name = activation
        self.gru_hidden_size = gru_hidden_size

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(action_dim))

        # Build GRU + critic layers
        self.gru_cell = nnx.GRUCell(
            in_features=obs_dim,
            hidden_features=gru_hidden_size,
            rngs=rngs,
        )

        _critic_layers = []
        in_dim = gru_hidden_size
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x, hidden_state):
        """Full forward pass with actor and GRU-based critic."""
        pi = self.actor(x)
        value, new_hidden_state = self.critic(x, hidden_state)
        return pi, value, new_hidden_state

    def actor(self, x):
        """Actor forward pass (no hidden state needed)."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.actor_layers:
            x = activation(layer(x))
        actor_mean = self.actor_output(x)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.actor_log_std.value)
        )

    def critic(self, x, hidden_state):
        """GRU-based critic forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh

        new_hidden_state, _ = self.gru_cell(hidden_state, x)

        c = new_hidden_state
        for layer in self.critic_layers:
            c = activation(layer(c))
        c = jnp.squeeze(self.critic_output_layer(c), axis=-1)

        return c, new_hidden_state
