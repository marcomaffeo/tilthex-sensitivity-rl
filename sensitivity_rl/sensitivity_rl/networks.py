import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import jax
import jax.numpy as jnp


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_layer_sizes: tuple = (256, 256)
    critic_layer_sizes: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor network
        actor = x
        for idx, layer_dim in enumerate(self.actor_layer_sizes):
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
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic network
        critic = x
        for idx, layer_dim in enumerate(self.critic_layer_sizes):
            critic = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"critic_dense{idx + 1}",
            )(critic)
            critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class SoftQNetwork(nn.Module):
    layer_sizes: tuple = (256, 256)
    activation: str = "relu"

    @nn.compact
    def __call__(self, state, action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = jnp.concatenate([state, action], axis=-1)
        for idx, layer_dim in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"dense{idx + 1}",
            )(x)
            x = activation(x)
        q_value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="output",
        )(x)
        return q_value


class Actor(nn.Module):
    action_dim: int
    layer_sizes: tuple = (256, 256)
    activation: str = "relu"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, state):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = state
        for idx, layer_dim in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"dense{idx + 1}",
            )(x)
            x = activation(x)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="output_mean",
        )(x)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="output_log_std",
        )(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (
            self.LOG_STD_MAX - self.LOG_STD_MIN
        ) * (log_std + 1)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return pi


def get_sac_action_dist(pi, scale_action, bias_action, _rng):
    """
    Sample from a Gaussian policy pi, apply tanh squashing, and rescale to the
    environment action space with (scale_action, bias_action). Also compute the
    corrected log-probability including the Jacobian of the tanh and the
    rescaling term, matching CleanRL/SpinUp conventions:
        log_prob = N(x_t|mu, sigma) - sum(log(scale * (1 - tanh(x_t)^2) + 1e-6))
    """
    x_t = pi.sample(seed=_rng)
    y_t = jnp.tanh(x_t)
    # Include both tanh correction and action rescaling in the log-prob
    # Note: scale_action is per-dimension; broadcast works as expected
    log_prob = pi.log_prob(x_t) - jnp.log(
        scale_action * (1 - y_t**2) + 1e-6
    ).sum(-1)
    mean_tanh = jnp.tanh(pi.mean())
    action = y_t * scale_action + bias_action
    mean = mean_tanh * scale_action + bias_action
    log_prob = jnp.expand_dims(log_prob, -1)
    return action, log_prob, mean


def get_ddpg_action_dist(pi, scale_action, bias_action, _rng):
    mean_tanh = jnp.tanh(pi.mean())
    mean = mean_tanh * scale_action + bias_action
    return mean, None, mean


class ActorCriticSens(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_layer_sizes: tuple = (256, 256)
    critic_layer_sizes: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor network with explicit names
        actor = x
        for idx, layer_dim in enumerate(self.actor_layer_sizes):
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
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic network with explicit names
        critic = x
        for idx, layer_dim in enumerate(self.critic_layer_sizes):
            critic = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"critic_dense{idx + 1}",
            )(critic)
            critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)

        # Critic network with explicit names
        critic_s = x
        for idx, layer_dim in enumerate(self.critic_layer_sizes):
            critic_s = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"critic_s_dense{idx + 1}",
            )(critic_s)
            critic_s = activation(critic_s)
        critic_s = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_s_output",
        )(critic_s)

        return pi, (
            jnp.squeeze(critic, axis=-1),
            jnp.squeeze(critic_s, axis=-1),
        )


class ActorCriticSensAug(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_layer_sizes: tuple = (256, 256)
    critic_layer_sizes: tuple = (256, 256)
    gru_hidden_size: int = 64

    @nn.compact
    def __call__(self, obs, hidden_state):
        """
        Args:
            obs: observation array [batch, obs_dim]
            hidden_state: GRU hidden state [batch, gru_hidden_size]
        Returns:
            value_s: sensitivity value estimate [batch]
            new_hidden_state: updated GRU hidden state [batch, gru_hidden_size]
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # GRU cell to process observation sequence
        gru_cell = nn.GRUCell(features=self.gru_hidden_size, name="gru_cell")
        new_hidden_state, gru_output = gru_cell(hidden_state, obs)

        # Critic network with explicit names, using GRU output
        critic_s = gru_output
        for idx, layer_dim in enumerate(self.critic_layer_sizes):
            critic_s = nn.Dense(
                layer_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"critic_s_dense{idx + 1}",
            )(critic_s)
            critic_s = activation(critic_s)
        critic_s = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_s_output",
        )(critic_s)

        return jnp.squeeze(critic_s, axis=-1), new_hidden_state
