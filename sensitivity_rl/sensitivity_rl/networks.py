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
