import jax

from envs import Pendulum

env = Pendulum()
env_params = env.default_params

rng = jax.random.PRNGKey(42)
obsv, env_state = env.reset(rng, env_params)

state = env_state.state

print(state)
print(env.get_obs(state))

jac_obs = jax.jacobian(env.get_obs)
print("dobs/dx: ", env.dobs_dx(state))

print("df/dx: ", env.df_dx(rng, state, jax.numpy.array([1.0]), env_params))
print("df/du: ", env.df_du(rng, state, jax.numpy.array([1.0]), env_params))
print("df/dp: ", env.df_dp(rng, state, jax.numpy.array([1.0]), env_params))
