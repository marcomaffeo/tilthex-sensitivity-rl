import sys
import os

# --- BLOCCO MAGICO V2 (PRIORITÀ ASSOLUTA) ---
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
inner_dir = os.path.join(current_dir, "sensitivity_rl")
if inner_dir not in sys.path:
    sys.path.insert(0, inner_dir)
# --------------------------------------------

import jax
import jax.numpy as jnp
import numpy as np
# Importiamo l'ambiente
try:
    from sensitivity_rl.envs.tilthex_v0 import TilthexV0
except ImportError:
    # Fallback se la struttura delle cartelle è diversa
    from envs.tilthex_v0 import TilthexV0

def debug_forces_physics():
    print("\n" + "="*40)
    print("=== DEBUG 1: FISICA E MAPPING (ZERO-CENTERED) ===")
    print("="*40)
    
    # Qui usiamo disable_jit per sicurezza, anche se è calcolo diretto
    with jax.disable_jit():
        env = TilthexV0()
        params = env.default_params
        
        # 1. FISICA
        weight = params.m * 9.81
        efficiency = jnp.cos(jnp.deg2rad(params.alpha))
        hover_force = weight / (params.n * efficiency)
        
        print(f"Massa: {params.m} kg | Peso: {weight:.2f} N")
        print(f"Hovering Force (per motore): {hover_force:.4f} N")
        
        # 2. RESCALE (Action 0)
        action_zero = jnp.zeros(6)
        mapped_zero = env.rescale_action(action_zero, params)
        
        print(f"\n[TEST CRUCIALE] Action = 0.0 -> Force = {mapped_zero[0]:.4f} N")
        
        diff = mapped_zero[0] - hover_force
        accel = (diff * 6) / params.m
        print(f"Differenza da Hover: {diff:+.4f} N")
        print(f"Accelerazione iniziale: {accel:.4f} m/s^2")
        
        if abs(accel) < 0.01:
            print("✅ SUCCESS: Il drone è perfettamente bilanciato!")
        else:
            print("❌ FAIL: Il drone ha ancora un bias di accelerazione.")

        # 3. INVESTIGAZIONE REVERSE (Nuova Formula)
        print(f"\n[Mapping Inverso - Nuova Logica]")
        target_8 = 8.0
        # Action = (Force - Hover) / 6.5
        action_8 = (target_8 - hover_force) / 6.5
        print(f"Per avere {target_8} N (Salita) -> Action: {action_8:.4f}")
        
        target_3 = 3.0
        action_3 = (target_3 - hover_force) / 6.5
        print(f"Per avere {target_3} N (Discesa) -> Action: {action_3:.4f}")


def debug_reward_landscape():
    print("\n" + "="*40)
    print("=== DEBUG 2: REWARD LANDSCAPE ===")
    print("="*40)
    
    # DISABILITIAMO IL JIT per evitare l'errore "Ambiguous truth value"
    # dovuto alla definizione di EnvParams
    with jax.disable_jit():
        env = TilthexV0()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        
        # Stato Iniziale "Perfetto"
        _, env_state = env.reset_env(key, params)
        
        # Forziamo la perfezione
        state_perfect = env_state.state.replace(
            pos=jnp.array([0., 0., 1.0]),
            vel=jnp.zeros(3),
            attitude=jnp.array([1., 0., 0., 0.]),
            omega=jnp.zeros(3)
        )
        env_state = env_state.replace(state=state_perfect, target_pos=jnp.array([0., 0., 1.0]))
        
        # Azione Neutra (Hovering)
        action = jnp.zeros(6)
        
        print("--- Test Distanza (Drone dritto, si allontana) ---")
        dists = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
        for d in dists:
            pos_new = jnp.array([d, 0., 1.0])
            s_mod = state_perfect.replace(pos=pos_new)
            es_mod = env_state.replace(state=s_mod)
            
            # Chiamata step_env in modalità eager (no jit)
            _, _, reward, _, _ = env.step_env(key, es_mod, action, params)
            
            check = "✅" if reward > 0 else "⚠️" if d < 0.2 else "✅"
            print(f"Dist: {d:4.2f}m -> Reward: {reward:8.4f} {check}")

        print("\n--- Test Inclinazione (Drone fermo sul target, si inclina) ---")
        angles = [0, 10, 30, 45, 90, 180]
        for deg in angles:
            rad = jnp.deg2rad(deg)
            q_new = jnp.array([jnp.cos(rad/2), jnp.sin(rad/2), 0.0, 0.0])
            
            s_mod = state_perfect.replace(attitude=q_new)
            es_mod = env_state.replace(state=s_mod)
            
            _, _, reward, _, _ = env.step_env(key, es_mod, action, params)
            print(f"Tilt: {deg:3d}° -> Reward: {reward:8.4f}")


def debug_observations_sanity():
    print("\n" + "="*40)
    print("=== DEBUG 3: OBSERVATION SANITY CHECK ===")
    print("="*40)
    
    # DISABILITIAMO IL JIT
    with jax.disable_jit():
        env = TilthexV0()
        params = env.default_params
        key = jax.random.PRNGKey(42)
        
        _, env_state = env.reset_env(key, params)
        obs_history = []
        
        print("Running 100 random steps (JIT Disabled)...")
        for _ in range(100):
            key, subkey = jax.random.split(key)
            action = jax.random.normal(subkey, shape=(6,))
            
            # Qui step_env viene eseguita in Python puro/NumPy mode
            obs, env_state, reward, done, _ = env.step_env(key, env_state, action, params)
            obs_history.append(obs)
            
            if done:
                 _, env_state = env.reset_env(subkey, params)

        obs_stack = jnp.stack(obs_history)
        
        has_nan = jnp.any(jnp.isnan(obs_stack))
        has_inf = jnp.any(jnp.isinf(obs_stack))
        
        print(f"Min Val: {jnp.min(obs_stack):.4f}")
        print(f"Max Val: {jnp.max(obs_stack):.4f}")
        print(f"Mean   : {jnp.mean(obs_stack):.4f}")
        
        if not has_nan and not has_inf and jnp.max(jnp.abs(obs_stack)) < 100:
            print("✅ SUCCESS: Le osservazioni sono pulite e limitate.")
        else:
            print(f"❌ FAIL: Trovati NaN? {has_nan}, Inf? {has_inf}")


def debug_crash_test():
    print("\n" + "="*40)
    print("=== DEBUG 4: CRASH TEST (TERMINAZIONE) ===")
    print("="*40)
    
    # DISABILITIAMO IL JIT
    with jax.disable_jit():
        env = TilthexV0()
        params = env.default_params
        key = jax.random.PRNGKey(99)
        _, env_state = env.reset_env(key, params)
        
        # Mettiamo il drone sotto terra (-10m)
        state_underground = env_state.state.replace(pos=jnp.array([0., 0., -10.0]))
        env_state = env_state.replace(state=state_underground)
        
        _, _, _, done, _ = env.step_env(key, env_state, jnp.zeros(6), params)
        
        print(f"Posizione Z = -10.0m -> Done: {done}")
        
        if done:
            print("✅ SUCCESS: L'ambiente termina correttamente se il drone cade.")
        else:
            print("⚠️ WARNING: L'ambiente NON termina se il drone va sotto terra!")
            print("   Suggerimento: Modifica 'is_terminal' per includere limiti di posizione.")

if __name__ == "__main__":
    # Esegue tutti i test
    debug_forces_physics()
    debug_reward_landscape()
    debug_observations_sanity()
    debug_crash_test()