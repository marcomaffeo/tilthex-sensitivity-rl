import jax
import jax.numpy as jnp
import json
import os
import sys
from pathlib import Path
from flax import serialization

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================
# Incolla qui il percorso della tua run
RUN_PATH = "runs/tilthex_v0/PPO_20251226-004156"
# ==============================================================================

# --- IMPORT FIX (Magia per trovare i file) ---
# Aggiungiamo la cartella corrente e la sottocartella 'sensitivity_rl' al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "sensitivity_rl"))

try:
    # Tentativo 1: Import diretto (se i file sono nella root)
    from envs.tilthex_v0 import TilthexV0
    from networks import ActorCritic
except ImportError:
    # Tentativo 2: Import dalla sottocartella (se lanci dalla root)
    try:
        from sensitivity_rl.envs.tilthex_v0 import TilthexV0
        from sensitivity_rl.networks import ActorCritic
    except ImportError as e:
        print("❌ ERRORE CRITICO DI IMPORTAZIONE")
        print("Non riesco a trovare 'networks.py' o 'envs/'.")
        print(f"Sto cercando in: {sys.path}")
        raise e
# ---------------------------------------------

def visualize():
    run_dir = Path(RUN_PATH)
    config_file = run_dir / "config.json"
    params_file = run_dir / "params.msgpack"

    if not params_file.exists():
        print(f"❌ ERRORE: Non trovo il file {params_file}")
        print(f"Controlla che il percorso '{RUN_PATH}' sia corretto.")
        return

    print(f"📂 Caricamento configurazione da: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    # 1. Inizializza Ambiente
    print("🛠️  Inizializzazione ambiente TilthexV0...")
    env = TilthexV0()
    env_params = env.default_params

    # 2. Inizializza Rete (Vuota)
    print("🧠 Ricostruzione struttura Rete Neurale...")
    actor_sizes = tuple(config.get("ACTOR_LAYER_SIZES", (256, 256)))
    critic_sizes = tuple(config.get("CRITIC_LAYER_SIZES", (256, 256)))
    
    network = ActorCritic(
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        actor_layer_sizes=actor_sizes,
        critic_layer_sizes=critic_sizes,
    )

    # Inizializziamo con pesi dummy
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
    network_params_structure = network.init(rng, dummy_obs)

    # 3. Carica i Pesi Addestrati
    print(f"📥 Caricamento pesi da: {params_file}")
    with open(params_file, "rb") as f:
        params_bytes = f.read()
    
    trained_params = serialization.from_bytes(network_params_structure, params_bytes)
    print("✅ Modello caricato con successo!")

    # 4. Loop di Visualizzazione
    print("\n" + "="*80)
    print(f"{'STEP':^5} | {'DISTANZA':^10} | {'FORZE MOTORI (Newton) [Min 0.31 - Max 12.0]':^50}")
    print("="*80)

    with jax.disable_jit():
        rng = jax.random.PRNGKey(12345) 
        reset_key, step_key = jax.random.split(rng)
        
        obs, env_state = env.reset(reset_key, env_params)
        target_pos = env_state.target_pos
        
        for t in range(300):
            step_key, subkey = jax.random.split(step_key)
            
            # Calcolo Azione
            pi, _ = network.apply(trained_params, obs)
            action = pi.mode()
            
            # Step Fisico
            obs, env_state, reward, done, info = env.step(
                subkey, env_state, action, env_params
            )
            
            # Lettura Forze
            forces = env_state.last_u
            curr_pos = env_state.state.pos
            dist = jnp.linalg.norm(curr_pos - target_pos)
            
            # Formattazione
            f_str = " | ".join([f"{f:5.2f}" for f in forces])
            
            status = ""
            if dist < 0.05: 
                status = "🎯 ARRIVATO"
            elif t > 10 and jnp.mean(forces) < 1.0: 
                status = "💀 MOTORI SPENTI"
            elif jnp.any(forces >= 11.9):
                status = "🔴 SATURAZIONE"

            print(f"{t:03d}   | {dist:8.4f} m | [{f_str}] {status}")
            
            if done:
                print(f"\n🛑 Episodio finito al passo {t}")
                break

if __name__ == "__main__":
    visualize()