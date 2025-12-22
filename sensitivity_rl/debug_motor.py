import sys
import os
import jax
import jax.numpy as jnp

# --- SOLITO SETUP PERCORSI ---
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
inner_dir = os.path.join(current_dir, "sensitivity_rl")
if inner_dir not in sys.path:
    sys.path.insert(0, inner_dir)

from sensitivity_rl.envs.tilthex_v0 import TilthexV0

def analyze_single_propeller():
    # Carichiamo l'ambiente
    # Usa disable_jit per poter stampare e debuggare liberamente
    with jax.disable_jit():
        env = TilthexV0()
        params = env.default_params
        
        print("\n" + "="*60)
        print("🔬 ANALISI DETTAGLIATA FORZA ELICA (Singolo Motore)")
        print("="*60)
        
        # 1. Recuperiamo i parametri base
        efficiency = jnp.cos(jnp.deg2rad(params.alpha))
        hover_force = (params.m * params.g) / (params.n * efficiency)
        
        # ATTENZIONE: JAX non ci permette di leggere variabili locali dentro le funzioni compilate.
        # Dobbiamo calcolare manualmente usando LA STESSA LOGICA che c'è in tilthex_v0.py
        # Se hai cambiato action_scale nel file, scrivilo anche qui per verificare!
        
        # --- PARAMETRO CRITICO DA VERIFICARE ---
        current_scale = 3.5  # <--- Assicurati che questo combaci col tuo file!
        # ---------------------------------------

        print(f"Parametri Attuali:")
        print(f" -> Hovering Force (Offset): {hover_force:.4f} N")
        print(f" -> Action Scale (Gain)    : {current_scale}")
        print(f" -> Limiti Fisici Motore   : [{params.min_input[0]:.2f}, {params.max_input[0]:.2f}] N")
        print("-"*60)
        print(f"{'ACTION (Rete)':^15} | {'FORZA RAW (N)':^15} | {'CLIPPED (N)':^15} | {'STATO'}")
        print("-"*60)

        # 2. Testiamo un range di azioni dalla rete
        # Da -1.5 (spinta forte giù) a +1.5 (spinta forte su)
        test_actions = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
        
        for val in test_actions:
            # Simuliamo l'azione per tutti i motori
            action = jnp.ones(6) * val
            
            # --- RIESECUZIONE LOGICA STEP_ENV ---
            # 1. Clipping Rete
            clipped_action = jnp.clip(action, params.min_action, params.max_action)
            
            # 2. Calcolo Forza (La tua formula nuova)
            # Input = Hover + (Action * Scale)
            u_raw = hover_force + (clipped_action * current_scale)
            
            # 3. Saturazione Fisica
            u_final = jnp.clip(u_raw, params.min_input, params.max_input)
            
            # Analisi singolo motore (indice 0)
            act_val = val
            raw_val = u_raw[0]
            fin_val = u_final[0]
            
            # Determiniamo lo stato
            status = "✅ OK"
            if fin_val == params.max_input[0]:
                status = "🔴 MAX (Saturazione)"
            elif fin_val == params.min_input[0]:
                status = "🔵 MIN (Spegnimento)"
            elif act_val != clipped_action[0]:
                status = "🔶 CLIPPED (Rete > 1)"

            print(f"{act_val:^15.2f} | {raw_val:^15.4f} | {fin_val:^15.4f} | {status}")

if __name__ == "__main__":
    analyze_single_propeller()