import math

# --- 1. CONFIGURAZIONE ---
# Inserisci qui i tuoi valori
PARAMS = {
    "m": 3.072,          # Massa (kg) - CAMBIA QUESTO SE DIVERSO
    "g": 9.81,
    "n": 6,            # Numero motori
    "alpha": 30.0,     # Angolo tilt
    "min_input": 0.3072,
    "max_input": 12.0
}

# --- 2. LOGICA MATEMATICA (Identica al tuo codice JAX) ---
def rescale_action_pure(action, p):
    # Efficienza verticale
    efficiency = math.cos(math.radians(p["alpha"]))
    
    # Forza Hovering per singolo motore
    hover_force_single = (p["m"] * p["g"]) / (p["n"] * efficiency)
    
    # Calcolo range
    range_up = p["max_input"] - hover_force_single
    range_down = hover_force_single - p["min_input"]
    
    # Logica asimmetrica (sostituisce jnp.where)
    if action >= 0:
        scale = range_up
    else:
        scale = range_down
        
    # Calcolo spinta
    u = hover_force_single + (action * scale)
    
    # Clipping manuale (sostituisce jnp.clip)
    u_clipped = max(p["min_input"], min(u, p["max_input"]))
    
    return u_clipped, hover_force_single

# --- 3. TEST ---
test_actions = [0.0, 1.0, -1.0]

print(f"--- TEST SENZA LIBRERIE (Massa: {PARAMS['m']} kg) ---")
print(f"Hover teorico richiesto (totale): {PARAMS['m']*9.81:.2f} N")

for action in test_actions:
    out, hov = rescale_action_pure(action, PARAMS)
    print(f"Action: {action:>4} -> Spinta: {out:.4f} N")

print("\n--- VERIFICA ---")
print(f"1. Se Action è  0.0 -> Deve essere {hov:.4f} N (Hover)")
print(f"2. Se Action è  1.0 -> Deve essere {PARAMS['max_input']} N (Max)")
print(f"3. Se Action è -1.0 -> Deve essere {PARAMS['min_input']} N (Min)")