import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Importazione della TUA matrice reale ---
# Se utils_allocation.py è nella stessa cartella, questo funziona.
# Altrimenti aggiusta l'import.
try:
    from utils_allocation import compute_allocation
    print("✅ utils_allocation importato correttamente.")
except ImportError:
    print("⚠️ ERRORE: Non trovo utils_allocation.py. Assicurati che sia nella stessa cartella.")
    # Fallback dummy se l'import fallisce (solo per non crashare, ma i dati saranno falsi)
    def compute_allocation(n, arm_length, alpha, beta, c_f, c_tau, com_default):
        return jnp.ones((3, n)), jnp.eye(3, n)

# --- 2. Parametri (Copiati dal tuo EnvParams) ---
class MockParams:
    m = 3.072
    g = 9.81
    n = 6
    min_input = 0.3072
    max_input = 12.0
    
    # Parametri geometrici per compute_allocation
    arm_length = 0.39
    alpha = jnp.full((6,), -20.0) # Tilt angle
    c_f = 12.0e-4
    c_tau = 2.4e-5
    beta = 0.0
    com_default = jnp.array([0.0, 0.0, 0.0])

params = MockParams()

# --- 3. Calcolo Matrice M Reale ---
GF, GT = compute_allocation(
    params.n, params.arm_length, params.alpha, 
    params.beta, params.c_f, params.c_tau, params.com_default
)
M = jnp.vstack([GF, GT])
print(f"Matrice M shape: {M.shape}")

# --- 4. Il Solver QP Aggiornato (Weighted PGD) ---
def solve_weighted_qp(W_des, M, step_size, n_iter, reg):
    avg_hover = (params.m * params.g) / params.n  # ~5.0 N
    u = jnp.full((params.n,), avg_hover)
    
    # PESI: [Fx, Fy, Fz, Tx, Ty, Tz]
    # Diamo priorità 1.0 alle forze e 20.0 alle coppie.
    # Questo dice al solver: "1 Nm di errore coppia vale quanto 20 N di errore forza"
    weights = jnp.array([1.0, 1.0, 1.0, 20.0, 20.0, 20.0])
    
    print(f"\n--- DEBUG START (lr={step_size}, iter={n_iter}, reg={reg}) ---")
    print(f"Target Wrench: {W_des}")
    
    loss_history = []
    
    for i in range(n_iter):
        # 1. Calcolo Errore Raw
        W_pred = M @ u
        raw_err = W_pred - W_des
        
        # 2. Applicazione Pesi (Weighted Least Squares)
        # Loss = 0.5 * sum((err * w)^2) + reg
        weighted_err = raw_err * weights
        
        fit_loss = 0.5 * jnp.sum(weighted_err**2)
        reg_loss = 0.5 * reg * jnp.sum(u**2)
        total_loss = fit_loss + reg_loss
        loss_history.append(total_loss)
        
        # 3. Gradiente Pesato
        # d/du = M.T @ (weighted_err * weights) + reg*u
        # Nota: moltiplichiamo per weights due volte (una per l'errore, una per la derivata interna)
        grad = M.T @ (weighted_err * weights) + reg * u
        
        # 4. Update & Clip
        u = u - step_size * grad
        u = jnp.clip(u, params.min_input, params.max_input)
        
        if i % 20 == 0 or i == n_iter - 1:
            # Stampiamo errore reale (non pesato) per capire la fisica
            print(f"Iter {i:3d} | Loss: {total_loss:.4f} | Err Fz: {raw_err[2]:.4f} | Err Tx: {raw_err[3]:.5f}")

    print(f"--- FINAL U: {u} ---")
    print(f"--- FINAL W_pred: {M @ u} ---")
    print(f"--- FINAL ERRORS: {(M @ u) - W_des} ---")
    return loss_history

# --- 5. Scenario di Test ---
# Scenario Stress Test: 
# Fz = 30N (vicino al limite mg), Fy = 5N (spinta laterale). Coppie = 0.
W_target = jnp.array([0.0, 5.0, 30.0, 0.0, 0.0, 0.0])

# LANCIA IL DEBUG
# step_size leggermente ridotto a 1e-1 per stabilità con i pesi alti
history = solve_weighted_qp(W_target, M, step_size=4e-3, n_iter=100, reg=1e-2)

# Plot convergenza
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.title("Convergenza Weighted QP Solver")
plt.xlabel("Iterazioni")
plt.ylabel("Loss Pesata")
plt.yscale("log")
plt.grid(True, which="both", ls="-")
plt.show()