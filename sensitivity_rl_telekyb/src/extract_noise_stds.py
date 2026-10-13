import numpy as np
import matplotlib.pyplot as plt

def analyze_multiple_hovering_logs(file_paths, plot=True):
    all_pos_noise = []
    all_vel_noise = []
    all_quat_noise = []
    all_avel_noise = []

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"  -> Load error: {e}")
            continue

        # Use the entire log
        pos_hover = data['pos']
        vel_hover = data['vel']
        quat_hover = data['quat']
        avel_hover = data['avel']

        # Remove mean to center the noise around zero
        all_pos_noise.append(pos_hover - np.mean(pos_hover, axis=0))
        all_vel_noise.append(vel_hover - np.mean(vel_hover, axis=0))
        all_quat_noise.append(quat_hover - np.mean(quat_hover, axis=0))
        all_avel_noise.append(avel_hover - np.mean(avel_hover, axis=0))

    if not all_pos_noise:
        print("No valid data processed.")
        return

    # Concatenate all centered noise arrays
    global_pos_noise = np.concatenate(all_pos_noise, axis=0)
    global_vel_noise = np.concatenate(all_vel_noise, axis=0)
    global_quat_noise = np.concatenate(all_quat_noise, axis=0)
    global_avel_noise = np.concatenate(all_avel_noise, axis=0)

    # Compute global standard deviation
    pos_std = np.std(global_pos_noise, axis=0, ddof=1)
    vel_std = np.std(global_vel_noise, axis=0, ddof=1)
    quat_std = np.std(global_quat_noise, axis=0, ddof=1)
    avel_std = np.std(global_avel_noise, axis=0, ddof=1)
    q_err_std = quat_std # Assume target error noise matches state quaternion noise

    print("\n--- STANDARD DEVIATION RESULTS (Copy to JAX) ---")
    print("jnp.array([")
    print(f"    {pos_std[0]:.6f}, {pos_std[1]:.6f}, {pos_std[2]:.6f},  # Position (x, y, z)")
    print(f"    {vel_std[0]:.6f}, {vel_std[1]:.6f}, {vel_std[2]:.6f},  # Velocity (vx, vy, vz)")
    print(f"    {quat_std[0]:.6f}, {quat_std[1]:.6f}, {quat_std[2]:.6f}, {quat_std[3]:.6f},  # Quaternion (w, x, y, z)")
    print(f"    {avel_std[0]:.6f}, {avel_std[1]:.6f}, {avel_std[2]:.6f},  # Angular Vel (wx, wy, wz)")
    print(f"    {q_err_std[0]:.6f}, {q_err_std[1]:.6f}, {q_err_std[2]:.6f}, {q_err_std[3]:.6f}   # Quaternion Target Error")
    print("])")
    print("--------------------------------------------------")

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'Combined Sensor Noise ({len(file_paths)} flights)', fontsize=14)

        axs[0,0].plot(global_pos_noise, alpha=0.7)
        axs[0,0].set_title('Position Noise')
        axs[0,0].legend(['X', 'Y', 'Z'], loc='upper right')

        axs[0,1].plot(global_vel_noise, alpha=0.7)
        axs[0,1].set_title('Linear Velocity Noise')
        axs[0,1].legend(['Vx', 'Vy', 'Vz'], loc='upper right')

        axs[1,0].plot(global_quat_noise, alpha=0.7)
        axs[1,0].set_title('Quaternion Noise')
        axs[1,0].legend(['W', 'X', 'Y', 'Z'], loc='upper right')
        
        axs[1,1].plot(global_avel_noise, alpha=0.7)
        axs[1,1].set_title('Angular Velocity Noise')
        axs[1,1].legend(['Wx', 'Wy', 'Wz'], loc='upper right')

        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Aggregated Samples')
            
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Flight logs to process
    file_paths = [
        "logs_policy/SPPO_noise/data.npz",
    ]
    
    analyze_multiple_hovering_logs(file_paths, plot=True)