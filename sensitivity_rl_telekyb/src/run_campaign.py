import os
import time
import subprocess
import shutil
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

TASK_TYPE = "hovering"  # "point_to_point" or "hovering"                   
POLICY_NAME = "SPPO"                               
TARGET_POSE = [1.0, 0.5, 1.2, 0.0, 0.0, 15.0]      
TEST_DURATION = 4.0

NUM_MC_RUNS = 10      
RANDOM_SEED = 42   

NOMINAL = {
    "MASS": 2.91118, "IXX": 0.082529, "IYY": 0.0846092, "IZZ": 0.154814,
    "CF": 12.0e-4, "COM_X": 0.000122973, "COM_Y": 0.00013006, "COM_Z": 0.0248586,
    "ALPHA_POS": 0.349066, "ALPHA_NEG": -0.349066  
}

RANGES = {
    "MASS":  (2.69, 3.29), "CF": (1.08e-3, 1.32e-3),
    "IXX":   (0.074, 0.090), "IYY": (0.076, 0.093), "IZZ": (0.140, 0.170),
    "COM_X": (0.001, 0.002), "COM_Y": (0.001, 0.002), "COM_Z": (0.020, 0.030),
    "ALPHA": (0.314159, 0.383972)
}

def generate_mc_perturbations():
    np.random.seed(RANDOM_SEED)
    perturbations = {}
    
    for i in range(1, NUM_MC_RUNS + 1):
        alpha_val = np.random.uniform(*RANGES["ALPHA"])
        
        valid_inertia = False
        while not valid_inertia:
            ixx = np.random.uniform(*RANGES["IXX"])
            iyy = np.random.uniform(*RANGES["IYY"])
            izz = np.random.uniform(*RANGES["IZZ"])
            
            if (ixx + iyy >= izz) and (ixx + izz >= iyy) and (iyy + izz >= ixx):
                valid_inertia = True

        p = {
            "MASS": np.random.uniform(*RANGES["MASS"]), 
            "CF": np.random.uniform(*RANGES["CF"]),
            "IXX": ixx, 
            "IYY": iyy, 
            "IZZ": izz,
            "COM_X": np.random.uniform(*RANGES["COM_X"]), 
            "COM_Y": np.random.uniform(*RANGES["COM_Y"]), 
            "COM_Z": np.random.uniform(*RANGES["COM_Z"]),
            "ALPHA_POS": alpha_val, 
            "ALPHA_NEG": -alpha_val
        }
        perturbations[f"mc_run_{i:03d}"] = p
        
    return perturbations

PERTURBATIONS = generate_mc_perturbations()

TEMPLATE_PATH = "models/mrsim-tilthex-laas/tilthex_laas_template.sdf" 
SDF_PATH = "models/mrsim-tilthex-laas/tilthex_laas.sdf"              

def cleanup(procs=None):
    print("[SYSTEM] Safely terminating Gazebo and Genomix...")
    if procs:
        for p in procs:
            try: p.terminate()
            except: pass
            try: p.kill() 
            except: pass

    subprocess.run("killall -9 ruby gz-server gz-sim-server gz", shell=True, stderr=subprocess.DEVNULL)
    
    genom_targets = ["optitrack-genom3", "pom-genom3", "rotorcraft-genom3", "nhfc-genom3", "maneuver-genom3", "dynamixel-genom3", "h2"]
    for t in genom_targets:
        subprocess.run(f"killall -9 {t}", shell=True, stderr=subprocess.DEVNULL)
        
    subprocess.run("pkill -9 -f 'policy_control_node_tilthex'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("rm -f /tmp/pty-hr* /tmp/h2* /tmp/pocolibs*", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(6)
    
def generate_sdf(overrides):
    with open(TEMPLATE_PATH, "r") as f:
        sdf_content = f.read()
    params = {k: overrides.get(k, NOMINAL[k]) for k in NOMINAL}
    for key, value in params.items():
        formatted_value = f"{float(value):.6f}"
        sdf_content = sdf_content.replace(f"{{{{{key}}}}}", formatted_value)
        sdf_content = sdf_content.replace(f"{{{key}}}", formatted_value)
    with open(SDF_PATH, "w") as f:
        f.write(sdf_content)

def run_test(test_name, overrides, target_log_dir, skip_plots=False, npz_name="error_data.npz"):
    print(f"\n{'='*50}\nSTARTING TEST: {test_name}\n{'='*50}")
    cleanup()
    generate_sdf(overrides)

    procs = []
    try:
        print("[START] Gazebo (Headless mode)...")
        p_sim = subprocess.Popen(
            ["pixi", "run", "gz", "sim", "-s", "-r", "worlds/myworld.world"], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(p_sim)
        time.sleep(8) 
        
        print("[START] Genomix Components...")
        p_comp = subprocess.Popen(
            ["pixi", "run", "components"], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) 
        procs.append(p_comp)
        time.sleep(10) 

        print(f"[START] Policy Node ({POLICY_NAME} -> Saving {npz_name} to {target_log_dir})...")
        policy_dir_path = f"configs/policy/{POLICY_NAME}"
        ros_cmd = [
            "pixi", "run", "policy-control-tilthex", 
            "--ros-args", 
            "-p", f"custom_log_dir:={target_log_dir}",
            "-p", f"policy_dir:={policy_dir_path}",
            "-p", f"policy_duration:={TEST_DURATION}",
            "-p", f"npz_filename:={npz_name}" 
        ]
        if skip_plots: ros_cmd.extend(["-p", "skip_plots:=True"])

        p_node = subprocess.Popen(ros_cmd)
        procs.append(p_node)
        time.sleep(8) 

        def call_srv(srv_name):
            subprocess.run(["ros2", "service", "call", srv_name, "std_srvs/srv/Trigger"], stdout=subprocess.DEVNULL)

        print("[SEQ] Setup & Start...")
        call_srv("/control/setup")
        time.sleep(2)
        call_srv("/control/start")
        time.sleep(4)

        print("[SEQ] Takeoff...")
        call_srv("/control/takeoff")
        time.sleep(5) 

        if TASK_TYPE == "point_to_point":
            print(f"[SEQ] Sending Point-to-Point Setpoint: {TARGET_POSE}")
            cmd = [
                "pixi", "run", "python", "src/policy_control_node_tilthex.py", "--setpoint",
                str(TARGET_POSE[0]), str(TARGET_POSE[1]), str(TARGET_POSE[2]),
                str(TARGET_POSE[3]), str(TARGET_POSE[4]), str(TARGET_POSE[5])
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0) 
        else:
            print("[SEQ] Hovering Task: Automatic setpoint on current position.")

        print(f"[SEQ] Policy Start! (Test duration: {TEST_DURATION}s)")
        call_srv("/policy/start")
        time.sleep(TEST_DURATION + 1.0)

        print("[SEQ] Policy Stop & Land...")
        call_srv("/policy/stop")
        time.sleep(1)
        call_srv("/policy/land")
        time.sleep(4)
        call_srv("/control/stop")

        print("[SYSTEM] Waiting for data saving...")
        time.sleep(6)
        print(f"[END] Test {test_name} completed.")
    
    except KeyboardInterrupt:
        print("\n[WARN] Manual interruption detected.")
    except Exception as e:
        print(f"\n[ERROR] Error during test {test_name}: {e}")
    finally:
        cleanup(procs)

def plot_aggregated(campaign_dir, perturbed_dir, raw_data_dir):
    print("\n[PLOT] Generating aggregated envelope plot...")
    run_files = list(Path(raw_data_dir).glob("mc_run_*.npz"))
    if not run_files: return

    all_pos_errors, all_att_errors, all_vel_errors, all_avel_errors, time_arrays = [], [], [], [], []
    for npz_file in run_files:
        data = np.load(npz_file)
        time_arrays.append(data['t'])
        all_pos_errors.append(data['pos_err'])
        all_att_errors.append(data['att_err'])
        all_vel_errors.append(data.get('vel_err', np.zeros_like(data['pos_err'])))
        all_avel_errors.append(data.get('avel_err', np.zeros_like(data['pos_err'])))

    max_len = max(len(arr) for arr in all_pos_errors)
    
    padded_pos = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in all_pos_errors])
    padded_att = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in all_att_errors])
    padded_vel = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in all_vel_errors])
    padded_avel = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in all_avel_errors])
    t_axis = time_arrays[np.argmax([len(t) for t in time_arrays])]

    mean_pos, std_pos = np.mean(padded_pos, axis=0), np.std(padded_pos, axis=0)
    mean_att, std_att = np.mean(padded_att, axis=0), np.std(padded_att, axis=0)
    mean_vel, std_vel = np.mean(padded_vel, axis=0), np.std(padded_vel, axis=0)
    mean_avel, std_avel = np.mean(padded_avel, axis=0), np.std(padded_avel, axis=0)

    steady_state_idx = int(max_len * 0.2)
    rmse_pos = np.sqrt(np.mean(padded_pos[:, steady_state_idx:]**2))
    rmse_att = np.sqrt(np.mean(padded_att[:, steady_state_idx:]**2))
    rmse_vel = np.sqrt(np.mean(padded_vel[:, steady_state_idx:]**2))
    rmse_avel = np.sqrt(np.mean(padded_avel[:, steady_state_idx:]**2))
    
    print(f"\n{'='*40}")
    print(f"[{POLICY_NAME}] FINAL MONTE CARLO METRICS")
    print(f"Steady-State RMSE Position: {rmse_pos:.4f} m")
    print(f"Steady-State RMSE Attitude: {rmse_att:.4f} rad")
    print(f"Steady-State RMSE Linear Vel: {rmse_vel:.4f} m/s")
    print(f"Steady-State RMSE Angular Vel: {rmse_avel:.4f} rad/s")
    print(f"{'='*40}\n")

    plt.style.use("seaborn-v0_8") 
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Overall {POLICY_NAME} Robustness Envelope ({len(run_files)} MC Runs)", fontsize=16)

    axs[0, 0].plot(t_axis, mean_pos, color="blue", linewidth=1.5)
    axs[0, 0].fill_between(t_axis, mean_pos - std_pos, mean_pos + std_pos, color="blue", alpha=0.2)
    axs[0, 0].text(0.5, 0.9, f"Final: {mean_pos[-1]:.3f} $\\pm$ {std_pos[-1]:.3f} m", transform=axs[0, 0].transAxes, ha="center", bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))
    axs[0, 0].set_title("Position Error [m]")
    axs[0, 0].grid(True, alpha=0.3)

    axs[1, 0].plot(t_axis, mean_att, color="red", linewidth=1.5)
    axs[1, 0].fill_between(t_axis, mean_att - std_att, mean_att + std_att, color="red", alpha=0.2)
    axs[1, 0].text(0.5, 0.9, f"Final: {mean_att[-1]:.3f} $\\pm$ {std_att[-1]:.3f} rad", transform=axs[1, 0].transAxes, ha="center", bbox=dict(boxstyle="round", fc="lightsalmon", alpha=0.5))
    axs[1, 0].set_title("Attitude Error [rad]")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t_axis, mean_vel, color="green", linewidth=1.5)
    axs[0, 1].fill_between(t_axis, mean_vel - std_vel, mean_vel + std_vel, color="green", alpha=0.2)
    axs[0, 1].text(0.5, 0.9, f"Final: {mean_vel[-1]:.3f} $\\pm$ {std_vel[-1]:.3f} m/s", transform=axs[0, 1].transAxes, ha="center", bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.5))
    axs[0, 1].set_title("Linear Velocity Error [m/s]")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 1].plot(t_axis, mean_avel, color="purple", linewidth=1.5)
    axs[1, 1].fill_between(t_axis, mean_avel - std_avel, mean_avel + std_avel, color="purple", alpha=0.2)
    axs[1, 1].text(0.5, 0.9, f"Final: {mean_avel[-1]:.3f} $\\pm$ {std_avel[-1]:.3f} rad/s", transform=axs[1, 1].transAxes, ha="center", bbox=dict(boxstyle="round", fc="thistle", alpha=0.5))
    axs[1, 1].set_title("Angular Velocity Error [rad/s]")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(perturbed_dir) / f"sensitivity_results_{POLICY_NAME}.png"
    fig.savefig(save_path, dpi=150)
    plt.close('all')
    print(f"[PLOT] Envelope plot saved successfully in {perturbed_dir}")

if __name__ == "__main__":
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    global CAMPAIGN_DIR
    
    CAMPAIGN_DIR = os.path.join("campaigns", f"{POLICY_NAME}_campaign_{TIMESTAMP}")
    NOMINAL_DIR = os.path.join(CAMPAIGN_DIR, "nominal")
    PERTURBED_DIR = os.path.join(CAMPAIGN_DIR, "perturbed")
    RAW_DATA_DIR = os.path.join(CAMPAIGN_DIR, "mc_data") # <- Rinominata per renderla visibile
    
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    os.makedirs(PERTURBED_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    config_src = os.path.join("configs", "policy", POLICY_NAME, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(CAMPAIGN_DIR, "config.json"))

    perturbations_file = os.path.join(CAMPAIGN_DIR, "perturbations_log.json")
    with open(perturbations_file, "w") as f:
        json.dump({"RUNS": PERTURBATIONS}, f, indent=4)
    
    cleanup()
    
    # === TEST NOMINALE ===
    nominal_npz_path = os.path.join(NOMINAL_DIR, "error_data.npz")
    attempt = 1
    while not os.path.exists(nominal_npz_path):
        if attempt > 1:
            print(f"\n[SYSTEM] RETRY ATTEMPT {attempt} for nominal run. Cooling down...")
            time.sleep(8) 
        run_test("nominal", {}, NOMINAL_DIR, skip_plots=False, npz_name="error_data.npz")
        attempt += 1
    
    # === TEST PERTURBATI ===
    for run_name, overrides in PERTURBATIONS.items():
        file_name = f"{run_name}.npz"
        target_npz_path = os.path.join(RAW_DATA_DIR, file_name)
        
        attempt = 1
        while not os.path.exists(target_npz_path):
            if attempt > 1:
                print(f"\n[SYSTEM] RETRY ATTEMPT {attempt} for {run_name}. Cooling down...")
                time.sleep(8)
            run_test(run_name, overrides, RAW_DATA_DIR, skip_plots=True, npz_name=file_name)
            attempt += 1
            
    plot_aggregated(CAMPAIGN_DIR, PERTURBED_DIR, RAW_DATA_DIR)