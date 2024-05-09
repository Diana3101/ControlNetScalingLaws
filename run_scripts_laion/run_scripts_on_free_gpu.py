import subprocess
import os
import time

def check_gpu_processes():
    # Check if there are any GPU processes running
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())

def run_after_gpu_processes(command_to_run):
    while check_gpu_processes() > 0:
        print("Waiting for GPU processes to finish...")
        time.sleep(600)  # Adjust the delay as needed

    print("All GPU processes finished. Running the command now.")
    subprocess.run(command_to_run, shell=True)

# Example usage:
bash_command = "./run_scripts_laion/run_scripts_depth.sh"
run_after_gpu_processes(bash_command)