import subprocess
import time
import torch

class GPUPowerMonitor:
    def __init__(self):
        self.start_time = None
        self.total_power = 0.0
        self.num_samples = 0

    def start_monitoring(self):
        self.start_time = time.time()

    def record_power(self):
        try:
            # Run nvidia-smi command to get GPU power consumption
            cmd = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd, shell=True)
            power_str = result.decode("utf-8").strip()
            power_str_first = power_str.split('\n')[0]  # Extract the first value
            power = float(power_str_first)
            self.total_power += power
            self.num_samples += 1
        except Exception as e:
            print(f"Error recording GPU power: {e}")

    def stop_monitoring(self):
        if self.num_samples > 0:
            average_power = self.total_power / self.num_samples
            self.start_time = None
            self.total_power = 0.0
            self.num_samples = 0
            return average_power
        else:
            return None
            
def monitor_gpu_power():
    if gpu_monitor.start_time is None:
        gpu_monitor.start_monitoring()
        print("Started monitoring GPU power consumption.")
    else:
        average_power = gpu_monitor.stop_monitoring()
        if average_power is not None:
            print(f"Average GPU power consumption: {average_power} watts")
        else:
            print("No data recorded for GPU power consumption.")

# Create an instance of the GPUPowerMonitor class
gpu_monitor = GPUPowerMonitor()

def simulate_gpu_computation():
    # Create random tensors and perform matrix multiplication on GPU
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    c = torch.matmul(a, b)

    # Record GPU power consumption during computation
    gpu_monitor.record_power()

# Example usage to simulate GPU computation
monitor_gpu_power()  # Start monitoring GPU power
for _ in range(100):  # Simulate GPU computation 100 times
    simulate_gpu_computation()
monitor_gpu_power()  # Stop monitoring and display average GPU power
