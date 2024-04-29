# GPU Power Monitor

This Python script monitors the power consumption of a GPU using the `nvidia-smi` command. It can be integrated into your code to track GPU power during specific operations.

## Installation

1. Ensure you have NVIDIA GPU drivers installed.
2. Install Python (3.6 or higher) and PyTorch.
3. Clone or download this repository.

## Usage

1. Import the `GPUPowerMonitor` class from `gpu_power_monitor.py` into your Python script.
2. Create an instance of `GPUPowerMonitor` (`gpu_monitor = GPUPowerMonitor()`).
3. Call `start_monitoring()` before starting the GPU-intensive operation.
4. Call `record_power()` within the operation to track GPU power.
5. After the operation, call `stop_monitoring()` to get the average GPU power consumption.

Example:

```python
from gpu_power_monitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

def simulate_gpu_computation():
    gpu_monitor.record_power()  # Record GPU power consumption
    # Perform GPU-intensive operations here

gpu_monitor.start_monitoring()  # Start monitoring GPU power
for _ in range(100):  # Simulate GPU computation 100 times
    simulate_gpu_computation()
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption: {average_power} watts")
