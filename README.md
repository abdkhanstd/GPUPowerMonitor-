**GPU Power Monitor**
=====================

This Python script monitors the power consumption of a GPU using the `nvidia-smi` command. It can be integrated into your code to track GPU power during specific operations.

**Getting Started**
---------------

### Prerequisites

* NVIDIA GPU drivers installed
* Python 3.6 or higher
* PyTorch/Keras/TensorFlow installed
* `nvidia-smi` command-line utility installed

### Installation

1. Clone or download this repository.
2. Install the required Python modules: `pip install torch`

### Usage
---------------

1. Import the `GPUPowerMonitor` class from `GPUPowerMonitor.py` into your Python script.
2. Create an instance of `GPUPowerMonitor` (`gpu_monitor = GPUPowerMonitor()`).
3. Call `start_monitoring()` before starting the GPU-intensive operation.
4. Call `record_power()` within the operation to track GPU power.
5. After the operation, call `stop_monitoring()` to get the average GPU power consumption.

**Example Usage**
----------------

### Simulating GPU Computation

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

def simulate_gpu_computation():
    gpu_monitor.record_power()  # Record GPU power consumption
    # Perform GPU-intensive operations here
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.matmul(a, b)

gpu_monitor.start_monitoring()  # Start monitoring GPU power
for _ in range(100):  # Simulate GPU computation 100 times
    simulate_gpu_computation()
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption: {average_power} watts")
```

### Training a Deep Learning Model

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch
import torch.nn as nn
import torch.optim as optim

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

gpu_monitor.start_monitoring()  # Start monitoring GPU power
for epoch in range(10):  # Train the model for 10 epochs
    for x, y in dataset:
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        gpu_monitor.record_power()  # Record GPU power consumption
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption during training: {average_power} watts")
```

### Measuring Power Consumption of a Single Operation

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

gpu_monitor.start_monitoring()  # Start monitoring GPU power
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = torch.matmul(a, b)
gpu_monitor.record_power()  # Record GPU power consumption
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption during matrix multiplication: {average_power} watts")
```

### Measuring Power Consumption of a GPU-Intensive Function

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

def gpu_intensive_function():
    gpu_monitor.record_power()  # Record GPU power consumption
    # Perform GPU-intensive operations here
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.matmul(a, b)
    return c

gpu_monitor.start_monitoring()  # Start monitoring GPU power
result = gpu_intensive_function()
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption during function execution: {average_power} watts")
```

### Measuring Power Consumption of a Loop with GPU Operations

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

gpu_monitor.start_monitoring()  # Start monitoring GPU power
for i in range(100):
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.matmul(a, b)
    gpu_monitor.record_power()  # Record GPU power consumption
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption during loop execution: {average_power} watts")
```

### Measuring Power Consumption of a PyTorch DataLoader

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch
from torch.utils.data import DataLoader

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

gpu_monitor.start_monitoring()  # Start monitoring GPU power
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in data_loader:
    # Perform GPU-intensive operations here
    gpu_monitor.record_power()  # Record GPU power consumption
average_power = gpu_monitor.stop_monitoring()  # Stop monitoring and get average power
print(f"Average GPU power consumption during data loading: {average_power} watts")
```

### Measuring Power Consumption with Baseline Subtraction

```python
from GPUPowerMonitor import GPUPowerMonitor
import torch

# Create an instance of GPUPowerMonitor
gpu_monitor = GPUPowerMonitor()

# Record baseline power consumption
gpu_monitor.start_monitoring()
baseline_power = gpu_monitor.record_power()
gpu_monitor.stop_monitoring()

# Perform GPU-intensive operations
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = torch.matmul(a, b)

# Record power consumption during operation
gpu_monitor.start_monitoring()
operation_power = gpu_monitor.record_power()
gpu_monitor.stop_monitoring()

# Calculate actual power consumption by subtracting baseline power
actual_power = operation_power - baseline_power
print(f"Actual GPU power consumption during operation: {actual_power} watts")
```

**Important Notes**
-----------------

* Make sure no other program is using the GPU while running this script, as it may affect the accuracy of the power consumption measurement.
* Alternatively, you can record the GPU power consumption before and after your computation, and subtract the baseline power consumption from the total power consumption to get the actual power consumption of your computation.

**Troubleshooting**
-----------------

* If you encounter issues with the `nvidia-smi` command, ensure that it is installed and configured correctly on your system.
* If you encounter issues with the `torch` module, ensure that it is installed and configured correctly on your system.

**License**
-------

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Contributing**
------------

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.
