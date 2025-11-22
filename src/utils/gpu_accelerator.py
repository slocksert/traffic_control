"""
GPU Accelerator Module for SUMO Traffic Control

This module provides GPU acceleration capabilities for traffic simulation
and AI agent computations using PyTorch.

Since SUMO itself doesn't support GPU natively, this module accelerates:
1. AI agent computations (Q-learning, neural networks)
2. Traffic data processing and analysis
3. Batch operations on simulation data
"""

import numpy as np
import logging
import os
from typing import Union, Any

os.environ.setdefault("HIP_PATH", "/opt/rocm")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault(
    "HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
)

try:
    import torch

    HAS_TORCH = True

    HAS_TORCH_CUDA = torch.cuda.is_available()
    if HAS_TORCH_CUDA:
        try:
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                HAS_TORCH_ROCM = True
                gpu_name = torch.cuda.get_device_name(0)
                print(
                    f"GPU DETECTED: PyTorch ROCm {torch.version.hip} - "
                    f"{torch.cuda.device_count()} AMD GPU(s) ({gpu_name})"
                )
            else:
                HAS_TORCH_ROCM = False
                gpu_name = torch.cuda.get_device_name(0)
                print(
                    f"GPU DETECTED: PyTorch CUDA - "
                    f"{torch.cuda.device_count()} GPU(s) ({gpu_name})"
                )
        except Exception as e:
            HAS_TORCH_ROCM = False
            print(f"GPU DETECTED: PyTorch CUDA - {torch.cuda.device_count()} GPU(s)")
    else:
        HAS_TORCH_ROCM = False
        print("GPU NOT DETECTED: PyTorch found but no GPU support")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.version, "hip"):
            print(f"  - ROCm version: {torch.version.hip}")
        print("  - Check if ROCm/CUDA drivers are properly installed")
except ImportError:
    HAS_TORCH = False
    HAS_TORCH_CUDA = False
    HAS_TORCH_ROCM = False
    print("GPU NOT DETECTED: PyTorch not installed")

try:
    from numba import cuda

    HAS_NUMBA_CUDA = True and cuda.is_available()
    if HAS_NUMBA_CUDA:
        print(f"Numba CUDA detected: {len(cuda.gpus)} GPU(s) available")
except ImportError:
    HAS_NUMBA_CUDA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUAccelerator:
    """
    GPU acceleration for SUMO traffic control computations.

    Supports PyTorch for GPU operations with automatic fallback to CPU.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU accelerator.

        Args:
            device_id: GPU device ID to use (default: 0)
        """
        self.device_id = device_id
        self.device = self._setup_device()
        self.gpu_operations = 0
        self.cpu_operations = 0

    def _setup_device(self) -> str:
        """
        Setup and configure GPU device.

        Returns:
            str: Device type ('torch', 'cpu')
        """
        if HAS_TORCH and HAS_TORCH_CUDA:
            try:
                torch.cuda.set_device(self.device_id)

                if HAS_TORCH_ROCM:
                    print(f"PyTorch ROCm setup successful on device {self.device_id}")
                    print(f"HIP version: {torch.version.hip}")
                else:
                    print(f"PyTorch CUDA setup successful on device {self.device_id}")

                return "torch"
            except Exception as e:
                print(f"PyTorch GPU setup failed: {e}")

        print("Using CPU for computations")
        return "cpu"

    def to_device(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """
        Move array to GPU device.

        Args:
            array: NumPy array to move

        Returns:
            Array on GPU device or original array if GPU unavailable
        """
        if self.device == "torch" and HAS_TORCH:
            return torch.tensor(
                array, dtype=torch.float32, device=f"cuda:{self.device_id}"
            )
        return array

    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Move array back to CPU.

        Args:
            array: Array to move to CPU

        Returns:
            NumPy array on CPU
        """
        if self.device == "torch" and HAS_TORCH:
            if hasattr(array, "cpu"):
                return array.cpu().numpy()
        return np.asarray(array)

    def qlearning_update(
        self,
        q_table: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        lr: float,
        gamma: float,
    ) -> np.ndarray:
        """
        GPU-accelerated Q-learning update.

        Args:
            q_table: Q-table to update
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            lr: Learning rate
            gamma: Discount factor

        Returns:
            Updated Q-table
        """
        if self.device == "torch" and HAS_TORCH:
            return self._torch_qlearning_update(
                q_table, states, actions, rewards, next_states, lr, gamma
            )
        else:
            return self._cpu_qlearning_update(
                q_table, states, actions, rewards, next_states, lr, gamma
            )

    def _torch_qlearning_update(
        self, q_table, states, actions, rewards, next_states, lr, gamma
    ):
        """PyTorch implementation of Q-learning update"""
        try:
            self.gpu_operations += 1
            device = f"cuda:{self.device_id}"

            # Convert to tensors
            q_tensor = torch.tensor(q_table, dtype=torch.float32, device=device)
            states_tensor = torch.tensor(states, dtype=torch.long, device=device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states_tensor = torch.tensor(
                next_states, dtype=torch.long, device=device
            )

            # Q-learning: Q(s,a) = Q(s,a) + lr * (r + gamma *
            # max(Q(s',a')) - Q(s,a))
            current_q = q_tensor[states_tensor, actions_tensor]
            next_q_max = torch.max(q_tensor[next_states_tensor], dim=1)[0]
            target_q = rewards_tensor + gamma * next_q_max

            # Update Q-table
            q_tensor[states_tensor, actions_tensor] += lr * (target_q - current_q)

            return q_tensor.cpu().numpy()

        except Exception as e:
            logger.warning(
                f"PyTorch Q-learning update failed: {e}, falling back to CPU"
            )
            return self._cpu_qlearning_update(
                q_table, states, actions, rewards, next_states, lr, gamma
            )

    def _cpu_qlearning_update(
        self, q_table, states, actions, rewards, next_states, lr, gamma
    ):
        """CPU implementation of Q-learning update"""
        self.cpu_operations += 1
        q_table = np.copy(q_table)

        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]

            current_q = q_table[state, action]
            next_q_max = np.max(q_table[next_state])
            target_q = reward + gamma * next_q_max
            q_table[state, action] += lr * (target_q - current_q)

        return q_table

    def traffic_metrics(
        self,
        waiting_times: np.ndarray,
        speeds: np.ndarray,
        positions: np.ndarray,
    ) -> dict:
        """
        GPU-accelerated traffic metrics calculation.

        Args:
            waiting_times: Vehicle waiting times
            speeds: Vehicle speeds
            positions: Vehicle positions

        Returns:
            Dictionary with calculated metrics
        """
        if self.device == "torch" and HAS_TORCH:
            return self._torch_traffic_metrics(waiting_times, speeds, positions)
        else:
            return self._cpu_traffic_metrics(waiting_times, speeds, positions)

    def _torch_traffic_metrics(self, waiting_times, speeds, positions):
        """PyTorch implementation of traffic metrics"""
        try:
            self.gpu_operations += 1
            device = f"cuda:{self.device_id}"

            wt_tensor = torch.tensor(waiting_times, dtype=torch.float32, device=device)
            speeds_tensor = torch.tensor(speeds, dtype=torch.float32, device=device)
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)

            # Calculate metrics
            avg_waiting = torch.mean(wt_tensor).item()
            max_waiting = torch.max(wt_tensor).item()
            avg_speed = torch.mean(speeds_tensor).item()
            min_speed = torch.min(speeds_tensor).item()

            # Calculate density (vehicles per km)
            if len(positions) > 0:
                road_length = torch.max(pos_tensor) - torch.min(pos_tensor)
                density = len(positions) / max(
                    road_length.item() / 1000, 0.001
                )  # vehicles/km
            else:
                density = 0.0

            return {
                "avg_waiting_time": avg_waiting,
                "max_waiting_time": max_waiting,
                "avg_speed": avg_speed,
                "min_speed": min_speed,
                "density": density,
                "total_vehicles": len(positions),
            }

        except Exception as e:
            logger.warning(f"PyTorch traffic metrics failed: {e}, falling back to CPU")
            return self._cpu_traffic_metrics(waiting_times, speeds, positions)

    def _cpu_traffic_metrics(self, waiting_times, speeds, positions):
        """CPU implementation of traffic metrics"""
        self.cpu_operations += 1

        if len(waiting_times) == 0:
            return {
                "avg_waiting_time": 0.0,
                "max_waiting_time": 0.0,
                "avg_speed": 0.0,
                "min_speed": 0.0,
                "density": 0.0,
                "total_vehicles": 0,
            }

        if len(positions) > 0:
            road_length = np.max(positions) - np.min(positions)
            density = len(positions) / max(road_length / 1000, 0.001)
        else:
            density = 0.0

        return {
            "avg_waiting_time": np.mean(waiting_times),
            "max_waiting_time": np.max(waiting_times),
            "avg_speed": np.mean(speeds),
            "min_speed": np.min(speeds),
            "density": density,
            "total_vehicles": len(positions),
        }

    def batch_state_processing(
        self, states: np.ndarray, state_space_size: int
    ) -> np.ndarray:
        """
        Process batch of states for neural network input.

        Args:
            states: Raw state data
            state_space_size: Size of state space for normalization

        Returns:
            Processed and normalized states
        """
        if self.device == "torch" and HAS_TORCH:
            device = f"cuda:{self.device_id}"
            states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
            processed = torch.clamp(states_tensor / state_space_size, 0, 1)
            return processed.cpu().numpy()
        else:
            return np.clip(states / state_space_size, 0, 1)

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.device in ["torch"]

    def get_device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "gpu_available": self.is_gpu_available(),
            "gpu_operations": self.gpu_operations,
            "cpu_operations": self.cpu_operations,
        }

        if self.device == "torch" and HAS_TORCH and HAS_TORCH_CUDA:
            try:
                if HAS_TORCH_ROCM:
                    info["backend"] = "ROCm"
                    info["hip_version"] = torch.version.hip
                else:
                    info["backend"] = "CUDA"
                    info["cuda_version"] = torch.version.cuda

                info["gpu_name"] = torch.cuda.get_device_name(self.device_id)
                info["gpu_memory"] = torch.cuda.get_device_properties(
                    self.device_id
                ).total_memory
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
                info["backend"] = "Unknown"
        else:
            info["backend"] = "CPU"

        return info


def check_gpu_support() -> dict:
    """
    Check what GPU acceleration options are available.

    Returns:
        Dictionary with availability of different GPU backends
    """
    support = {
        "torch": HAS_TORCH,
        "torch_cuda": HAS_TORCH_CUDA,
        "torch_rocm": HAS_TORCH_ROCM,
        "numba_cuda": HAS_NUMBA_CUDA,
        "any_gpu": HAS_TORCH_CUDA or HAS_NUMBA_CUDA,
    }

    if HAS_TORCH_ROCM:
        support.update(
            {
                "rocm_path": os.environ.get("ROCM_PATH", "/opt/rocm"),
                "hip_path": os.environ.get("HIP_PATH", "/opt/rocm"),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
                "rocm_installed": os.path.exists("/opt/rocm"),
                "rocm_version": torch.version.hip if HAS_TORCH else None,
                "hip_version": torch.version.hip if HAS_TORCH else None,
                "torch_devices": (torch.cuda.device_count() if HAS_TORCH_CUDA else 0),
                "gpu_backend": "ROCm",
            }
        )
    elif HAS_TORCH_CUDA:
        support.update(
            {
                "cuda_version": torch.version.cuda if HAS_TORCH else None,
                "torch_devices": (torch.cuda.device_count() if HAS_TORCH_CUDA else 0),
                "gpu_backend": "CUDA",
            }
        )

    return support


def get_accelerator(device_id: int = 0) -> GPUAccelerator:
    """
    Get a GPU accelerator instance.

    Args:
        device_id: GPU device ID to use

    Returns:
        Configured GPUAccelerator instance
    """
    return GPUAccelerator(device_id)


if __name__ == "__main__":
    print("GPU Acceleration Test")
    print("=" * 50)

    support = check_gpu_support()
    print("GPU Support:")
    for key, value in support.items():
        print(f"  {key}: {value}")

    print("\nTesting accelerator...")
    accelerator = get_accelerator()

    info = accelerator.get_device_info()
    print("Device Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTesting operations...")

    test_array = np.random.rand(100, 10)
    gpu_array = accelerator.to_device(test_array)
    cpu_array = accelerator.to_cpu(gpu_array)

    print(f"Array moved to GPU and back: {np.allclose(test_array, cpu_array)}")

    print("GPU acceleration setup complete!")
