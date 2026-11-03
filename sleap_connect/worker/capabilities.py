"""Worker capabilities detection and job compatibility checking.

This module provides GPU hardware detection, job compatibility evaluation,
and resource utilization reporting for sleap-connect workers.
"""

import logging
from typing import Any, Dict, List, Optional


class WorkerCapabilities:
    """Manages worker hardware capabilities and job compatibility.

    This class handles GPU detection, job requirement matching, duration estimation,
    and resource utilization reporting for worker nodes.

    Attributes:
        gpu_id: GPU device ID to use (default 0).
        gpu_memory_mb: Total GPU memory in megabytes.
        gpu_model: GPU model name (e.g., "NVIDIA RTX 3090").
        cuda_version: CUDA version string (e.g., "11.8").
        supported_models: List of supported model types.
        supported_job_types: List of supported job types.
        status: Current worker status ("available", "busy", "reserved").
    """

    def __init__(
        self,
        gpu_id: int = 0,
        supported_models: Optional[List[str]] = None,
        supported_job_types: Optional[List[str]] = None,
    ):
        """Initialize worker capabilities.

        Args:
            gpu_id: GPU device ID to use (default 0).
            supported_models: List of supported model types. Defaults to
                ["base", "centroid", "topdown"].
            supported_job_types: List of supported job types. Defaults to
                ["training", "inference"].
        """
        self.gpu_id = gpu_id
        self.gpu_memory_mb = self._detect_gpu_memory()
        self.gpu_model = self._detect_gpu_model()
        self.cuda_version = self._detect_cuda_version()
        self.supported_models = supported_models or ["base", "centroid", "topdown"]
        self.supported_job_types = supported_job_types or ["training", "inference"]
        self.status = "available"  # Managed by StateManager, but read here

    def _detect_gpu_memory(self) -> int:
        """Detect GPU memory in MB.

        Returns:
            GPU memory in MB, or 0 if no GPU available.
        """
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > self.gpu_id:
                return torch.cuda.get_device_properties(self.gpu_id).total_memory // (
                    1024 * 1024
                )
        except (ImportError, RuntimeError) as e:
            logging.warning(f"Failed to detect GPU memory: {e}")
        return 0

    def _detect_gpu_model(self) -> str:
        """Detect GPU model name.

        Returns:
            GPU model name, or result of nvidia-smi fallback, or "CPU".
        """
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > self.gpu_id:
                name = torch.cuda.get_device_properties(self.gpu_id).name
                if name:
                    return name
        except (ImportError, RuntimeError) as e:
            logging.warning(f"Failed to detect GPU model via torch: {e}")

        # Fallback: try nvidia-smi (handles HPC environments where torch name is empty)
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu=name",
                    "--format=csv,noheader",
                    f"--id={self.gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                name = result.stdout.strip().splitlines()[0].strip()
                if name:
                    return name
        except Exception as e:
            logging.warning(f"Failed to detect GPU model via nvidia-smi: {e}")

        return "CPU"

    def _detect_cuda_version(self) -> str:
        """Detect CUDA version.

        Returns:
            CUDA version string, or "N/A" if CUDA not available.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.version.cuda if torch.version.cuda else "N/A"
        except (ImportError, RuntimeError) as e:
            logging.warning(f"Failed to detect CUDA version: {e}")
        return "N/A"

    def check_job_compatibility(self, request: Dict) -> bool:
        """Check if this worker can handle the job.

        Evaluates job requirements against worker capabilities including
        GPU memory, model type support, and job type support.

        Args:
            request: Job request dictionary containing:
                - config: Job configuration with model_type
                - requirements: Hardware requirements (min_gpu_memory_mb)
                - job_type: Type of job ("training" or "inference")

        Returns:
            True if worker can handle the job, False otherwise.
        """
        job_spec = request.get("config", {})
        requirements = request.get("requirements", {})

        # Check GPU memory
        min_gpu_mb = requirements.get("min_gpu_memory_mb", 0)
        if self.gpu_memory_mb < min_gpu_mb:
            logging.info(
                f"Job requires {min_gpu_mb}MB GPU memory, worker has {self.gpu_memory_mb}MB"
            )
            return False

        # Check model support
        model_type = job_spec.get("model_type")
        if model_type and model_type not in self.supported_models:
            logging.info(
                f"Job requires model type '{model_type}', worker supports {self.supported_models}"
            )
            return False

        # Check job type
        job_type = request.get("job_type")
        if job_type and job_type not in self.supported_job_types:
            logging.info(
                f"Job type '{job_type}' not supported, worker supports {self.supported_job_types}"
            )
            return False

        return True

    def estimate_job_duration(self, request: Dict) -> int:
        """Estimate job duration in minutes.

        Provides rough estimates based on job type and configuration.

        Args:
            request: Job request dictionary containing:
                - job_type: "training" or "inference"
                - config: Job configuration (epochs for training)
                - dataset_info: Dataset information (frame_count for inference)

        Returns:
            Estimated duration in minutes.
        """
        job_type = request.get("job_type", "training")
        config = request.get("config", {})

        if job_type == "training":
            epochs = config.get("epochs", 100)
            # Rough estimate: 0.5 minutes per epoch
            return int(epochs * 0.5)
        elif job_type == "inference":
            frame_count = request.get("dataset_info", {}).get("frame_count", 1000)
            # Rough estimate: 100 frames per minute
            return max(1, int(frame_count / 100))

        return 60  # Default 60 minutes

    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization as a float between 0.0 and 1.0.
        """
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > self.gpu_id:
                # Simple check: return 0.0 if available, 0.9 if busy
                return 0.0 if self.status == "available" else 0.9
        except (ImportError, RuntimeError):
            pass
        return 0.0

    def get_available_memory(self) -> int:
        """Get available GPU memory in MB.

        Returns:
            Available memory in MB.
        """
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > self.gpu_id:
                free_memory = torch.cuda.mem_get_info(self.gpu_id)[0]
                return int(free_memory / (1024 * 1024))
        except (ImportError, RuntimeError):
            pass
        return self.gpu_memory_mb

    def to_metadata_dict(self) -> Dict:
        """Convert capabilities to metadata dictionary for registration.

        Returns:
            Dictionary with worker capabilities for signaling server metadata.
        """
        return {
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_model": self.gpu_model,
            "cuda_version": self.cuda_version,
            "supported_models": self.supported_models,
            "supported_job_types": self.supported_job_types,
        }
