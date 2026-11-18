"""
Unit tests for hardware profiler.
"""

import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from sidweave.config import Settings
from sidweave.hardware import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    HardwareProfiler,
    MemoryInfo,
    OSInfo,
)


class TestCPUInfo:
    """Test CPUInfo dataclass."""

    def test_to_dict(self):
        cpu = CPUInfo(
            vendor="Intel",
            model="Core i7-12700K",
            cores_physical=12,
            cores_logical=20,
            frequency_mhz=3600.0,
            instruction_sets=["avx2", "avx512f"],
            architecture="x86_64",
        )

        d = cpu.to_dict()
        assert d["vendor"] == "Intel"
        assert d["model"] == "Core i7-12700K"
        assert d["cores_physical"] == 12
        assert d["cores_logical"] == 20
        assert d["frequency_mhz"] == 3600.0
        assert "avx2" in d["instruction_sets"]
        assert d["architecture"] == "x86_64"


class TestMemoryInfo:
    """Test MemoryInfo dataclass."""

    def test_to_dict(self):
        mem = MemoryInfo(
            total_gb=32.0,
            available_gb=24.5,
            swap_total_gb=8.0,
            swap_free_gb=8.0,
        )

        d = mem.to_dict()
        assert d["total_gb"] == 32.0
        assert d["available_gb"] == 24.5
        assert d["swap_total_gb"] == 8.0
        assert d["swap_free_gb"] == 8.0


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_to_dict_available(self):
        gpu = GPUInfo(
            vendor="NVIDIA",
            model="RTX 4090",
            vram_gb=24.0,
            driver_version="550.0",
            compute_capability="8.9",
            device_id="GPU-123",
        )

        d = gpu.to_dict()
        assert d["vendor"] == "NVIDIA"
        assert d["model"] == "RTX 4090"
        assert d["vram_gb"] == 24.0
        assert d["is_available"] is True

    def test_to_dict_unavailable(self):
        gpu = GPUInfo()

        d = gpu.to_dict()
        assert d["vendor"] is None
        assert d["is_available"] is False


class TestOSInfo:
    """Test OSInfo dataclass."""

    def test_to_dict(self):
        os_info = OSInfo(
            name="Linux",
            version="6.8.0",
            arch="x86_64",
            python_version="3.12.3",
        )

        d = os_info.to_dict()
        assert d["name"] == "Linux"
        assert d["version"] == "6.8.0"
        assert d["arch"] == "x86_64"
        assert d["python_version"] == "3.12.3"


class TestHardwareProfile:
    """Test HardwareProfile dataclass."""

    def test_to_dict(self):
        cpu = CPUInfo(vendor="Intel", model="i7", cores_physical=8, cores_logical=16)
        mem = MemoryInfo(total_gb=32.0, available_gb=20.0)
        gpu = GPUInfo(vendor="NVIDIA", model="RTX 3080", vram_gb=10.0)
        os_info = OSInfo(name="Linux", version="6.5", arch="x86_64", python_version="3.12")

        profile = HardwareProfile(
            cpu=cpu,
            memory=mem,
            gpu=gpu,
            os=os_info,
            available_backends=["cpu", "cuda", "ollama"],
            disk_free_gb=500.0,
        )

        d = profile.to_dict()
        assert d["cpu"]["vendor"] == "Intel"
        assert d["memory"]["total_gb"] == 32.0
        assert d["gpu"]["vendor"] == "NVIDIA"
        assert d["available_backends"] == ["cpu", "cuda", "ollama"]
        assert d["disk_free_gb"] == 500.0

    def test_get_inference_backends(self):
        cpu = CPUInfo(vendor="Intel", model="i7", cores_physical=8, cores_logical=16)
        mem = MemoryInfo(total_gb=32.0, available_gb=20.0)
        gpu = GPUInfo(vendor="NVIDIA", model="RTX 3080", vram_gb=10.0)
        os_info = OSInfo(name="Linux", version="6.5", arch="x86_64", python_version="3.12")

        profile = HardwareProfile(
            cpu=cpu,
            memory=mem,
            gpu=gpu,
            os=os_info,
            available_backends=["cpu", "cuda", "ollama"],
        )

        backends = profile.get_inference_backends()
        assert "cpu" in backends
        assert "cuda" in backends
        assert "tensorrt" in backends


class TestHardwareProfiler:
    """Test HardwareProfiler class."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def profiler(self, settings):
        return HardwareProfiler(settings)

    def test_init(self, profiler):
        assert profiler.settings is not None
        assert profiler._cached_profile is None

    @patch("sidweave.hardware.profiler.psutil.cpu_count")
    @patch("sidweave.hardware.profiler.psutil.cpu_freq")
    @patch("sidweave.hardware.profiler.psutil.virtual_memory")
    @patch("sidweave.hardware.profiler.psutil.swap_memory")
    @patch("sidweave.hardware.profiler.shutil.disk_usage")
    @patch("sidweave.hardware.profiler.shutil.which")
    @patch("sidweave.hardware.profiler.subprocess.run")
    @patch("pathlib.Path.open", new_callable=mock_open, read_data="vendor_id\t: GenuineIntel\nmodel name\t: Intel Core i7-12700K\nflags\t: avx2 avx512f\n")
    def test_detect_cpu_linux(
        self,
        mock_open_file,
        mock_run,
        mock_which,
        mock_disk,
        mock_swap,
        mock_vmem,
        mock_freq,
        mock_cpu_count,
        profiler,
    ):
        """Test CPU detection on Linux."""
        mock_cpu_count.side_effect = [8, 16]  # physical, logical
        mock_freq.return_value = MagicMock(current=3600.0)

        mock_vmem.return_value = MagicMock(total=32*1024**3, available=20*1024**3)
        mock_swap.return_value = MagicMock(total=8*1024**3, free=8*1024**3)
        mock_disk.return_value = MagicMock(free=500*1024**3)
        mock_which.return_value = None

        # Mock GPU detection to return unavailable (nvidia-smi, rocm-smi, lspci x2)
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="nvidia-smi not found"),  # nvidia-smi fails
            MagicMock(returncode=1, stderr="rocm-smi not found"),  # rocm-smi fails
            MagicMock(returncode=1, stderr="lspci not found"),  # lspci for amd fails
            MagicMock(returncode=1, stderr="lspci not found"),  # lspci for intel fails
        ]

        # We need to mock sys.platform to return "linux" and platform for OSInfo
        with patch.object(sys, "platform", "linux"):
            with patch("sidweave.hardware.profiler.platform.system", return_value="Linux"):
                with patch("sidweave.hardware.profiler.platform.machine", return_value="x86_64"):
                    with patch("sidweave.hardware.profiler.platform.release", return_value="6.5.0"):
                        with patch("sidweave.hardware.profiler.platform.python_version", return_value="3.12.3"):
                            profile = profiler._detect_sync()

        assert profile.cpu.vendor == "Intel"
        assert profile.cpu.cores_physical == 8
        assert profile.cpu.cores_logical == 16
        assert profile.cpu.frequency_mhz == 3600.0
        assert "avx2" in profile.cpu.instruction_sets
        assert profile.memory.total_gb == 32.0

    @patch("sidweave.hardware.profiler.psutil.cpu_count")
    @patch("sidweave.hardware.profiler.psutil.cpu_freq")
    @patch("sidweave.hardware.profiler.psutil.virtual_memory")
    @patch("sidweave.hardware.profiler.psutil.swap_memory")
    @patch("sidweave.hardware.profiler.shutil.disk_usage")
    @patch("sidweave.hardware.profiler.shutil.which")
    @patch("sidweave.hardware.profiler.subprocess.run")
    @patch("pathlib.Path.open", new_callable=mock_open, read_data="vendor_id\t: GenuineIntel\nmodel name\t: Intel Core i7\nflags\t: avx2\n")
    def test_detect_nvidia_gpu(
        self,
        mock_open_file,
        mock_run,
        mock_which,
        mock_disk,
        mock_swap,
        mock_vmem,
        mock_freq,
        mock_cpu_count,
        profiler,
    ):
        """Test NVIDIA GPU detection."""
        mock_cpu_count.side_effect = [8, 16]
        mock_freq.return_value = MagicMock(current=3600.0)
        mock_vmem.return_value = MagicMock(total=32*1024**3, available=20*1024**3)
        mock_swap.return_value = MagicMock(total=8*1024**3, free=8*1024**3)
        mock_disk.return_value = MagicMock(free=500*1024**3)
        mock_which.return_value = None

        # Mock nvidia-smi success
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 24576, 550.0, 8.9, GPU-123\n"),  # nvidia-smi
            MagicMock(returncode=1, stderr="rocm-smi not found"),  # rocm-smi fails
            MagicMock(returncode=1, stderr="lspci not found"),  # lspci for amd fails
            MagicMock(returncode=1, stderr="lspci not found"),  # lspci for intel fails
        ]

        with patch.object(sys, "platform", "linux"):
            with patch("sidweave.hardware.profiler.platform.system", return_value="Linux"):
                with patch("sidweave.hardware.profiler.platform.machine", return_value="x86_64"):
                    with patch("sidweave.hardware.profiler.platform.release", return_value="6.5.0"):
                        with patch("sidweave.hardware.profiler.platform.python_version", return_value="3.12.3"):
                            profile = profiler._detect_sync()

        assert profile.gpu.is_available is True
        assert profile.gpu.vendor == "NVIDIA"
        assert profile.gpu.model == "NVIDIA GeForce RTX 4090"
        assert profile.gpu.vram_gb == 24.0
        assert "cuda" in profile.available_backends
        assert "tensorrt" in profile.available_backends

    def test_detect_caching(self, profiler):
        """Test that detection results are cached."""
        with patch.object(profiler, "_detect_sync") as mock_detect:
            mock_detect.return_value = HardwareProfile(
                cpu=CPUInfo(vendor="Test", model="Test", cores_physical=4, cores_logical=8),
                memory=MemoryInfo(total_gb=16.0, available_gb=8.0),
                gpu=GPUInfo(),
                os=OSInfo(name="Linux", version="6.0", arch="x86_64", python_version="3.12"),
            )

            # First call
            import asyncio
            result1 = asyncio.run(profiler.detect())

            # Second call should use cache
            result2 = asyncio.run(profiler.detect())

            assert mock_detect.call_count == 1
            assert result1 is result2

    def test_force_refresh(self, profiler):
        """Test force refresh bypasses cache."""
        with patch.object(profiler, "_detect_sync") as mock_detect:
            mock_detect.return_value = HardwareProfile(
                cpu=CPUInfo(vendor="Test", model="Test", cores_physical=4, cores_logical=8),
                memory=MemoryInfo(total_gb=16.0, available_gb=8.0),
                gpu=GPUInfo(),
                os=OSInfo(name="Linux", version="6.0", arch="x86_64", python_version="3.12"),
            )

            import asyncio
            result1 = asyncio.run(profiler.detect())
            result2 = asyncio.run(profiler.detect(force_refresh=True))

            assert mock_detect.call_count == 2
