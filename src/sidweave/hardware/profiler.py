"""
Hardware detection and profiling implementation.
"""

import asyncio
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from sidweave.config import Settings

# Optional Windows-only import
try:
    import wmi  # type: ignore[import-not-found]
except ImportError:
    wmi = None


# Constants
NVIDIA_SMI_FIELDS = 4
NVIDIA_DEVICE_ID_INDEX = 4


@dataclass
class CPUInfo:
    """CPU information."""
    vendor: str
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: float | None = None
    instruction_sets: list[str] = field(default_factory=list)
    architecture: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "model": self.model,
            "cores_physical": self.cores_physical,
            "cores_logical": self.cores_logical,
            "frequency_mhz": self.frequency_mhz,
            "instruction_sets": self.instruction_sets,
            "architecture": self.architecture,
        }


@dataclass
class MemoryInfo:
    """Memory information."""
    total_gb: float
    available_gb: float
    swap_total_gb: float = 0.0
    swap_free_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gb": self.total_gb,
            "available_gb": self.available_gb,
            "swap_total_gb": self.swap_total_gb,
            "swap_free_gb": self.swap_free_gb,
        }


@dataclass
class GPUInfo:
    """GPU information."""
    vendor: str | None = None
    model: str | None = None
    vram_gb: float | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    device_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "model": self.model,
            "vram_gb": self.vram_gb,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "device_id": self.device_id,
            "is_available": self.is_available,
        }

    @property
    def is_available(self) -> bool:
        return self.vendor is not None


@dataclass
class OSInfo:
    """Operating system information."""
    name: str
    version: str
    arch: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "arch": self.arch,
            "python_version": self.python_version,
        }


@dataclass
class HardwareProfile:
    """Complete hardware profile."""
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: GPUInfo
    os: OSInfo
    available_backends: list[str] = field(default_factory=list)
    disk_free_gb: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "gpu": self.gpu.to_dict(),
            "os": self.os.to_dict(),
            "available_backends": self.available_backends,
            "disk_free_gb": self.disk_free_gb,
            "raw_data": self.raw_data,
        }

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()

    def get_inference_backends(self) -> list[str]:
        """Get list of available inference backends."""
        backends = ["cpu"]

        if self.gpu.is_available:
            if self.gpu.vendor == "NVIDIA":
                backends.extend(["cuda", "tensorrt"])
            elif self.gpu.vendor == "AMD":
                backends.append("rocm")
            elif self.gpu.vendor == "Apple":
                backends.append("metal")

        # Check for specific backends
        if shutil.which("llama-server") or shutil.which("llama.cpp"):
            backends.append("llama.cpp")

        return list(set(backends))


class HardwareProfiler:
    """
    Hardware detection and profiling.

    Detects CPU, memory, GPU, OS, and available inference backends.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cached_profile: HardwareProfile | None = None

    async def detect(self, force_refresh: bool = False) -> HardwareProfile:
        """
        Detect hardware configuration.

        Args:
            force_refresh: Force re-detection even if cached

        Returns:
            HardwareProfile with all detected information
        """
        if self._cached_profile and not force_refresh:
            return self._cached_profile

        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        profile = await loop.run_in_executor(None, self._detect_sync)

        self._cached_profile = profile
        return profile

    def _detect_sync(self) -> HardwareProfile:
        """Synchronous hardware detection."""
        cpu = self._detect_cpu()
        memory = self._detect_memory()
        gpu = self._detect_gpu()
        os_info = self._detect_os()
        disk_free = self._get_disk_free_gb()
        backends = self._detect_backends(cpu, gpu)

        return HardwareProfile(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            os=os_info,
            available_backends=backends,
            disk_free_gb=disk_free,
            raw_data={
                "cpu_raw": self._get_cpu_raw(),
                "memory_raw": self._get_memory_raw(),
                "gpu_raw": self._get_gpu_raw(),
            }
        )

    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU information."""
        # Physical/logical cores
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical = psutil.cpu_count(logical=True) or 1

        # Frequency
        freq = psutil.cpu_freq()
        frequency_mhz = freq.current if freq else None

        # Architecture
        arch = platform.machine().lower()

        # Vendor and model
        vendor = "Unknown"
        model = "Unknown"
        instruction_sets: list[str] = []

        if sys.platform == "linux":
            vendor, model, instruction_sets = self._parse_cpuinfo_linux()
        elif sys.platform == "darwin":
            vendor, model = self._parse_cpuinfo_macos()
        elif sys.platform == "win32":
            vendor, model, instruction_sets = self._parse_cpuinfo_windows()

        return CPUInfo(
            vendor=vendor,
            model=model,
            cores_physical=cores_physical,
            cores_logical=cores_logical,
            frequency_mhz=frequency_mhz,
            instruction_sets=instruction_sets,
            architecture=arch,
        )

    def _parse_cpuinfo_linux(self) -> tuple[str, str, list[str]]:
        """Parse /proc/cpuinfo on Linux."""
        vendor = "Unknown"
        model = "Unknown"
        instruction_sets = []

        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            with cpuinfo_path.open() as f:
                for line in f:
                    if line.startswith("vendor_id"):
                        vendor = line.split(":")[1].strip()
                    elif line.startswith("model name"):
                        model = line.split(":")[1].strip()
                    elif line.startswith("flags"):
                        flags = line.split(":")[1].strip().split()
                        instruction_sets = list(set(flags))
                        break
        except Exception:
            pass

        # Normalize vendor
        vendor_lower = vendor.lower()
        if "genuineintel" in vendor_lower:
            vendor = "Intel"
        elif "authenticamd" in vendor_lower:
            vendor = "AMD"
        elif "arm" in vendor_lower:
            vendor = "ARM"

        return vendor, model, instruction_sets

    def _parse_cpuinfo_macos(self) -> tuple[str, str]:
        """Parse CPU info on macOS."""
        vendor = "Apple"
        model = "Unknown"

        try:
            # Use sysctl
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
                check=False,
            )
            if result.returncode == 0:
                model = result.stdout.strip()
        except Exception:
            pass

        # Try to get more details
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.vendor"],
                capture_output=True, text=True, timeout=2,
                check=False,
            )
            if result.returncode == 0:
                vendor_raw = result.stdout.strip().lower()
                if "intel" in vendor_raw:
                    vendor = "Intel"
                elif "amd" in vendor_raw:
                    vendor = "AMD"
        except Exception:
            pass

        return vendor, model

    def _parse_cpuinfo_windows(self) -> tuple[str, str, list[str]]:
        """Parse CPU info on Windows."""
        vendor = "Unknown"
        model = "Unknown"
        instruction_sets: list[str] = []

        vendor, model = self._get_windows_cpu_vendor_model(vendor, model)
        instruction_sets = self._get_windows_instruction_sets(instruction_sets)

        return vendor, model, instruction_sets

    def _get_windows_cpu_vendor_model(
        self, vendor: str, model: str
    ) -> tuple[str, str]:
        """Get CPU vendor and model on Windows."""
        if wmi is not None:
            try:
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    vendor = processor.Manufacturer or "Unknown"
                    model = processor.Name or "Unknown"
                    break
            except Exception:
                pass

        if vendor == "Unknown" or model == "Unknown":
            try:
                cmd = (
                    "Get-WmiObject Win32_Processor | "
                    "Select-Object Manufacturer, Name | Format-List"
                )
                result = subprocess.run(
                    ["powershell", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "Manufacturer" in line:
                            vendor = line.split(":")[1].strip()
                        elif "Name" in line:
                            model = line.split(":")[1].strip()
            except Exception:
                pass

        return vendor, model

    def _get_windows_instruction_sets(
        self, instruction_sets: list[str]
    ) -> list[str]:
        """Get CPU instruction sets on Windows."""
        try:
            cmd = "[Environment]::GetEnvironmentVariable('PROCESSOR_IDENTIFIER')"
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                identifier = result.stdout.strip()
                if "Intel64" in identifier:
                    instruction_sets.append("x86_64")
                if "AMD64" in identifier:
                    instruction_sets.append("x86_64")
        except Exception:
            pass
        return instruction_sets

    def _detect_memory(self) -> MemoryInfo:
        """Detect memory information."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryInfo(
            total_gb=round(mem.total / (1024**3), 2),
            available_gb=round(mem.available / (1024**3), 2),
            swap_total_gb=round(swap.total / (1024**3), 2),
            swap_free_gb=round(swap.free / (1024**3), 2),
        )

    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU information."""
        # Try NVIDIA first
        gpu = self._detect_nvidia_gpu()
        if gpu.is_available:
            return gpu

        # Try AMD
        gpu = self._detect_amd_gpu()
        if gpu.is_available:
            return gpu

        # Try Apple Silicon
        gpu = self._detect_apple_gpu()
        if gpu.is_available:
            return gpu

        # Try Intel
        gpu = self._detect_intel_gpu()
        if gpu.is_available:
            return gpu

        return GPUInfo()

    def _detect_nvidia_gpu(self) -> GPUInfo:
        """Detect NVIDIA GPU via nvidia-smi."""
        NVIDIA_SMI_FIELDS = 4
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_capability,uuid",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.hardware.gpu_detection_timeout,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]  # First GPU
                parts = [p.strip() for p in line.split(",")]

                if len(parts) >= NVIDIA_SMI_FIELDS:
                    has_device_id = len(parts) > NVIDIA_DEVICE_ID_INDEX
                    device_id = parts[NVIDIA_DEVICE_ID_INDEX] if has_device_id else None
                    return GPUInfo(
                        vendor="NVIDIA",
                        model=parts[0],
                        vram_gb=round(int(parts[1]) / 1024, 2),
                        driver_version=parts[2],
                        compute_capability=parts[3],
                        device_id=device_id,
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return GPUInfo()

    def _detect_amd_gpu(self) -> GPUInfo:
        """Detect AMD GPU via rocm-smi or lspci."""
        # Try rocm-smi
        try:
            cmd = [
                "rocm-smi",
                "--showproductname",
                "--showvram",
                "--showdriverversion",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.hardware.gpu_detection_timeout,
                check=False,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                # Parse rocm-smi output (format varies)
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "GPU" in line and ":" in line:
                        model = line.split(":")[1].strip()
                        return GPUInfo(
                            vendor="AMD",
                            model=model,
                        )
        except (FileNotFoundError, Exception):
            pass

        # Try lspci
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            for line in result.stdout.splitlines():
                is_gpu = "VGA" in line or "Display" in line
                is_amd = "AMD" in line or "ATI" in line or "Radeon" in line
                if is_gpu and is_amd:
                    return GPUInfo(
                        vendor="AMD",
                        model=line.split(":")[-1].strip(),
                    )
        except (FileNotFoundError, Exception):
            pass

        return GPUInfo()

    def _detect_apple_gpu(self) -> GPUInfo:
        """Detect Apple Silicon GPU."""
        if sys.platform != "darwin":
            return GPUInfo()

        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip() == "1":
                # Get GPU cores
                gpu_cores = "Unknown"
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType", "-json"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    data = json.loads(result.stdout)
                    for display in data.get("SPDisplaysDataType", []):
                        if "sppci_model" in display:
                            gpu_cores = display.get("sppci_cores", "Unknown")
                            break
                except Exception:
                    pass

                # Get total memory (unified)
                mem = psutil.virtual_memory()

                return GPUInfo(
                    vendor="Apple",
                    model=f"Apple Silicon GPU ({gpu_cores} cores)",
                    vram_gb=round(mem.total / (1024**3) * 0.75, 2),  # Estimate
                    compute_capability="Metal",
                )
        except Exception:
            pass

        return GPUInfo()

    def _detect_intel_gpu(self) -> GPUInfo:
        """Detect Intel GPU (Arc, integrated)."""
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            for line in result.stdout.splitlines():
                is_gpu = "VGA" in line or "Display" in line
                is_intel_gpu = (
                    "Intel" in line
                    and ("Arc" in line or "Iris" in line or "UHD" in line or "HD Graphics" in line)
                )
                if is_gpu and is_intel_gpu:
                    return GPUInfo(
                        vendor="Intel",
                        model=line.split(":")[-1].strip(),
                    )
        except (FileNotFoundError, Exception):
            pass

        return GPUInfo()

    def _detect_os(self) -> OSInfo:
        """Detect OS information."""
        return OSInfo(
            name=platform.system(),
            version=platform.release(),
            arch=platform.machine(),
            python_version=platform.python_version(),
        )

    def _get_disk_free_gb(self) -> float:
        """Get free disk space in GB."""
        try:
            data_dir = self.settings.storage.data_dir
            usage = shutil.disk_usage(data_dir)
            return round(usage.free / (1024**3), 2)
        except Exception:
            return 0.0

    def _detect_backends(self, cpu: CPUInfo, gpu: GPUInfo) -> list[str]:
        """Detect available inference backends."""
        backends = ["cpu"]

        # Check for llama.cpp
        if shutil.which("llama-server") or shutil.which("llama-cli"):
            backends.append("llama.cpp")

        # Check for Ollama
        if shutil.which("ollama"):
            backends.append("ollama")

        # GPU backends
        if gpu.is_available:
            if gpu.vendor == "NVIDIA":
                backends.extend(["cuda", "tensorrt"])
            elif gpu.vendor == "AMD":
                backends.append("rocm")
            elif gpu.vendor == "Apple":
                backends.append("metal")
            elif gpu.vendor == "Intel":
                backends.append("openvino")

        # CPU instruction sets
        instruction_sets = set(cpu.instruction_sets)
        if any(s in instruction_sets for s in ["avx2", "avx512f", "amx"]):
            backends.append("mkldnn")

        return list(set(backends))

    def _get_cpu_raw(self) -> dict[str, Any]:
        """Get raw CPU info for debugging."""
        try:
            if sys.platform == "linux":
                cpuinfo_path = Path("/proc/cpuinfo")
                with cpuinfo_path.open() as f:
                    return {"cpuinfo": f.read()[:2000]}
        except Exception:
            pass
        return {}

    def _get_memory_raw(self) -> dict[str, Any]:
        """Get raw memory info."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "virtual": mem._asdict(),
            "swap": swap._asdict(),
        }

    def _get_gpu_raw(self) -> dict[str, Any]:
        """Get raw GPU info for debugging."""
        raw = {}
        for cmd, key in [
            (["nvidia-smi", "-q"], "nvidia_smi"),
            (["rocm-smi", "-a"], "rocm_smi"),
            (["lspci", "-nn"], "lspci"),
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
                if result.returncode == 0:
                    raw[key] = result.stdout[:2000]
            except Exception:
                pass
        return raw
