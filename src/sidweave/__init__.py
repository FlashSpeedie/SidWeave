"""
SidWeave - Adaptive Runtime for AI Agents

An adaptive runtime that makes existing AI agents more capable, efficient, and reliable
without requiring users to replace their agents or completely change their workflow.
"""

__version__ = "0.1.0"
__author__ = "SidEra Labs"
__email__ = "team@sidweave.dev"

from sidweave.benchmark import (
    BUILTIN_BENCHMARKS,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
    BenchmarkTaskResult,
    load_benchmark_tasks,
    save_benchmark_tasks,
)
from sidweave.config import Settings, get_settings
from sidweave.core.runtime import SidWeaveRuntime

__all__ = [
    "BUILTIN_BENCHMARKS",
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "BenchmarkTaskResult",
    "Settings",
    "SidWeaveRuntime",
    "__version__",
    "get_settings",
    "load_benchmark_tasks",
    "save_benchmark_tasks",
]
