"""
Benchmark runner for comparing agent performance with and without SidWeave.

Provides:
- Benchmark task definitions
- Baseline and SidWeave execution modes
- Persistent result storage
- Comparative reporting
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sidweave.config import Settings, get_settings
from sidweave.hardware import HardwareProfiler
from sidweave.models import ModelCapability
from sidweave.models.registry import ModelRegistry
from sidweave.providers import (
    ChatCompletionRequest,
    ChatMessage,
    ProviderGateway,
)
from sidweave.routing import RoutingConstraints, RoutingEngine
from sidweave.storage import (
    BenchmarkRepository,
    BenchmarkRun,
    get_session,
)

try:
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable
    RICH_AVAILABLE = True
except ImportError:
    RichConsole = None  # type: ignore
    RichTable = None  # type: ignore
    RICH_AVAILABLE = False

Console = RichConsole
Table = RichTable

logger = logging.getLogger(__name__)


class BenchmarkMode(StrEnum):
    """Benchmark execution mode."""
    BASELINE = "baseline"
    SIDWEAVE = "sidweave"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    name: str
    prompt: str
    category: str = "general"
    expected_capabilities: list[str] = field(default_factory=list)
    timeout: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTaskResult:
    """Result of a single benchmark task."""
    task_name: str
    success: bool
    latency_ms: float | None = None
    tokens_used: int | None = None
    model_used: str | None = None
    provider_used: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""
    run_id: str
    name: str
    mode: BenchmarkMode
    agent_name: str
    started_at: datetime
    completed_at: datetime | None = None
    task_results: list[BenchmarkTaskResult] = field(default_factory=list)
    hardware_profile: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def total_tasks(self) -> int:
        return len(self.task_results)

    @property
    def passed_tasks(self) -> int:
        return sum(1 for r in self.task_results if r.success)

    @property
    def failed_tasks(self) -> int:
        return self.total_tasks - self.passed_tasks

    @property
    def avg_latency_ms(self) -> float | None:
        latencies = [r.latency_ms for r in self.task_results if r.latency_ms]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.task_results if r.tokens_used)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "mode": self.mode.value,
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "task_results": [asdict(r) for r in self.task_results],
            "hardware_profile": self.hardware_profile,
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """
    Runs benchmarks comparing baseline and SidWeave execution modes.

    In baseline mode, tasks are executed directly via the provider gateway
    without routing optimization. In SidWeave mode, the routing engine
    selects the best model for each task.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider_gateway: ProviderGateway | None = None,
        routing_engine: RoutingEngine | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider_gateway = provider_gateway
        self.routing_engine = routing_engine
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the benchmark runner."""
        if self.provider_gateway is None:
            self.provider_gateway = ProviderGateway(self.settings)
            await self.provider_gateway.initialize()

        if self.routing_engine is None:
            self.routing_engine = RoutingEngine(self.settings)
            await self.routing_engine.initialize()

        # Ensure model registry is initialized with NVIDIA NIM models
        model_registry = ModelRegistry(self.settings)
        await model_registry.initialize()

        self._initialized = True
        logger.info("Benchmark runner initialized")

    async def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        mode: BenchmarkMode,
        agent_name: str = "test-agent",
        benchmark_name: str = "benchmark",
    ) -> BenchmarkResult:
        """
        Run a complete benchmark.

        Args:
            tasks: List of benchmark tasks to execute
            mode: Execution mode (baseline or sidweave)
            agent_name: Name of the agent being benchmarked
            benchmark_name: Name for this benchmark run

        Returns:
            BenchmarkResult with all task results and aggregated metrics
        """
        if not self._initialized:
            await self.initialize()

        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.now(UTC)

        logger.info(f"Starting benchmark {benchmark_name} ({mode.value}) with {len(tasks)} tasks")

        # Get hardware profile for context
        hardware_profile = None
        try:
            profiler = HardwareProfiler(self.settings)
            hw = await profiler.detect()
            hardware_profile = hw.to_dict()
        except Exception:
            logger.warning("Could not detect hardware profile")

        result = BenchmarkResult(
            run_id=run_id,
            name=benchmark_name,
            mode=mode,
            agent_name=agent_name,
            started_at=started_at,
            hardware_profile=hardware_profile,
        )

        # Execute each task
        for task in tasks:
            task_result = await self._execute_task(task, mode)
            result.task_results.append(task_result)
            logger.info(f"Task {task.name}: {'PASS' if task_result.success else 'FAIL'}")

        result.completed_at = datetime.now(UTC)

        # Persist results
        await self._persist_result(result)

        logger.info(
            f"Benchmark {benchmark_name} ({mode.value}) completed: "
            f"{result.passed_tasks}/{result.total_tasks} passed, "
            f"avg latency: {result.avg_latency_ms:.1f}ms, "
            f"tokens: {result.total_tokens}"
        )

        return result

    async def _execute_task(
        self,
        task: BenchmarkTask,
        mode: BenchmarkMode,
    ) -> BenchmarkTaskResult:
        """Execute a single benchmark task."""
        start_time = time.perf_counter()

        try:
            if mode == BenchmarkMode.BASELINE:
                task_result = await self._execute_baseline(task)
            else:
                task_result = await self._execute_sidweave(task)

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("Task %s failed", task.name)
            return BenchmarkTaskResult(
                task_name=task.name,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )

        else:
            latency_ms = (time.perf_counter() - start_time) * 1000
            task_result.latency_ms = latency_ms
            return task_result

    async def _execute_baseline(self, task: BenchmarkTask) -> BenchmarkTaskResult:
        """Execute task in baseline mode (direct provider call)."""
        assert self.provider_gateway is not None
        providers = await self.provider_gateway.list_providers()
        if not providers:
            raise RuntimeError("No providers available for baseline")  # noqa: TRY003

        # Find first healthy provider with available models
        provider = None
        model_name = None
        for p in providers:
            if p.healthy and p.models:
                provider = p
                model_name = p.models[0]
                break

        if provider is None:
            # Fallback: try to find any model from any provider
            for p in providers:
                if p.models:
                    provider = p
                    model_name = p.models[0]
                    break

        if provider is None or model_name is None:
            raise RuntimeError("No providers with available models for baseline")  # noqa: TRY003

        request = ChatCompletionRequest(
            model=model_name,
            messages=[ChatMessage(role="user", content=task.prompt)],
            max_tokens=1000,
            temperature=0.7,
        )

        assert self.provider_gateway is not None
        response = await self.provider_gateway.chat_completion(request)

        usage = response.usage or {}
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

        finish_reason = None
        if response.choices:
            choice = response.choices[0]
            if isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")

        return BenchmarkTaskResult(
            task_name=task.name,
            success=True,
            tokens_used=total_tokens,
            model_used=response.model,
            provider_used=response.provider,
            metadata={
                "latency_ms": response.latency_ms,
                "finish_reason": finish_reason,
            },
        )

    async def _execute_sidweave(self, task: BenchmarkTask) -> BenchmarkTaskResult:
        """Execute task using SidWeave routing."""
        capabilities = None
        if task.expected_capabilities:
            capabilities = [ModelCapability(c) for c in task.expected_capabilities]

        constraints = None
        if capabilities:
            constraints = RoutingConstraints(required_capabilities=capabilities)

        assert self.routing_engine is not None
        # Route to best model
        routing_result = await self.routing_engine.route(
            task_category=task.category,
            required_capabilities=capabilities,
            constraints=constraints,
        )

        selected_model = routing_result.selected_model
        selected_provider = routing_result.selected_provider

        # Execute with selected model
        request = ChatCompletionRequest(
            model=selected_model.provider_model_id,
            messages=[ChatMessage(role="user", content=task.prompt)],
            max_tokens=1000,
            temperature=0.7,
        )

        assert self.provider_gateway is not None
        response = await self.provider_gateway.chat_completion(
            request,
            preferred_provider=selected_provider.name,
        )

        usage = response.usage or {}
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

        finish_reason = None
        if response.choices:
            choice = response.choices[0]
            if isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")

        return BenchmarkTaskResult(
            task_name=task.name,
            success=True,
            tokens_used=total_tokens,
            model_used=selected_model.name,
            provider_used=selected_provider.name,
            metadata={
                "latency_ms": response.latency_ms,
                "routing_decision_time_ms": routing_result.decision_time_ms,
                "fallback_models": [m.name for m in routing_result.fallback_models],
                "routing_rationale": routing_result.rationale,
                "finish_reason": finish_reason,
            },
        )

    async def _persist_result(self, result: BenchmarkResult) -> None:
        """Persist benchmark result to database."""
        async with get_session() as session:
            repo = BenchmarkRepository(session)
            run = BenchmarkRun(
                name=result.name,
                description=result.metadata.get("description"),
                benchmark_version="0.1.0",
                sidweave_version="0.1.0",
                mode=result.mode.value,
                agent_name=result.agent_name,
                total_tasks=result.total_tasks,
                passed_tasks=result.passed_tasks,
                failed_tasks=result.failed_tasks,
                avg_latency_ms=result.avg_latency_ms or 0.0,
                total_tokens=result.total_tokens,
                total_cost=0.0,
                tool_success_rate=result.success_rate,
                verification_rate=0.0,
                results=result.to_dict(),
                hardware_profile=result.hardware_profile,
            )
            await repo.add_run(run)

    async def run_comparison(
        self,
        tasks: list[BenchmarkTask],
        agent_name: str = "test-agent",
        benchmark_name: str = "comparison",
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Run both baseline and SidWeave modes and return both results.

        Args:
            tasks: List of benchmark tasks
            agent_name: Name of the agent
            benchmark_name: Base name for the runs

        Returns:
            Tuple of (baseline_result, sidweave_result)
        """
        logger.info(f"Running comparison benchmark: {benchmark_name}")

        # Run baseline
        baseline = await self.run_benchmark(
            tasks=tasks,
            mode=BenchmarkMode.BASELINE,
            agent_name=agent_name,
            benchmark_name=f"{benchmark_name}_baseline",
        )

        # Run SidWeave
        sidweave = await self.run_benchmark(
            tasks=tasks,
            mode=BenchmarkMode.SIDWEAVE,
            agent_name=agent_name,
            benchmark_name=f"{benchmark_name}_sidweave",
        )

        return baseline, sidweave

    def print_comparison(self, baseline: BenchmarkResult, sidweave: BenchmarkResult) -> None:
        """Print a comparison report."""
        if not RICH_AVAILABLE:
            raise RuntimeError("rich library required for comparison")  # noqa: TRY003
        console = Console()

        table = Table(title="Benchmark Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("SidWeave", style="green")
        table.add_column("Delta", style="magenta")

        def fmt_latency(v: float | None) -> str:
            return f"{v:.1f}ms" if v else "N/A"

        def fmt_rate(v: float) -> str:
            return f"{v:.1%}"

        def fmt_tokens(v: int) -> str:
            return f"{v:,}"

        def delta_str(
            baseline_val: float | None,
            sidweave_val: float | None,
            higher_better: bool = False,
        ) -> str:
            if baseline_val is None or sidweave_val is None:
                return "N/A"
            if baseline_val == 0:
                return "N/A"
            pct = ((sidweave_val - baseline_val) / baseline_val) * 100
            if higher_better:
                sign = "+" if pct > 0 else ""
            else:
                sign = "+" if pct < 0 else ""
                pct = -pct
            return f"{sign}{pct:.1f}%"

        table.add_row("Total Tasks", str(baseline.total_tasks), str(sidweave.total_tasks), "")
        table.add_row("Passed", str(baseline.passed_tasks), str(sidweave.passed_tasks), "")
        table.add_row("Failed", str(baseline.failed_tasks), str(sidweave.failed_tasks), "")
        sr_base = fmt_rate(baseline.success_rate)
        sr_sw = fmt_rate(sidweave.success_rate)
        sr_delta = delta_str(baseline.success_rate, sidweave.success_rate, True)
        table.add_row("Success Rate", sr_base, sr_sw, sr_delta)

        lat_base = fmt_latency(baseline.avg_latency_ms)
        lat_sw = fmt_latency(sidweave.avg_latency_ms)
        lat_delta = delta_str(baseline.avg_latency_ms, sidweave.avg_latency_ms, False)
        table.add_row("Avg Latency", lat_base, lat_sw, lat_delta)

        tok_base = fmt_tokens(baseline.total_tokens)
        tok_sw = fmt_tokens(sidweave.total_tokens)
        tok_delta = delta_str(baseline.total_tokens, sidweave.total_tokens, False)
        table.add_row("Total Tokens", tok_base, tok_sw, tok_delta)

        console.print(table)


# Built-in example benchmarks
BUILTIN_BENCHMARKS: dict[str, list[BenchmarkTask]] = {
    "quick": [
        BenchmarkTask(
            name="simple_math",
            prompt="What is 42 * 17?",
            category="reasoning",
            expected_capabilities=["reasoning"],
        ),
        BenchmarkTask(
            name="code_snippet",
            prompt="Write a Python function to reverse a string.",
            category="coding",
            expected_capabilities=["code_generation"],
        ),
        BenchmarkTask(
            name="creative_writing",
            prompt="Write a haiku about artificial intelligence.",
            category="chat",
            expected_capabilities=["chat", "completion"],
        ),
    ],
    "coding": [
        BenchmarkTask(
            name="python_class",
            prompt="Create a Python class for a binary search tree with insert and search methods.",
            category="coding",
            expected_capabilities=["code_generation", "reasoning"],
        ),
        BenchmarkTask(
            name="debug_code",
            prompt="Find the bug in this code: def factorial(n): return n * factorial(n-1)",
            category="coding",
            expected_capabilities=["code_generation", "reasoning", "tools"],
        ),
        BenchmarkTask(
            name="api_design",
            prompt=(
                "Design a REST API for a todo list application. "
                "Include endpoints, request/response formats."
            ),
            category="coding",
            expected_capabilities=["code_generation", "reasoning"],
        ),
    ],
    "reasoning": [
        BenchmarkTask(
            name="logic_puzzle",
            prompt="If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
            category="reasoning",
            expected_capabilities=["reasoning"],
        ),
        BenchmarkTask(
            name="math_problem",
            prompt=(
                "A train leaves Chicago at 60mph. Another leaves Detroit at 80mph. "
                "They are 300 miles apart. When do they meet?"
            ),
            category="reasoning",
            expected_capabilities=["reasoning"],
        ),
    ],
}


def load_benchmark_tasks(source: str | Path) -> list[BenchmarkTask]:
    """Load benchmark tasks from a JSON file."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError("Benchmark file missing")  # noqa: TRY003

    with path.open() as f:
        data = json.load(f)

    return [BenchmarkTask(**item) for item in data]


def save_benchmark_tasks(tasks: list[BenchmarkTask], path: str | Path) -> None:
    """Save benchmark tasks to a JSON file."""
    path = Path(path)
    data = [asdict(t) for t in tasks]
    with path.open("w") as f:
        json.dump(data, f, indent=2)


__all__ = [
    "BUILTIN_BENCHMARKS",
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "BenchmarkTaskResult",
    "load_benchmark_tasks",
    "save_benchmark_tasks",
]
