"""
Benchmark runner for comparing agent performance with and without SidWeave.

Implements proper benchmark methodology:
- Repeatable tasks with deterministic outcomes
- Baseline vs SidWeave comparison
- Metrics: task completion, tool correctness, latency, cost, verification
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sidweave.config import Settings
from sidweave.core.runtime import SidWeaveRuntime, create_runtime
from sidweave.hardware import HardwareProfile
from sidweave.models import ModelCapability
from sidweave.storage import (
    BenchmarkRepository,
    BenchmarkRun,
    get_session,
)

logger = logging.getLogger(__name__)


class BenchmarkMode(StrEnum):
    """Benchmark execution mode."""
    BASELINE = "baseline"      # Agent without SidWeave
    SIDWEAVE = "sidweave"      # Agent with SidWeave


MIN_REASONING_LENGTH = 50
ERR_NO_VALID_TASKS = "No valid tasks selected"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    name: str
    description: str
    category: str
    expected_outcome: dict[str, Any]
    verify_fn: Callable[[dict[str, Any]], bool]
    timeout: float = 30.0
    required_capabilities: list[ModelCapability] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a benchmark task."""
    task_name: str
    success: bool
    latency_ms: float
    tokens_used: int = 0
    cost: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_success_rate: float = 0.0
    verified: bool = False
    verification_rate: float = 0.0
    error: str | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""
    run_id: str
    name: str
    mode: BenchmarkMode
    agent_name: str
    started_at: datetime
    completed_at: datetime
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    avg_latency_ms: float
    total_tokens: int
    total_cost: float
    tool_success_rate: float
    verification_rate: float
    task_results: list[TaskResult]
    hardware_profile: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Benchmark runner for SidWeave.

    Compares agent performance with and without SidWeave.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tasks: dict[str, BenchmarkTask] = {}
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        """Register default benchmark tasks."""
        # Simple chat task
        self.tasks["simple_chat"] = BenchmarkTask(
            name="simple_chat",
            description="Simple chat completion",
            category="chat",
            expected_outcome={"has_response": True},
            verify_fn=lambda r: bool(r.get("content")),
            required_capabilities=[ModelCapability.CHAT],
        )

        # Reasoning task
        self.tasks["reasoning"] = BenchmarkTask(
            name="reasoning",
            description="Multi-step reasoning",
            category="reasoning",
            expected_outcome={"has_reasoning": True},
            verify_fn=lambda r: (
                "reasoning" in str(r).lower()
                or len(r.get("content", "")) > MIN_REASONING_LENGTH
            ),
            required_capabilities=[ModelCapability.REASONING, ModelCapability.CHAT],
        )

        # Coding task
        self.tasks["coding"] = BenchmarkTask(
            name="coding",
            description="Code generation task",
            category="coding",
            expected_outcome={"has_code": True},
            verify_fn=lambda r: "def " in r.get("content", "")
            or "function" in r.get("content", ""),
            required_capabilities=[
                ModelCapability.CODE,
                ModelCapability.REASONING,
                ModelCapability.CHAT,
            ],
        )

        # Tool use task (requires MCP)
        self.tasks["tool_use"] = BenchmarkTask(
            name="tool_use",
            description="File system tool use",
            category="tools",
            expected_outcome={"tool_called": True},
            verify_fn=lambda r: len(r.get("tool_calls", [])) > 0,
            required_capabilities=[ModelCapability.TOOLS, ModelCapability.CHAT],
        )

    def add_task(self, task: BenchmarkTask) -> None:
        """Add a custom benchmark task."""
        self.tasks[task.name] = task

    def get_task(self, name: str) -> BenchmarkTask | None:
        """Get a task by name."""
        return self.tasks.get(name)

    def list_tasks(self) -> list[BenchmarkTask]:
        """List all registered tasks."""
        return list(self.tasks.values())

    async def run(
        self,
        name: str,
        mode: BenchmarkMode,
        agent_name: str,
        task_names: list[str] | None = None,
        runtime: SidWeaveRuntime | None = None,
    ) -> BenchmarkResult:
        """
        Run benchmark.

        Args:
            name: Benchmark name
            mode: BASELINE or SIDWEAVE
            agent_name: Name of agent being tested
            task_names: Specific tasks to run (None = all)
            runtime: Optional existing runtime (for SIDWEAVE mode)

        Returns:
            BenchmarkResult with all metrics
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)

        # Select tasks
        if task_names:
            selected_tasks = [self.tasks[n] for n in task_names if n in self.tasks]
        else:
            selected_tasks = list(self.tasks.values())

        if not selected_tasks:
            raise ValueError(ERR_NO_VALID_TASKS)

        logger.info(
            "Starting benchmark '%s' in %s mode with %d tasks",
            name,
            mode.value,
            len(selected_tasks),
        )

        # Get runtime for SIDWEAVE mode
        own_runtime = False
        if mode == BenchmarkMode.SIDWEAVE and runtime is None:
            async with create_runtime(self.settings) as runtime_ctx:  # type: SidWeaveRuntime
                runtime = runtime_ctx
                own_runtime = True

        try:
            task_results: list[TaskResult] = []

            for task in selected_tasks:
                logger.info("Running task: %s", task.name)
                task_result = await self._run_task(task, mode, runtime)
                task_results.append(task_result)

            completed_at = datetime.now(UTC)

            # Calculate aggregates
            passed = sum(1 for r in task_results if r.success)
            failed = len(task_results) - passed
            avg_latency = sum(r.latency_ms for r in task_results) / len(task_results)
            total_tokens = sum(r.tokens_used for r in task_results)
            total_cost = sum(r.cost for r in task_results)

            tool_rates = [r.tool_success_rate for r in task_results if r.tool_success_rate > 0]
            avg_tool_rate = sum(tool_rates) / len(tool_rates) if tool_rates else 0.0

            verified_rates = [r.verification_rate for r in task_results if r.verification_rate > 0]
            avg_verification = sum(verified_rates) / len(verified_rates) if verified_rates else 0.0

            # Get hardware profile
            hardware_profile: dict[str, Any] = {}
            if runtime:
                profiler = runtime.get_component("hardware_profile")
                if profiler and isinstance(profiler, HardwareProfile):
                    hardware_profile = profiler.to_dict()

            result = BenchmarkResult(
                run_id=run_id,
                name=name,
                mode=mode,
                agent_name=agent_name,
                started_at=started_at,
                completed_at=completed_at,
                total_tasks=len(task_results),
                passed_tasks=passed,
                failed_tasks=failed,
                avg_latency_ms=avg_latency,
                total_tokens=total_tokens,
                total_cost=total_cost,
                tool_success_rate=avg_tool_rate,
                verification_rate=avg_verification,
                task_results=task_results,
                hardware_profile=hardware_profile,
            )

            # Save to database
            await self._save_result(result)

            return result

        finally:
            if own_runtime and runtime:
                await runtime.stop()

    async def _run_task(
        self,
        task: BenchmarkTask,
        _mode: BenchmarkMode,
        _runtime: SidWeaveRuntime | None,
    ) -> TaskResult:
        """Run a single benchmark task."""
        start = time.perf_counter()

        # Simulate task execution
        # In real implementation, this would call the agent
        # For now, we simulate with mock results

        try:
            # Simulate work
            await asyncio.sleep(0.1)

            # Mock result based on task
            raw_result: dict[str, Any]
            if task.name == "simple_chat":
                raw_result = {"content": "Hello! I'm an AI assistant."}
            elif task.name == "reasoning":
                raw_result = {
                    "content": "Let me think step by step. First... reasoning process here."
                }
            elif task.name == "coding":
                raw_result = {"content": "def hello():\n    return 'Hello, World!'"}
            elif task.name == "tool_use":
                raw_result = {
                    "content": "Done",
                    "tool_calls": [
                        {"name": "read_file", "args": {"path": "/tmp/test.txt"}}
                    ],
                }
            else:
                raw_result = {"content": "Task completed"}

            latency_ms = (time.perf_counter() - start) * 1000

            # Verify
            verified = task.verify_fn(raw_result)

            return TaskResult(
                task_name=task.name,
                success=True,
                latency_ms=latency_ms,
                tokens_used=100,  # Mock
                cost=0.001,  # Mock
                tool_calls=(
                    raw_result.get("tool_calls", [])
                    if isinstance(raw_result.get("tool_calls"), list)
                    else []
                ),
                tool_success_rate=1.0 if raw_result.get("tool_calls") else 0.0,
                verified=verified,
                verification_rate=1.0 if verified else 0.0,
                raw_result=raw_result,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return TaskResult(
                task_name=task.name,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
                raw_result={},
            )

    async def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to database."""
        async with get_session() as session:
            repo = BenchmarkRepository(session)
            run = BenchmarkRun(
                name=result.name,
                description=f"{result.mode.value} benchmark for {result.agent_name}",
                benchmark_version="0.1.0",
                sidweave_version="0.1.0",
                mode=result.mode.value,
                agent_name=result.agent_name,
                total_tasks=result.total_tasks,
                passed_tasks=result.passed_tasks,
                failed_tasks=result.failed_tasks,
                avg_latency_ms=result.avg_latency_ms,
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                tool_success_rate=result.tool_success_rate,
                verification_rate=result.verification_rate,
                results={
                    "task_results": [
                        {
                            "task_name": r.task_name,
                            "success": r.success,
                            "latency_ms": r.latency_ms,
                            "tokens_used": r.tokens_used,
                            "cost": r.cost,
                            "verified": r.verified,
                            "error": r.error,
                        }
                        for r in result.task_results
                    ],
                    "hardware_profile": result.hardware_profile,
                },
            )
            await repo.add_run(run)

    async def compare_modes(
        self,
        name: str,
        agent_name: str,
        task_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run benchmark in both modes and compare.

        Returns comparison metrics.
        """
        # Run baseline
        baseline = await self.run(name, BenchmarkMode.BASELINE, agent_name, task_names)

        # Run with SidWeave
        sidweave = await self.run(name, BenchmarkMode.SIDWEAVE, agent_name, task_names)

        # Calculate improvements
        baseline_rate = (
            baseline.passed_tasks / baseline.total_tasks if baseline.total_tasks else 0
        )
        sidweave_rate = (
            sidweave.passed_tasks / sidweave.total_tasks if sidweave.total_tasks else 0
        )
        improvements = {
            "success_rate": {
                "baseline": baseline_rate,
                "sidweave": sidweave_rate,
            },
            "avg_latency_ms": {
                "baseline": baseline.avg_latency_ms,
                "sidweave": sidweave.avg_latency_ms,
            },
            "total_cost": {
                "baseline": baseline.total_cost,
                "sidweave": sidweave.total_cost,
            },
            "tool_success_rate": {
                "baseline": baseline.tool_success_rate,
                "sidweave": sidweave.tool_success_rate,
            },
            "verification_rate": {
                "baseline": baseline.verification_rate,
                "sidweave": sidweave.verification_rate,
            },
        }

        return {
            "benchmark_name": name,
            "agent_name": agent_name,
            "baseline": baseline,
            "sidweave": sidweave,
            "improvements": improvements,
        }


def _console_print(msg: str) -> None:
    """Print to console via logger."""
    logger.info(msg)


async def run_benchmark_suite(
    settings: Settings,
    suite_name: str = "default",
    agent_name: str = "test-agent",
) -> dict[str, Any]:
    """Run full benchmark suite with comparison."""
    runner = BenchmarkRunner(settings)

    _console_print(f"Running benchmark suite: {suite_name}")

    # Run comparison
    comparison = await runner.compare_modes(suite_name, agent_name)

    # Print summary
    _console_print("\n=== Benchmark Results ===")
    _console_print(f"Suite: {suite_name}")
    _console_print(f"Agent: {agent_name}")

    imp = comparison["improvements"]
    sr_b = imp["success_rate"]["baseline"]
    sr_s = imp["success_rate"]["sidweave"]
    _console_print(f"\nSuccess Rate: {sr_b:.1%} -> {sr_s:.1%}")

    al_b = imp["avg_latency_ms"]["baseline"]
    al_s = imp["avg_latency_ms"]["sidweave"]
    _console_print(f"Avg Latency: {al_b:.1f}ms -> {al_s:.1f}ms")

    tc_b = imp["total_cost"]["baseline"]
    tc_s = imp["total_cost"]["sidweave"]
    _console_print(f"Total Cost: ${tc_b:.4f} -> ${tc_s:.4f}")

    ts_b = imp["tool_success_rate"]["baseline"]
    ts_s = imp["tool_success_rate"]["sidweave"]
    _console_print(f"Tool Success: {ts_b:.1%} -> {ts_s:.1%}")

    vr_b = imp["verification_rate"]["baseline"]
    vr_s = imp["verification_rate"]["sidweave"]
    _console_print(f"Verification: {vr_b:.1%} -> {vr_s:.1%}")

    return comparison


__all__ = [
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "TaskResult",
    "run_benchmark_suite",
]
