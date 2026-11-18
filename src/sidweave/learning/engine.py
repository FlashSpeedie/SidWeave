"""
Learning and evaluation engine for continuous improvement.

Records task outcomes, analyzes performance, and generates routing recommendations.
Does not automatically modify production routing - provides reports for human review.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, Union

from sidweave.config import Settings
from sidweave.storage import (
    BenchmarkRun,
    Model,
    ModelRepository,
    TaskRepository,
    get_session,
)

logger = logging.getLogger(__name__)


class TaskLike(Protocol):
    """Protocol defining the common interface for task-like objects."""
    category: str
    success: bool
    total_latency_ms: float | None
    total_cost: float | None
    model_id: int
    provider: str


# Type alias for task objects that can be either TaskRecordInput or BenchmarkRun
TaskLikeType = Union["TaskRecordInput", "BenchmarkRun"]


# Type alias for any task-like object (used for MyPy compatibility)
TaskData = Any


# Constants for anomaly detection thresholds
MIN_TASKS_FOR_CATEGORY_RECOMMENDATION = 10
MIN_TASKS_FOR_COST_OPTIMIZATION = 10
MIN_COST_FOR_COST_OPTIMIZATION = 10.0
MIN_TASKS_FOR_CATEGORY_ANOMALY = 20
MIN_SUCCESS_RATE_FOR_CATEGORY_ANOMALY = 0.5
MIN_TASKS_FOR_MODEL_ANOMALY = 20
MIN_SUCCESS_RATE_FOR_MODEL_ANOMALY = 0.3
MIN_TASKS_FOR_PROVIDER_ANOMALY = 10
MIN_SUCCESS_RATE_FOR_PROVIDER_ANOMALY = 0.6
MIN_SUCCESS_RATE_FOR_MODEL_DEGRADATION = 0.3


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    generated_at: datetime
    period: str
    total_tasks: int
    success_rate: float
    avg_latency_ms: float
    total_cost: float
    by_category: dict[str, Any] = field(default_factory=dict)
    by_model: dict[str, Any] = field(default_factory=dict)
    by_provider: dict[str, Any] = field(default_factory=dict)
    routing_recommendations: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


@dataclass
class RoutingRecommendation:
    """Recommendation for routing improvement."""
    task_category: str
    current_model: str
    recommended_model: str
    reason: str
    confidence: float
    expected_improvement: dict[str, Any]


@dataclass
class TaskRecordInput:
    """Input for recording a task."""
    category: str
    description: str
    model_id: int
    provider: str
    total_latency_ms: float | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_success_rate: float | None = None
    verified: bool = False
    verification_rate: float | None = None
    success: bool | None = None
    error: str | None = None
    user_feedback: str | None = None


class LearningEngine:
    """
    Learning and evaluation engine.

    Analyzes completed tasks and routing decisions to generate
    performance reports and routing recommendations.

    Does NOT automatically modify production routing.
    All recommendations require human review.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = False
        self._running = False
        self._background_task: asyncio.Task[Any] | None = None

    async def initialize(self) -> None:
        """Initialize learning engine."""
        self._initialized = True
        logger.info("Learning engine initialized")

    async def record_task(self, task_input: TaskRecordInput) -> None:
        """Record a completed task for learning."""
        async with get_session() as session:
            repo = TaskRepository(session)
            await repo.add_task_from_input(task_input)

    async def _calculate_aggregates(
        self,
        tasks: list[TaskData],
    ) -> tuple[int, float, float, float]:
        """Calculate aggregate metrics from tasks."""
        total_tasks = len(tasks)
        successful = sum(1 for t in tasks if t.success)
        success_rate = successful / total_tasks if total_tasks > 0 else 0.0

        latencies = [t.total_latency_ms for t in tasks if t.total_latency_ms]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        costs = [t.total_cost for t in tasks if t.total_cost]
        total_cost = sum(costs)

        return total_tasks, success_rate, avg_latency, total_cost

    async def _calculate_category_stats(self, tasks: list[TaskData]) -> dict[str, dict[str, Any]]:
        """Calculate statistics grouped by category."""
        by_category: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "success": 0, "latency": [], "cost": []}
        )
        for task in tasks:
            cat = task.category
            by_category[cat]["total"] += 1
            if task.success:
                by_category[cat]["success"] += 1
            if task.total_latency_ms:
                by_category[cat]["latency"].append(task.total_latency_ms)
            if task.total_cost:
                by_category[cat]["cost"].append(task.total_cost)

        category_stats: dict[str, dict[str, Any]] = {}
        for cat, data in by_category.items():
            lat = data["latency"]
            category_stats[cat] = {
                "total": data["total"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                "avg_latency_ms": sum(lat) / len(lat) if lat else 0,
                "total_cost": sum(data["cost"]),
            }
        return category_stats

    async def _calculate_model_stats(
        self, tasks: list[TaskData], model_repo: ModelRepository
    ) -> dict[str, dict[str, Any]]:
        """Calculate statistics grouped by model."""
        by_model: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "success": 0, "latency": [], "cost": []}
        )
        for task in tasks:
            if task.model_id:
                by_model[task.model_id]["total"] += 1
                if task.success:
                    by_model[task.model_id]["success"] += 1
                if task.total_latency_ms:
                    by_model[task.model_id]["latency"].append(task.total_latency_ms)
                if task.total_cost:
                    by_model[task.model_id]["cost"].append(task.total_cost)

        model_stats: dict[str, dict[str, Any]] = {}
        for mid, data in by_model.items():
            model = await model_repo.get(Model, mid)
            name = model.name if model else f"model_{mid}"
            lat = data["latency"]
            model_stats[name] = {
                "total": data["total"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                "avg_latency_ms": sum(lat) / len(lat) if lat else 0,
                "total_cost": sum(data["cost"]),
            }
        return model_stats

    async def _calculate_provider_stats(self, tasks: list[TaskData]) -> dict[str, dict[str, Any]]:
        """Calculate statistics grouped by provider."""
        by_provider: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "success": 0, "latency": [], "cost": []}
        )
        for task in tasks:
            prov = task.provider
            by_provider[prov]["total"] += 1
            if task.success:
                by_provider[prov]["success"] += 1
            if task.total_latency_ms:
                by_provider[prov]["latency"].append(task.total_latency_ms)
            if task.total_cost:
                by_provider[prov]["cost"].append(task.total_cost)

        provider_stats: dict[str, dict[str, Any]] = {}
        for prov, data in by_provider.items():
            lat = data["latency"]
            provider_stats[prov] = {
                "total": data["total"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                "avg_latency_ms": sum(lat) / len(lat) if lat else 0,
                "total_cost": sum(data["cost"]),
            }
        return provider_stats

    async def generate_performance_report(
        self,
        period_days: int = 7,
        category: str | None = None,
    ) -> PerformanceReport:
        """Generate performance report for the specified period."""
        since = datetime.now(UTC) - timedelta(days=period_days)

        async with get_session() as session:
            task_repo = TaskRepository(session)
            model_repo = ModelRepository(session)

            # Get tasks in period
            tasks = await task_repo.get_recent_tasks(_category=category, limit=10000)
            tasks = [t for t in tasks if t.created_at >= since]

            if not tasks:
                return PerformanceReport(
                    generated_at=datetime.now(UTC),
                    period=f"{period_days}d",
                    total_tasks=0,
                    success_rate=0.0,
                    avg_latency_ms=0.0,
                    total_cost=0.0,
                    anomalies=[],
                )

            # Calculate aggregates
            _total_tasks, success_rate, avg_latency, total_cost = (
                await self._calculate_aggregates(tasks)
            )

            # Calculate stats by dimension
            category_stats = await self._calculate_category_stats(tasks)
            model_stats = await self._calculate_model_stats(tasks, model_repo)
            provider_stats = await self._calculate_provider_stats(tasks)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                category_stats, model_stats, provider_stats
            )

            # Detect anomalies
            anomalies = self._detect_anomalies(category_stats, model_stats, provider_stats)

        return PerformanceReport(
            generated_at=datetime.now(UTC),
            period=f"{period_days}d",
            total_tasks=len(tasks),
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            total_cost=total_cost,
            by_category=category_stats,
            by_model=model_stats,
            by_provider=provider_stats,
            routing_recommendations=recommendations,
            anomalies=anomalies,
        )

    def _generate_recommendations(
        self,
        by_category: dict[str, dict[str, Any]],
        by_model: dict[str, dict[str, Any]],
        by_provider: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate routing recommendations based on performance data."""
        recommendations = []

        # Find best model per category
        for category, cat_data in by_category.items():
            if cat_data["total"] < MIN_TASKS_FOR_CATEGORY_RECOMMENDATION:
                continue

            # Find models used for this category
            category_models = {m: d for m, d in by_model.items()}  # noqa: C416

            if not category_models:
                continue

            # Find best performing model
            best_model = max(category_models.items(), key=lambda x: x[1]["success_rate"])
            worst_model = min(category_models.items(), key=lambda x: x[1]["success_rate"])

            if best_model[1]["success_rate"] > worst_model[1]["success_rate"] + 0.1:
                recommendations.append(
                    f"Category '{category}': Consider preferring '{best_model[0]}' "
                    f"(success: {best_model[1]['success_rate']:.1%}) over '{worst_model[0]}' "
                    f"(success: {worst_model[1]['success_rate']:.1%})"
                )

        # Cost optimization recommendations
        for provider, prov_data in by_provider.items():
            if (
                prov_data["total"] > MIN_TASKS_FOR_COST_OPTIMIZATION
                and prov_data.get("total_cost", 0) > MIN_COST_FOR_COST_OPTIMIZATION
            ):
                recommendations.append(
                    f"Provider '{provider}': High cost (${prov_data['total_cost']:.2f}). "
                    f"Consider routing more tasks to local models."
                )

        return recommendations

    def _detect_anomalies(
        self,
        by_category: dict[str, dict[str, Any]],
        by_model: dict[str, dict[str, Any]],
        by_provider: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Detect performance anomalies."""
        anomalies = []

        # Sudden success rate drops
        for category, data in by_category.items():
            if (
                data["total"] > MIN_TASKS_FOR_CATEGORY_ANOMALY
                and data["success_rate"] < MIN_SUCCESS_RATE_FOR_CATEGORY_ANOMALY
            ):
                anomalies.append(
                    f"Category '{category}': Low success rate ({data['success_rate']:.1%})"
                )

        # Model degradation
        for model, data in by_model.items():
            if (
                data["total"] > MIN_TASKS_FOR_MODEL_ANOMALY
                and data["success_rate"] < MIN_SUCCESS_RATE_FOR_MODEL_ANOMALY
            ):
                anomalies.append(
                    f"Model '{model}': Very low success rate ({data['success_rate']:.1%})"
                )

        # Provider issues
        for provider, data in by_provider.items():
            if (
                data["total"] > MIN_TASKS_FOR_PROVIDER_ANOMALY
                and data["success_rate"] < MIN_SUCCESS_RATE_FOR_PROVIDER_ANOMALY
            ):
                anomalies.append(
                    f"Provider '{provider}': Low success rate ({data['success_rate']:.1%})"
                )

        return anomalies

    async def get_routing_recommendations(self) -> list[str]:
        """Get specific routing recommendations."""
        # This would analyze routing decisions vs outcomes
        # For now, return empty list
        return []

    async def run_background(self) -> None:
        """Run background learning tasks."""
        self._running = True

        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                if not self._running:
                    break

                # Generate daily report
                report = await self.generate_performance_report(period_days=1)

                if report.anomalies:
                    logger.warning(f"Performance anomalies detected: {report.anomalies}")

                if report.routing_recommendations:
                    logger.info(f"Routing recommendations: {report.routing_recommendations}")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Background learning error")

    async def shutdown(self) -> None:
        """Shutdown learning engine."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background_task


__all__ = [
    "LearningEngine",
    "PerformanceReport",
    "RoutingRecommendation",
]
