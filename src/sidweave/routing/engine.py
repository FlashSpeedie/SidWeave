"""
Adaptive routing engine for model selection.

Implements explainable deterministic routing based on:
- Task category and required capabilities
- Hardware availability and constraints
- Historical performance measurements
- Cost and latency preferences
- Model availability
"""

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

from sidweave.config import Settings
from sidweave.hardware import HardwareProfile
from sidweave.models import ModelCapability, ModelInfo
from sidweave.providers import Provider, ProviderType
from sidweave.storage import (
    MeasurementRepository,
    ModelRepository,
    RoutingDecision,
    RoutingRepository,
    get_session,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(StrEnum):
    """Routing strategies."""
    BALANCED = "balanced"      # Balance latency, cost, quality
    LATENCY = "latency"        # Minimize latency
    COST = "cost"              # Minimize cost
    QUALITY = "quality"        # Maximize quality (throughput/accuracy)
    LOCAL_FIRST = "local_first"  # Prefer local models


@dataclass
class RoutingConstraints:
    """Constraints for routing decision."""
    max_latency_ms: int | None = None
    max_cost_per_1k_tokens: float | None = None
    required_capabilities: list[ModelCapability] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.BALANCED


@dataclass
class ModelScore:
    """Score breakdown for a model."""
    model: ModelInfo
    provider: Provider
    total_score: float
    latency_score: float
    cost_score: float
    quality_score: float
    capability_score: float
    hardware_score: float
    availability_score: float
    rationale: str
    estimated_latency_ms: float | None = None
    estimated_cost: float | None = None


@dataclass
class RoutingResult:
    """Result of routing decision."""
    selected_model: ModelInfo
    selected_provider: Provider
    fallback_models: list[ModelInfo]
    scores: dict[str, ModelScore]  # model_name -> score
    rationale: str
    constraints: RoutingConstraints
    decision_time_ms: float


class RoutingEngine:
    """
    Deterministic adaptive routing engine.
    Selects the best model and provider based on task requirements,
    hardware constraints, and historical performance.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = True

    async def initialize(self) -> None:
        """Initialize routing engine."""
        logger.info("Routing engine initialized")

    async def route(
        self,
        task_category: str,
        required_capabilities: list[ModelCapability] | None = None,
        constraints: RoutingConstraints | None = None,
        hardware_profile: HardwareProfile | None = None,
        _preferred_model: str | None = None,
    ) -> RoutingResult:
        """
        Route a request to the best model/provider.
        Args:
            task_category: Category of task (e.g., "coding", "reasoning", "chat")
            required_capabilities: Required model capabilities
            constraints: Routing constraints
            hardware_profile: Current hardware profile
            _preferred_model: Optional preferred model name
        Returns:
            RoutingResult with selected model and rationale
        """
        start = time.perf_counter()

        if constraints is None:
            constraints = RoutingConstraints()

        # Apply settings defaults
        if constraints.strategy == RoutingStrategy.BALANCED:
            constraints.strategy = RoutingStrategy(self.settings.routing.default_strategy)

        if self.settings.routing.max_latency_ms and not constraints.max_latency_ms:
            constraints.max_latency_ms = self.settings.routing.max_latency_ms

        if self.settings.routing.max_cost_per_1k_tokens and not constraints.max_cost_per_1k_tokens:
            constraints.max_cost_per_1k_tokens = self.settings.routing.max_cost_per_1k_tokens

        if self.settings.routing.require_capabilities and not constraints.required_capabilities:
            constraints.required_capabilities = [
                ModelCapability(c) for c in self.settings.routing.require_capabilities
            ]

        # Get available models and providers
        async with get_session() as session:
            model_repo = ModelRepository(session)
            meas_repo = MeasurementRepository(session)

            # Get all available models
            db_models = await model_repo.list_models(available_only=True)

        # Convert to ModelInfo dataclasses
        models = [
            ModelInfo(
                name=m.name,
                provider=m.provider,
                provider_model_id=m.provider_model_id,
                capabilities=[ModelCapability(c) for c in m.capabilities],
                context_length=m.context_length,
                max_output_tokens=m.max_output_tokens,
                min_ram_gb=m.min_ram_gb,
                min_vram_gb=m.min_vram_gb,
                recommended_backend=m.recommended_backend,
                quantization=m.quantization,
                parameter_count=m.parameter_count,
                is_local=m.is_local,
                is_available=m.is_available,
                metadata=m.metadata_,
            )
            for m in db_models
        ]

        # Get provider gateway to check provider health
        # Note: In practice, this would be injected. For now, we'll work with model data.

        # Filter models by capabilities
        capable_models = self._filter_by_capabilities(models, required_capabilities or [])

        # Filter by constraints
        filtered_models = self._apply_constraints(capable_models, constraints)

        if not filtered_models:
            raise ValueError("No models match the required capabilities and constraints")  # noqa: TRY003

        # Score each model
        scores = await self._score_models(
            filtered_models,
            task_category,
            constraints,
            hardware_profile,
            meas_repo,
            model_repo,
        )

        # Select best model
        best_score = max(scores.values(), key=lambda s: s.total_score)

        # Build fallback list
        fallback_scores = sorted(
            [s for s in scores.values() if s.model.name != best_score.model.name],
            key=lambda s: s.total_score,
            reverse=True
        )
        fallback_models = [
            s.model for s in fallback_scores[:self.settings.routing.max_fallback_attempts]
        ]

        # Build rationale
        rationale = self._build_rationale(best_score, scores, constraints, fallback_models)

        decision_time = (time.perf_counter() - start) * 1000

        # Record routing decision
        await self._record_decision(
            task_category=task_category,
            required_capabilities=[c.value for c in (required_capabilities or [])],
            constraints=constraints,
            selected_model=best_score.model,
            selected_provider=best_score.provider,
            fallback_models=[m.name for m in fallback_models],
            rationale=rationale,
            scores={name: score.total_score for name, score in scores.items()},
        )

        return RoutingResult(
            selected_model=best_score.model,
            selected_provider=best_score.provider,
            fallback_models=fallback_models,
            scores=scores,
            rationale=rationale,
            constraints=constraints,
            decision_time_ms=decision_time,
        )

    def _filter_by_capabilities(
        self,
        models: list[ModelInfo],
        required_capabilities: list[ModelCapability],
    ) -> list[ModelInfo]:
        """Filter models by required capabilities."""
        if not required_capabilities:
            return models

        return [
            m for m in models
            if all(c in m.capabilities for c in required_capabilities)
        ]

    def _apply_constraints(
        self,
        models: list[ModelInfo],
        constraints: RoutingConstraints,
    ) -> list[ModelInfo]:
        """Apply routing constraints to filter models."""
        filtered = models

        # Exclude specific models
        if constraints.excluded_models:
            filtered = [m for m in filtered if m.name not in constraints.excluded_models]

        # Filter by max latency (using historical data)
        if constraints.max_latency_ms:
            # This would need historical data - for now, we estimate
            pass

        # Filter by cost
        if constraints.max_cost_per_1k_tokens:
            # This would need pricing data - for now, we estimate
            pass

        return filtered

    async def _score_models(  # noqa: PLR0913, PLR0917
        self,
        models: list[ModelInfo],
        task_category: str,
        constraints: RoutingConstraints,
        hardware_profile: HardwareProfile | None,
        meas_repo: MeasurementRepository,
        model_repo: ModelRepository,
    ) -> dict[str, ModelScore]:
        """Score each model based on multiple factors."""
        scores = {}

        for model in models:
            # Get historical measurements - need model ID from database
            db_model = await model_repo.get_by_name(model.name)
            if not db_model:
                avg_latency = None
                avg_throughput = None
            else:
                avg_latency = await meas_repo.get_average_latency(db_model.id, "latency")
                avg_throughput = await meas_repo.get_average_throughput(db_model.id)

            # Calculate individual scores (0-1)
            latency_score = self._calculate_latency_score(avg_latency, constraints)
            cost_score = self._calculate_cost_score(model, constraints)
            quality_score = self._calculate_quality_score(avg_throughput, model, task_category)
            capability_score = self._calculate_capability_score(model, constraints)
            hardware_score = self._calculate_hardware_score(model, hardware_profile)
            availability_score = self._calculate_availability_score(model)

            # Weight scores based on strategy
            weights = self._get_strategy_weights(constraints.strategy)

            total_score = (
                weights["latency"] * latency_score +
                weights["cost"] * cost_score +
                weights["quality"] * quality_score +
                weights["capability"] * capability_score +
                weights["hardware"] * hardware_score +
                weights["availability"] * availability_score
            )

            # Build rationale
            rationale = self._build_model_rationale(
                model, latency_score, cost_score, quality_score,
                capability_score, hardware_score, availability_score
            )


            # Convert provider string to ProviderType
            try:
                provider_type = ProviderType(model.provider)
            except ValueError:
                provider_type = ProviderType.LITELLM

            scores[model.name] = ModelScore(
                model=model,
                provider=Provider(name=model.provider, type=provider_type),
                total_score=total_score,
                latency_score=latency_score,
                cost_score=cost_score,
                quality_score=quality_score,
                capability_score=capability_score,
                hardware_score=hardware_score,
                availability_score=availability_score,
                rationale=rationale,
                estimated_latency_ms=avg_latency,
            )

        return scores

    def _calculate_latency_score(  # noqa: PLR0911
        self,
        avg_latency: float | None,
        constraints: RoutingConstraints,
    ) -> float:
        """Calculate latency score (lower is better)."""
        if avg_latency is None:
            return 0.5  # Unknown latency

        if constraints.max_latency_ms:
            if avg_latency <= constraints.max_latency_ms:
                return 1.0
            # Penalize proportionally
            return max(
                0.0,
                1.0 - (avg_latency - constraints.max_latency_ms) / constraints.max_latency_ms
            )

        # No constraint - prefer lower latency
        # Normalize against typical range (100-5000ms)
        LATENCY_EXCELLENT = 200
        LATENCY_GOOD = 1000
        LATENCY_FAIR = 3000
        if avg_latency < LATENCY_EXCELLENT:
            return 1.0
        if avg_latency <= LATENCY_GOOD:
            return 0.8
        if avg_latency <= LATENCY_FAIR:
            return 0.5
        return 0.2

    def _calculate_cost_score(self, model: ModelInfo, constraints: RoutingConstraints) -> float:
        """Calculate cost score (lower is better)."""
        # Local models have zero marginal cost
        if model.is_local:
            return 1.0

        # For cloud models, we'd need pricing data
        # For now, estimate based on parameter count
        if constraints.max_cost_per_1k_tokens:
            # Would compare estimated cost against constraint
            return 0.5

        return 0.7  # Default for cloud models

    def _calculate_quality_score(
        self,
        avg_throughput: float | None,
        model: ModelInfo,
        task_category: str,
    ) -> float:
        """Calculate quality score based on throughput and task fit."""
        score = 0.5

        # Throughput factor
        THROUGHPUT_HIGH = 50
        THROUGHPUT_MEDIUM = 20
        THROUGHPUT_LOW = 10
        if avg_throughput:
            if avg_throughput >= THROUGHPUT_HIGH:
                score += 0.3
            elif avg_throughput >= THROUGHPUT_MEDIUM:
                score += 0.2
            elif avg_throughput >= THROUGHPUT_LOW:
                score += 0.1

        # Task-specific bonuses
        task_capabilities = {
            "coding": [ModelCapability.CODE, ModelCapability.REASONING],
            "reasoning": [ModelCapability.REASONING, ModelCapability.CODE],
            "chat": [ModelCapability.CHAT],
            "embedding": [ModelCapability.EMBEDDING],
            "vision": [ModelCapability.VISION],
            "tools": [ModelCapability.TOOLS],
        }

        required = task_capabilities.get(task_category, [])
        matches = sum(1 for c in required if c in model.capabilities)
        if required:
            score += 0.3 * (matches / len(required))

        # Context length bonus
        CONTEXT_LARGE = 32768
        CONTEXT_MEDIUM = 8192
        if model.context_length >= CONTEXT_LARGE:
            score += 0.1
        elif model.context_length >= CONTEXT_MEDIUM:
            score += 0.05

        return min(1.0, score)

    def _calculate_capability_score(
        self,
        model: ModelInfo,
        constraints: RoutingConstraints,
    ) -> float:
        """Calculate capability match score."""
        if not constraints.required_capabilities:
            return 1.0

        matches = sum(
            1 for c in constraints.required_capabilities
            if c in model.capabilities
        )
        return matches / len(constraints.required_capabilities)

    def _calculate_hardware_score(
        self,
        model: ModelInfo,
        hardware_profile: HardwareProfile | None,
    ) -> float:
        """Calculate hardware compatibility score."""
        if not hardware_profile:
            return 0.5

        score = 0.5

        # Check RAM
        if model.min_ram_gb and hardware_profile.memory.total_gb:
            if hardware_profile.memory.total_gb >= model.min_ram_gb * 1.2:
                score += 0.3
            elif hardware_profile.memory.total_gb >= model.min_ram_gb:
                score += 0.1

        # Check VRAM for GPU models
        if model.min_vram_gb and hardware_profile.gpu.is_available:
            if hardware_profile.gpu.vram_gb and (
                hardware_profile.gpu.vram_gb >= model.min_vram_gb * 1.2
            ):
                score += 0.2
            elif hardware_profile.gpu.vram_gb and hardware_profile.gpu.vram_gb >= model.min_vram_gb:
                score += 0.1
            else:
                # VRAM insufficient but GPU available - penalty
                score -= 0.3
        elif model.min_vram_gb and not hardware_profile.gpu.is_available:
            # Model needs GPU but none available
            score -= 0.3

        # Backend compatibility
        if model.recommended_backend in hardware_profile.available_backends:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _calculate_availability_score(self, model: ModelInfo) -> float:
        """Calculate availability score."""
        if not model.is_available:
            return 0.0
        return 1.0

    def _get_strategy_weights(self, strategy: RoutingStrategy) -> dict[str, float]:
        """Get score weights for routing strategy."""
        weights = {
            RoutingStrategy.BALANCED: {
                "latency": 0.25,
                "cost": 0.20,
                "quality": 0.25,
                "capability": 0.15,
                "hardware": 0.10,
                "availability": 0.05,
            },
            RoutingStrategy.LATENCY: {
                "latency": 0.50,
                "cost": 0.10,
                "quality": 0.15,
                "capability": 0.10,
                "hardware": 0.10,
                "availability": 0.05,
            },
            RoutingStrategy.COST: {
                "latency": 0.10,
                "cost": 0.50,
                "quality": 0.15,
                "capability": 0.10,
                "hardware": 0.10,
                "availability": 0.05,
            },
            RoutingStrategy.QUALITY: {
                "latency": 0.10,
                "cost": 0.10,
                "quality": 0.50,
                "capability": 0.15,
                "hardware": 0.10,
                "availability": 0.05,
            },
            RoutingStrategy.LOCAL_FIRST: {
                "latency": 0.15,
                "cost": 0.30,
                "quality": 0.20,
                "capability": 0.15,
                "hardware": 0.15,
                "availability": 0.05,
            },
        }
        return weights.get(strategy, weights[RoutingStrategy.BALANCED])

    def _build_model_rationale(  # noqa: PLR0912, PLR0913, PLR0917
        self,
        model: ModelInfo,
        latency_score: float,
        cost_score: float,
        quality_score: float,
        capability_score: float,
        hardware_score: float,
        _availability_score: float,
    ) -> str:
        """Build human-readable rationale for model score."""
        parts = []
        SCORE_HIGH = 0.8
        SCORE_MEDIUM = 0.5

        if latency_score >= SCORE_HIGH:
            parts.append("low latency")
        elif latency_score >= SCORE_MEDIUM:
            parts.append("moderate latency")
        else:
            parts.append("higher latency")

        if cost_score >= SCORE_HIGH:
            parts.append("low cost")
        elif cost_score >= SCORE_MEDIUM:
            parts.append("moderate cost")
        else:
            parts.append("higher cost")

        if quality_score >= SCORE_HIGH:
            parts.append("high quality")
        elif quality_score >= SCORE_MEDIUM:
            parts.append("good quality")
        else:
            parts.append("standard quality")

        if capability_score >= 1.0:
            parts.append("all required capabilities")
        elif capability_score >= SCORE_MEDIUM:
            parts.append("partial capabilities")
        else:
            parts.append("missing capabilities")

        if hardware_score >= SCORE_HIGH:
            parts.append("hardware compatible")
        elif hardware_score >= SCORE_MEDIUM:
            parts.append("partial hardware fit")
        else:
            parts.append("hardware constraints")

        return f"{model.name} ({model.provider}): {', '.join(parts)}"

    def _build_rationale(
        self,
        best_score: ModelScore,
        all_scores: dict[str, ModelScore],
        constraints: RoutingConstraints,
        fallback_models: list[ModelInfo],
    ) -> str:
        """Build overall routing rationale."""
        lines = [
            f"Selected {best_score.model.name} via {best_score.provider.name}",
            f"Strategy: {constraints.strategy.value}",
            f"Total score: {best_score.total_score:.3f}",
            "",
            "Score breakdown:",
            f"  Latency: {best_score.latency_score:.2f}",
            f"  Cost: {best_score.cost_score:.2f}",
            f"  Quality: {best_score.quality_score:.2f}",
            f"  Capability: {best_score.capability_score:.2f}",
            f"  Hardware: {best_score.hardware_score:.2f}",
            f"  Availability: {best_score.availability_score:.2f}",
            "",
            f"Rationale: {best_score.rationale}",
        ]

        if fallback_models:
            lines.append("")
            lines.append("Fallback options:")
            for i, model in enumerate(fallback_models[:3], 1):
                score = all_scores.get(model.name)
                if score:
                    lines.append(f"  {i}. {model.name} (score: {score.total_score:.3f})")

        return "\n".join(lines)

    async def _record_decision(  # noqa: PLR0913, PLR0917
        self,
        task_category: str,
        required_capabilities: list[str],
        constraints: RoutingConstraints,
        selected_model: ModelInfo,
        selected_provider: Provider,
        fallback_models: list[str],
        rationale: str,
        scores: dict[str, float],
    ) -> None:
        """Record routing decision to database."""
        try:
            async with get_session() as session:
                repo = RoutingRepository(session)
                # Convert fallback model names to IDs
                fallback_model_ids = []
                if fallback_models:
                    model_registry = ModelRepository(session)
                    for name in fallback_models:
                        model = await model_registry.get_by_name(name)
                        if model:
                            fallback_model_ids.append(model.id)

                decision = RoutingDecision(
                    task_category=task_category,
                    task_description=f"Routing for {task_category}",
                    required_capabilities=required_capabilities,
                    constraints={
                        "max_latency_ms": constraints.max_latency_ms,
                        "max_cost_per_1k_tokens": constraints.max_cost_per_1k_tokens,
                        "strategy": constraints.strategy.value,
                        "preferred_providers": constraints.preferred_providers,
                    },
                    selected_model_id=selected_model.id if hasattr(selected_model, "id") else 0,
                    selected_model_name=selected_model.name,
                    selected_provider=selected_provider.name,
                    fallback_model_ids=fallback_model_ids,
                    rationale=rationale,
                    scores=scores,
                )
                await repo.add_decision(decision)
        except Exception as e:
            logger.warning(f"Failed to record routing decision: {e}")


__all__ = [
    "ModelScore",
    "RoutingConstraints",
    "RoutingEngine",
    "RoutingResult",
    "RoutingStrategy",
]
