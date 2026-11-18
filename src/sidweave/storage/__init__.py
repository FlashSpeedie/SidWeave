"""
Storage layer for SidWeave using SQLAlchemy 2.0 async.

Provides database models, repositories, and session management.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    desc,
    func,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from sidweave.config import Settings, get_settings

if TYPE_CHECKING:
    from sidweave.learning.engine import TaskRecordInput

# Global engine for default sessions
_default_engine: AsyncEngine | None = None


def set_default_engine(engine: AsyncEngine) -> None:
    """Set the default engine for get_session()."""
    global _default_engine  # noqa: PLW0603
    _default_engine = engine


def get_default_engine() -> AsyncEngine | None:
    """Get the default engine."""
    return _default_engine


T = TypeVar("T")


class Base(DeclarativeBase):
    """Base class for all database models."""


class HardwareProfile(Base):
    """Hardware profile detected at runtime."""

    __tablename__ = "hardware_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # CPU
    cpu_vendor: Mapped[str] = mapped_column(String(128))
    cpu_model: Mapped[str] = mapped_column(String(256))
    cpu_cores_physical: Mapped[int] = mapped_column(Integer)
    cpu_cores_logical: Mapped[int] = mapped_column(Integer)
    cpu_frequency_mhz: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpu_instruction_sets: Mapped[list[str]] = mapped_column(JSON, default=list)
    cpu_architecture: Mapped[str] = mapped_column(String(64))

    # Memory
    memory_total_gb: Mapped[float] = mapped_column(Float)
    memory_available_gb: Mapped[float] = mapped_column(Float)
    swap_total_gb: Mapped[float] = mapped_column(Float, default=0.0)
    swap_free_gb: Mapped[float] = mapped_column(Float, default=0.0)

    # GPU
    gpu_vendor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gpu_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    gpu_vram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_driver_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gpu_compute_capability: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gpu_device_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # OS
    os_name: Mapped[str] = mapped_column(String(128))
    os_version: Mapped[str] = mapped_column(String(128))
    os_arch: Mapped[str] = mapped_column(String(64))
    os_python_version: Mapped[str] = mapped_column(String(32))

    # Disk
    disk_free_gb: Mapped[float] = mapped_column(Float, default=0.0)

    # Raw data for debugging
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "cpu": {
                "vendor": self.cpu_vendor,
                "model": self.cpu_model,
                "cores_physical": self.cpu_cores_physical,
                "cores_logical": self.cpu_cores_logical,
                "frequency_mhz": self.cpu_frequency_mhz,
                "instruction_sets": self.cpu_instruction_sets,
                "architecture": self.cpu_architecture,
            },
            "memory": {
                "total_gb": self.memory_total_gb,
                "available_gb": self.memory_available_gb,
                "swap_total_gb": self.swap_total_gb,
                "swap_free_gb": self.swap_free_gb,
            },
            "gpu": {
                "vendor": self.gpu_vendor,
                "model": self.gpu_model,
                "vram_gb": self.gpu_vram_gb,
                "driver_version": self.gpu_driver_version,
                "compute_capability": self.gpu_compute_capability,
                "device_id": self.gpu_device_id,
                "is_available": self.gpu_vendor is not None,
            },
            "os": {
                "name": self.os_name,
                "version": self.os_version,
                "arch": self.os_arch,
                "python_version": self.os_python_version,
            },
            "disk_free_gb": self.disk_free_gb,
            "raw_data": self.raw_data,
        }


class Model(Base):
    """Model registry entry."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Identity
    name: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    provider_model_id: Mapped[str] = mapped_column(String(256))

    # Capabilities
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    context_length: Mapped[int] = mapped_column(Integer, default=4096)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_ram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_vram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quantization: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parameter_count: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Status
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)

    # Metadata
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    measurements: Mapped[list["ModelMeasurement"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


class ModelMeasurement(Base):
    """Single model profiling measurement."""

    __tablename__ = "model_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Model reference
    model_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("models.id"), index=True, nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(256), index=True)
    model: Mapped[Model | None] = relationship(back_populates="measurements")

    # Test configuration
    test_type: Mapped[str] = mapped_column(String(64), index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    max_tokens: Mapped[int] = mapped_column(Integer, default=100)
    backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hardware_profile_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("hardware_profiles.id"), nullable=True
    )

    # Status
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput_tps: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_usage_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_memory_usage_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # pending, passed, failed, unknown
    verification_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Duration
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Benchmark context
    benchmark_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("benchmark_runs.id"), nullable=True
    )

    # Execution context
    execution_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tool_executions.id"), nullable=True
    )


class RoutingDecision(Base):
    """Record of a routing decision."""

    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Task
    task_category: Mapped[str] = mapped_column(String(64), index=True)
    task_description: Mapped[str] = mapped_column(Text)
    required_capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Selected model
    selected_model_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("models.id"), nullable=True
    )
    selected_model_name: Mapped[str] = mapped_column(String(256))
    selected_provider: Mapped[str] = mapped_column(String(128))

    # Fallback models
    fallback_model_ids: Mapped[list[int]] = mapped_column(JSON, default=list)

    # Rationale
    rationale: Mapped[str] = mapped_column(Text)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    constraints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Outcome
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # pending, passed, failed, unknown
    verification_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class ToolExecution(Base):
    """Tool execution record for MCP integration."""

    __tablename__ = "tool_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Identity
    execution_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128), unique=True, nullable=True, index=True
    )

    # Tool info
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    server_name: Mapped[str] = mapped_column(String(128), index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Preconditions
    preconditions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    # Postconditions
    postconditions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    # Execution
    status: Mapped[str] = mapped_column(String(32), default="requested", index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # pending, passed, failed, unknown
    verification_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Approval
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Benchmark context
    benchmark_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("benchmark_runs.id"), nullable=True
    )


class BenchmarkRun(Base):
    """Benchmark run record."""

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Identity
    name: Mapped[str] = mapped_column(String(256), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Benchmark configuration
    benchmark_version: Mapped[str] = mapped_column(String(32))
    sidweave_version: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(32), index=True)  # baseline, sidweave
    agent_name: Mapped[str] = mapped_column(String(128))

    # Aggregated results
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    passed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    tool_success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    verification_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Results
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # Detailed per-task results

    # Hardware context
    hardware_profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database(settings: Settings | None = None) -> AsyncEngine:
    """Initialize database engine and create tables."""
    global _engine, _session_factory  # noqa: PLW0603, _default_engine  # noqa: PLW0603

    settings = settings or get_settings()
    database_url = settings.database_url

    _engine = create_async_engine(
        database_url,
        echo=settings.telemetry.enabled and settings.telemetry.log_level == "DEBUG",
        pool_pre_ping=True,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _default_engine = _engine

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return _engine


async def close_database() -> None:
    """Close database engine."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None


@asynccontextmanager
async def get_session(engine: AsyncEngine | None = None) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async_session: async_sessionmaker[AsyncSession]

    if engine is None:
        engine = get_default_engine()
        if engine is None:
            await init_database()

    if engine is not None:
        # Create a new session factory for this engine
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    else:
        if _session_factory is None:
            await init_database()
        async_session = _session_factory  # type: ignore[assignment]

    # At this point, async_session is guaranteed to be not None
    assert async_session is not None
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class BaseRepository:
    """Base repository with common operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, obj: T) -> T:
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def get(self, model: type[T], obj_id: int) -> T | None:
        return await self.session.get(model, obj_id)

    async def get_by_name(self, model: type[T], name: str) -> T | None:
        stmt = select(model).where(model.name == name)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, model: type[T], limit: int = 100, offset: int = 0) -> list[T]:
        stmt = select(model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ModelRepository(BaseRepository):
    """Model repository."""

    async def list_models(  # noqa: PLR0913, PLR0917
        self,
        provider: str | None = None,
        capability: str | None = None,
        is_local: bool | None = None,
        available_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Model]:

        stmt = select(Model)
        if provider:
            stmt = stmt.where(Model.provider == provider)
        if capability:
            stmt = stmt.where(Model.capabilities.contains([capability]))
        if is_local is not None:
            stmt = stmt.where(Model.is_local == is_local)
        if available_only:
            stmt = stmt.where(Model.is_available.is_(True))

        stmt = stmt.order_by(Model.name).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_or_create(
        self,
        name: str,
        provider: str,
        provider_model_id: str,
        **kwargs: Any,
    ) -> Model:
        model = await self.get_by_name(name)
        if model:
            return model

        model = Model(
            name=name,
            provider=provider,
            provider_model_id=provider_model_id,
            **kwargs,
        )
        return await self.add(model)

    async def upsert(
        self,
        name: str,
        provider: str,
        provider_model_id: str,
        **kwargs: Any,
    ) -> Model:
        """Create or update a model."""
        model = await self.get_by_name(name)
        if model:
            for key, value in kwargs.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            model.provider = provider
            model.provider_model_id = provider_model_id
            await self.session.flush()
            return model

        model = Model(
            name=name,
            provider=provider,
            provider_model_id=provider_model_id,
            **kwargs,
        )
        return await self.add(model)

    async def get_by_name(self, name: str) -> Model | None:  # type: ignore[override]
        stmt = select(Model).where(Model.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class MeasurementRepository(BaseRepository):
    """Model measurement repository."""

    async def add_measurement(self, measurement: ModelMeasurement) -> ModelMeasurement:
        return await self.add(measurement)

    async def get_latest_measurements(
        self,
        model_id: int,
        test_type: str,
        limit: int = 10,
    ) -> list[ModelMeasurement]:

        stmt = (
            select(ModelMeasurement)
            .where(
                ModelMeasurement.model_id == model_id,
                ModelMeasurement.test_type == test_type,
                ModelMeasurement.success.is_(True),
            )
            .order_by(desc(ModelMeasurement.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_average_latency(self, model_id: int, test_type: str = "latency") -> float | None:

        stmt = (
            select(func.avg(ModelMeasurement.latency_ms))
            .where(
                ModelMeasurement.model_id == model_id,
                ModelMeasurement.test_type == test_type,
                ModelMeasurement.success.is_(True),
                ModelMeasurement.latency_ms.is_not(None),
            )
        )
        result = await self.session.execute(stmt)
        val = result.scalar()
        return float(val) if val is not None else None

    async def get_average_throughput(self, model_id: int) -> float | None:

        stmt = (
            select(func.avg(ModelMeasurement.throughput_tps))
            .where(
                ModelMeasurement.model_id == model_id,
                ModelMeasurement.test_type == "throughput",
                ModelMeasurement.success.is_(True),
                ModelMeasurement.throughput_tps.is_not(None),
            )
        )
        result = await self.session.execute(stmt)
        val = result.scalar()
        return float(val) if val is not None else None

    async def get_by_model(self, model_name: str) -> list[ModelMeasurement]:
        """Get all measurements for a model by name."""

        stmt = select(ModelMeasurement).where(ModelMeasurement.model_name == model_name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class RoutingRepository(BaseRepository):
    """Routing decision repository."""

    async def add_decision(self, decision: RoutingDecision) -> RoutingDecision:
        return await self.add(decision)

    async def get_recent_decisions(
        self,
        task_category: str | None = None,
        limit: int = 100,
    ) -> list[RoutingDecision]:

        stmt = select(RoutingDecision).order_by(desc(RoutingDecision.created_at)).limit(limit)
        if task_category:
            stmt = stmt.where(RoutingDecision.task_category == task_category)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_success_rate_by_model(self, model_id: int) -> float:

        stmt = (
            select(
                func.count(RoutingDecision.id),
                func.sum(RoutingDecision.success.cast(Integer))
            )
            .where(RoutingDecision.selected_model_id == model_id)
        )
        result = await self.session.execute(stmt)
        total, successful = result.one()
        if total == 0:
            return 0.0
        return float(successful) / float(total)


class ExecutionRepository(BaseRepository):
    """Tool execution repository."""

    async def add_execution(self, execution: ToolExecution) -> ToolExecution:
        return await self.add(execution)

    async def get_by_execution_id(self, execution_id: str) -> ToolExecution | None:
        stmt = select(ToolExecution).where(ToolExecution.execution_id == execution_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, key: str) -> ToolExecution | None:
        stmt = select(ToolExecution).where(ToolExecution.idempotency_key == key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(  # noqa: PLR0913, PLR0917
        self,
        execution_id: str,
        status: str,
        error: str | None = None,
        completed_at: datetime | None = None,
        duration_ms: float | None = None,
        verification_status: str | None = None,
        result: dict[str, Any] | None = None,
        verification_details: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(UTC)
        values = {"status": status, "updated_at": now}
        if error is not None:
            values["error"] = error
        if completed_at is not None:
            values["completed_at"] = completed_at
        if duration_ms is not None:
            values["duration_ms"] = duration_ms
        if verification_status is not None:
            values["verification_status"] = verification_status
        if result is not None:
            values["result"] = result
        if verification_details is not None:
            values["verification_details"] = verification_details
        stmt = (
            update(ToolExecution)
            .where(ToolExecution.execution_id == execution_id)
            .values(**values)
        )
        await self.session.execute(stmt)

    async def reconcile(self, execution_id: str) -> ToolExecution | None:
        """Reconcile an execution with unknown outcome."""
        execution = await self.get_by_execution_id(execution_id)
        if not execution:
            return None

        # Try to determine actual state
        # This is a simplified version - real implementation would check tool state
        return execution

    async def list_pending_reconciliation(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolExecution]:
        """List executions that need reconciliation."""
        stmt = (
            select(ToolExecution)
            .where(
                ToolExecution.status.in_([
                    "unknown",
                    "recovery_required",
                    "executing",
                    "verification_pending",
                ]),
            )
            .order_by(ToolExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_pending_reconciliation(self) -> int:
        """Count executions needing reconciliation."""
        stmt = (
            select(func.count(ToolExecution.id))
            .where(
                ToolExecution.status.in_([
                    "unknown",
                    "recovery_required",
                    "executing",
                    "verification_pending",
                ]),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List available tools on a server."""
        stmt = select(ToolExecution).where(
            ToolExecution.server_name == server_name,
            ToolExecution.status == "completed",
        ).order_by(ToolExecution.created_at.desc()).limit(100)
        result = await self.session.execute(stmt)
        executions = result.scalars().all()

        tools = {}
        for ex in executions:
            if ex.tool_name not in tools:
                tools[ex.tool_name] = {
                    "name": ex.tool_name,
                    "description": ex.arguments.get("description", ""),
                    "input_schema": ex.arguments.get("input_schema", {}),
                }
        return list(tools.values())


class TaskRepository(BaseRepository):
    """Benchmark task repository."""

    async def add_task(self, task: BenchmarkRun) -> BenchmarkRun:
        return await self.add(task)

    async def add_task_from_input(self, task_input: "TaskRecordInput") -> BenchmarkRun:
        """Add a task from TaskRecordInput by converting to BenchmarkRun."""

        task = BenchmarkRun(
            name=task_input.category,
            description=task_input.description,
            benchmark_version="0.1.0",
            sidweave_version="0.1.0",
            mode="sidweave",
            agent_name="learning-agent",
            total_tasks=1,
            passed_tasks=1 if task_input.success else 0,
            failed_tasks=0 if task_input.success else 1,
            avg_latency_ms=task_input.total_latency_ms or 0.0,
            total_tokens=task_input.total_tokens or 0,
            total_cost=task_input.total_cost or 0.0,
            tool_success_rate=task_input.tool_success_rate or 0.0,
            verification_rate=task_input.verification_rate or 0.0,
            results={},
        )
        return await self.add(task)

    async def get_recent_tasks(
        self, _category: str | None = None, limit: int = 10000
    ) -> list[BenchmarkRun]:
        """Get recent tasks, optionally filtered by category."""
        return await self.list_runs(benchmark_name=None, mode=None, limit=limit)

    async def list_runs(
        self,
        benchmark_name: str | None = None,
        mode: str | None = None,
        limit: int = 100,
    ) -> list[BenchmarkRun]:
        stmt = select(BenchmarkRun).order_by(desc(BenchmarkRun.created_at))
        if benchmark_name:
            stmt = stmt.where(BenchmarkRun.name == benchmark_name)
        if mode:
            stmt = stmt.where(BenchmarkRun.mode == mode)
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def compare_modes(self, benchmark_name: str) \
        -> dict[str, tuple[float, float, float, float, float, int]]:
        """Compare baseline vs sidweave modes for a benchmark."""

        stmt = (
            select(
                BenchmarkRun.mode,
                func.avg(BenchmarkRun.avg_latency_ms),
                func.avg(BenchmarkRun.total_cost),
                func.avg(BenchmarkRun.tool_success_rate),
                func.avg(BenchmarkRun.verification_rate),
                func.count(BenchmarkRun.id),
            )
            .where(BenchmarkRun.name == benchmark_name)
            .group_by(BenchmarkRun.mode)
        )
        result = await self.session.execute(stmt)

        comparison: dict[str, tuple[float, float, float, float, float, int]] = {}
        for row in result.all():
            mode, avg_latency, avg_cost, tool_success, _verification, count = row
            comparison[mode] = (
                float(avg_latency) if avg_latency else 0.0,
                float(avg_cost) if avg_cost else 0.0,
                float(tool_success) if tool_success else 0.0,
                float(0),  # verification rate not directly available
                float(0),  # success rate not directly available
                int(count) if count else 0,
            )
        return comparison


class HardwareProfileRepository(BaseRepository):
    """Hardware profile repository."""

    async def add_hardware_profile(self, profile: Any) -> HardwareProfile:
        """Add or update a hardware profile.

        Accepts either a HardwareProfile dataclass or SQLAlchemy HardwareProfile model.
        """
        # Convert dataclass to SQLAlchemy model if needed
        if not isinstance(profile, HardwareProfile):
            # Assume it's the dataclass from hardware module
            profile_dict = profile.to_dict()
            hw_profile = HardwareProfile(
                cpu_vendor=profile_dict["cpu"]["vendor"],
                cpu_model=profile_dict["cpu"]["model"],
                cpu_cores_physical=profile_dict["cpu"]["cores_physical"],
                cpu_cores_logical=profile_dict["cpu"]["cores_logical"],
                cpu_frequency_mhz=profile_dict["cpu"]["frequency_mhz"],
                cpu_instruction_sets=profile_dict["cpu"]["instruction_sets"],
                cpu_architecture=profile_dict["cpu"]["architecture"],
                memory_total_gb=profile_dict["memory"]["total_gb"],
                memory_available_gb=profile_dict["memory"]["available_gb"],
                swap_total_gb=profile_dict["memory"]["swap_total_gb"],
                swap_free_gb=profile_dict["memory"]["swap_free_gb"],
                gpu_vendor=profile_dict["gpu"]["vendor"],
                gpu_model=profile_dict["gpu"]["model"],
                gpu_vram_gb=profile_dict["gpu"]["vram_gb"],
                gpu_driver_version=profile_dict["gpu"]["driver_version"],
                gpu_compute_capability=profile_dict["gpu"]["compute_capability"],
                gpu_device_id=profile_dict["gpu"]["device_id"],
                os_name=profile_dict["os"]["name"],
                os_version=profile_dict["os"]["version"],
                os_arch=profile_dict["os"]["arch"],
                os_python_version=profile_dict["os"]["python_version"],
                disk_free_gb=profile_dict["disk_free_gb"],
                raw_data=profile_dict["raw_data"],
            )
            profile = hw_profile
        else:
            hw_profile = profile

        # Check if a similar profile already exists
        stmt = select(HardwareProfile).where(
            HardwareProfile.cpu_vendor == hw_profile.cpu_vendor,
            HardwareProfile.cpu_model == hw_profile.cpu_model,
            HardwareProfile.cpu_cores_physical == hw_profile.cpu_cores_physical,
            HardwareProfile.cpu_cores_logical == hw_profile.cpu_cores_logical,
            HardwareProfile.memory_total_gb == hw_profile.memory_total_gb,
            HardwareProfile.gpu_vendor == hw_profile.gpu_vendor,
            HardwareProfile.gpu_model == hw_profile.gpu_model,
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            # Update existing profile (exclude auto-managed fields)
            exclude_fields = {"id", "created_at", "updated_at"}
            for key, value in hw_profile.to_dict().items():
                if key not in exclude_fields:
                    setattr(existing, key, value)
            await self.session.flush()
            await self.session.refresh(existing)
            return existing

        return await self.add(hw_profile)

    async def get_latest_profile(self) -> HardwareProfile | None:
        """Get the most recent hardware profile."""
        stmt = select(HardwareProfile).order_by(desc(HardwareProfile.created_at)).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class BenchmarkRepository(BaseRepository):
    """Benchmark run repository."""

    async def add_run(self, run: BenchmarkRun) -> BenchmarkRun:
        return await self.add(run)

    async def get_recent_runs(self, limit: int = 50) -> list[BenchmarkRun]:
        stmt = select(BenchmarkRun).order_by(desc(BenchmarkRun.created_at)).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def compare_modes(self, benchmark_name: str) \
        -> dict[str, tuple[float, float, float, float, float, int]]:
        """Compare baseline vs sidweave modes for a benchmark."""

        stmt = (
            select(
                BenchmarkRun.mode,
                func.avg(BenchmarkRun.avg_latency_ms),
                func.avg(BenchmarkRun.total_cost),
                func.avg(BenchmarkRun.tool_success_rate),
                func.avg(BenchmarkRun.verification_rate),
                func.count(BenchmarkRun.id),
            )
            .where(BenchmarkRun.name == benchmark_name)
            .group_by(BenchmarkRun.mode)
        )
        result = await self.session.execute(stmt)

        comparison: dict[str, tuple[float, float, float, float, float, int]] = {}
        for row in result.all():
            mode, avg_latency, avg_cost, tool_success, _verification, count = row
            comparison[mode] = (
                float(avg_latency) if avg_latency else 0.0,
                float(avg_cost) if avg_cost else 0.0,
                float(tool_success) if tool_success else 0.0,
                0.0,
                0.0,
                int(count) if count else 0,
            )
        return comparison


__all__ = [
    "Base",
    "BenchmarkRepository",
    "BenchmarkRun",
    "ExecutionRepository",
    "HardwareProfile",
    "HardwareProfileRepository",
    "MeasurementRepository",
    "Model",
    "ModelMeasurement",
    "ModelRepository",
    "RoutingDecision",
    "RoutingRepository",
    "TaskRepository",
    "ToolExecution",
    "close_database",
    "get_session",
    "init_database",
]
