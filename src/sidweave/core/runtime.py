"""
SidWeave Runtime - Main orchestrator for all components.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from sidweave.config import Settings, get_settings
from sidweave.execution import ExecutionEngine
from sidweave.hardware import HardwareProfile, HardwareProfiler
from sidweave.learning import LearningEngine
from sidweave.models import ModelRegistry
from sidweave.providers import ProviderGateway
from sidweave.routing import RoutingEngine

if TYPE_CHECKING:
    pass
from sidweave.storage import close_database, init_database
from sidweave.telemetry import init_telemetry

if TYPE_CHECKING:
    from sidweave.execution import ExecutionEngine
    from sidweave.hardware import HardwareProfile
    from sidweave.learning import LearningEngine
    from sidweave.models import ModelRegistry
    from sidweave.routing import RoutingEngine

logger = logging.getLogger(__name__)


class RuntimeState(Enum):
    """Runtime lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ComponentStatus:
    """Status of a runtime component."""
    name: str
    healthy: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class SidWeaveRuntime:
    """
    Main runtime orchestrator.

    Manages lifecycle of all SidWeave components:
    - Hardware Profiler
    - Model Registry
    - Provider Gateway
    - Routing Engine
    - Execution Engine
    - Learning Engine
    - Telemetry
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._components: dict[
            str,
            (
                ProviderGateway
                | HardwareProfile
                | ModelRegistry
                | RoutingEngine
                | ExecutionEngine
                | LearningEngine
            ),
        ] = {}
        self._component_status: dict[str, ComponentStatus] = {}
        self._state = RuntimeState.STOPPED
        self._startup_tasks: list[asyncio.Task[Any]] = []
        self._shutdown_tasks: list[asyncio.Task[Any]] = []
        self._periodic_tasks: list[asyncio.Task[Any]] = []

    def get_component(self, name: str) -> ProviderGateway | None:
        """Get a component by name."""
        return cast(
            "ProviderGateway | None",
            self._components.get(name),
        )

    def get_status(self) -> dict[str, ComponentStatus]:
        """Get status of all components."""
        return self._component_status.copy()

    # Component initialization methods

    async def _init_telemetry(self) -> None:
        """Initialize OpenTelemetry."""
        self._component_status["telemetry"] = ComponentStatus(
            name="telemetry",
            healthy=False,
            message="Initializing..."
        )

        init_telemetry(self.settings)

        self._component_status["telemetry"] = ComponentStatus(
            name="telemetry",
            healthy=True,
            message="OpenTelemetry initialized"
        )

    async def _init_storage(self) -> None:
        """Initialize database storage."""
        self._component_status["storage"] = ComponentStatus(
            name="storage",
            healthy=False,
            message="Initializing..."
        )

        await init_database(self.settings)

        self._component_status["storage"] = ComponentStatus(
            name="storage",
            healthy=True,
            message="Database initialized"
        )

    async def _init_hardware_profiler(self) -> None:
        """Initialize hardware profiler."""
        self._component_status["hardware"] = ComponentStatus(
            name="hardware",
            healthy=False,
            message="Initializing..."
        )

        profiler = HardwareProfiler(self.settings)
        profile = await profiler.detect()

        self._components["hardware_profile"] = profile
        self._component_status["hardware"] = ComponentStatus(
            name="hardware",
            healthy=True,
            message="Hardware profile initialized"
        )

    async def _init_model_registry(self) -> None:
        """Initialize model registry."""
        self._component_status["models"] = ComponentStatus(
            name="models",
            healthy=False,
            message="Initializing..."
        )

        model_registry = ModelRegistry(self.settings)
        await model_registry.initialize()

        self._components["model_registry"] = model_registry
        self._component_status["models"] = ComponentStatus(
            name="models",
            healthy=True,
            message="Model registry initialized"
        )

    async def _init_provider_gateway(self) -> None:
        """Initialize provider gateway."""
        self._component_status["providers"] = ComponentStatus(
            name="providers",
            healthy=False,
            message="Initializing..."
        )

        provider_gateway = ProviderGateway(self.settings)
        await provider_gateway.initialize()

        self._components["provider_gateway"] = provider_gateway
        self._component_status["providers"] = ComponentStatus(
            name="providers",
            healthy=True,
            message="Provider gateway initialized"
        )

    async def _init_routing_engine(self) -> None:
        """Initialize routing engine."""
        self._component_status["routing"] = ComponentStatus(
            name="routing",
            healthy=False,
            message="Initializing..."
        )

        routing_engine = RoutingEngine(self.settings)
        await routing_engine.initialize()

        self._components["routing_engine"] = routing_engine
        self._component_status["routing"] = ComponentStatus(
            name="routing",
            healthy=True,
            message="Routing engine initialized"
        )

    async def _init_execution_engine(self) -> None:
        """Initialize verified execution engine."""
        self._component_status["execution"] = ComponentStatus(
            name="execution",
            healthy=False,
            message="Initializing..."
        )

        execution_engine = ExecutionEngine(self.settings)
        await execution_engine.initialize()

        self._components["execution_engine"] = execution_engine
        self._component_status["execution"] = ComponentStatus(
            name="execution",
            healthy=True,
            message="Execution engine initialized"
        )

    async def _init_learning_engine(self) -> None:
        """Initialize learning and evaluation engine."""
        self._component_status["learning"] = ComponentStatus(
            name="learning",
            healthy=False,
            message="Initializing..."
        )

        learning_engine = LearningEngine(self.settings)
        await learning_engine.initialize()

        self._components["learning_engine"] = learning_engine
        self._component_status["learning"] = ComponentStatus(
            name="learning",
            healthy=True,
            message="Learning engine initialized"
        )

    async def start(self) -> None:
        """Start all runtime components."""
        if self._state != RuntimeState.STOPPED:
            logger.warning("Runtime already started or starting")
            return

        self._state = RuntimeState.STARTING
        logger.info("Starting SidWeave runtime...")

        # Initialize all components in order
        init_methods = [
            ("telemetry", self._init_telemetry),
            ("storage", self._init_storage),
            ("hardware", self._init_hardware_profiler),
            ("models", self._init_model_registry),
            ("providers", self._init_provider_gateway),
            ("routing", self._init_routing_engine),
            ("execution", self._init_execution_engine),
            ("learning", self._init_learning_engine),
        ]

        for name, init_method in init_methods:
            try:
                logger.info("Initializing %s...", name)
                await init_method()
                logger.info("%s initialized", name)
            except Exception as exc:
                logger.exception("Failed to initialize %s", name)
                self._component_status[name] = ComponentStatus(
                    name=name,
                    healthy=False,
                    message=f"Initialization failed: {exc}"
                )
                self._state = RuntimeState.ERROR
                raise

        self._state = RuntimeState.RUNNING
        logger.info("SidWeave runtime started successfully")

        # Start periodic tasks
        self._periodic_tasks = [
            asyncio.create_task(self._periodic_hardware_profile()),
            asyncio.create_task(self._periodic_model_health_check()),
        ]

    async def stop(self) -> None:
        """Stop all runtime components gracefully."""
        if self._state == RuntimeState.STOPPED:
            return

        self._state = RuntimeState.STOPPING
        logger.info("Stopping SidWeave runtime...")

        # Cancel periodic tasks
        for task in self._periodic_tasks:
            task.cancel()
        await asyncio.gather(*self._periodic_tasks, return_exceptions=True)

        # Stop components in reverse order
        for name in reversed(list(self._components.keys())):
            component = self._components[name]
            try:
                logger.info("Shutting down %s...", name)
                if hasattr(component, "shutdown"):
                    await component.shutdown()
                elif hasattr(component, "stop"):
                    await component.stop()
                logger.info("Shutdown %s", name)
            except Exception:
                logger.exception("Error shutting down %s", name)

        self._components.clear()
        self._state = RuntimeState.STOPPED
        logger.info("SidWeave runtime stopped")
        await close_database()

    async def _periodic_hardware_profile(self) -> None:
        """Periodically re-profile hardware."""
        while self._state == RuntimeState.RUNNING:
            await asyncio.sleep(self.settings.hardware.profile_interval_sec)
            if self._state != RuntimeState.RUNNING:
                break

            profiler = HardwareProfiler(self.settings)
            try:
                profile = await profiler.detect()
                self._components["hardware_profile"] = profile
                logger.debug("Hardware re-profiling completed")
            except Exception:
                logger.exception("Hardware re-profile failed")

    async def _periodic_model_health_check(self) -> None:
        """Periodically check model health."""
        while self._state == RuntimeState.RUNNING:
            await asyncio.sleep(self.settings.provider.health_check_interval_sec)
            if self._state != RuntimeState.RUNNING:
                break

            gateway = self._components.get("provider_gateway")
            if not gateway:
                return
            assert isinstance(gateway, ProviderGateway)

            try:
                await gateway.health_check()
            except Exception:
                logger.exception("Model health check failed")

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Context manager for runtime lifecycle."""
        await self.start()
        try:
            yield
        finally:
            await self.stop()


@asynccontextmanager
async def create_runtime(settings: Settings | None = None) -> AsyncIterator[SidWeaveRuntime]:
    """Create and start a runtime instance."""
    settings = settings or get_settings()
    runtime = SidWeaveRuntime(settings)
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()
