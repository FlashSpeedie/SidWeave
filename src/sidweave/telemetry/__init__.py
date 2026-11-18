"""
Telemetry and observability using OpenTelemetry.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from sidweave.config import Settings

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def init_telemetry(settings: Settings) -> None:
    """Initialize OpenTelemetry tracing and metrics."""
    global _tracer_provider, _meter_provider  # noqa: PLW0603

    if not settings.telemetry.enabled:
        return

    # Create resource
    resource = Resource.create({
        "service.name": settings.telemetry.service_name,
        "service.version": settings.version,
    })

    # Initialize tracing
    _tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(_tracer_provider)

    # Configure exporters
    if settings.telemetry.otel_endpoint:
        # OTLP exporter
        span_exporter = OTLPSpanExporter(
            endpoint=settings.telemetry.otel_endpoint,
            headers=settings.telemetry.otel_headers,
        )
        _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    else:
        # Console exporter for development
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    # Initialize metrics
    metric_readers = []
    if settings.telemetry.otel_endpoint:
        metric_exporter = OTLPMetricExporter(
            endpoint=settings.telemetry.otel_endpoint,
            headers=settings.telemetry.otel_headers,
        )
        metric_readers.append(
            PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=settings.telemetry.export_interval_ms,
            )
        )
    else:
        # Console metric exporter for development
        metric_readers.append(
            PeriodicExportingMetricReader(
                exporter=ConsoleMetricExporter(),
                export_interval_millis=settings.telemetry.export_interval_ms,
            )
        )

    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers,
    )
    metrics.set_meter_provider(_meter_provider)

    # Auto-instrumentation
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()

    # Create custom metrics
    _create_custom_metrics()


def _create_custom_metrics() -> None:
    """Create custom SidWeave metrics."""
    meter = metrics.get_meter("sidweave")

    # Routing metrics
    meter.create_counter(
        name="sidweave.routing.decisions",
        description="Number of routing decisions made",
        unit="1",
    )

    meter.create_histogram(
        name="sidweave.routing.latency",
        description="Routing decision latency",
        unit="ms",
    )

    # Provider metrics
    meter.create_counter(
        name="sidweave.provider.requests",
        description="Number of provider requests",
        unit="1",
    )

    meter.create_histogram(
        name="sidweave.provider.latency",
        description="Provider request latency",
        unit="ms",
    )

    meter.create_counter(
        name="sidweave.provider.tokens",
        description="Token usage per provider",
        unit="tokens",
    )

    meter.create_counter(
        name="sidweave.provider.errors",
        description="Number of provider errors",
        unit="1",
    )

    # Tool execution metrics
    meter.create_counter(
        name="sidweave.tool.executions",
        description="Number of tool executions",
        unit="1",
    )

    meter.create_histogram(
        name="sidweave.tool.latency",
        description="Tool execution latency",
        unit="ms",
    )

    meter.create_counter(
        name="sidweave.tool.verifications",
        description="Tool verification results",
        unit="1",
    )

    # Benchmark metrics
    meter.create_histogram(
        name="sidweave.benchmark.latency",
        description="Benchmark task latency",
        unit="ms",
    )

    meter.create_counter(
        name="sidweave.benchmark.results",
        description="Benchmark results",
        unit="1",
    )


def get_tracer(name: str = "sidweave") -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_meter(name: str = "sidweave") -> metrics.Meter:
    """Get a meter instance."""
    return metrics.get_meter(name)


@asynccontextmanager
async def trace_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> AsyncIterator[trace.Span]:
    """Context manager for tracing operations."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


# Decorator for tracing functions
def trace_function(
    name: str | None = None, attributes: dict[str, Any] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace a function."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                return result
        return wrapper
    return decorator


# Custom span attributes for SidWeave
class SpanAttributes:
    """Standard span attribute keys for SidWeave."""

    # Routing
    ROUTING_TASK_CATEGORY = "sidweave.routing.task_category"
    ROUTING_SELECTED_MODEL = "sidweave.routing.selected_model"
    ROUTING_SELECTED_PROVIDER = "sidweave.routing.selected_provider"
    ROUTING_STRATEGY = "sidweave.routing.strategy"
    ROUTING_FALLBACK_COUNT = "sidweave.routing.fallback_count"
    ROUTING_RATIONALE = "sidweave.routing.rationale"

    # Provider
    PROVIDER_NAME = "sidweave.provider.name"
    PROVIDER_MODEL = "sidweave.provider.model"
    PROVIDER_LATENCY_MS = "sidweave.provider.latency_ms"
    PROVIDER_TOKENS_PROMPT = "sidweave.provider.tokens_prompt"
    PROVIDER_TOKENS_COMPLETION = "sidweave.provider.tokens_completion"
    PROVIDER_TOKENS_TOTAL = "sidweave.provider.tokens_total"
    PROVIDER_ERROR = "sidweave.provider.error"

    # Tool execution
    TOOL_NAME = "sidweave.tool.name"
    TOOL_SERVER = "sidweave.tool.server"
    TOOL_ARGUMENTS = "sidweave.tool.arguments"
    TOOL_STATUS = "sidweave.tool.status"
    TOOL_LATENCY_MS = "sidweave.tool.latency_ms"
    TOOL_VERIFICATION_STATUS = "sidweave.tool.verification_status"
    TOOL_IDEMPOTENCY_KEY = "sidweave.tool.idempotency_key"

    # Benchmark
    BENCHMARK_NAME = "sidweave.benchmark.name"
    BENCHMARK_MODE = "sidweave.benchmark.mode"
    BENCHMARK_TASK = "sidweave.benchmark.task"
    BENCHMARK_PASSED = "sidweave.benchmark.passed"

    # Hardware
    HARDWARE_CPU_VENDOR = "sidweave.hardware.cpu_vendor"
    HARDWARE_CPU_CORES = "sidweave.hardware.cpu_cores"
    HARDWARE_GPU_VENDOR = "sidweave.hardware.gpu_vendor"
    HARDWARE_GPU_VRAM_GB = "sidweave.hardware.gpu_vram_gb"
    HARDWARE_MEMORY_GB = "sidweave.hardware.memory_gb"


def shutdown_telemetry() -> None:
    """Shutdown telemetry providers."""
    global _tracer_provider, _meter_provider  # noqa: PLW0603

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None

    if _meter_provider:
        _meter_provider.shutdown()
        _meter_provider = None
