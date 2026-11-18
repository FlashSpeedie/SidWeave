"""
SidWeave CLI - Command-line interface for the adaptive AI runtime.
"""

import asyncio
import contextlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import typer
import uvicorn
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sidweave.benchmark import (
    BUILTIN_BENCHMARKS,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
    load_benchmark_tasks,
)
from sidweave.config import Settings, get_settings
from sidweave.core.runtime import SidWeaveRuntime, create_runtime
from sidweave.hardware import HardwareProfile, HardwareProfiler
from sidweave.models import ModelCapability, ModelRegistry
from sidweave.providers import ProviderGateway
from sidweave.routing import RoutingConstraints, RoutingEngine, RoutingStrategy
from sidweave.storage import close_database

app = typer.Typer(
    name="sidweave",
    help="SidWeave - Adaptive Runtime for AI Agents",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


# Module-level typer.Option constants to avoid B008
_BENCHMARK_MODE_OPT = typer.Option(
    "sidweave", "--mode", "-m", help="Mode: baseline, sidweave, or comparison"
)
_BENCHMARK_AGENT_OPT = typer.Option("test-agent", "--agent", "-a", help="Agent name")
_BENCHMARK_TASKS_OPT = typer.Option(
    None,
    "--tasks",
    "-t",
    help="Comma-separated task names (from builtin: quick, coding, reasoning)",
)
_BENCHMARK_TASK_FILE_OPT = typer.Option(
    None, "--task-file", "-f", help="Path to JSON file with custom tasks"
)
_BENCHMARK_COMPARE_OPT = typer.Option(
    False, "--compare", "-c", help="Run both baseline and sidweave modes for comparison"
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark command."""
    name: str
    mode: str = "sidweave"
    agent: str = "test-agent"
    tasks: str | None = None
    task_file: Path | None = None
    compare: bool = False


def _build_benchmark_config(  # noqa: PLR0913
    name: str,
    *,
    mode: str = "sidweave",
    agent: str = "test-agent",
    tasks: str | None = None,
    task_file: Path | None = None,
    compare: bool = False,
) -> BenchmarkConfig:
    """Build benchmark configuration from CLI arguments."""
    return BenchmarkConfig(
        name=name,
        mode=mode,
        agent=agent,
        tasks=tasks,
        task_file=task_file,
        compare=compare,
    )


def get_runtime(settings: Settings | None = None) -> SidWeaveRuntime:
    """Create and start runtime."""
    settings = settings or get_settings()
    return SidWeaveRuntime(settings)


@app.command()
def init(
    data_dir: Path | None = typer.Option(  # noqa: B008
        None, "--data-dir", "-d", help="Data directory for SidWeave"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing configuration"
    ),
) -> None:
    """Initialize SidWeave configuration and data directory."""
    settings = get_settings()

    if data_dir:
        settings.storage.data_dir = data_dir

    if settings.storage.data_dir.exists() and not force:
        msg = f"[yellow]Data directory already exists: {settings.storage.data_dir}[/yellow]"
        console.print(msg)
        console.print("Use --force to overwrite")
        return

    with Progress(
        SpinnerColumn(spinner_name="simpleDots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing SidWeave...", total=None)

        settings.ensure_data_dir()

        # Create example .env file
        env_example = settings.storage.data_dir / ".env.example"
        if not env_example.exists():
            env_content = """# SidWeave Configuration
# Copy to .env and modify as needed

# Runtime
SIDWEAVE_LOG_LEVEL=INFO
SIDWEAVE_ENVIRONMENT=development

# Server
SIDWEAVE_SERVER_HOST=127.0.0.1
SIDWEAVE_SERVER_PORT=8080

# Ollama
SIDWEAVE_PROVIDER_OLLAMA_HOST=http://localhost:11434
SIDWEAVE_PROVIDER_OLLAMA_TIMEOUT=30

# Cloud Providers (optional)
SIDWEAVE_PROVIDER_ENABLE_CLOUD=false
SIDWEAVE_PROVIDER_LITELLM_MASTER_KEY=

# Routing
SIDWEAVE_ROUTING_DEFAULT_STRATEGY=balanced
SIDWEAVE_ROUTING_PREFER_LOCAL=true
SIDWEAVE_ROUTING_FALLBACK_TO_CLOUD=true

# Telemetry
SIDWEAVE_TELEMETRY_ENABLED=true
SIDWEAVE_TELEMETRY_OTEL_ENDPOINT=
"""
            env_example.write_text(env_content)

        progress.update(task, description="SidWeave initialized successfully!")

    console.print(Panel.fit(
        f"[green]SidWeave initialized![/green]\n\n"
        f"Data directory: [cyan]{settings.storage.data_dir}[/cyan]\n"
        f"Config file: [cyan]{env_example}[/cyan]\n\n"
        f"Next steps:\n"
        f"1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and configure\n"
        f"2. Run [cyan]sidweave doctor[/cyan] to verify setup\n"
        f"3. Run [cyan]sidweave serve[/cyan] to start the runtime",
        title="Welcome to SidWeave",
    ))


@app.command()
def doctor(
    _verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics"),
) -> None:
    """Run diagnostics to verify SidWeave setup."""
    settings = get_settings()

    console.print(Panel.fit("SidWeave Diagnostics", style="bold blue"))

    checks = []

    # Check data directory
    data_dir = settings.storage.data_dir
    checks.append(("Data directory", data_dir.exists(), str(data_dir)))

    # Check database
    db_path = settings.database_path
    checks.append(("Database", db_path.exists(), str(db_path)))

    # Check Ollama
    ollama_available = False
    with contextlib.suppress(Exception):
        ollama_available = shutil.which("ollama") is not None
    status = "Available" if ollama_available else "Not found"
    checks.append(("Ollama CLI", ollama_available, status))

    # Check Python packages
    required_packages = [
        "ollama", "litellm", "mcp", "fastapi", "uvicorn",
        "sqlalchemy", "pydantic", "typer", "rich", "httpx",
    ]

    for pkg in required_packages:
        try:
            __import__(pkg)
            checks.append((f"Package: {pkg}", True, "OK"))
        except ImportError:
            checks.append((f"Package: {pkg}", False, "MISSING"))

    # Display results
    table = Table(title="Diagnostic Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    all_passed = True
    for name, passed, details in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, details)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print("\n[red]Some checks failed. Run with --verbose for details.[/red]")
        sys.exit(1)


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Server host"
    ),
    port: int = typer.Option(
        8080, "--port", "-p", help="Server port"
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload"
    ),
) -> None:
    """Start the SidWeave runtime server."""
    settings = get_settings()
    settings.server.host = host
    settings.server.port = port
    settings.server.reload = reload

    console.print(Panel.fit(
        f"Starting SidWeave server on [cyan]{host}:{port}[/cyan]",
        title="SidWeave Runtime",
    ))

    try:
        uvicorn.run(
            "sidweave.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level=settings.server.log_level.lower(),
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Server error: {e}[/red]")
        sys.exit(1)


@app.command()
def models(
    action: str = typer.Argument("list", help="Action: list, profile, stats"),
    model_name: str | None = typer.Argument(None, help="Model name for profile/stats"),
    test_types: str | None = typer.Option(
        None, "--tests", "-t", help="Comma-separated test types (latency,throughput)"
    ),
) -> None:
    """Manage and profile models."""
    # Use Typer's native async support instead of asyncio.run()
    _run_models_action_sync(action, model_name, test_types)


def _run_models_action_sync(
    action: str,
    model_name: str | None,
    test_types: str | None,
) -> None:
    """Run models action with proper async handling."""

    async def _run_models_action(
        action: str,
        model_name: str | None,
        test_types: str | None,
    ) -> None:
        """Run models action."""
        settings = get_settings()

        async with create_runtime(settings) as runtime:  # type: SidWeaveRuntime
            model_registry = runtime.get_component("model_registry")
            assert isinstance(model_registry, ModelRegistry)

            # Fix: Use correct key "hardware_profile" instead of "hardware_profiler"
            hardware_profile = runtime.get_component("hardware_profile")
            if hardware_profile is None:
                # If not cached, detect it
                hardware_profiler = HardwareProfiler(settings)
                profile = await hardware_profiler.detect()
            else:
                profile = cast(HardwareProfile, hardware_profile)
                assert isinstance(profile, HardwareProfile)

            if action == "list":
                await _list_models(model_registry)
            elif action == "profile":
                if not model_name:
                    console.print("[red]Model name required for profile action[/red]")
                    return
                await _profile_model(model_registry, model_name, test_types, profile)
            elif action == "stats":
                await _show_model_stats(model_registry, model_name)
            else:
                console.print(f"[red]Unknown action: {action}[/red]")

    # Run the async function properly
    try:
        asyncio.run(_run_models_action(action, model_name, test_types))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # We're already in an event loop, run directly
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_run_models_action(action, model_name, test_types))
        else:
            raise


async def _list_models(model_registry: ModelRegistry) -> None:
    """List available models."""
    models = await model_registry.list_models()

    table = Table(title=f"Available Models ({len(models)})")
    table.add_column("Name", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Capabilities", style="yellow")
    table.add_column("Context", style="blue")
    table.add_column("Local", style="magenta")
    table.add_column("Available", style="green")

    for m in models:
        caps = ", ".join([c.value for c in m.capabilities])
        table.add_row(
            m.name,
            m.provider,
            caps,
            str(m.context_length),
            "[+]" if m.is_local else "[-]",
            "[+]" if m.is_available else "[-]",
        )

    console.print(table)


async def _profile_model(
    model_registry: ModelRegistry,
    model_name: str | None,
    test_types: str | None,
    profile: HardwareProfile,
) -> None:
    """Profile a model."""
    if not model_name:
        console.print("[red]Model name required for profile action[/red]")
        return

    tests = test_types.split(",") if test_types else ["latency", "throughput"]

    console.print(f"Profiling [cyan]{model_name}[/cyan] with tests: {tests}")

    with Progress(
        SpinnerColumn(spinner_name="simpleDots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        _ = progress.add_task("Running profiling...", total=None)
        results = await model_registry.profile_model(model_name, tests, profile)

    table = Table(title=f"Profiling Results: {model_name}")
    table.add_column("Test Type", style="cyan")
    table.add_column("Latency (ms)", style="green")
    table.add_column("Throughput (tok/s)", style="blue")
    table.add_column("Success", style="magenta")
    table.add_column("Error", style="red")

    for r in results:
        table.add_row(
            r.test_type,
            f"{r.latency_ms:.1f}" if r.latency_ms else "N/A",
            f"{r.throughput_tps:.1f}" if r.throughput_tps else "N/A",
            "[+]" if r.success else "[-]",
            r.error or "",
        )

    console.print(table)


async def _show_model_stats(model_registry: ModelRegistry, model_name: str | None) -> None:
    """Show model statistics."""
    if not model_name:
        console.print("[red]Model name required for stats action[/red]")
        return

    stats = await model_registry.get_model_stats(model_name)

    if not stats:
        console.print(f"[yellow]No stats available for {model_name}[/yellow]")
        return

    table = Table(title=f"Model Stats: {model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in stats.items():
        if v is not None:
            table.add_row(k, str(v))

    console.print(table)


@app.command()
def route(
    task: str = typer.Argument(..., help="Task category (e.g., coding, chat, reasoning)"),
    capabilities: str | None = typer.Option(
        None, "--capabilities", "-c", help="Comma-separated required capabilities"
    ),
    strategy: str = typer.Option(
        "balanced", "--strategy", "-s", help="Routing strategy"
    ),
    max_latency: int | None = typer.Option(
        None, "--max-latency", help="Maximum latency in ms"
    ),
) -> None:
    """Test routing decision for a task."""
    _run_route_sync(task, capabilities, strategy, max_latency)


def _run_route_sync(
    task: str,
    capabilities: str | None,
    strategy: str,
    max_latency: int | None,
) -> None:
    """Run route action with proper async handling."""

    async def _route() -> None:
        settings = get_settings()

        async with create_runtime(settings) as runtime:  # type: SidWeaveRuntime
            routing_engine = runtime.get_component("routing_engine")
            assert isinstance(routing_engine, RoutingEngine)

            # Fix: Use correct key "hardware_profile"
            hardware_profile = runtime.get_component("hardware_profile")
            if hardware_profile is None:
                hardware_profiler = HardwareProfiler(settings)
                profile = await hardware_profiler.detect()
            else:
                profile = cast(HardwareProfile, hardware_profile)
                assert isinstance(profile, HardwareProfile)

            caps = [ModelCapability(c) for c in capabilities.split(",")] if capabilities else None

            constraints = RoutingConstraints(
                max_latency_ms=max_latency,
                strategy=RoutingStrategy(strategy),
            )

            console.print(f"Routing for task: [cyan]{task}[/cyan]")
            if caps:
                cap_str = ", ".join(c.value for c in caps)
                console.print(f"Required capabilities: [yellow]{cap_str}[/yellow]")

            with Progress(
                SpinnerColumn(spinner_name="simpleDots"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                _ = progress.add_task("Computing routing decision...", total=None)
                result = await routing_engine.route(
                    task_category=task,
                    required_capabilities=caps,
                    constraints=constraints,
                    hardware_profile=profile,
                )

            console.print(Panel.fit(
                f"[bold]Selected Model:[/bold] [cyan]{result.selected_model.name}[/cyan]\n"
                f"[bold]Provider:[/bold] [green]{result.selected_provider.name}[/green]\n"
                f"[bold]Decision Time:[/bold] [yellow]{result.decision_time_ms:.1f}ms[/yellow]\n\n"
                f"[bold]Rationale:[/bold]\n{result.rationale}",
                title="Routing Decision",
            ))

            if result.fallback_models:
                fallback_table = Table(title="Fallback Models")
                fallback_table.add_column("Model", style="cyan")
                fallback_table.add_column("Score", style="green")

                for model in result.fallback_models[:3]:
                    score = result.scores.get(model.name)
                    if score:
                        fallback_table.add_row(model.name, f"{score.total_score:.3f}")

                console.print(fallback_table)

    # Run the async function properly
    try:
        asyncio.run(_route())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_route())
        else:
            raise


@app.command()
def status() -> None:
    """Show runtime status."""
    settings = get_settings()

    console.print(Panel.fit("SidWeave Status", style="bold blue"))

    # Configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Data Directory", str(settings.storage.data_dir))
    table.add_row("Database", str(settings.database_path))
    table.add_row("Ollama Host", settings.provider.ollama_host)
    cloud_status = "Enabled" if settings.provider.enable_cloud_providers else "Disabled"
    table.add_row("Cloud Providers", cloud_status)
    table.add_row("Routing Strategy", settings.routing.default_strategy)
    telemetry_status = "Enabled" if settings.telemetry.enabled else "Disabled"
    table.add_row("Telemetry", telemetry_status)

    console.print(table)

    # Hardware
    console.print("\n")
    _run_hardware_sync()


def _run_hardware_sync() -> None:
    """Run hardware detection with proper async handling."""

    async def _hardware() -> None:
        settings = get_settings()
        hardware_profiler = HardwareProfiler(settings)
        profile = await hardware_profiler.detect()

        hw_table = Table(title="Hardware Profile")
        hw_table.add_column("Component", style="cyan")
        hw_table.add_column("Details", style="green")

        cpu_str = (
            f"{profile.cpu.vendor} {profile.cpu.model} "
            f"({profile.cpu.cores_physical}C/{profile.cpu.cores_logical}T)"
        )
        hw_table.add_row("CPU", cpu_str)

        mem_str = (
            f"{profile.memory.total_gb:.1f}GB total, "
            f"{profile.memory.available_gb:.1f}GB available"
        )
        hw_table.add_row("Memory", mem_str)

        if profile.gpu.is_available:
            gpu_str = (
                f"{profile.gpu.vendor} {profile.gpu.model} "
                f"({profile.gpu.vram_gb:.1f}GB VRAM)"
            )
            hw_table.add_row("GPU", gpu_str)
        else:
            hw_table.add_row("GPU", "None detected")

        os_str = f"{profile.os.name} {profile.os.version} ({profile.os.arch})"
        hw_table.add_row("OS", os_str)

        backends = ", ".join(profile.available_backends) if profile.available_backends else "None"
        hw_table.add_row("Backends", backends)

        console.print(hw_table)

    try:
        asyncio.run(_hardware())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_hardware())
        else:
            raise


@app.command()
def benchmark(  # noqa: PLR0913, PLR0917
    name: str = typer.Argument(..., help="Benchmark name"),
    mode: str = _BENCHMARK_MODE_OPT,
    agent: str = _BENCHMARK_AGENT_OPT,
    tasks: str | None = _BENCHMARK_TASKS_OPT,
    task_file: Path | None = _BENCHMARK_TASK_FILE_OPT,
    compare: bool = _BENCHMARK_COMPARE_OPT,
) -> None:
    """Run benchmark comparison."""
    config = _build_benchmark_config(
        name=name,
        mode=mode,
        agent=agent,
        tasks=tasks,
        task_file=task_file,
        compare=compare,
    )
    try:
        asyncio.run(_run_benchmark(config))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_run_benchmark(config))
        else:
            raise


async def _run_benchmark(config: BenchmarkConfig) -> None:
    """Run benchmark with the given configuration."""
    settings = get_settings()

    # Initialize components
    provider_gateway = ProviderGateway(settings)
    await provider_gateway.initialize()

    routing_engine = RoutingEngine(settings)
    await routing_engine.initialize()

    runner = BenchmarkRunner(
        settings=settings,
        provider_gateway=provider_gateway,
        routing_engine=routing_engine,
    )

    task_list = _resolve_task_list(config)
    if not task_list:
        console.print("[red]No tasks to run[/red]")
        return

    _print_benchmark_header(config, task_list)

    if config.mode == "comparison" or config.compare:
        await _run_comparison_benchmark(runner, config, task_list)
    else:
        await _run_single_benchmark(runner, config, task_list)

    await provider_gateway.shutdown()
    await close_database()


def _resolve_task_list(config: BenchmarkConfig) -> list[BenchmarkTask]:
    """Resolve the task list from configuration."""
    if config.task_file:
        return load_benchmark_tasks(config.task_file)

    if config.tasks:
        task_list: list[BenchmarkTask] = []
        for task_name in config.tasks.split(","):
            if task_name in BUILTIN_BENCHMARKS:
                task_list.extend(BUILTIN_BENCHMARKS[task_name])
            else:
                console.print(f"[red]Unknown builtin benchmark: {task_name}[/red]")
                return []
        return task_list

    # Default to quick benchmark
    return BUILTIN_BENCHMARKS["quick"]


def _print_benchmark_header(config: BenchmarkConfig, task_list: list["BenchmarkTask"]) -> None:
    """Print benchmark header panel."""
    mode_display = config.mode
    if config.mode == "comparison" or config.compare:
        mode_display = "comparison (baseline vs sidweave)"
    console.print(Panel.fit(
        f"Running benchmark: [cyan]{config.name}[/cyan]\n"
        f"Mode: [yellow]{mode_display}[/yellow] | Agent: [green]{config.agent}[/green] "
        f"| Tasks: [blue]{len(task_list)}[/blue]",
        title="Benchmark",
    ))


async def _run_comparison_benchmark(
    runner: BenchmarkRunner,
    config: BenchmarkConfig,
    task_list: list["BenchmarkTask"],
) -> None:
    """Run comparison benchmark (baseline vs sidweave)."""
    console.print("[cyan]Running comparison: baseline vs sidweave[/cyan]")
    baseline, sidweave = await runner.run_comparison(
        tasks=task_list,
        agent_name=config.agent,
        benchmark_name=config.name,
    )
    runner.print_comparison(baseline, sidweave)


async def _run_single_benchmark(
    runner: BenchmarkRunner,
    config: BenchmarkConfig,
    task_list: list["BenchmarkTask"],
) -> None:
    """Run single mode benchmark."""
    benchmark_mode = BenchmarkMode(config.mode)
    result = await runner.run_benchmark(
        tasks=task_list,
        mode=benchmark_mode,
        agent_name=config.agent,
        benchmark_name=config.name,
    )
    _print_benchmark_results(config, result)


def _print_benchmark_results(config: BenchmarkConfig, result: "BenchmarkResult") -> None:
    """Print benchmark results table and summary."""
    table = Table(title=f"Benchmark Results: {config.name} ({config.mode})")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Latency", style="yellow")
    table.add_column("Tokens", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Provider", style="cyan")

    for tr in result.task_results:
        status = "[green]PASS[/green]" if tr.success else "[red]FAIL[/red]"
        latency = f"{tr.latency_ms:.1f}ms" if tr.latency_ms else "N/A"
        tokens = str(tr.tokens_used) if tr.tokens_used else "N/A"
        model = tr.model_used or "N/A"
        provider = tr.provider_used or "N/A"
        table.add_row(tr.task_name, status, latency, tokens, model, provider)

    console.print(table)

    summary = Table(title="Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Total Tasks", str(result.total_tasks))
    summary.add_row("Passed", str(result.passed_tasks))
    summary.add_row("Failed", str(result.failed_tasks))
    summary.add_row("Success Rate", f"{result.success_rate:.1%}")
    avg_latency = f"{result.avg_latency_ms:.1f}ms" if result.avg_latency_ms else "N/A"
    summary.add_row("Avg Latency", avg_latency)
    summary.add_row("Total Tokens", f"{result.total_tokens:,}")
    console.print(summary)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config in editor"),
) -> None:
    """Manage configuration."""
    settings = get_settings()

    if show:
        config_dict = settings.model_dump()
        console.print(yaml.dump(config_dict, default_flow_style=False))
    elif edit:
        env_file = settings.storage.data_dir / ".env"
        if env_file.exists():
            subprocess.run(["code", str(env_file)], check=False)
        else:
            console.print("[red]No .env file found. Run 'sidweave init' first.[/red]")
    else:
        console.print("Use --show to display config or --edit to edit")


@app.command()
def version() -> None:
    """Show version information."""
    settings = get_settings()
    console.print(f"SidWeave v{settings.version}")
    console.print(f"Python: {sys.version.split()[0]}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """SidWeave - Adaptive Runtime for AI Agents"""
    if ctx.invoked_subcommand is None:
        console.print(Panel.fit(
            "[bold]SidWeave[/bold] - Adaptive Runtime for AI Agents\n\n"
            "Make existing AI agents more capable, efficient, and reliable\n"
            "without requiring users to replace their agents.\n\n"
            "Run [cyan]sidweave --help[/cyan] for available commands.",
            title="Welcome to SidWeave",
        ))


if __name__ == "__main__":
    app()
