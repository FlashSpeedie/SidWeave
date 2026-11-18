"""
Configuration management for SidWeave.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareSettings(BaseSettings):
    """Hardware detection and profiling settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_HARDWARE_")

    # Detection
    enable_gpu_detection: bool = True
    gpu_detection_timeout: float = 5.0

    # Profiling
    profile_warmup_runs: int = 2
    profile_measurement_runs: int = 5
    profile_timeout: float = 60.0

    # Periodic profiling
    profile_interval_sec: int = 3600


class ProviderSettings(BaseSettings):
    """Provider integration settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_PROVIDER_")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 30.0
    ollama_keep_alive: str = "5m"

    # LiteLLM (cloud providers)
    litellm_master_key: SecretStr | None = None
    litellm_base_url: str | None = None
    enable_cloud_providers: bool = False

    # Fallback
    prefer_local: bool = True
    fallback_to_cloud: bool = True

    # Health check
    health_check_interval_sec: int = 300


class RoutingSettings(BaseSettings):
    """Adaptive routing settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_ROUTING_")

    # Strategy
    default_strategy: str = "balanced"  # balanced, latency, cost, quality
    enable_learning: bool = True
    min_samples_for_learning: int = 10

    # Constraints
    max_latency_ms: int | None = None
    max_cost_per_1k_tokens: float | None = None
    require_capabilities: list[str] = Field(default_factory=list)

    # Fallback
    max_fallback_attempts: int = 3
    fallback_on_error: bool = True


class ExecutionSettings(BaseSettings):
    """Verified execution settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_EXECUTION_")

    # MCP
    mcp_servers: list[str] = Field(default_factory=list)
    mcp_timeout: float = 30.0

    # Verification
    require_preconditions: bool = True
    require_postconditions: bool = True
    enable_idempotency: bool = True

    # Recovery
    max_retry_attempts: int = 2
    retry_on_transient: bool = True
    require_approval_for: list[str] = Field(default_factory=list)


class TelemetrySettings(BaseSettings):
    """Telemetry and observability settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_TELEMETRY_")

    enabled: bool = True
    service_name: str = "sidweave"
    otel_endpoint: str | None = None
    otel_headers: dict[str, str] = Field(default_factory=dict)
    sample_rate: float = 1.0
    export_interval_ms: int = 5000
    log_level: str = "INFO"


class StorageSettings(BaseSettings):
    """Storage settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_STORAGE_")

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".sidweave")
    db_filename: str = "sidweave.db"
    enable_wal: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7


class ServerSettings(BaseSettings):
    """API server settings."""
    model_config = SettingsConfigDict(env_prefix="SIDWEAVE_SERVER_")

    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    access_log: bool = True


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    # Meta
    app_name: str = "SidWeave"
    version: str = "0.1.0"
    environment: str = "development"  # development, staging, production
    debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, console

    # Sub-settings
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @property
    def database_path(self) -> Path:
        """Full path to SQLite database."""
        return self.storage.data_dir / self.storage.db_filename

    @property
    def database_url(self) -> str:
        """Async SQLite database URL."""
        if self.storage.db_filename == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        return f"sqlite+aiosqlite:///{self.database_path}"

    def ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        (self.storage.data_dir / "logs").mkdir(exist_ok=True)
        (self.storage.data_dir / "cache").mkdir(exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
        _settings.ensure_data_dir()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None
