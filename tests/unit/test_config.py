"""
Unit tests for configuration.
"""



from sidweave.config import Settings, get_settings, reset_settings


class TestSettings:
    """Test configuration settings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.app_name == "SidWeave"
        assert settings.version == "0.1.0"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

        # Server defaults
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080
        assert settings.server.workers == 1

        # Provider defaults
        assert settings.provider.ollama_host == "http://localhost:11434"
        assert settings.provider.enable_cloud_providers is False
        assert settings.provider.prefer_local is True

        # Routing defaults
        assert settings.routing.default_strategy == "balanced"
        assert settings.routing.enable_learning is True

        # Storage defaults
        assert settings.storage.db_filename == "sidweave.db"
        assert settings.storage.enable_wal is True

    def test_database_path(self):
        """Test database path property."""
        settings = Settings()
        expected = settings.storage.data_dir / settings.storage.db_filename
        assert settings.database_path == expected

    def test_database_url(self):
        """Test database URL property."""
        settings = Settings()
        expected = f"sqlite+aiosqlite:///{settings.database_path}"
        assert settings.database_url == expected

    def test_ensure_data_dir(self, tmp_path):
        """Test data directory creation."""
        settings = Settings()
        settings.storage.data_dir = tmp_path / "test_data"

        assert not settings.storage.data_dir.exists()
        settings.ensure_data_dir()
        assert settings.storage.data_dir.exists()
        assert (settings.storage.data_dir / "logs").exists()
        assert (settings.storage.data_dir / "cache").exists()

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SIDWEAVE_SERVER_PORT", "9090")
        monkeypatch.setenv("SIDWEAVE_PROVIDER_OLLAMA_HOST", "http://custom:11434")

        reset_settings()
        settings = get_settings()

        assert settings.log_level == "DEBUG"
        assert settings.server.port == 9090
        assert settings.provider.ollama_host == "http://custom:11434"

    def test_nested_settings_env_override(self, monkeypatch):
        """Test nested settings with env prefix."""
        monkeypatch.setenv("SIDWEAVE_HARDWARE_ENABLE_GPU_DETECTION", "false")
        monkeypatch.setenv("SIDWEAVE_HARDWARE_GPU_DETECTION_TIMEOUT", "10.0")

        reset_settings()
        settings = get_settings()

        assert settings.hardware.enable_gpu_detection is False
        assert settings.hardware.gpu_detection_timeout == 10.0

    def test_secret_handling(self, monkeypatch):
        """Test secret value handling."""
        monkeypatch.setenv("SIDWEAVE_PROVIDER_LITELLM_MASTER_KEY", "secret-key-123")

        reset_settings()
        settings = get_settings()

        # SecretStr should be set
        assert settings.provider.litellm_master_key is not None
        assert settings.provider.litellm_master_key.get_secret_value() == "secret-key-123"

    def test_settings_singleton(self):
        """Test settings caching."""
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

        reset_settings()
        s3 = get_settings()
        assert s3 is not s1


class TestHardwareSettings:
    """Test hardware-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.hardware.enable_gpu_detection is True
        assert settings.hardware.gpu_detection_timeout == 5.0
        assert settings.hardware.profile_warmup_runs == 2
        assert settings.hardware.profile_measurement_runs == 5


class TestProviderSettings:
    """Test provider-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.provider.ollama_timeout == 30.0
        assert settings.provider.ollama_keep_alive == "5m"
        assert settings.provider.enable_cloud_providers is False
        assert settings.provider.prefer_local is True
        assert settings.provider.fallback_to_cloud is True


class TestRoutingSettings:
    """Test routing-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.routing.default_strategy == "balanced"
        assert settings.routing.enable_learning is True
        assert settings.routing.min_samples_for_learning == 10
        assert settings.routing.max_fallback_attempts == 3
        assert settings.routing.fallback_on_error is True


class TestExecutionSettings:
    """Test execution-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.execution.mcp_timeout == 30.0
        assert settings.execution.require_preconditions is True
        assert settings.execution.require_postconditions is True
        assert settings.execution.enable_idempotency is True
        assert settings.execution.max_retry_attempts == 2
        assert settings.execution.retry_on_transient is True


class TestTelemetrySettings:
    """Test telemetry-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.telemetry.enabled is True
        assert settings.telemetry.service_name == "sidweave"
        assert settings.telemetry.sample_rate == 1.0
        assert settings.telemetry.export_interval_ms == 5000


class TestStorageSettings:
    """Test storage-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.storage.db_filename == "sidweave.db"
        assert settings.storage.enable_wal is True
        assert settings.storage.backup_interval_hours == 24
        assert settings.storage.max_backups == 7


class TestServerSettings:
    """Test server-specific settings."""

    def setup_method(self):
        reset_settings()

    def test_defaults(self):
        settings = Settings()
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8080
        assert settings.server.workers == 1
        assert settings.server.reload is False
        assert settings.server.log_level == "info"
        assert settings.server.access_log is True
