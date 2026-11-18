"""
Unit tests for storage layer.
"""

import asyncio

import pytest
import pytest_asyncio

from sidweave.config import reset_settings
from sidweave.execution import ExecutionStatus
from sidweave.storage import (
    BenchmarkRepository,
    BenchmarkRun,
    ExecutionRepository,
    MeasurementRepository,
    Model,
    ModelMeasurement,
    ModelRepository,
    RoutingDecision,
    RoutingRepository,
    TaskRepository,
    ToolExecution,
    get_session,
)
from sidweave.storage import (
    HardwareProfile as DBHardwareProfile,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_settings_fixture():
    """Reset settings before each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.mark.asyncio
async def test_init_database(db_engine):
    """Test database initialization creates tables."""
    async with get_session(db_engine) as session:
        from sqlalchemy import inspect
        def get_tables(sync_conn):
            inspector = inspect(sync_conn)
            return inspector.get_table_names()

        conn = await session.connection()
        tables = await conn.run_sync(get_tables)

        assert "hardware_profiles" in tables
        assert "models" in tables
        assert "model_measurements" in tables
        assert "routing_decisions" in tables
        assert "tool_executions" in tables


@pytest.mark.asyncio
async def test_session_context_manager(db_session):
    """Test session context manager."""
    session = db_session
    from sqlalchemy import text
    result = await session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_session_rollback_on_error(db_session):
    """Test session rolls back on error."""
    try:
        session = db_session
        from sqlalchemy import text
        await session.execute(text("INSERT INTO models (name, provider, provider_model_id) VALUES ('test', 'test', 'test')"))
        await session.execute(text("INVALID SQL"))
    except Exception:
        pass

    # Verify rollback happened - no partial data
    session = db_session
    from sqlalchemy import select
    result = await session.execute(select(Model).where(Model.name == "test"))
    assert result.scalar_one_or_none() is None


class TestModelRepository:
    """Test ModelRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield ModelRepository(db_session)

    @pytest.mark.asyncio
    async def test_get_or_create(self, repo):
        """Test get_or_create."""
        model = await repo.get_or_create(
            name="test-model",
            provider="ollama",
            provider_model_id="test-model",
        )

        assert model.name == "test-model"
        assert model.provider == "ollama"
        assert model.id is not None

        # Second call should return same model
        model2 = await repo.get_or_create(
            name="test-model",
            provider="ollama",
            provider_model_id="test-model",
        )
        assert model2.id == model.id

    @pytest.mark.asyncio
    async def test_get_by_name(self, repo):
        """Test get_by_name."""
        await repo.get_or_create("test-model", "ollama", "test-model")
        model = await repo.get_by_name("test-model")
        assert model is not None
        assert model.name == "test-model"

        model = await repo.get_by_name("nonexistent")
        assert model is None

    @pytest.mark.asyncio
    async def test_list_models(self, repo):
        """Test list_models."""
        await repo.get_or_create("model1", "ollama", "model1")
        await repo.get_or_create("model2", "ollama", "model2")

        models = await repo.list_models()
        assert len(models) >= 2

        names = [m.name for m in models]
        assert "model1" in names
        assert "model2" in names


class TestMeasurementRepository:
    """Test MeasurementRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield MeasurementRepository(db_session)

    @pytest.mark.asyncio
    async def test_add_measurement(self, repo, db_session):
        """Test adding measurement."""
        session = db_session
        model_repo = ModelRepository(session)
        model = await model_repo.get_or_create("test", "ollama", "test")
        model_id = model.id

        session = db_session
        meas_repo = MeasurementRepository(session)
        measurement = await meas_repo.add_measurement(
            ModelMeasurement(
                model_id=model_id,
                model_name="test",
                test_type="latency",
                prompt_tokens=10,
                completion_tokens=50,
                latency_ms=150.5,
                throughput_tps=33.2,
                max_tokens=100,
                backend="ollama",
                success=True,
            )
        )

        assert measurement.id is not None
        assert measurement.model_id == model_id
        assert measurement.latency_ms == 150.5

    @pytest.mark.asyncio
    async def test_get_average_latency(self, repo, db_session):
        """Test getting average latency."""
        session = db_session
        model_repo = ModelRepository(session)
        model = await model_repo.get_or_create("test", "ollama", "test")
        model_id = model.id

        session = db_session
        meas_repo = MeasurementRepository(session)
        await meas_repo.add_measurement(
            ModelMeasurement(
                model_id=model_id,
                model_name="test",
                test_type="latency",
                prompt_tokens=10,
                completion_tokens=50,
                latency_ms=100.0,
                max_tokens=100,
                backend="ollama",
                success=True,
            )
        )
        await meas_repo.add_measurement(
            ModelMeasurement(
                model_id=model_id,
                model_name="test",
                test_type="latency",
                prompt_tokens=10,
                completion_tokens=50,
                latency_ms=200.0,
                max_tokens=100,
                backend="ollama",
                success=True,
            )
        )
        await meas_repo.add_measurement(
            ModelMeasurement(
                model_id=model_id,
                model_name="test",
                test_type="latency",
                prompt_tokens=10,
                completion_tokens=50,
                latency_ms=300.0,
                max_tokens=100,
                backend="ollama",
                success=False,
            )
        )

        session = db_session
        meas_repo = MeasurementRepository(session)
        avg = await meas_repo.get_average_latency(model_id)
        assert avg == 150.0  # (100 + 200) / 2 = 150 (only successful measurements)


class TestRoutingRepository:
    """Test RoutingRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield RoutingRepository(db_session)

    @pytest.mark.asyncio
    async def test_add_decision(self, repo, db_session):
        """Test adding routing decision."""
        session = db_session
        model_repo = ModelRepository(session)
        model = await model_repo.get_or_create("test", "ollama", "test")
        model_id = model.id

        session = db_session
        routing_repo = RoutingRepository(session)
        decision = await routing_repo.add_decision(
            RoutingDecision(
                task_category="coding",
                task_description="Test routing decision",
                selected_model_id=model_id,
                selected_model_name=model.name,
                selected_provider="ollama",
                fallback_model_ids=[],
                rationale="Best model for coding",
                scores={"test": 0.9},
            )
        )

        assert decision.id is not None

    @pytest.mark.asyncio
    async def test_get_success_rate_by_model(self, repo, db_session):
        """Test getting success rate by model."""
        session = db_session
        model_repo = ModelRepository(session)
        model = await model_repo.get_or_create("test", "ollama", "test")
        model_id = model.id

        session = db_session
        routing_repo = RoutingRepository(session)
        for i in range(5):
            await routing_repo.add_decision(
                RoutingDecision(
                    task_category="coding",
                    task_description="Test routing decision",
                    selected_model_id=model_id,
                    selected_model_name=model.name,
                    selected_provider="ollama",
                    fallback_model_ids=[],
                    rationale="Test",
                    scores={},
                    success=i < 4,  # 4 out of 5 success
                )
            )

        session = db_session
        routing_repo = RoutingRepository(session)
        rate = await routing_repo.get_success_rate_by_model(model_id)
        assert rate == 0.8  # 4/5


class TestExecutionRepository:
    """Test ExecutionRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield ExecutionRepository(db_session)

    @pytest.mark.asyncio
    async def test_add_and_get_execution(self, repo, db_session):
        """Test adding and retrieving execution."""
        session = db_session
        execution = ToolExecution(
            execution_id="test-123",
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
            idempotency_key="idem-123",
            status=ExecutionStatus.REQUESTED.value,
        )
        execution = await repo.add_execution(execution)

        assert execution.execution_id == "test-123"
        assert execution.tool_name == "read_file"

        # Retrieve by execution_id (using same repo/session)
        retrieved = await repo.get_by_execution_id("test-123")
        assert retrieved is not None
        assert retrieved.idempotency_key == "idem-123"

    @pytest.mark.asyncio
    async def test_get_by_idempotency_key(self, repo, db_session):
        """Test getting execution by idempotency key."""
        session = db_session
        repo = ExecutionRepository(session)
        execution = ToolExecution(
            execution_id="test-456",
            tool_name="write_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test2.txt", "content": "hello"},
            idempotency_key="idem-456",
            status=ExecutionStatus.REQUESTED.value,
        )
        execution = await repo.add_execution(execution)

        retrieved = await repo.get_by_idempotency_key("idem-456")
        assert retrieved is not None
        assert retrieved.execution_id == "test-456"

    @pytest.mark.asyncio
    async def test_update_status(self, repo, db_session):
        """Test updating execution status."""
        session = db_session
        repo = ExecutionRepository(session)
        execution = ToolExecution(
            execution_id="test-789",
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test3.txt"},
            idempotency_key="idem-789",
            status=ExecutionStatus.REQUESTED.value,
        )
        execution = await repo.add_execution(execution)

        await repo.update_status("test-789", "completed")

        retrieved = await repo.get_by_execution_id("test-789")
        assert retrieved is not None
        assert retrieved.status == "completed"


class TestTaskRepository:
    """Test TaskRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield TaskRepository(db_session)

    @pytest.mark.asyncio
    async def test_add_and_get_tasks(self, repo, db_session):
        """Test adding and retrieving tasks."""
        session = db_session
        model_repo = ModelRepository(session)
        model = await model_repo.get_or_create("test", "ollama", "test")
        model_id = model.id

        session = db_session
        task_repo = TaskRepository(session)
        task = BenchmarkRun(
            name="test_benchmark",
            description="Test benchmark",
            benchmark_version="0.1.0",
            sidweave_version="0.1.0",
            mode="sidweave",
            agent_name="test_agent",
            total_tasks=1,
            passed_tasks=1,
            failed_tasks=0,
            avg_latency_ms=100.0,
            total_tokens=100,
            total_cost=0.001,
            tool_success_rate=1.0,
            verification_rate=1.0,
            results={},
        )
        task = await task_repo.add_task(task)

        assert task.id is not None
        assert task.name == "test_benchmark"


class TestBenchmarkRepository:
    """Test BenchmarkRepository."""

    @pytest_asyncio.fixture
    async def repo(self, db_session):
        yield BenchmarkRepository(db_session)

    @pytest.mark.asyncio
    async def test_add_and_get_runs(self, repo, db_session):
        """Test adding and retrieving benchmark runs."""
        session = db_session
        run = BenchmarkRun(
            name="test-benchmark",
            mode="sidweave",
            agent_name="test-agent",
            total_tasks=5,
            passed_tasks=4,
            failed_tasks=1,
            avg_latency_ms=150.0,
            total_tokens=1000,
            total_cost=0.01,
            benchmark_version="0.1.0",
            sidweave_version="0.1.0",
        )
        run = await repo.add_run(run)

        assert run.id is not None
        assert run.name == "test-benchmark"

        runs = await repo.get_recent_runs(limit=10)
        assert len(runs) >= 1
        assert runs[0].name == "test-benchmark"


class TestModels:
    """Test model definitions."""

    def test_hardware_profile_model(self):
        """Test HardwareProfile model creation."""
        profile = DBHardwareProfile(
            cpu_vendor="Intel",
            cpu_model="i7",
            cpu_cores_physical=8,
            cpu_cores_logical=16,
            memory_total_gb=32.0,
            memory_available_gb=20.0,
            os_name="Linux",
            os_version="6.5",
            os_arch="x86_64",
        )
        assert profile.cpu_vendor == "Intel"
        assert profile.cpu_cores_physical == 8

        # Test to_dict
        data = profile.to_dict()
        assert data["cpu"]["vendor"] == "Intel"
        assert data["memory"]["total_gb"] == 32.0
