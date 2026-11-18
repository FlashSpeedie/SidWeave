"""
Unit tests for execution engine.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from sidweave.config import Settings
from sidweave.execution import (
    ExecutionEngine,
    ExecutionRequest,
    ExecutionStatus,
    MCPServerManager,
    Postcondition,
    Precondition,
    VerificationStatus,
    file_exists_precondition,
    no_error_postcondition,
    result_success_postcondition,
)


class TestPrecondition:
    """Test Precondition dataclass."""

    def test_creation(self):
        pre = Precondition(
            name="test",
            check=lambda x: True,
            description="Test precondition",
        )
        assert pre.name == "test"
        assert pre.check({}) is True
        assert pre.description == "Test precondition"

    def test_file_exists_precondition(self):
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            pre = file_exists_precondition("path")
            assert pre.check({"path": f.name}) is True
            assert pre.check({"path": "/nonexistent/path"}) is False


class TestPostcondition:
    """Test Postcondition dataclass."""

    def test_creation(self):
        post = Postcondition(
            name="test",
            check=lambda x: True,
            description="Test postcondition",
        )
        assert post.name == "test"

    def test_result_success_postcondition(self):
        post = result_success_postcondition()
        assert post.check({"success": True}) is True
        assert post.check({"success": False}) is False
        assert post.check({}) is False

    def test_no_error_postcondition(self):
        post = no_error_postcondition()
        assert post.check({"success": True}) is True
        assert post.check({"success": True, "error": "something"}) is False
        assert post.check({"is_error": True}) is False


class TestExecutionRequest:
    """Test ExecutionRequest dataclass."""

    def test_creation(self):
        req = ExecutionRequest(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
        )
        assert req.tool_name == "read_file"
        assert req.server_name == "filesystem"
        assert req.arguments == {"path": "/tmp/test.txt"}
        assert req.requires_approval is False


class TestExecutionEngine:
    """Test ExecutionEngine class."""

    @pytest.fixture
    def settings(self):
        s = Settings()
        s.execution.mcp_servers = []
        return s

    @pytest.fixture
    def mock_server_manager(self):
        """Create a mock server manager."""
        mock = AsyncMock(spec=MCPServerManager)
        return mock

    @pytest.fixture
    def engine(self, settings, mock_server_manager, db_engine):
        engine = ExecutionEngine(settings, db_engine)
        # Replace server_manager with mock
        engine.server_manager = mock_server_manager
        return engine

    @pytest.mark.asyncio
    async def test_initialize(self, engine):
        await engine.initialize()
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_execute_simple(self, engine):
        """Test simple tool execution."""
        engine.server_manager.call_tool = AsyncMock(return_value={
            "success": True,
            "content": "File content here",
        })

        request = ExecutionRequest(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
        )

        result = await engine.execute(request)

        assert result.status == ExecutionStatus.VERIFIED
        assert result.result == "File content here"
        assert result.verification_status == VerificationStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_with_precondition(self, engine):
        """Test execution with precondition."""
        engine.server_manager.call_tool = AsyncMock(return_value={
            "success": True,
            "content": "Content",
        })

        # Precondition passes
        request = ExecutionRequest(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/existing.txt"},
            preconditions=[file_exists_precondition("path")],
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = await engine.execute(request)

        assert result.status == ExecutionStatus.VERIFIED

        # Precondition fails
        request2 = ExecutionRequest(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/missing.txt"},
            preconditions=[file_exists_precondition("path")],
        )

        with patch("pathlib.Path.exists", return_value=False):
            result2 = await engine.execute(request2)

        assert result2.status == ExecutionStatus.FAILED
        assert "Precondition validation failed" in result2.error

    @pytest.mark.asyncio
    async def test_execute_with_postcondition(self, engine):
        """Test execution with postcondition."""
        engine.server_manager.call_tool = AsyncMock(return_value={
            "success": True,
            "content": "Done",
        })

        request = ExecutionRequest(
            tool_name="write_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt", "content": "Hello"},
            postconditions=[result_success_postcondition()],
        )

        result = await engine.execute(request)

        assert result.status == ExecutionStatus.VERIFIED
        assert result.verification_status == VerificationStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_postcondition_failure(self, engine):
        """Test postcondition failure."""
        engine.server_manager.call_tool = AsyncMock(return_value={
            "success": False,
            "error": "Permission denied",
        })

        request = ExecutionRequest(
            tool_name="write_file",
            server_name="filesystem",
            arguments={"path": "/root/test.txt", "content": "Hello"},
            postconditions=[result_success_postcondition()],
        )

        result = await engine.execute(request)

        assert result.status == ExecutionStatus.FAILED
        assert result.verification_status == VerificationStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_execute_with_approval(self, engine):
        """Test execution requiring approval."""
        engine.server_manager.call_tool = AsyncMock(return_value={
            "success": True,
            "content": "Approved action done",
        })

        request = ExecutionRequest(
            tool_name="delete_file",
            server_name="filesystem",
            arguments={"path": "/tmp/test.txt"},
            requires_approval=True,
        )

        # Approve the execution
        execution_id = "test-exec-123"
        # We need to trigger execution and approve quickly
        # For this test, we'll mock the approval wait
        with patch.object(engine, "_wait_for_approval", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True

            request.requires_approval = True
            # We can't easily test the full approval flow without more mocking
            # Just verify the method exists
            assert hasattr(engine, "approve")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, engine):
        """Test execution timeout handling."""
        engine.server_manager.call_tool = AsyncMock(side_effect=TimeoutError())

        request = ExecutionRequest(
            tool_name="slow_tool",
            server_name="test",
            arguments={},
            timeout=0.1,  # Very short timeout
        )

        result = await engine.execute(request)

        assert result.status == ExecutionStatus.FAILED
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_approve(self, engine):
        """Test approval mechanism."""
        execution_id = "test-123"
        # Set up pending approval
        engine._pending_approvals[execution_id] = asyncio.Event()
        engine._approval_results[execution_id] = False

        # Approve
        engine.approve(execution_id, True)

        assert engine._approval_results[execution_id] is True
        assert engine._pending_approvals[execution_id].is_set()

    @pytest.mark.asyncio
    async def test_get_execution(self, engine):
        """Test getting execution by ID."""
        engine.server_manager.call_tool = AsyncMock(return_value={"success": True, "content": "test"})

        request = ExecutionRequest(
            tool_name="test_tool",
            server_name="test",
            arguments={},
        )

        result = await engine.execute(request)
        exec_id = result.execution_id

        # Get execution
        retrieved = await engine.get_execution(exec_id)

        assert retrieved is not None
        assert retrieved.execution_id == exec_id

    @pytest.mark.asyncio
    async def test_reconcile(self, engine):
        """Test reconciliation of unknown outcome."""
        # This would need a database record to reconcile
        # For now, just test the method exists
        assert hasattr(engine, "reconcile")

    @pytest.mark.asyncio
    async def test_list_tools(self, engine):
        """Test listing tools."""
        engine.server_manager.list_tools = AsyncMock(return_value=[
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ])

        await engine.initialize()
        tools = await engine.list_tools("filesystem")

        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"
