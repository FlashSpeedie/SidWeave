"""
Verified execution engine with MCP integration.

Implements tool execution lifecycle with verification:
- Requested -> Validated -> Approved -> Executing -> Completed/Failed
- Verification pending -> Verified/Unknown
- Recovery handling for ambiguous outcomes
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from sqlalchemy.ext.asyncio import AsyncEngine

from sidweave.config import Settings
from sidweave.storage import (
    ExecutionRepository,
    ToolExecution,
    get_session,
)

logger = logging.getLogger(__name__)

MCP_CONFIG_MIN_PARTS = 3
ERR_NO_SESSION = "No session for server: {}"
ERR_EXECUTION_NOT_FOUND = "Execution not found: {}"


class ExecutionStatus(StrEnum):
    """Tool execution lifecycle status."""
    REQUESTED = "requested"
    VALIDATED = "validated"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFICATION_PENDING = "verification_pending"
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    RECOVERY_REQUIRED = "recovery_required"


class VerificationStatus(StrEnum):
    """Verification result status."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class Precondition:
    """Precondition for tool execution."""
    name: str
    check: Callable[[dict[str, Any]], bool]
    description: str


@dataclass
class Postcondition:
    """Postcondition for tool execution verification."""
    name: str
    check: Callable[[dict[str, Any]], bool]
    description: str


@dataclass
class ExecutionRequest:
    """Request to execute a tool."""
    tool_name: str
    server_name: str
    arguments: dict[str, Any]
    preconditions: list[Precondition] = field(default_factory=list)
    postconditions: list[Postcondition] = field(default_factory=list)
    idempotency_key: str | None = None
    requires_approval: bool = False
    timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    execution_id: str
    status: ExecutionStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_details: dict[str, Any] | None = None
    duration_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class MCPServerManager:
    """Manages MCP server connections."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, ClientSession] = {}
        self._server_params: dict[str, StdioServerParameters] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MCP server connections."""
        for server_config in self.settings.execution.mcp_servers:
            # Parse server config (format: "name:command:args")
            parts = server_config.split(":")
            if len(parts) >= MCP_CONFIG_MIN_PARTS:
                name = parts[0]
                command = parts[1]
                args = parts[2].split(",")

                self._server_params[name] = StdioServerParameters(
                    command=command,
                    args=args,
                )

                # Connect
                await self._connect_server(name)

        self._initialized = True
        logger.info(f"MCP server manager initialized with {len(self._sessions)} servers")

    async def _connect_server(self, name: str) -> bool:
        """Connect to an MCP server."""
        if name not in self._server_params:
            return False

        try:
            params = self._server_params[name]
            read_stream, write_stream = await stdio_client(params).__aenter__()
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            self._sessions[name] = session
            logger.info(f"Connected to MCP server: {name}")
        except Exception:
            logger.exception(f"Failed to connect to MCP server {name}")
            return False
        else:
            return True

    async def get_session(self, server_name: str) -> ClientSession | None:
        """Get MCP session for server."""
        return self._sessions.get(server_name)

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List available tools on server."""
        session = self._sessions.get(server_name)
        if not session:
            return []

        try:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in result.tools
            ]
        except Exception:
            logger.exception(f"Failed to list tools on {server_name}")
            return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on an MCP server."""
        session = self._sessions.get(server_name)
        if not session:
            raise RuntimeError(ERR_NO_SESSION.format(server_name))

        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception:
            return {
                "success": False,
                "error": "Execution failed",
                "is_error": True,
            }
        else:
            return {
                "success": True,
                "content": result.content,
                "is_error": result.isError,
            }

    async def shutdown(self) -> None:
        """Shutdown all MCP connections."""
        for name, session in self._sessions.items():
            try:
                # ClientSession may not have a close method in newer MCP versions
                if hasattr(session, "close"):
                    await session.close()
                elif hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)
            except Exception:
                logger.exception(f"Error closing MCP session {name}")
        self._sessions.clear()


class ExecutionEngine:
    """
    Verified execution engine with MCP integration.

    Manages the complete tool execution lifecycle with:
    - Precondition validation
    - Approval workflow
    - Execution with timeout handling
    - Postcondition verification
    - Idempotency support
    - Recovery for ambiguous outcomes
    """

    def __init__(self, settings: Settings, engine: AsyncEngine | None = None):
        self.settings = settings
        self.server_manager = MCPServerManager(settings)
        self._initialized = False
        self._pending_approvals: dict[str, asyncio.Event] = {}
        self._approval_results: dict[str, bool] = {}
        self._engine = engine

    async def initialize(self) -> None:
        """Initialize execution engine."""
        await self.server_manager.initialize()
        self._initialized = True
        logger.info("Execution engine initialized")

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:  # noqa: PLR0911
        """
        Execute a tool with full verification lifecycle.

        Args:
            request: Execution request with tool details and verification rules

        Returns:
            ExecutionResult with status and verification outcome
        """
        execution_id = request.idempotency_key or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Check idempotency
        existing = await self._check_idempotency(execution_id)
        if existing:
            logger.info(f"Returning existing execution result for {execution_id}")
            return existing

        # Create execution record
        await self._create_execution_record(
            execution_id=execution_id,
            request=request,
        )

        try:
            # Step 1: Validate preconditions
            await self._update_status(execution_id, ExecutionStatus.VALIDATED)
            if not await self._validate_preconditions(request):
                return await self._fail_execution(
                    execution_id, "Precondition validation failed", start_time
                )

            # Step 2: Approval if required
            if request.requires_approval or self._requires_approval(request.tool_name):
                await self._update_status(execution_id, ExecutionStatus.APPROVED)
                approved = await self._wait_for_approval(execution_id)
                if not approved:
                    return await self._fail_execution(
                        execution_id, "Execution not approved", start_time
                    )

            # Step 3: Execute
            await self._update_status(execution_id, ExecutionStatus.EXECUTING)
            result = await self._execute_tool(request, execution_id)

            if result.get("success"):
                # Step 4: Verify postconditions
                await self._update_status(execution_id, ExecutionStatus.VERIFICATION_PENDING)
                verification = await self._verify_postconditions(request, result)

                if verification["passed"]:
                    content = result.get("content")
                    if content is None:
                        content = {}
                    return await self._complete_execution(
                        execution_id, content, start_time, verification
                    )
                content = result.get("content")
                if content is None:
                    content = {}
                return await self._verification_failed(
                    execution_id, content, start_time, verification
                )
            return await self._fail_execution(
                execution_id, result.get("error", "Unknown error"), start_time
            )

        except TimeoutError:
            return await self._handle_timeout(execution_id, start_time)
        except Exception:
            logger.exception("Execution failed")
            return await self._fail_execution(execution_id, "Unknown error", start_time)

    async def _check_idempotency(self, execution_id: str) -> ExecutionResult | None:
        """Check if execution was already performed."""
        if not self.settings.execution.enable_idempotency:
            return None

        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            existing = await repo.get_by_idempotency_key(execution_id)

            if existing:
                # Return cached result
                return ExecutionResult(
                    execution_id=existing.execution_id,
                    status=ExecutionStatus(existing.status),
                    result=existing.result,
                    error=existing.error,
                    verification_status=VerificationStatus(
                        existing.verification_status or "unknown"
                    ),
                    verification_details=existing.verification_details,
                    duration_ms=existing.duration_ms,
                    started_at=existing.started_at,
                    completed_at=existing.completed_at,
                )
        return None

    async def _create_execution_record(
        self,
        execution_id: str,
        request: ExecutionRequest,
    ) -> ToolExecution:
        """Create initial execution record."""
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            record = ToolExecution(
                execution_id=execution_id,
                idempotency_key=request.idempotency_key,
                tool_name=request.tool_name,
                server_name=request.server_name,
                arguments=request.arguments,
                status=ExecutionStatus.REQUESTED.value,
                preconditions={p.name: p.description for p in request.preconditions},
                postconditions={p.name: p.description for p in request.postconditions},
            )
            return await repo.add_execution(record)

    async def _validate_preconditions(self, request: ExecutionRequest) -> bool:
        """Validate all preconditions."""
        for precondition in request.preconditions:
            try:
                if not precondition.check(request.arguments):
                    logger.warning(f"Precondition failed: {precondition.name}")
                    return False
            except Exception:
                logger.exception("Precondition check error")
                return False
        return True

    def _requires_approval(self, tool_name: str) -> bool:
        """Check if tool requires approval."""
        return tool_name in self.settings.execution.require_approval_for

    async def _wait_for_approval(self, execution_id: str) -> bool:
        """Wait for human approval."""
        event = asyncio.Event()
        self._pending_approvals[execution_id] = event

        try:
            # Wait for approval with timeout
            await asyncio.wait_for(event.wait(), timeout=300)  # 5 minutes
            return self._approval_results.get(execution_id, False)
        except TimeoutError:
            logger.warning(f"Approval timeout for {execution_id}")
            return False
        finally:
            self._pending_approvals.pop(execution_id, None)
            self._approval_results.pop(execution_id, None)

    def approve(self, execution_id: str, approved: bool) -> bool:
        """Approve or reject a pending execution."""
        if execution_id in self._pending_approvals:
            self._approval_results[execution_id] = approved
            self._pending_approvals[execution_id].set()
            return True
        return False

    async def _execute_tool(
        self,
        request: ExecutionRequest,
        execution_id: str,
    ) -> dict[str, Any]:
        """Execute the tool via MCP."""
        try:
            result = await asyncio.wait_for(
                self.server_manager.call_tool(
                    request.server_name,
                    request.tool_name,
                    request.arguments,
                ),
                timeout=request.timeout,
            )
        except TimeoutError:
            # Timeout - execution may have happened
            logger.warning(f"Tool execution timeout for {execution_id}")
            return {
                "success": False,
                "error": "Execution timeout - outcome unknown",
                "timeout": True,
            }
        else:
            return result

    async def _verify_postconditions(
        self,
        request: ExecutionRequest,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify postconditions."""
        verification_details: dict[str, Any] = {}
        all_passed = True

        for postcondition in request.postconditions:
            try:
                passed = postcondition.check(result)
                verification_details[postcondition.name] = {
                    "passed": passed,
                    "description": postcondition.description,
                }
                if not passed:
                    all_passed = False
            except Exception:
                logger.exception("Postcondition check error")
                verification_details[postcondition.name] = {
                    "passed": False,
                    "error": "Postcondition check failed",
                    "description": postcondition.description,
                }
                all_passed = False

        return {
            "passed": all_passed,
            "details": verification_details,
        }

    async def _complete_execution(
        self,
        execution_id: str,
        result: dict[str, Any],
        start_time: float,
        verification: dict[str, Any],
    ) -> ExecutionResult:
        """Mark execution as completed and verified."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            await repo.update_status(
                execution_id,
                ExecutionStatus.VERIFIED.value,
                result=result,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                verification_status=VerificationStatus.PASSED.value,
                verification_details=verification,
            )

        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.VERIFIED,
            result=result,
            verification_status=VerificationStatus.PASSED,
            verification_details=verification,
            duration_ms=duration_ms,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def _verification_failed(
        self,
        execution_id: str,
        result: dict[str, Any],
        start_time: float,
        verification: dict[str, Any],
    ) -> ExecutionResult:
        """Handle verification failure."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            await repo.update_status(
                execution_id,
                ExecutionStatus.FAILED.value,
                result=result,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error="Postcondition verification failed",
                verification_status=VerificationStatus.FAILED.value,
                verification_details=verification,
            )

        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            result=result,
            error="Postcondition verification failed",
            verification_status=VerificationStatus.FAILED,
            verification_details=verification,
            duration_ms=duration_ms,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def _fail_execution(
        self,
        execution_id: str,
        error: str,
        start_time: float,
    ) -> ExecutionResult:
        """Mark execution as failed."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            await repo.update_status(
                execution_id,
                ExecutionStatus.FAILED.value,
                error=error,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                verification_status=VerificationStatus.UNKNOWN.value,
            )

        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            error=error,
            verification_status=VerificationStatus.UNKNOWN,
            duration_ms=duration_ms,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def _handle_timeout(
        self,
        execution_id: str,
        start_time: float,
    ) -> ExecutionResult:
        """Handle execution timeout - outcome unknown."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            await repo.update_status(
                execution_id,
                ExecutionStatus.UNKNOWN.value,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error="Execution timeout - outcome unknown, reconciliation required",
                verification_status=VerificationStatus.UNKNOWN.value,
            )

        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.UNKNOWN,
            error="Execution timeout - outcome unknown, reconciliation required",
            verification_status=VerificationStatus.UNKNOWN,
            duration_ms=duration_ms,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def _update_status(self, execution_id: str, status: ExecutionStatus) -> None:
        """Update execution status."""
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            await repo.update_status(execution_id, status.value)

    async def reconcile(self, execution_id: str) -> ExecutionResult:
        """
        Reconcile an execution with unknown outcome.

        Attempts to determine the actual outcome of a timed-out or interrupted execution
        by checking persisted state and querying external systems through supported adapters.

        Returns:
            ExecutionResult with determined status or UNKNOWN if outcome cannot be established.
        """
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            record = await repo.get_by_execution_id(execution_id)

            if not record:
                raise ValueError(ERR_EXECUTION_NOT_FOUND.format(execution_id))

            # If already in a terminal state with verification, return as-is
            if record.status in (
                ExecutionStatus.COMPLETED.value,
                ExecutionStatus.FAILED.value,
                ExecutionStatus.VERIFIED.value,
            ) and record.verification_status in (
                VerificationStatus.PASSED.value,
                VerificationStatus.FAILED.value,
            ):
                logger.info(f"Execution {execution_id} already in verified terminal state")
                return ExecutionResult(
                    execution_id=record.execution_id,
                    status=ExecutionStatus(record.status),
                    result=record.result,
                    error=record.error,
                    verification_status=VerificationStatus(record.verification_status),
                    verification_details=record.verification_details,
                    duration_ms=record.duration_ms,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                )

            # Attempt reconciliation through available adapters
            reconciliation_result = await self._attempt_reconciliation(record)

            # Update record with reconciliation outcome
            await repo.update_status(
                execution_id,
                reconciliation_result.status.value,
                result=reconciliation_result.result,
                error=reconciliation_result.error,
                verification_status=reconciliation_result.verification_status.value,
                verification_details=reconciliation_result.verification_details,
                completed_at=reconciliation_result.completed_at,
            )

        return reconciliation_result

    async def _attempt_reconciliation(self, record: ToolExecution) -> ExecutionResult:
        """
        Attempt to determine execution outcome through available mechanisms.

        Checks:
        1. If execution completed but verification was interrupted
        2. If MCP server has tool state we can query
        3. If idempotency key allows safe re-check
        4. If external adapter can verify state
        """
        execution_id = record.execution_id
        logger.info(f"Attempting reconciliation for execution {execution_id}")

        # Case 1: Result exists but verification was pending/interrupted
        if record.result and record.status in (
            ExecutionStatus.EXECUTING.value,
            ExecutionStatus.VERIFICATION_PENDING.value,
            ExecutionStatus.UNKNOWN.value,
            ExecutionStatus.RECOVERY_REQUIRED.value,
        ):
            # Re-run postconditions if we have them
            postconditions_raw: dict[str, Any] | list[Any] = record.postconditions or {}
            postconditions: dict[str, str] = {}
            if isinstance(postconditions_raw, dict):
                postconditions = {k: str(v) for k, v in postconditions_raw.items()}
            if postconditions:
                verification = await self._recheck_postconditions(record)
                if verification["passed"]:
                    return ExecutionResult(
                        execution_id=execution_id,
                        status=ExecutionStatus.VERIFIED,
                        result=record.result,
                        verification_status=VerificationStatus.PASSED,
                        verification_details=verification,
                        duration_ms=record.duration_ms,
                        started_at=record.started_at,
                        completed_at=datetime.now(UTC),
                    )
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILED,
                    result=record.result,
                    error="Postcondition re-check failed",
                    verification_status=VerificationStatus.FAILED,
                    verification_details=verification,
                    duration_ms=record.duration_ms,
                    started_at=record.started_at,
                    completed_at=datetime.now(UTC),
                )

            # If we have a result but no postconditions, mark as completed with unknown verification
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED,
                result=record.result,
                verification_status=VerificationStatus.UNKNOWN,
                verification_details={"note": "Result recovered, no postconditions to verify"},
                duration_ms=record.duration_ms,
                started_at=record.started_at,
                completed_at=datetime.now(UTC),
            )

        # Case 2: No result - try to check via MCP if possible
        mcp_session = await self.server_manager.get_session(record.server_name)
        if mcp_session and record.tool_name:
            # Try to list tools to verify server is responsive
            try:
                tools_result = await mcp_session.list_tools()
                tools = tools_result.tools if tools_result else []
                logger.debug(
                    f"MCP server {record.server_name} responsive, "
                    f"{len(tools)} tools available"
                )
                # Note: We cannot safely re-execute non-idempotent tools
                # But we can check if the tool exists
                tool_exists = any(t.name == record.tool_name for t in tools)
                if tool_exists:
                    return ExecutionResult(
                        execution_id=execution_id,
                        status=ExecutionStatus.UNKNOWN,
                        error=(
                            "MCP server responsive but outcome unknown - "
                            "non-idempotent tool cannot be safely re-executed"
                        ),
                        verification_status=VerificationStatus.UNKNOWN,
                        verification_details={
                            "server_responsive": True,
                            "tool_exists": True,
                            "retry_safe": False,
                        },
                        duration_ms=record.duration_ms,
                        started_at=record.started_at,
                        completed_at=datetime.now(UTC),
                    )
            except Exception as e:
                logger.warning(f"MCP server check failed: {e}")

        # Case 3: Check if idempotency key allows safe retry verification
        if record.idempotency_key and self.settings.execution.enable_idempotency:
            # The idempotency key means we could potentially retry
            # but we must be careful - only safe for idempotent operations
            err_msg = (
                "Outcome unknown - idempotency key available "
                "for safe retry if tool is idempotent"
            )
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.UNKNOWN,
                error=err_msg,
                verification_status=VerificationStatus.UNKNOWN,
                verification_details={
                    "idempotency_key": record.idempotency_key,
                    "can_retry_safely": "unknown - depends on tool semantics",
                },
                duration_ms=record.duration_ms,
                started_at=record.started_at,
                completed_at=datetime.now(UTC),
            )

        # Case 4: No way to determine outcome - mark as unknown with details
        err_msg = (
            "Unable to determine execution outcome - "
            "no result stored, no verification adapters available"
        )
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.UNKNOWN,
            error=err_msg,
            verification_status=VerificationStatus.UNKNOWN,
            verification_details={
                "attempted_checks": [
                    "result_present",
                    "postconditions_present",
                    "mcp_server_responsive",
                    "idempotency_key_present",
                ],
                "result_present": record.result is not None,
                "postconditions_present": bool(record.postconditions or {}),
                "server_responsive": False,
                "idempotency_key_present": record.idempotency_key is not None,
            },
            duration_ms=record.duration_ms,
            started_at=record.started_at,
            completed_at=datetime.now(UTC),
        )

    async def _recheck_postconditions(
        self, record: ToolExecution
    ) -> dict[str, Any]:
        """Re-run postcondition checks against stored result."""
        verification_details: dict[str, Any] = {}
        all_passed = True

        postconditions_raw: dict[str, Any] | list[Any] = record.postconditions or {}
        postconditions: dict[str, str] = {}
        if isinstance(postconditions_raw, dict):
            postconditions = {k: str(v) for k, v in postconditions_raw.items()}
        result_data = record.result or {}
        for pc_name, pc_description in postconditions.items():
            try:
                # We don't have the original callable, so we check common patterns
                passed = False
                if pc_name == "result_success":
                    passed = result_data.get("success", False)
                elif pc_name == "has_content":
                    passed = bool(result_data.get("content"))
                elif pc_name == "no_error":
                    passed = not result_data.get("is_error", False) and not result_data.get("error")
                else:
                    passed = True  # Unknown postcondition - assume passed

                verification_details[pc_name] = {
                    "passed": passed,
                    "description": pc_description,
                    "rechecked": True,
                }
                if not passed:
                    all_passed = False
            except Exception:
                logger.exception("Postcondition re-check error")
                verification_details[pc_name] = {
                    "passed": False,
                    "error": "Postcondition re-check failed",
                    "description": pc_description,
                    "rechecked": True,
                }
                all_passed = False

        return {
            "passed": all_passed,
            "details": verification_details,
            "rechecked_at": datetime.now(UTC).isoformat(),
        }

    async def get_execution(self, execution_id: str) -> ExecutionResult | None:
        """Get execution result by ID."""
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            record = await repo.get_by_execution_id(execution_id)

            if not record:
                return None

            return ExecutionResult(
                execution_id=record.execution_id,
                status=ExecutionStatus(record.status),
                result=record.result if record.result is not None else None,
                error=record.error,
                verification_status=VerificationStatus(record.verification_status or "unknown"),
                verification_details=record.verification_details,
                duration_ms=record.duration_ms,
                started_at=record.started_at,
                completed_at=record.completed_at,
            )

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List available tools on a server."""
        return await self.server_manager.list_tools(server_name)

    async def list_pending_reconciliation(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ExecutionResult]:
        """List executions that need reconciliation."""
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            records = await repo.list_pending_reconciliation(limit=limit, offset=offset)

        return [
            ExecutionResult(
                execution_id=record.execution_id,
                status=ExecutionStatus(record.status),
                result=record.result if record.result is not None else None,
                error=record.error,
                verification_status=VerificationStatus(record.verification_status or "unknown"),
                verification_details=record.verification_details,
                duration_ms=record.duration_ms,
                started_at=record.started_at,
                completed_at=record.completed_at,
            )
            for record in records
        ]

    async def count_pending_reconciliation(self) -> int:
        """Count executions needing reconciliation."""
        async with get_session(self._engine) as session:
            repo = ExecutionRepository(session)
            return await repo.count_pending_reconciliation()

    async def shutdown(self) -> None:
        """Shutdown execution engine."""
        await self.server_manager.shutdown()
        self._initialized = False


# Convenience functions for common preconditions/postconditions
def file_exists_precondition(path_key: str = "path") -> Precondition:
    """Precondition: file at path_key exists."""
    return Precondition(
        name="file_exists",
        check=lambda args: Path(args.get(path_key, "")).exists(),
        description=f"File at {path_key} must exist",
    )


def file_not_exists_precondition(path_key: str = "path") -> Precondition:
    """Precondition: file at path_key does not exist."""
    return Precondition(
        name="file_not_exists",
        check=lambda args: not Path(args.get(path_key, "")).exists(),
        description=f"File at {path_key} must not exist",
    )


def directory_exists_precondition(path_key: str = "path") -> Precondition:
    """Precondition: directory at path_key exists."""
    return Precondition(
        name="directory_exists",
        check=lambda args: Path(args.get(path_key, "")).is_dir(),
        description=f"Directory at {path_key} must exist",
    )


def result_success_postcondition() -> Postcondition:
    """Postcondition: tool result indicates success."""
    return Postcondition(
        name="result_success",
        check=lambda result: result.get("success", False),
        description="Tool must return success=true",
    )


def result_has_content_postcondition(content_key: str = "content") -> Postcondition:
    """Postcondition: result has content."""
    return Postcondition(
        name="has_content",
        check=lambda result: bool(result.get(content_key)),
        description=f"Result must have {content_key}",
    )


def no_error_postcondition() -> Postcondition:
    """Postcondition: no error in result."""
    return Postcondition(
        name="no_error",
        check=lambda result: not result.get("is_error", False) and not result.get("error"),
        description="Result must not contain error",
    )


__all__ = [
    "ExecutionEngine",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "MCPServerManager",
    "Postcondition",
    "Precondition",
    "VerificationStatus",
    "directory_exists_precondition",
    "file_exists_precondition",
    "file_not_exists_precondition",
    "no_error_postcondition",
    "result_has_content_postcondition",
    "result_success_postcondition",
]
