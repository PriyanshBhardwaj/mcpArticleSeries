"""
Minimal raw JSON-RPC server for the MCP 2026-07-28 article example.

Transport: newline-delimited JSON over stdio.
Scope: one request at a time; no HTTP, subscriptions, cancellation, or
multi-round-trip elicitation flow. This is a teaching implementation, not a
production MCP server.
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

SUPPORTED_VERSIONS = ["2026-07-28"]

META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT = "io.modelcontextprotocol/clientInfo"
META_SERVER = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_VERSION = -32022


class RpcError(Exception):
    """An expected protocol error that should be returned to the caller."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


INSTRUCTIONS = (
    "Use catalogue tools only for service metadata. "
    "Do not infer ownership or dependencies when a lookup returns no result."
)


@dataclass(frozen=True)
class Service:
    name: str
    owner: str
    dependencies: tuple[str, ...]
    runbook_url: str


CATALOG: dict[str, Service] = {
    "checkout": Service(
        "checkout", "commerce-platform", ("payments", "inventory"),
        "https://runbooks.example.invalid/checkout",
    ),
    "payments": Service(
        "payments", "payments-platform", ("ledger",),
        "https://runbooks.example.invalid/payments",
    ),
    "inventory": Service(
        "inventory", "supply-platform", (),
        "https://runbooks.example.invalid/inventory",
    ),
    "ledger": Service(
        "ledger", "finance-platform", (),
        "https://runbooks.example.invalid/ledger",
    ),
}


def _request_meta(params: dict[str, Any]) -> dict[str, Any]:
    """Validate the required request metadata without selecting a version."""
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        raise RpcError(INVALID_PARAMS, "params._meta must be an object")

    if META_VERSION not in meta:
        raise RpcError(INVALID_PARAMS, f"missing _meta['{META_VERSION}']")
    if META_CAPS not in meta:
        raise RpcError(INVALID_PARAMS, f"missing _meta['{META_CAPS}']")

    version = meta[META_VERSION]
    capabilities = meta[META_CAPS]
    if not isinstance(version, str) or not version:
        raise RpcError(INVALID_PARAMS, "protocol version must be a non-empty string")
    if not isinstance(capabilities, dict):
        raise RpcError(INVALID_PARAMS, "client capabilities must be an object")

    if META_CLIENT in meta:
        client_info = meta[META_CLIENT]
        if not isinstance(client_info, dict):
            raise RpcError(INVALID_PARAMS, "clientInfo must be an object")
        if not isinstance(client_info.get("name"), str) or not client_info["name"].strip():
            raise RpcError(INVALID_PARAMS, "clientInfo.name must be a non-empty string")
        if not isinstance(client_info.get("version"), str) or not client_info["version"].strip():
            raise RpcError(INVALID_PARAMS, "clientInfo.version must be a non-empty string")

    return meta


def check_envelope(params: dict[str, Any]) -> dict[str, Any]:
    """Validate required metadata and the requested protocol version."""
    meta = _request_meta(params)
    requested = meta[META_VERSION]
    if requested not in SUPPORTED_VERSIONS:
        raise RpcError(
            UNSUPPORTED_VERSION,
            "Unsupported protocol version",
            {"supported": SUPPORTED_VERSIONS, "requested": requested},
        )
    return meta[META_CAPS]


def server_meta() -> dict[str, Any]:
    return {META_SERVER: {"name": "service-catalog", "version": "0.1.0"}}


def tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Return structured data and its text serialization for compatibility."""
    return {
        "resultType": "complete",
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "structuredContent": payload,
        "isError": False,
        "_meta": server_meta(),
    }


def tool_error(message: str) -> dict[str, Any]:
    """Return an execution-level error that belongs in a tool result."""
    return {
        "resultType": "complete",
        "content": [{"type": "text", "text": message}],
        "isError": True,
        "_meta": server_meta(),
    }


def discover(params: dict[str, Any]) -> dict[str, Any]:
    # Discovery is still a request: it carries metadata and its version is
    # validated just like tools/list and tools/call.
    check_envelope(params)
    return {
        "resultType": "complete",
        "supportedVersions": SUPPORTED_VERSIONS,
        "capabilities": {"tools": {}},
        "instructions": INSTRUCTIONS,
        "ttlMs": 60_000,
        "cacheScope": "public",
        "_meta": server_meta(),
    }


def list_tools(params: dict[str, Any]) -> dict[str, Any]:
    check_envelope(params)
    service_schema = {
        "type": "object",
        "properties": {"service": {"type": "string"}},
        "required": ["service"],
        "additionalProperties": False,
    }
    return {
        "resultType": "complete",
        "tools": [
            {
                "name": "get_service_owner",
                "description": "Owning team and runbook for one service.",
                "inputSchema": service_schema,
            },
            {
                "name": "list_service_dependencies",
                "description": "Direct dependencies only.",
                "inputSchema": service_schema,
            },
            {
                "name": "archive_service",
                "description": "Archive a service. Needs a human to confirm.",
                "inputSchema": service_schema,
            },
        ],
        "ttlMs": 60_000,
        "cacheScope": "public",
        "_meta": server_meta(),
    }


def call_tool(params: dict[str, Any]) -> dict[str, Any]:
    capabilities = check_envelope(params)
    name = params.get("name")
    arguments = params.get("arguments", {})

    if not isinstance(name, str) or not name:
        raise RpcError(INVALID_PARAMS, "tools/call requires a non-empty tool name")

    known_tools = {
        "get_service_owner",
        "list_service_dependencies",
        "archive_service",
    }
    if name not in known_tools:
        # An unknown tool is a protocol-level invalid-params error, not a
        # tool execution result with isError=true.
        raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}")

    if not isinstance(arguments, dict):
        raise RpcError(INVALID_PARAMS, "tool arguments must be an object")

    if set(arguments) != {"service"}:
        return tool_error(f"tool arguments must contain exactly the required 'service' property")

    service_name = arguments["service"]
    if not isinstance(service_name, str) or not service_name.strip():
       return tool_error(f"tool argument 'service' must be a non-empty string")

    entry = CATALOG.get(service_name.strip().lower())
    if entry is None:
        return tool_error(f"unknown service: {service_name!r}")

    if name == "get_service_owner":
        return tool_result(
            {"service": entry.name, "owner": entry.owner, "runbook_url": entry.runbook_url}
        )

    if name == "list_service_dependencies":
        return tool_result(
            {"service": entry.name, "dependencies": list(entry.dependencies)}
        )

    # This example requires form elicitation for the human-confirmation step.
    # The actual multi-round-trip confirmation flow is intentionally omitted.
    elicitation = capabilities.get("elicitation")
    if not isinstance(elicitation, dict) or not isinstance(elicitation.get("form"), dict):
        raise RpcError(
            MISSING_CLIENT_CAPABILITY,
            "archive_service requires form elicitation",
            {"requiredCapabilities": {"elicitation": {"form": {}}}},
        )

    # Do not perform a destructive operation: confirmation is not implemented.
    return tool_error("confirmation flow not implemented in this example")


METHODS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "server/discover": discover,
    "tools/list": list_tools,
    "tools/call": call_tool,
}


def _error_response(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _valid_id(value: Any) -> bool:
    # MCP RequestId is string | integer; bool is an int subclass in Python.
    return (isinstance(value, (str, int)) and not isinstance(value, bool))


def handle(message: Any) -> dict[str, Any] | None:
    """Dispatch one JSON value; notifications never receive a response."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error_response(None, INVALID_REQUEST, "invalid request")

    method = message.get("method")
    if not isinstance(method, str) or not method:
        return _error_response(message.get("id"), INVALID_REQUEST, "invalid request")

    if "id" not in message:
        # This is a notification. Even an unknown notification gets no reply.
        return None

    request_id = message["id"]
    if not _valid_id(request_id):
        return _error_response(None, INVALID_REQUEST, "invalid request id")

    params = message.get("params", {})
    if not isinstance(params, dict):
        return _error_response(request_id, INVALID_PARAMS, "params must be an object")

    handler = METHODS.get(method)
    if handler is None:
        return _error_response(request_id, METHOD_NOT_FOUND, f"method not found: {method}")

    try:
        result = handler(params)
    except RpcError as exc:
        return _error_response(request_id, exc.code, exc.message, exc.data)
    except Exception:
        # Keep protocol stdout clean while returning the standard internal error.
        traceback.print_exc(file=sys.stderr)
        return _error_response(request_id, INTERNAL_ERROR, "internal error")

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    """Read one newline-delimited JSON-RPC message at a time."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = _error_response(None, PARSE_ERROR, "parse error")
        else:
            reply = handle(message)

        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
