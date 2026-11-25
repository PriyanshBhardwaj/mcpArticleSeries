"""
Minimal synchronous modern-MCP client for the raw stdio example.

This client supports the 2026-07-28 protocol only. It negotiates a mutually
supported modern version, but does not fall back to the legacy initialise
handshake. It is a teaching client, not a production transport implementation.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from typing import Any

PREFERRED_VERSIONS = ["2026-07-28"]
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT = "io.modelcontextprotocol/clientInfo"


class RpcFailure(Exception):
    """A JSON-RPC error response received from the server."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message", "JSON-RPC request failed")))
        self.error = error


class Client:
    def __init__(
        self,
        command: list[str],
        capabilities: dict[str, Any] | None = None,
        client_info: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError("command must be a non-empty list of strings")
        if capabilities is not None and not isinstance(capabilities, dict):
            raise TypeError("capabilities must be a dictionary")

        if client_info is None:
            client_info = {"name": "catalogue-example-client", "version": "0.1.0"}
        if (
            not isinstance(client_info, dict)
            or not isinstance(client_info.get("name"), str)
            or not client_info["name"].strip()
            or not isinstance(client_info.get("version"), str)
            or not client_info["version"].strip()
        ):
            raise ValueError("client_info must contain non-empty string name and version")

        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            self.proc.kill()
            raise RuntimeError("failed to create server stdin/stdout pipes")

        self.ids = itertools.count(1)
        self.version = PREFERRED_VERSIONS[0]
        self.capabilities = dict(capabilities or {})
        self.client_info = dict(client_info)
        self._closed = False

    def _meta(self) -> dict[str, Any]:
        return {
            META_VERSION: self.version,
            META_CAPS: self.capabilities,
            META_CLIENT: self.client_info,
        }

    def _send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed or self.proc.poll() is not None:
            raise RuntimeError("server process is not running")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary")

        request_id = next(self.ids)
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params if params is not None else {},
        }
        assert self.proc.stdin is not None and self.proc.stdout is not None

        try:
            self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("failed while communicating with the server") from exc

        if not line:
            raise RuntimeError(
                f"server closed stdout without replying (exit code: {self.proc.poll()})"
            )
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"server returned invalid JSON: {line!r}") from exc

        if not isinstance(reply, dict):
            raise RuntimeError("server response must be a JSON object")
        response_id = reply.get("id")
        if (
            reply.get("jsonrpc") != "2.0"
            or type(response_id) is not type(request_id)
            or response_id != request_id
        ):
            raise RuntimeError("response does not match the outstanding request")

        if "error" in reply:
            error = reply["error"]
            if not isinstance(error, dict) or not isinstance(error.get("code"), int):
                raise RuntimeError("server returned a malformed JSON-RPC error")
            raise RpcFailure(error)
        if "result" not in reply or not isinstance(reply["result"], dict):
            raise RuntimeError("server response must contain an object result")
        return reply["result"]

    @staticmethod
    def _supported_versions(error: RpcFailure) -> list[str]:
        data = error.error.get("data")
        supported = data.get("supported") if isinstance(data, dict) else None
        if not isinstance(supported, list) or not all(
            isinstance(version, str) and version for version in supported
        ):
            raise RuntimeError("UnsupportedProtocolVersionError has invalid data.supported")
        return supported

    def _choose_version(self, supported: list[str]) -> str:
        for preferred in PREFERRED_VERSIONS:
            if preferred in supported:
                return preferred
        raise RuntimeError(
            f"no mutually supported protocol version; server supports {supported!r}"
        )

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and retry once only after an explicit version error."""
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        request_params = dict(params or {})
        request_params["_meta"] = self._meta()

        try:
            return self._send(method, request_params)
        except RpcFailure as failure:
            if failure.error.get("code") != -32022:
                raise
            self.version = self._choose_version(self._supported_versions(failure))
            request_params["_meta"] = self._meta()
            # _send allocates a new JSON-RPC ID for this retry.
            return self._send(method, request_params)

    def discover(self) -> dict[str, Any]:
        """Discover server support, retrying discovery after a version error."""
        params = {"_meta": self._meta()}
        try:
            result = self._send("server/discover", params)
        except RpcFailure as failure:
            if failure.error.get("code") != -32022:
                raise
            self.version = self._choose_version(self._supported_versions(failure))
            result = self._send("server/discover", {"_meta": self._meta()})

        supported = result.get("supportedVersions")
        if not isinstance(supported, list) or not all(
            isinstance(version, str) and version for version in supported
        ):
            raise RuntimeError("server/discover returned an invalid supportedVersions list")
        self.version = self._choose_version(supported)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

        if self.proc.stdout is not None and not self.proc.stdout.closed:
            self.proc.stdout.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


if __name__ == "__main__":
    command = [sys.executable, "service_catalog_raw.py"]

    with Client(command) as client:
        discovery = client.discover()
        print(json.dumps(discovery, indent=2))

        result = client.request(
            "tools/call",
            {"name": "get_service_owner", "arguments": {"service": "checkout"}},
        )

        print(json.dumps(result, indent=2))
