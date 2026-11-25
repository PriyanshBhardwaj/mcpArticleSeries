from __future__ import annotations

import json
import subprocess
import sys
import unittest

from client import Client, RpcFailure
from service_catalog_raw import handle

CMD = [sys.executable, "service_catalog_raw.py"]
VERSION = "2026-07-28"
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT = "io.modelcontextprotocol/clientInfo"


def raw(
    method: str,
    params: dict | None = None,
    *,
    caps: dict | None = None,
    version: str = VERSION,
) -> dict:
    """Send one request through a fresh stdio server process."""
    with Client(CMD, capabilities=caps) as client:
        client.version = version
        request_params = dict(params or {})
        request_params["_meta"] = {
            META_VERSION: version,
            META_CAPS: caps if caps is not None else {},
            META_CLIENT: {"name": "test-client", "version": "1.0"},
        }
        return client._send(method, request_params)


class ServiceCatalogTests(unittest.TestCase):
    def error_of(self, call) -> dict:
        try:
            call()
        except RpcFailure as failure:
            return failure.error
        self.fail("expected RpcFailure")

    def test_discovery_returns_version_capabilities_and_server_identity(self) -> None:
        result = raw("server/discover")
        self.assertEqual(result["resultType"], "complete")
        self.assertEqual(result["supportedVersions"], [VERSION])
        self.assertEqual(result["capabilities"], {"tools": {}})
        self.assertEqual(
            result["_meta"]["io.modelcontextprotocol/serverInfo"],
            {"name": "service-catalog", "version": "0.1.0"},
        )

    def test_discovery_rejects_unsupported_request_version(self) -> None:
        error = self.error_of(lambda: raw("server/discover", version="2025-11-25"))
        self.assertEqual(error["code"], -32022)
        self.assertEqual(
            error["data"],
            {"supported": [VERSION], "requested": "2025-11-25"},
        )

    def test_client_retries_discovery_after_version_error(self) -> None:
        with Client(CMD) as client:
            client.version = "2025-11-25"
            result = client.discover()
            self.assertEqual(result["supportedVersions"], [VERSION])
            self.assertEqual(client.version, VERSION)

    def test_tool_lookup_returns_structured_and_text_content(self) -> None:
        result = raw(
            "tools/call",
            {"name": "get_service_owner", "arguments": {"service": "checkout"}},
        )
        expected = {
            "service": "checkout",
            "owner": "commerce-platform",
            "runbook_url": "https://runbooks.example.invalid/checkout",
        }
        self.assertEqual(result["resultType"], "complete")
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"], expected)
        self.assertEqual(json.loads(result["content"][0]["text"]), expected)

    def test_missing_envelope_is_invalid_params(self) -> None:
        with Client(CMD) as client:
            error = self.error_of(lambda: client._send("tools/list", {}))
        self.assertEqual(error["code"], -32602)

    def test_missing_client_capabilities_is_invalid_params(self) -> None:
        with Client(CMD) as client:
            error = self.error_of(
                lambda: client._send(
                    "tools/list", {"_meta": {META_VERSION: VERSION}}
                )
            )
        self.assertEqual(error["code"], -32602)

    def test_invalid_client_info_is_invalid_params(self) -> None:
        invalid_values = (None, {"name": "missing-version"}, {"name": "client", "version": ""})
        for client_info in invalid_values:
            with self.subTest(client_info=client_info), Client(CMD) as client:
                params = {
                    "_meta": {
                        META_VERSION: VERSION,
                        META_CAPS: {},
                        META_CLIENT: client_info,
                    }
                }
                error = self.error_of(lambda: client._send("tools/list", params))
                self.assertEqual(error["code"], -32602)

    def test_unsupported_version_returns_supported_and_requested(self) -> None:
        error = self.error_of(lambda: raw("tools/list", version="2025-11-25"))
        self.assertEqual(error["code"], -32022)
        self.assertEqual(
            error["data"],
            {"supported": [VERSION], "requested": "2025-11-25"},
        )

    def test_client_retries_once_on_version_error(self) -> None:
        with Client(CMD) as client:
            client.version = "2025-11-25"
            result = client.request("tools/list")
            self.assertEqual(client.version, VERSION)
            self.assertEqual(result["resultType"], "complete")
            self.assertEqual(
                {tool["name"] for tool in result["tools"]},
                {"get_service_owner", "list_service_dependencies", "archive_service"},
            )

    def test_unknown_tool_is_protocol_invalid_params(self) -> None:
        error = self.error_of(
            lambda: raw(
                "tools/call",
                {"name": "not_a_tool", "arguments": {"service": "checkout"}},
            )
        )
        self.assertEqual(error["code"], -32602)
        self.assertIn("Unknown tool", error["message"])

    def test_invalid_tool_arguments_are_invalid_params(self) -> None:
        for arguments in ("checkout", None, [], {"service": "checkout", "extra": 1}, {}):
            with self.subTest(arguments=arguments):
                error = self.error_of(
                    lambda: raw(
                        "tools/call",
                        {"name": "get_service_owner", "arguments": arguments},
                    )
                )
                self.assertEqual(error["code"], -32602)

    def test_unknown_service_is_tool_execution_error(self) -> None:
        result = raw(
            "tools/call",
            {"name": "get_service_owner", "arguments": {"service": "missing"}},
        )
        self.assertTrue(result["isError"])
        self.assertIn("unknown service", result["content"][0]["text"])

    def test_capability_gate_requires_form_elicitation(self) -> None:
        params = {
            "name": "archive_service",
            "arguments": {"service": "checkout"},
        }

        missing = self.error_of(lambda: raw("tools/call", params))
        self.assertEqual(missing["code"], -32021)
        self.assertEqual(
            missing["data"],
            {"requiredCapabilities": {"elicitation": {"form": {}}}},
        )

        parent_only = self.error_of(
            lambda: raw("tools/call", params, caps={"elicitation": {}})
        )
        self.assertEqual(parent_only["code"], -32021)

        malformed_form = self.error_of(
            lambda: raw("tools/call", params, caps={"elicitation": {"form": None}})
        )
        self.assertEqual(malformed_form["code"], -32021)

        accepted = raw(
            "tools/call", params, caps={"elicitation": {"form": {}}}
        )
        self.assertEqual(accepted["resultType"], "complete")
        self.assertTrue(accepted["isError"])
        self.assertIn("confirmation flow not implemented", accepted["content"][0]["text"])

    def test_unknown_method_is_method_not_found(self) -> None:
        error = self.error_of(
            lambda: raw("initialize", {"protocolVersion": "2025-11-25"})
        )
        self.assertEqual(error["code"], -32601)

    def test_json_rpc_float_id_is_invalid(self) -> None:
        response = handle(
            {"jsonrpc": "2.0", "id": 1.5, "method": "tools/list", "params": {}}
        )
        self.assertEqual(response["error"]["code"], -32600)
        self.assertIsNone(response["id"])

    def test_notification_gets_no_reply(self) -> None:
        process = subprocess.Popen(
            CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/unknown-example",
            "params": {},
        }
        process.stdin.write(json.dumps(notification) + "\n")
        process.stdin.write("{not json\n")
        process.stdin.flush()
        first = json.loads(process.stdout.readline())
        process.stdin.close()
        process.wait(timeout=5)
        process.stdout.close()
        self.assertEqual(first["error"]["code"], -32700)
        self.assertIsNone(first["id"])

    def test_malformed_params_return_invalid_params(self) -> None:
        response = handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": [],
            }
        )
        self.assertEqual(response["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main(verbosity=2)
