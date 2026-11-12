"""OpenPI-compatible MessagePack WebSocket policy client and test server."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

_SOCKET_OPTIONS = {
    "compression": None,
    "max_size": None,
    "ping_interval": None,
    "ping_timeout": None,
}


def _encode_numpy(value: Any) -> Any:
    """Encode NumPy values with the byte-key schema used by OpenPI clients."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {array.dtype}")
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"unsupported MessagePack value: {type(value).__name__}")


def _decode_numpy(value: dict[bytes, Any]) -> Any:
    """Restore arrays and scalars encoded by OpenPI's msgpack_numpy module."""

    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        shape = tuple(int(size) for size in value[b"shape"])
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(message: dict[str, Any]) -> bytes:
    """Encode one OpenPI mapping into a binary MessagePack frame."""

    return msgpack.packb(message, default=_encode_numpy, use_bin_type=True)


def unpack_message(payload: bytes) -> dict[str, Any]:
    """Decode one required OpenPI binary frame into a mapping."""

    message = msgpack.unpackb(payload, object_hook=_decode_numpy)
    if not isinstance(message, dict):
        raise ValueError("WebSocket payload must be a mapping")
    return message


class PolicyClient:
    """Synchronous client for one OpenPI-compatible policy connection."""

    def __init__(self, host: str, port: int, receive_timeout: float | None = None) -> None:
        """Connect and read metadata; optional receive timeout is seconds per response."""

        self.receive_timeout = receive_timeout
        self.connection = connect(f"ws://{host}:{int(port)}", proxy=None, **_SOCKET_OPTIONS)
        self.metadata = self._receive()

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Send one observation and return the untrusted candidate action array."""

        self.connection.send(pack_message(observation))
        response = self._receive()
        if "error" in response:
            raise RuntimeError(f"policy server error: {response['error']}")
        if response.get("protocol_version", 1) != 1:
            raise ValueError("unsupported policy response protocol version")
        if "actions" not in response:
            raise ValueError("policy response does not contain actions")
        return np.asarray(response["actions"], dtype=np.float32)

    def close(self) -> None:
        """Close the policy socket."""

        self.connection.close()

    def _receive(self) -> dict[str, Any]:
        """Receive and decode one required binary mapping."""

        payload = self.connection.recv(timeout=self.receive_timeout)
        if not isinstance(payload, bytes):
            raise ValueError("policy server must use binary WebSocket frames")
        return unpack_message(payload)


class PolicyServer:
    """Small OpenPI-compatible server used by dry-run and unit tests."""

    def __init__(
        self, policy: Any, host: str = "0.0.0.0", port: int = 8765,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store an ``infer(observation)`` policy, endpoint and optional handshake mapping."""

        self.policy = policy
        self.host = host
        self.port = int(port)
        self.metadata = {"server": "control_robot_mock"} if metadata is None else metadata

    def create_server(self) -> Any:
        """Create a stoppable synchronous WebSocket server."""

        return serve(self._handle, self.host, self.port, **_SOCKET_OPTIONS)

    def serve_forever(self) -> None:
        """Serve requests until interrupted."""

        with self.create_server() as server:
            server.serve_forever()

    def _handle(self, websocket: ServerConnection) -> None:
        """Return validated policy results for each binary observation frame."""

        websocket.send(pack_message(self.metadata))
        for payload in websocket:
            try:
                if not isinstance(payload, bytes):
                    raise ValueError("policy requests must use binary frames")
                observation = unpack_message(payload)
                actions = np.asarray(self.policy.infer(observation), dtype=np.float32)
                response = {"actions": actions}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            websocket.send(pack_message(response))
