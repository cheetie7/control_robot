"""Validate the local Web dashboard and Rerun server configuration."""

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

_MEMORY_LIMIT = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?(?:%|[KMGT]i?B)$", re.IGNORECASE)
_LOG_LEVELS = {"debug", "info", "warn", "error", "fatal"}


@dataclass(frozen=True)
class RerunConfig:
    """Ports and resource limits for one local Rerun Web Viewer."""

    enabled: bool
    grpc_port: int
    web_port: int
    server_memory_limit: str
    max_fps: float
    preview_scale: float


@dataclass(frozen=True)
class DashboardConfig:
    """Validated HTTP, diagnostics and visualization configuration."""

    enabled: bool
    host: str
    port: int
    open_browser: bool
    status_rate_hz: float
    rosout_min_level: str
    rosout_history_size: int
    service_timeout_sec: float
    rerun: RerunConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DashboardConfig":
        """Build dashboard settings from the top-level YAML mapping.

        Args:
            value: Mapping below ``dashboard`` or ``None``.

        Returns:
            Immutable scalar configuration. The first release intentionally
            accepts only local host bindings because its APIs can move hardware.
        """

        data = _mapping(value, "dashboard")
        enabled = _boolean(data.get("enabled", False), "dashboard.enabled")
        host = data.get("host", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("dashboard.host must be 127.0.0.1 or localhost")
        port = _port(data.get("port", 8080), "dashboard.port")
        open_browser = _boolean(
            data.get("open_browser", True), "dashboard.open_browser"
        )
        status_rate_hz = _positive_number(
            data.get("status_rate_hz", 2.0), "dashboard.status_rate_hz"
        )
        rosout_min_level = data.get("rosout_min_level", "warn")
        if rosout_min_level not in _LOG_LEVELS:
            raise ValueError(f"dashboard.rosout_min_level must be one of {sorted(_LOG_LEVELS)}")
        history_size = data.get("rosout_history_size", 200)
        if not isinstance(history_size, int) or isinstance(history_size, bool) or history_size <= 0:
            raise ValueError("dashboard.rosout_history_size must be a positive integer")
        service_timeout_sec = _positive_number(
            data.get("service_timeout_sec", 10.0), "dashboard.service_timeout_sec"
        )

        rerun_data = _mapping(data.get("rerun"), "dashboard.rerun")
        rerun_enabled = _boolean(
            rerun_data.get("enabled", True), "dashboard.rerun.enabled"
        )
        grpc_port = _port(rerun_data.get("grpc_port", 9876), "dashboard.rerun.grpc_port")
        web_port = _port(rerun_data.get("web_port", 9090), "dashboard.rerun.web_port")
        memory_limit = rerun_data.get("server_memory_limit", "16MiB")
        if not isinstance(memory_limit, str) or not _MEMORY_LIMIT.fullmatch(memory_limit):
            raise ValueError("dashboard.rerun.server_memory_limit is invalid")
        max_fps = _positive_number(
            rerun_data.get("max_fps", 10.0), "dashboard.rerun.max_fps"
        )
        preview_scale = _positive_number(
            rerun_data.get("preview_scale", 0.5), "dashboard.rerun.preview_scale"
        )
        if preview_scale > 1.0:
            raise ValueError("dashboard.rerun.preview_scale must be at most 1.0")
        ports = [port, grpc_port, web_port]
        if len(ports) != len(set(ports)):
            raise ValueError("dashboard and Rerun ports must be unique")
        return cls(
            enabled=enabled,
            host=host,
            port=port,
            open_browser=open_browser,
            status_rate_hz=status_rate_hz,
            rosout_min_level=rosout_min_level,
            rosout_history_size=history_size,
            service_timeout_sec=service_timeout_sec,
            rerun=RerunConfig(
                enabled=rerun_enabled,
                grpc_port=grpc_port,
                web_port=web_port,
                server_memory_limit=memory_limit,
                max_fps=max_fps,
                preview_scale=preview_scale,
            ),
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a YAML mapping or raise a field-specific error."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: Any, name: str) -> bool:
    """Return a strict YAML boolean."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _port(value: Any, name: str) -> int:
    """Return a valid non-privileged TCP port."""

    if not isinstance(value, int) or isinstance(value, bool) or not 1024 <= value <= 65535:
        raise ValueError(f"{name} must be an integer in [1024, 65535]")
    return value


def _positive_number(value: Any, name: str) -> float:
    """Return one positive finite scalar float."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)
