"""FastAPI application and ROS executor for the local recording dashboard."""

import asyncio
import threading
import webbrowser
from pathlib import Path

import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rclpy.executors import MultiThreadedExecutor
from starlette.responses import Response as StarletteResponse

from xarm_ros2_teleop.dashboard.ros_gateway import DashboardRosGateway
from xarm_ros2_teleop.rl.intervention import RL_SERVICE_COMMANDS


class _NoCacheStaticFiles(StaticFiles):
    """Static file mount whose responses always bypass browser caches.

    Revalidating on every request keeps the operator console in lockstep with
    the deployed assets during active dashboard iteration; the ETag still
    avoids re-downloading unchanged bodies.
    """

    def file_response(self, *args, **kwargs) -> StarletteResponse:
        """Return one static file response that browsers must revalidate."""

        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


class RecordingSessionRequest(BaseModel):
    """Web payload for one pre-recording dataset session configuration."""

    instruction: str
    dataset_root: str
    write_mode: str


def create_app(gateway: DashboardRosGateway, static_root: Path) -> FastAPI:
    """Create the local versioned HTTP API around one ROS gateway.

    Args:
        gateway: Spinning ROS node that owns all state and service clients.
        static_root: Directory containing ``index.html``, CSS and JavaScript.

    Returns:
        Configured FastAPI application; no server socket is opened here.
    """

    app = FastAPI(title="control_robot dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", _NoCacheStaticFiles(directory=static_root), name="static")

    @app.get("/")
    async def index():
        """Return the packaged single-page recording console."""

        # Console assets change with each deploy; browsers must always
        # revalidate instead of serving stale operator UI from heuristic cache.
        return FileResponse(
            static_root / "index.html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @app.get("/healthz")
    async def healthz():
        """Return HTTP process readiness independent of hardware health."""

        return {"ok": True}

    @app.get("/api/v1/status")
    async def status():
        """Return the latest immutable system snapshot."""

        return gateway.snapshot()

    @app.post("/api/v1/recording/configure")
    async def configure_recording(request: RecordingSessionRequest):
        """Set task, exact dataset directory and write behavior before capture."""

        return await run_in_threadpool(
            gateway.configure_recording_session,
            request.instruction,
            request.dataset_root,
            request.write_mode,
        )

    @app.websocket("/api/v1/events")
    async def events(websocket: WebSocket):
        """Stream complete snapshots on a two-hertz interval."""

        await websocket.accept()
        try:
            while True:
                await websocket.send_json(gateway.snapshot())
                await asyncio.sleep(1.0 / gateway.config.status_rate_hz)
        except WebSocketDisconnect:
            return

    for action in (
        "staging",
        "start",
        "save",
        "optimal",
        "failure",
        "discard",
        "finish",
    ):

        def make_handler(operation: str):
            """Return one HTTP handler bound to a recording Trigger service."""

            async def handler():
                """Execute the captured recording operation outside the event loop."""

                return await run_in_threadpool(gateway.call_recording_service, operation)

            handler.__name__ = f"recording_{operation}"
            return handler

        app.add_api_route(
            f"/api/v1/recording/{action}",
            make_handler(action),
            methods=["POST"],
        )

    @app.post("/api/v1/inference/stop")
    async def stop_inference():
        """Stop the active remote-policy rollout without changing motion arming."""

        return await run_in_threadpool(gateway.call_inference_stop)

    for action in ("success", "failure"):

        def make_evaluation_handler(operation: str):
            """Return one HTTP handler for a labeled evaluation outcome."""

            async def handler():
                """Save the active rollout with the captured outcome label."""

                return await run_in_threadpool(
                    gateway.call_evaluation_service, operation
                )

            handler.__name__ = f"evaluation_{operation}"
            return handler

        app.add_api_route(
            f"/api/v1/evaluation/{action}",
            make_evaluation_handler(action),
            methods=["POST"],
        )

    for action in RL_SERVICE_COMMANDS:

        def make_rl_handler(operation: str):
            """Return one HTTP handler bound to an RL Trigger service."""

            async def handler():
                """Execute the captured RL operation outside the event loop."""

                return await run_in_threadpool(gateway.call_rl_service, operation)

            handler.__name__ = f"rl_{operation}"
            return handler

        app.add_api_route(
            f"/api/v1/rl/{action}",
            make_rl_handler(action),
            methods=["POST"],
        )
    return app


def main(args=None) -> None:
    """Run ROS callbacks in a thread and FastAPI on the configured local port."""

    rclpy.init(args=args)
    gateway = DashboardRosGateway()
    system = gateway.system
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(gateway)
    executor_thread = threading.Thread(target=executor.spin, name="dashboard-ros", daemon=True)
    executor_thread.start()
    static_root = Path(__file__).parent / "static"
    app = create_app(gateway, static_root)
    url = f"http://{system.dashboard.host}:{system.dashboard.port}"
    if system.dashboard.open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(
            app,
            host=system.dashboard.host,
            port=system.dashboard.port,
            log_level="info",
        )
    finally:
        executor.shutdown()
        executor_thread.join(timeout=1.0)
        gateway.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
