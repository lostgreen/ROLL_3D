"""Manage only the Blender subprocess started by this harness."""

import os
from pathlib import Path
import socket
import subprocess
import time


def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


class HeadlessBlender:
    def __init__(self, port=9876, log_path=None, blender_bin=None, startup_timeout=60):
        self.port = port
        self.log_path = Path(log_path or f"runs/blender-{port}.log").resolve()
        self.blender_bin = blender_bin or os.environ.get("BLENDER_BIN") or os.environ.get("BLENDER", "blender")
        self.startup_timeout = startup_timeout
        self.process = None
        self._log = None

    def start(self):
        if _port_open("127.0.0.1", self.port):
            raise RuntimeError(f"Blender port {self.port} is already occupied")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        script = Path(__file__).with_name("blender_server.py").resolve()
        try:
            self.process = subprocess.Popen(
                [self.blender_bin, "--background", "--factory-startup", "--threads", "8",
                 "--python", str(script), "--", "--port", str(self.port)],
                stdout=self._log, stderr=subprocess.STDOUT,
                env={**os.environ, "BLENDER_MCP_DISABLE_TELEMETRY": "true"},
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Blender exited {self.process.returncode}; see {self.log_path}")
                if _port_open("127.0.0.1", self.port):
                    return self
                time.sleep(0.2)
            raise TimeoutError(f"Blender startup timed out; see {self.log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log:
            self._log.close()
