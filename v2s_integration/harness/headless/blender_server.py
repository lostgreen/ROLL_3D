"""Execute addon commands on Blender's main thread without a GUI event loop."""

import argparse
import importlib.util
import json
from pathlib import Path
import queue
import signal
import socket
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    addon_path = Path(__file__).resolve().parents[1] / "blender-mcp/addon.py"
    spec = importlib.util.spec_from_file_location("sceneact_blender_addon", addon_path)
    addon = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = addon
    spec.loader.exec_module(addon)
    addon.register()
    pending = queue.Queue()
    stopping = threading.Event()

    class HeadlessServer(addon.BlenderMCPServer):
        def _handle_client(self, client):
            client.settimeout(0.5)
            buffer = b""
            try:
                while not stopping.is_set():
                    try:
                        data = client.recv(8192)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buffer += data
                    if len(buffer) > 4 * 1024 * 1024:
                        raise ValueError("command exceeds 4 MiB")
                    try:
                        command = json.loads(buffer)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    buffer = b""
                    done = threading.Event()
                    result = []
                    pending.put((command, result, done))
                    while not done.wait(0.2):
                        if stopping.is_set():
                            return
                    client.sendall(json.dumps(result[0]).encode())
            except OSError:
                pass
            finally:
                client.close()

    server = HeadlessServer(host="127.0.0.1", port=args.port)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    server.start()
    if not server.running:
        raise RuntimeError("Blender command server did not bind")
    try:
        while not stopping.is_set():
            try:
                command, result, done = pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result.append(server.execute_command(command))
            except Exception as exc:
                result.append({"status": "error", "message": str(exc)})
            finally:
                done.set()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
