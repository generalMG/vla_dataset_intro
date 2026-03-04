"""
VLA Dataset Orchestrator — runs on WSL2 side.

Connects to Isaac Sim via the MCP socket bridge and drives data collection
episodes by sending execute_script commands that invoke vla_collector.py.

Usage:
    python orchestrate_vla.py [--episodes 50] [--max-steps 300] [--host <ip>] [--port 8767]
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VLA-Orchestrator")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _detect_windows_host() -> str:
    """Auto-detect the Windows host IP when running inside WSL2."""
    env_host = os.environ.get("ISAAC_SIM_HOST")
    if env_host:
        return env_host

    try:
        gw = subprocess.check_output(
            ["ip", "route", "show", "default"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = gw.strip().splitlines()
        if lines:
            parts = lines[0].split()
            if len(parts) >= 3:
                return parts[2]
    except Exception:
        pass

    return "localhost"


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _build_port_candidates(explicit_port: Optional[int]) -> list[int]:
    """Build a list of ports to try in order."""
    candidates: list[int] = []

    if explicit_port is not None:
        candidates.append(int(explicit_port))

    env_port = os.environ.get("ISAAC_SIM_PORT")
    if env_port:
        try:
            candidates.append(int(env_port))
        except ValueError:
            log.warning("Ignoring invalid ISAAC_SIM_PORT value: %s", env_port)

    # Try the known ports used by this project (newer first, legacy second).
    candidates.extend([8767, 8766])
    return _ordered_unique(candidates)


ISAAC_HOST = _detect_windows_host()
ISAAC_PORT = None  # None = auto-try candidate ports
DATASET_DIR = "/mnt/d/mg_ai_research/workspace/whatnot/vla_dataset"
COLLECTOR_SCRIPT = Path(__file__).parent / "vla_collector.py"


# ---------------------------------------------------------------------------
# Isaac Sim Socket Connection
# ---------------------------------------------------------------------------
class IsaacConnection:
    """Minimal socket connection to Isaac Sim MCP extension."""

    def __init__(self, host: str = ISAAC_HOST, port: Optional[int] = ISAAC_PORT):
        self.host = host
        self.port_candidates = _build_port_candidates(port)
        self.port = None
        self.sock = None

    def connect(self) -> bool:
        if self.sock:
            return True

        last_error = None
        for candidate in self.port_candidates:
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, candidate))
                sock.settimeout(None)

                self.sock = sock
                self.port = candidate
                log.info("Connected to Isaac Sim at %s:%s", self.host, candidate)
                return True
            except Exception as e:
                last_error = e
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

        self.sock = None
        log.error(
            "Connection failed. host=%s tried_ports=%s last_error=%s",
            self.host,
            self.port_candidates,
            last_error,
        )
        return False

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def reconnect(self) -> bool:
        self.disconnect()
        return self.connect()

    def _recv_full(self, timeout: float = 600.0) -> bytes:
        """Receive a complete JSON response, potentially in multiple chunks."""
        self.sock.settimeout(timeout)
        chunks = []

        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                if not chunks:
                    raise ConnectionError("Connection closed before receiving data")
                break

            chunks.append(chunk)
            try:
                data = b"".join(chunks)
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                continue

        data = b"".join(chunks)
        json.loads(data.decode("utf-8"))  # Raise if incomplete
        return data

    def send_command(self, command_type: str, params: dict = None, timeout: float = 600.0) -> dict:
        """Send a command and return the parsed response."""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac Sim")

        payload = {"type": command_type, "params": params or {}}
        try:
            self.sock.sendall(json.dumps(payload).encode("utf-8"))
            raw = self._recv_full(timeout=timeout)
            response = json.loads(raw.decode("utf-8"))
            if response.get("status") == "error":
                raise RuntimeError(response.get("message", "Unknown error"))
            return response.get("result", {})
        except (socket.timeout, ConnectionError, BrokenPipeError, ConnectionResetError, OSError) as e:
            log.error("Socket error: %s", e)
            self.sock = None
            raise

    def execute_script(self, code: str, timeout: float = 600.0) -> dict:
        """Convenience wrapper for execute_script commands."""
        return self.send_command("execute_script", {"code": code}, timeout=timeout)


# ---------------------------------------------------------------------------
# Script builders — generate Python code strings to send to Isaac Sim
# ---------------------------------------------------------------------------
def _read_collector_source() -> str:
    with open(COLLECTOR_SCRIPT, "r", encoding="utf-8") as f:
        return f.read()


def build_init_script() -> str:
    collector_src = _read_collector_source()
    return f'''\
_vla_command = "init"
_vla_kwargs = {{}}

{collector_src}
'''


def build_episode_script(episode_id: int, max_steps: int = 300) -> str:
    collector_src = _read_collector_source()
    return f'''\
_vla_command = "run_episode"
_vla_kwargs = {{"episode_id": {episode_id}, "max_steps": {max_steps}}}

{collector_src}
'''


def build_status_script() -> str:
    collector_src = _read_collector_source()
    return f'''\
_vla_command = "status"
_vla_kwargs = {{}}

{collector_src}
'''


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------
def generate_manifest(episodes_metadata: list, dataset_dir: str = DATASET_DIR):
    manifest = {
        "dataset_name": "franka_cube_manipulation",
        "num_episodes": len(episodes_metadata),
        "tasks": sorted(set(ep.get("task_type", "unknown") for ep in episodes_metadata)),
        "camera_views": ["wrist_rgb", "top_rgb", "front_rgb", "side_rgb"],
        "image_resolution": [640, 480],
        "action_dim": 9,
        "state_dim": 9,
        "episodes": episodes_metadata,
    }

    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("Manifest saved to %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------
def run_collection(
    num_episodes: int = 50,
    max_steps: int = 300,
    host: str = ISAAC_HOST,
    port: Optional[int] = ISAAC_PORT,
):
    isaac = IsaacConnection(host=host, port=port)

    if not isaac.connect():
        log.error("Could not connect to Isaac Sim. Is the MCP extension running?")
        sys.exit(1)

    # --- Step 1: Initialize ---
    log.info("Initializing VLA collector in Isaac Sim...")
    try:
        init_result = isaac.execute_script(build_init_script(), timeout=120.0)
        log.info("Init result: %s", json.dumps(init_result, indent=2))

        if isinstance(init_result, dict):
            res = init_result.get("result", init_result)
            if isinstance(res, dict) and res.get("status") == "error":
                log.error("Init failed: %s", res.get("message"))
                sys.exit(1)
    except Exception as e:
        log.error("Init failed: %s", e)
        sys.exit(1)

    # --- Step 2: Run episodes ---
    episodes_metadata = []
    successful = 0
    failed = 0

    for ep_idx in range(num_episodes):
        log.info("\n%s", "=" * 60)
        log.info("Episode %s/%s", ep_idx + 1, num_episodes)
        log.info("%s", "=" * 60)

        try:
            if not isaac.sock:
                log.info("Reconnecting to Isaac Sim...")
                if not isaac.reconnect():
                    log.error("Reconnection failed, skipping episode")
                    failed += 1
                    continue

            ep_result = isaac.execute_script(
                build_episode_script(ep_idx, max_steps),
                timeout=600.0,
            )

            if isinstance(ep_result, dict):
                res = ep_result.get("result", ep_result)
                if isinstance(res, dict):
                    if res.get("status") == "success":
                        metadata = res.get("metadata", {})
                        episodes_metadata.append(metadata)
                        successful += 1
                        log.info(
                            "Episode %s completed: task=%s, frames=%s, success=%s, duration=%ss",
                            ep_idx,
                            metadata.get("task_type"),
                            metadata.get("num_frames"),
                            metadata.get("success"),
                            metadata.get("duration_s"),
                        )
                    else:
                        log.error("Episode %s error: %s", ep_idx, res.get("message"))
                        err_meta = res.get("metadata") or {}
                        if err_meta:
                            log.error(
                                "Episode %s validation_errors=%s metrics=%s",
                                ep_idx,
                                err_meta.get("validation_errors"),
                                err_meta.get("validation_metrics"),
                            )
                        failed += 1
                else:
                    log.warning("Unexpected result format: %s", res)
                    failed += 1
            else:
                log.warning("Unexpected result type: %s", type(ep_result))
                failed += 1

        except Exception as e:
            log.error("Episode %s exception: %s", ep_idx, e)
            failed += 1
            isaac.reconnect()

        log.info("Progress: %s successful, %s failed, %s remaining", successful, failed, num_episodes - ep_idx - 1)

    # --- Step 3: Generate manifest ---
    if episodes_metadata:
        manifest_path = generate_manifest(episodes_metadata)
        log.info("\nDataset collection complete!")
        log.info("  Episodes: %s successful, %s failed", successful, failed)
        log.info("  Manifest: %s", manifest_path)
    else:
        log.warning("No episodes collected successfully.")

    isaac.disconnect()
    return episodes_metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VLA Dataset Collection Orchestrator")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=300, help="Max simulation steps per episode")
    parser.add_argument("--host", type=str, default=ISAAC_HOST, help="Isaac Sim host")
    parser.add_argument(
        "--port",
        type=int,
        default=ISAAC_PORT,
        help="Isaac Sim port. If omitted, tries ISAAC_SIM_PORT, then 8767, then 8766.",
    )
    args = parser.parse_args()

    run_collection(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
