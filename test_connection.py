"""Quick connection test to Isaac Sim MCP extension."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Iterable, Optional


def detect_windows_host() -> str:
    env_host = os.environ.get("ISAAC_SIM_HOST")
    if env_host:
        return env_host
    try:
        gw = subprocess.check_output(["ip", "route", "show", "default"], text=True, stderr=subprocess.DEVNULL)
        lines = gw.strip().splitlines()
        if lines:
            parts = lines[0].split()
            if len(parts) >= 3:
                return parts[2]
    except Exception:
        pass
    return "localhost"


def ordered_unique(values: Iterable[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_port_candidates(explicit_port: Optional[int]) -> list[int]:
    ports = []
    if explicit_port is not None:
        ports.append(int(explicit_port))

    env_port = os.environ.get("ISAAC_SIM_PORT")
    if env_port:
        try:
            ports.append(int(env_port))
        except ValueError:
            print(f"Ignoring invalid ISAAC_SIM_PORT value: {env_port}")

    ports.extend([8767, 8766])
    return ordered_unique(ports)


def connect_first(host: str, ports: list[int], timeout: float = 5.0):
    last_error = None
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, port))
            print(f"Connected to {host}:{port}")
            return s, port
        except Exception as e:
            last_error = e
            print(f"  Failed {host}:{port} -> {e}")
    raise ConnectionError(f"Could not connect to {host} on ports {ports}. Last error: {last_error}")


def send_and_recv(sock, command_type, params=None, timeout=60.0):
    payload = {"type": command_type, "params": params or {}}
    raw = json.dumps(payload).encode("utf-8")
    print(f"  Sending {len(raw)} bytes...")
    sock.sendall(raw)

    sock.settimeout(timeout)
    chunks = []
    start = time.time()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            elapsed = time.time() - start
            print(f"  Timeout after {elapsed:.1f}s, received {sum(len(c) for c in chunks)} bytes so far")
            raise
        if not chunk:
            elapsed = time.time() - start
            total = sum(len(c) for c in chunks)
            print(f"  Connection closed after {elapsed:.1f}s, received {total} bytes total")
            if total > 0:
                print(f"  Partial data: {b''.join(chunks)[:500]}")
            raise ConnectionError("Connection closed before complete response")
        chunks.append(chunk)
        try:
            data = b"".join(chunks)
            result = json.loads(data.decode("utf-8"))
            elapsed = time.time() - start
            print(f"  Got response in {elapsed:.1f}s ({len(data)} bytes)")
            return result
        except json.JSONDecodeError:
            continue


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim MCP socket preflight")
    parser.add_argument("--host", default=detect_windows_host(), help="Isaac Sim host")
    parser.add_argument("--port", type=int, default=None, help="Specific port. If omitted, tries 8767 then 8766")
    args = parser.parse_args()

    host = args.host
    ports = build_port_candidates(args.port)

    print(f"Target host: {host}")
    print(f"Port candidates: {ports}")
    print()

    # Test 1: TCP connect only
    print("=== Test 1: TCP connect ===")
    try:
        s, connected_port = connect_first(host, ports, timeout=5.0)
        print(f"  Connected! Local: {s.getsockname()}, Remote: {s.getpeername()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Test 2: get_scene_info
    print("\n=== Test 2: get_scene_info ===")
    try:
        r = send_and_recv(s, "get_scene_info", timeout=60)
        print(f"  OK: {json.dumps(r, indent=2)[:300]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Reconnecting...")
        s.close()
        time.sleep(1)
        s, connected_port = connect_first(host, [connected_port], timeout=5.0)

    # Test 3: tiny execute_script
    print("\n=== Test 3: execute_script (result=42) ===")
    try:
        r = send_and_recv(s, "execute_script", {"code": "result = 42"}, timeout=60)
        print(f"  OK: {json.dumps(r, indent=2)[:300]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Reconnecting...")
        s.close()
        time.sleep(1)
        s, connected_port = connect_first(host, [connected_port], timeout=5.0)

    # Test 4: list stage prims
    print("\n=== Test 4: execute_script (stage prims) ===")
    code = """
stage = omni.usd.get_context().get_stage()
prims = [str(p.GetPath()) for p in stage.Traverse()]
result = {"num_prims": len(prims), "first_5": prims[:5]}
"""
    try:
        r = send_and_recv(s, "execute_script", {"code": code}, timeout=60)
        print(f"  OK: {json.dumps(r, indent=2)[:300]}")
    except Exception as e:
        print(f"  FAILED: {e}")

    s.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
