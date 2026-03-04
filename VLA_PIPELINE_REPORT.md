# VLA Dataset Synthesis Pipeline — Implementation Report

## Overview

This document covers the full implementation of a Vision-Language-Action (VLA) dataset synthesis pipeline using Isaac Sim (Windows) controlled from WSL2 via a socket-based MCP extension. The pipeline collects robot manipulation episodes (Franka + colored cubes) and saves them as LeRobot-compatible HDF5 files.

**Final working state:** 5/5 episodes collected successfully, ~3.3MB per episode, ~1.1s per episode at 51 frames each.

---

## Architecture

```
WSL2 (Linux)                              Windows (Isaac Sim)
┌──────────────────────────┐              ┌───────────────────────────────┐
│ orchestrate_vla.py       │              │ extension.py (MCP Extension)  │
│  - IsaacConnection       │──TCP────────→│  - _server_loop (thread)      │
│  - run_collection()      │  Port 8767   │  - _handle_client (thread)    │
│  - embeds collector src  │  0.0.0.0     │  - threading.Event sync       │
│  - generates manifest    │              │  - execute_script_async()     │
└──────────────────────────┘              └───────────────────────────────┘
         │                                         │
         │  (sends full vla_collector.py            │ exec(code, local_ns)
         │   with _vla_command/_vla_kwargs)         │
         │                                         ▼
         └───────────────────────────────→  vla_collector.py
                                            (runs inside Isaac Sim)
                                            - VLACollector singleton
                                            - dispatch() command router
                                            - async_main() for episodes
                                            - saves HDF5 to D:\...\vla_dataset\
```

---

## Files Created/Modified

| File | Location | Action | Purpose |
|------|----------|--------|---------|
| `extension.py` | `C:\users\user\isaac-sim-mcp\isaac.sim.mcp_extension\...` | Modified | MCP socket server (runs in Isaac Sim) |
| `vla_collector.py` | `D:\mg_ai_research\workspace\whatnot\vla_dataset\` | Created | Data collection module (runs inside Isaac Sim via execute_script) |
| `orchestrate_vla.py` | `D:\mg_ai_research\workspace\whatnot\vla_dataset\` | Created | WSL2 orchestrator (drives collection via socket commands) |
| `test_connection.py` | `D:\mg_ai_research\workspace\whatnot\vla_dataset\` | Created | Connection diagnostic tool |
| `dataset_manifest.json` | `D:\mg_ai_research\workspace\whatnot\vla_dataset\` | Generated | Dataset metadata (LeRobot-compatible) |

**CRITICAL:** The extension file exists in TWO locations. Isaac Sim loads from the **Windows path** (`C:\users\user\isaac-sim-mcp\...`), NOT from the WSL2 path (`/home/mg_server/isaac-sim-mcp/...`). Always edit or copy to the Windows path. From WSL2:
```bash
# Windows path accessible from WSL2:
/mnt/c/users/user/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/extension.py

# To sync WSL2 edits → Windows:
cp /home/mg_server/isaac-sim-mcp/.../extension.py "/mnt/c/users/user/isaac-sim-mcp/.../extension.py"
```

---

## Isaac Sim Environment & Connection Architecture

### Isaac Sim Installation

Isaac Sim runs on **Windows** (not WSL2). Key paths:

| Component | Windows Path | WSL2 Accessible Path |
|-----------|-------------|---------------------|
| Isaac Sim App | `C:\users\user\isaacsim\` | `/mnt/c/users/user/isaacsim/` |
| Isaac Sim Source/Extensions | `C:\users\user\isaacsim\isaacsim\source\extensions\` | `/mnt/c/users/user/isaacsim/isaacsim/source/extensions/` |
| MCP Extension (custom) | `C:\users\user\isaac-sim-mcp\isaac.sim.mcp_extension\` | `/mnt/c/users/user/isaac-sim-mcp/isaac.sim.mcp_extension/` |
| Isaac Sim Python/pip env | `C:\Users\user\AppData\Local\ov\data\Kit\Isaac-Sim Full\5.1\pip3-envs\default\` | N/A |
| Dataset output | `D:\mg_ai_research\workspace\whatnot\vla_dataset\` | `/mnt/d/mg_ai_research/workspace/whatnot/vla_dataset/` |

**Isaac Sim version:** 5.1.0 GA (Kit-based application on NVIDIA Omniverse)

### How the MCP Extension is Loaded

The MCP extension (`isaac.sim.mcp_extension`) is a custom Kit extension that Isaac Sim discovers via its **extension search paths**. When Isaac Sim starts:

1. Isaac Sim scans extension directories listed in its config (including user-specified paths)
2. The MCP extension at `C:\users\user\isaac-sim-mcp\isaac.sim.mcp_extension\` is found via the extension search path
3. If the extension is **enabled** in Isaac Sim's Extension Manager UI, Kit calls `on_startup()` on the extension class
4. `on_startup()` spawns a background thread running `_server_loop()` that binds a TCP socket on `0.0.0.0:8767`
5. The extension stays active as long as Isaac Sim is running and the extension is enabled

To verify: check the Isaac Sim console for `"Isaac Sim MCP server started on 0.0.0.0:8767"` on startup. If you see a warning triangle in the Extension Manager, the extension failed to load — check the console log for the error.

### Full Connection Flow

```
┌─ WSL2 (Linux) ────────────────────────────────────────────────────────┐
│                                                                        │
│  orchestrate_vla.py                                                    │
│    │                                                                   │
│    ├─ detect_windows_host()                                            │
│    │   └─ `ip route show default` → parse gateway IP (192.168.64.1)   │
│    │                                                                   │
│    ├─ IsaacConnection.connect(host="192.168.64.1", port=8767)          │
│    │   └─ TCP socket.connect() ─────────────────────┐                  │
│    │                                                 │                  │
│    ├─ For each episode:                              │                  │
│    │   ├─ Read vla_collector.py source               │                  │
│    │   ├─ Prepend: _vla_command="run_episode"        │                  │
│    │   ├─ Prepend: _vla_kwargs={episode_id, ...}     │                  │
│    │   ├─ send_command("execute_script", {code: ...})│                  │
│    │   │   └─ JSON payload sent via TCP ─────────────┤                  │
│    │   └─ Block waiting for JSON response            │                  │
│    │                                                 │                  │
│    └─ Generate dataset_manifest.json                 │                  │
│                                                      │                  │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                            ┌───── TCP port 8767 ──────┘
                            │     (0.0.0.0 binding)
                            ▼
┌─ Windows (Isaac Sim) ─────────────────────────────────────────────────┐
│                                                                        │
│  extension.py — _server_loop() (background thread)                     │
│    │                                                                   │
│    ├─ socket.accept() → new client                                     │
│    ├─ _handle_client() in new thread                                   │
│    │   ├─ recv() JSON payload                                          │
│    │   ├─ Parse command: {"type": "execute_script", "params": {code}}  │
│    │   │                                                               │
│    │   ├─ Schedule async on Kit event loop:                            │
│    │   │   run_coroutine(execute_wrapper())                            │
│    │   │                                                               │
│    │   ├─ done_event.wait(timeout=300s)  ◄── thread blocks here        │
│    │   │                                                               │
│    │   │   ┌─ Kit async event loop ────────────────────────┐           │
│    │   │   │                                               │           │
│    │   │   │  execute_script_async(code)                   │           │
│    │   │   │    ├─ exec(code, local_ns)                    │           │
│    │   │   │    │   └─ vla_collector.py runs:               │           │
│    │   │   │    │       ├─ dispatch("run_episode")          │           │
│    │   │   │    │       └─ defines async_main()             │           │
│    │   │   │    │                                          │           │
│    │   │   │    ├─ await async_main()                      │           │
│    │   │   │    │   ├─ Setup cameras, randomize cubes      │           │
│    │   │   │    │   ├─ for step in range(max_steps):       │           │
│    │   │   │    │   │   ├─ await next_update_async()       │           │
│    │   │   │    │   │   ├─ capture_frame() (RGB, depth)    │           │
│    │   │   │    │   │   └─ read_robot_state()              │           │
│    │   │   │    │   └─ save_episode_hdf5()                 │           │
│    │   │   │    │       └─ writes to D:\...\vla_dataset\   │           │
│    │   │   │    │                                          │           │
│    │   │   │    └─ result = local_ns.get("result")         │           │
│    │   │   │                                               │           │
│    │   │   └─ done_event.set()  ──► unblocks handler       │           │
│    │   │                           thread                  │           │
│    │   │                                                               │
│    │   ├─ response = response_holder[0]                                │
│    │   └─ client.sendall(JSON response)  ──► back to WSL2              │
│    │                                                                   │
└────┼───────────────────────────────────────────────────────────────────┘
     │
     └─ response received by IsaacConnection._recv_full()
        → orchestrate_vla.py processes metadata, loops to next episode
```

### Key Integration Details

1. **One-shot script embedding**: The orchestrator reads the ENTIRE `vla_collector.py` source and embeds it in each `execute_script` call. This means the collector code is re-exec'd each time but uses a module-level singleton (`_COLLECTOR`) to persist state between calls within the same Isaac Sim session.

2. **Command routing via globals**: Before the embedded collector source, the orchestrator prepends `_vla_command = "run_episode"` and `_vla_kwargs = {episode_id: N}`. The collector's entry point reads these globals and dispatches accordingly.

3. **Async bridging**: The extension bridges two threading models:
   - **Client handler**: blocking thread (can call `socket.recv/sendall`)
   - **Kit event loop**: async (required for `next_update_async()`, USD operations)
   - The `threading.Event` pattern bridges them: handler blocks, async coroutine runs on Kit loop, signals Event when done

4. **Data path**: HDF5 files are written by Isaac Sim's Python (running on Windows) to `D:\mg_ai_research\workspace\whatnot\vla_dataset\`. WSL2 can read these same files via `/mnt/d/mg_ai_research/workspace/whatnot/vla_dataset/` for visualization and training.

5. **Scene prerequisite**: The Franka Cortex example scene must be loaded and simulation **playing** before running the orchestrator. The scene provides `/World/Franka` (robot), `/World/Obs/{Color}Cube` (cubes), and the Cortex decider network for autonomous behaviors.

---

## Debugging Journey & Issues Resolved

### Issue 1: WSL2 Cannot Reach Isaac Sim on `localhost`

**Symptom:** `[Errno 111] Connection refused` when connecting to `localhost:8766`.

**Root Cause:** Isaac Sim runs on Windows and binds to `localhost` (127.0.0.1). WSL2 is a separate network namespace — `localhost` in WSL2 resolves to WSL2's own loopback, not Windows.

**Resolution:** Connect to the Windows host IP instead. The Windows host IP (as seen from WSL2) is the default gateway:
```python
def _detect_windows_host() -> str:
    """Auto-detect Windows host IP from WSL2 gateway."""
    gw = os.popen("ip route show default 2>/dev/null").read()
    if gw:
        return gw.strip().split()[2]  # e.g., "192.168.64.1"
    return "localhost"
```

The extension also needed to bind to `0.0.0.0` (all interfaces) instead of `localhost` so WSL2 connections on the virtual adapter are accepted. See Issue 3 for complications with this.

### Issue 2: Extension Editing the Wrong File

**Symptom:** Changes to `extension.py` had no effect. Isaac Sim console still showed old errors (`self.stop()` instead of `self._stop()`).

**Root Cause:** We edited `/home/mg_server/isaac-sim-mcp/.../extension.py` (WSL2 filesystem), but Isaac Sim loads from `C:\users\user\isaac-sim-mcp\.../extension.py` (Windows filesystem). These are separate copies.

**Resolution:** Always edit or copy to the Windows-accessible path:
```bash
cp /home/mg_server/isaac-sim-mcp/.../extension.py \
   "/mnt/c/users/user/isaac-sim-mcp/.../extension.py"
```

**Lesson:** Always check the traceback paths in Isaac Sim error logs to confirm which file is loaded.

### Issue 3: Port Binding — `0.0.0.0` vs Port Proxy Conflict

**Symptom:** `[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions` when extension tries to bind.

**Root Cause (Attempt 1):** We changed the extension to bind `0.0.0.0:8766`. Windows blocked this (Hyper-V / permission issue).

**Resolution (Attempt 1):** Reverted to `localhost:8766` and set up a Windows port proxy:
```powershell
netsh interface portproxy add v4tov4 listenport=8766 listenaddress=0.0.0.0 connectport=8766 connectaddress=127.0.0.1
```

**Root Cause (Attempt 2):** The port proxy itself binds `0.0.0.0:8766`, which then conflicts with the extension binding `localhost:8766` — the port is already in use.

**Final Resolution:** Changed to a fresh port `8767` and bound directly to `0.0.0.0`:
```python
self.port = self._settings.get("/exts/isaac.sim.mcp/server, port") or 8767
self.host = self._settings.get("/exts/isaac.sim.mcp/server.host") or "0.0.0.0"
```
Then removed the port proxy:
```powershell
netsh interface portproxy delete v4tov4 listenport=8766 listenaddress=0.0.0.0
```
And added a firewall rule:
```powershell
netsh advfirewall firewall add rule name="Isaac Sim MCP 8767" dir=in action=allow protocol=TCP localport=8767
```

**Lesson:** Avoid port proxies when possible — they create hidden port conflicts. Use a clean port that nothing else reserves and bind directly to `0.0.0.0`.

**Useful diagnostic commands:**
```powershell
# Check what's using a port:
netstat -ano | findstr :8766

# Check Hyper-V port exclusions:
netsh interface ipv4 show excludedportrange protocol=tcp

# Check active port proxies:
netsh interface portproxy show v4tov4

# Kill a process holding a port:
taskkill /PID <PID> /F
```

### Issue 4: Zombie Socket from Old Isaac Sim Process

**Symptom:** New Isaac Sim instance fails to bind port 8766 even after restart. `netstat` shows dozens of `FIN_WAIT_2` and `CLOSE_WAIT` connections on that port from a PID that doesn't match the current Isaac Sim.

**Root Cause:** Previous Isaac Sim process wasn't fully killed — the old socket server thread kept the port open.

**Resolution:** Found the PID via `netstat -ano | findstr :8766` and killed it:
```powershell
taskkill /PID 6464 /F
```

**Lesson:** Always verify the old process is fully dead before restarting. Check Task Manager for lingering `kit.exe` or `isaac-sim` processes.

### Issue 5: `run_coroutine` Never Executing (Original Extension Bug)

**Symptom:** TCP connection succeeds, data is sent, but server never responds. Connection closes after ~30 seconds with 0 bytes received.

**Root Cause:** The original `_handle_client` dispatched commands via `run_coroutine(execute_wrapper())` which schedules an async coroutine on Kit's main event loop. The response was sent from inside the coroutine. But the coroutine either never ran or ran after the client handler had already moved on. The client thread looped back to `recv()` while the async coroutine was still queued.

**Resolution:** Replaced the fire-and-forget pattern with a `threading.Event` synchronization:

```python
# OLD (broken):
async def execute_wrapper():
    response = self.execute_command(command)
    client.sendall(json.dumps(response).encode('utf-8'))  # sent from async
run_coroutine(execute_wrapper())
# Handler immediately loops back to recv() — response never sent

# NEW (working):
import threading as _threading
response_holder = [None]
done_event = _threading.Event()

async def execute_wrapper():
    try:
        response = await self.execute_command_async(command)
        response_holder[0] = response
    finally:
        done_event.set()  # Signal completion

run_coroutine(execute_wrapper())

# Block client thread until async coroutine completes (up to 5 min)
if done_event.wait(timeout=300.0):
    response = response_holder[0]
else:
    response = {"status": "error", "message": "Execution timed out"}

# Send response from the client thread (not from async)
client.sendall(json.dumps(response).encode('utf-8'))
```

Also increased recv buffer from 16384 to 65536 bytes to handle large scripts.

### Issue 6: `self.stop()` → `self._stop()` Bug

**Symptom:** `AttributeError: 'MCPExtension' object has no attribute 'stop'` on startup failure.

**Root Cause:** The error handler in `_start()` called `self.stop()` but the method is named `self._stop()`.

**Resolution:** Changed line 130:
```python
# Before:
self.stop()
# After:
self._stop()
```

### Issue 7: `execute_script` Return Values Not Propagated

**Symptom:** Scripts that set `result = 42` returned `null` for the result field.

**Root Cause:** Line 355 in extension.py had `result = None` instead of reading from the exec'd namespace. The correct line was commented out.

**Resolution:**
```python
# Before:
# result = local_ns.get("result", None)
result = None

# After:
result = local_ns.get("result", None)
```

### Issue 8: USD Xform API Errors

**Symptom:** `pxr.Tf.ErrorException: Empty typeName for xformOp:translate` when creating cameras.

**Root Cause:** Tried to set `prim.GetAttribute("xformOp:translate").Set(...)` on a freshly defined `UsdGeom.Camera` that doesn't have xform ops yet.

**Resolution:** Use `UsdGeom.Xformable` API to add ops:
```python
# WRONG:
cam.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(...))

# CORRECT:
xform = UsdGeom.Xformable(cam.GetPrim())
xform.AddTranslateOp().Set(Gf.Vec3d(...))
```

### Issue 9: Cube Xform Op Type Mismatch

**Symptom:** `xformOp orient has typeName 'quatd' which does not match the requested precision 'PrecisionFloat'`

**Root Cause:** Cubes already have `xformOp:orient` as `quatd` type. Calling `ClearXformOpOrder()` + `AddOrientOp()` creates a new op with float precision that conflicts with the existing attribute.

**Resolution:** Read and set existing xform attributes directly instead of clearing/re-adding:
```python
# WRONG:
xform.ClearXformOpOrder()
xform.AddOrientOp().Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

# CORRECT:
orient_attr = prim.GetAttribute("xformOp:orient")
if orient_attr and orient_attr.IsValid():
    orient_attr.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
```

### Issue 10: `UsdGeom.DistantLight` Doesn't Exist

**Symptom:** `module 'pxr.UsdGeom' has no attribute 'DistantLight'`

**Root Cause:** Light types are in `UsdLux`, not `UsdGeom`. In newer USD versions the light schema moved.

**Resolution:** Check `prim.GetTypeName()` string instead of `prim.IsA()`:
```python
# WRONG:
if prim.IsA(UsdGeom.DistantLight):

# CORRECT:
if prim.GetTypeName() in ("DistantLight", "DomeLight", "SphereLight", "RectLight", "DiskLight"):
```

### Issue 11: Missing `h5py` in Isaac Sim Python Environment

**Symptom:** `No module named 'h5py'` when running episode collection.

**Resolution:** Install via Isaac Sim's Script Editor:
```python
import omni.kit.pipapi
omni.kit.pipapi.install("h5py")
```

Or via pip directly:
```cmd
"C:\Users\user\AppData\Local\ov\data\Kit\Isaac-Sim Full\5.1\pip3-envs\default\Scripts\pip.exe" install h5py
```

### Issue 12: Synchronous `step()` Call Inside Kit

**Symptom:** `Synchronous call to 'step' can only be performed in a standalone workflow and may not be made from within Kit.`

**Root Cause:** Called `app.update()`, `world.step()`, `SimulationContext.step()`, or `rep.orchestrator.step()` synchronously from within Kit's own event loop (inside an async coroutine scheduled by `run_coroutine`).

**Resolution (multi-part):**

**Part A — Simulation stepping:** Use `await omni.kit.app.get_app().next_update_async()` instead of any sync step call:
```python
# WRONG (any of these):
app.update()
world.step(render=True)
sim.step(render=True)

# CORRECT (inside async function):
await omni.kit.app.get_app().next_update_async()
```

**Part B — Replicator capture:** Remove `rep.orchestrator.step()` from frame capture. When the sim is already playing and `next_update_async()` advances frames, annotators get fresh data automatically:
```python
# WRONG:
def capture_frame(self):
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
    data = annotator.get_data()

# CORRECT:
def capture_frame(self):
    # Called after next_update_async() — annotators already have fresh data
    data = annotator.get_data()
```

**Part C — Async script support in extension:** Added `execute_script_async()` method that `exec()`s the code and then awaits `async_main()` if the script defines one:
```python
async def execute_script_async(self, code: str):
    local_ns = {"omni": omni, "carb": carb, ...}
    exec(code, local_ns)

    # If script defined async_main, await it
    async_main = local_ns.get("async_main", None)
    if async_main is not None and asyncio.iscoroutinefunction(async_main):
        await async_main()

    result = local_ns.get("result", None)
    return {"status": "success", "result": result}
```

**Part D — Collector async pattern:** The collector's dispatch returns `"NEEDS_ASYNC"` for episode commands, which triggers an `async_main()` definition that the extension awaits:
```python
# In vla_collector.py entry point:
if _dispatch_result == "NEEDS_ASYNC":
    async def async_main():
        global result
        collector = _get_collector()
        metadata = await collector.run_episode_async()
        result = {"status": "success", "metadata": metadata}
```

---

## Extension Changes Summary (extension.py)

All changes to `C:\users\user\isaac-sim-mcp\isaac.sim.mcp_extension\isaac_sim_mcp_extension\extension.py`:

| Change | Line(s) | Before | After |
|--------|---------|--------|-------|
| Port | 86 | `8766` | `8767` |
| Host binding | 87 | `"localhost"` | `"0.0.0.0"` |
| Error handler | 130 | `self.stop()` | `self._stop()` |
| Recv buffer | 199 | `client.recv(16384)` | `client.recv(65536)` |
| Client handler | 211-246 | Fire-and-forget `run_coroutine` | `threading.Event` sync pattern |
| Async dispatch | 217 | `self.execute_command(command)` | `await self.execute_command_async(command)` |
| New method | 282-296 | N/A | `execute_command_async()` — routes execute_script to async path |
| New method | 385-423 | N/A | `execute_script_async()` — exec + await async_main |
| Result return | 367 | `result = None` | `result = local_ns.get("result", None)` |

---

## Network Configuration (Windows)

### Required Firewall Rule
```powershell
# Admin PowerShell:
netsh advfirewall firewall add rule name="Isaac Sim MCP 8767" dir=in action=allow protocol=TCP localport=8767
```

### Port Proxy (NOT needed with current setup, but documented for reference)
```powershell
# If extension MUST bind to localhost, use port proxy:
netsh interface portproxy add v4tov4 listenport=8767 listenaddress=0.0.0.0 connectport=8767 connectaddress=127.0.0.1

# Remove port proxy:
netsh interface portproxy delete v4tov4 listenport=8767 listenaddress=0.0.0.0

# WARNING: Port proxy binds 0.0.0.0:<port>, which conflicts if the extension
# also tries to bind to any address on the same port. Use a different port or
# ensure the extension binds to localhost only when using port proxy.
```

### Diagnostic Commands
```powershell
# What's using a port:
netstat -ano | findstr :8767

# All port proxies:
netsh interface portproxy show v4tov4

# Hyper-V reserved ports:
netsh interface ipv4 show excludedportrange protocol=tcp

# Kill a process:
taskkill /PID <PID> /F
```

From WSL2:
```bash
# Get Windows host IP:
ip route show default | awk '{print $3}'

# Quick port test:
python3 -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('192.168.64.1', 8767)); print('OK'); s.close()"
```

---

## HDF5 Dataset Format

Each episode is saved as `episode_XXXXXX.hdf5`:

```
episode_XXXXXX.hdf5
├── action                          (N, 9)       float32  — target joint positions (shifted by 1 frame)
├── language_instruction            string                 — natural language task description
├── observation/
│   ├── images/
│   │   ├── wrist_rgb               (N, 256, 256, 3)  uint8    — wrist camera RGB
│   │   └── overhead_rgb            (N, 256, 256, 3)  uint8    — overhead camera RGB
│   ├── depth/
│   │   ├── wrist_depth             (N, 256, 256)     float32  — wrist camera depth
│   │   └── overhead_depth          (N, 256, 256)     float32  — overhead camera depth
│   ├── state                       (N, 9)            float32  — joint positions (7 arm + 2 finger)
│   ├── ee_pose                     (N, 7)            float32  — end-effector [x,y,z,qw,qx,qy,qz]
│   ├── gripper_width               (N,)              float32  — finger joint sum
│   ├── joint_velocities            (N, 9)            float32  — joint velocities
│   └── timestamps                  (N,)              float64  — seconds since episode start
└── attrs:
    ├── episode_id                  int
    ├── task_type                   string ("pick_place" | "stacking")
    ├── target_object               string ("RedCube", "BlueCube", etc.)
    ├── language_instruction        string
    ├── success                     bool
    └── num_frames                  int
```

Images are gzip-compressed (level 4). Typical episode: ~51 frames, ~3.3MB.

---

## Usage

### Prerequisites
1. Isaac Sim running on Windows with the Franka Cortex scene loaded
2. MCP extension enabled (check for `"Isaac Sim MCP server started on 0.0.0.0:8767"`)
3. Simulation playing (press Play in Isaac Sim)
4. `h5py` installed in Isaac Sim's Python environment
5. Windows firewall allows port 8767

### Test Connection
```bash
source ~/envs/selfdev/bin/activate
cd /mnt/d/mg_ai_research/workspace/whatnot/vla_dataset
python test_connection.py
```

### Collect Dataset
```bash
# Small test:
python orchestrate_vla.py --episodes 5 --max-steps 100

# Full collection:
python orchestrate_vla.py --episodes 50 --max-steps 300

# Custom host/port:
python orchestrate_vla.py --episodes 50 --host 192.168.64.1 --port 8767
```

### Inspect Data
```python
import h5py
f = h5py.File("episode_000000.hdf5", "r")
print(f["observation/images/wrist_rgb"].shape)   # (N, 256, 256, 3)
print(f["language_instruction"][()])               # b"pick up the red cube..."
print(dict(f.attrs))                               # episode metadata
f.close()
```

---

## Current Issues and Progress

### What Works
- TCP socket bridge from WSL2 → Windows Isaac Sim (port 8767, `0.0.0.0` binding)
- `execute_script` with result return and async support (`async_main` pattern)
- Scene discovery: Franka at `/World/Franka`, 4 cubes at `/World/Obs/{Color}Cube`
- Camera creation: wrist camera on `panda_hand`, overhead camera above workspace
- Replicator render products and annotators (RGB + depth) attached to both cameras
- Cube randomization: non-overlapping positions within workspace bounds
- Light randomization: intensity variation ±20%
- HDF5 saving: correct LeRobot-compatible structure with gzip compression
- Orchestrator: auto Windows IP detection, reconnection, manifest generation
- 5/5 episodes collected successfully (~1.1s per episode, 51 frames each)

### Critical Issues Found After First Collection

**Issue A: Robot is NOT moving — flat joint trajectories**

All joint position and velocity traces are completely flat across all frames. The EE position doesn't change. The Cortex block-stacking behavior is running on its own physics callback (`_on_physics_step`), but our `await next_update_async()` loop is not capturing the robot's motion.

Evidence from visualization:
- Joint positions: constant values across all 51 frames per episode
- Joint velocities: ~0 across all frames
- EE position: flat at (0.39, 0.01, 0.44) — the home position
- Gripper width: ~2.5e-7 (essentially 0) — never opens or closes

Root cause candidates:
1. **`dynamic_control` interface not reading live physics state** — The `_init_articulation` method uses `omni.isaac.dynamic_control` to read DOF states, but this may return stale/default values when called from an async coroutine on Kit's event loop rather than from a physics callback.
2. **USD xform attributes are not updated during simulation** — The fallback path reads USD attributes like `state:angular:physics:position`, which may not reflect the live physics simulation state.
3. **Timing mismatch** — `next_update_async()` advances the frame, but the physics solver and Cortex behavior may have already run by the time we read joint states, or the DC interface may not be synced.

Potential fixes to investigate:
- Hook into the Cortex behavior's physics callback (`_on_physics_step`) to record data at each physics step
- Use `ArticulationView` from `omni.isaac.core` with an active physics sim view
- Read joint states from the Cortex robot object directly: `robot.arm.get_fk_p()`, `robot.arm.articulation_subset.get_joints_state()`
- Register a custom physics callback via `world.add_physics_callback()` that records state synchronously during physics stepping

**Issue B: Wrist camera is nearly black**

RGB values range [0, 1] or [0, 6] out of 255. The camera sees almost nothing useful.

Evidence:
- Row 1 of image grid: completely black frames across all timesteps
- Wrist depth shows valid range [0.035, 0.081] — the camera IS sensing something very close (3.5-8.1cm), likely the inside of the gripper

Root cause: The wrist camera is positioned at `Gf.Vec3d(0.0, 0.0, 0.06)` relative to `panda_hand` — this places it 6cm along the Z-axis of the hand frame, which points INTO the gripper fingers rather than outward toward the workspace.

Potential fixes:
- Rotate the camera to face downward/outward: add a `RotateXYZOp` to point the camera away from the hand
- Move the camera further out along a different axis (e.g., negative Z to look forward)
- Attach to `panda_link7` instead of `panda_hand` for a better mounting point
- Reference the Franka URDF to understand the hand frame orientation before choosing the offset

**Issue C: Overhead camera is very dark**

RGB values range [0, 202] but images appear very dark with low contrast. The depth channel works (shows scene structure via viridis colormap), but the RGB is barely visible.

Potential causes:
- Scene lighting may be insufficient for the overhead viewing angle
- Camera exposure/aperture settings may need adjustment
- The `focalLength=24.0` may not be optimal for the 1.2m height

**Issue D: Cubes are randomized but robot doesn't interact with them**

The `randomize_cubes()` function moves cubes via USD xform attributes, but this may not properly update the physics simulation. The Cortex behavior tracks cubes via `CortexObject.get_world_pose()` which reads from the physics engine, not from USD xforms. So the behavior may still see cubes at their original positions.

Potential fixes:
- Use `DynamicCuboid.set_world_pose()` or the physics API to move cubes
- Reset the Cortex behavior context after randomization so it re-discovers cube positions
- Or use `rigid_body.set_world_pose()` via the DC interface

**Issue E: State and Action data are identical**

Since joint positions don't change, `action[t] = state[t+1] = state[t]` — all action values equal state values. This makes the dataset useless for training a policy. This resolves automatically once Issue A is fixed.

### Summary: What Must Be Fixed Before Useful Data Collection

| Priority | Issue | Impact |
|----------|-------|--------|
| P0 | Robot state not captured during motion | No action/state signal at all |
| P0 | Cubes not moved in physics engine | Behavior doesn't see randomized positions |
| P1 | Wrist camera orientation | No useful wrist images |
| P1 | Overhead camera brightness | Images too dark for visual learning |
| P2 | Episode success heuristic | Always marks "success" since velocities start at 0 |

---

## Known Limitations & Future Work

1. **Single-socket connection** — The extension only accepts one client at a time (`socket.listen(1)`). Multiple orchestrators cannot run simultaneously.

2. **No task-aligned behavior** — The Cortex block-stacking behavior always stacks in a fixed order (Blue, Yellow, Green, Red). The randomly generated language instructions (pick-place, stack) don't control what the robot actually does. Future work: modify the decider network context to accept task parameters, or create custom behaviors per task.

3. **Episode termination** — Currently uses max_steps or velocity-near-zero heuristic. Should instead detect task completion (cube at target position, stack formed, etc.).

4. **Data rate** — 51 frames at ~1.1s means the sim runs at ~46 FPS for data collection. For longer episodes with actual manipulation, expect 5-30 seconds per episode depending on task complexity.

---

## Quick Troubleshooting Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Connection refused` from WSL2 | Extension not running, wrong port, or firewall | Check Isaac Sim console for startup message; verify port; check firewall |
| `WinError 10013` on extension start | Port already in use (port proxy, old process) | `netstat -ano \| findstr :<port>`, kill conflicting process, remove port proxy |
| Connection closes with 0 bytes | `run_coroutine` not executing | Ensure `threading.Event` sync pattern is in `_handle_client` |
| `Empty typeName for xformOp` | Wrong xform API for setting transforms | Use `UsdGeom.Xformable.AddTranslateOp()` not raw attribute set |
| `Synchronous call to step` | Sync sim step inside Kit async | Use `await next_update_async()` in `async_main()` |
| Changes to extension have no effect | Editing WSL2 copy, not Windows copy | Copy to `/mnt/c/users/user/isaac-sim-mcp/...`; restart Isaac Sim |
| `No module named 'h5py'` | Missing pip package in Isaac Sim env | `omni.kit.pipapi.install("h5py")` in Script Editor |
