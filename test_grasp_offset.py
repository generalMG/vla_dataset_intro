import json
import socket

def send_script(code: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 8767))
    payload = {"type": "execute_script", "params": {"code": code}}
    sock.sendall(json.dumps(payload).encode("utf-8"))
    
    chunks = []
    while True:
        chunk = sock.recv(65536)
        if not chunk: break
        chunks.append(chunk)
        try:
            data = b"".join(chunks)
            res = json.loads(data.decode("utf-8"))
            return res
        except:
            pass
    return None

code = """
import numpy as np
from pxr import UsdGeom, Usd
stage = omni.usd.get_context().get_stage()

def get_pos(path):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(xf.ExtractTranslation())

hand = get_pos('/World/Franka/panda_hand')
rf = get_pos('/World/Franka/panda_rightfinger')
lf = get_pos('/World/Franka/panda_leftfinger')

result = {
    'hand': hand.tolist(),
    'rf': rf.tolist(),
    'lf': lf.tolist(),
    'offset_rf': (rf - hand).tolist(),
    'offset_lf': (lf - hand).tolist()
}
"""
print(send_script(code))
