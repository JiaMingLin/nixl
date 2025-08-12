import os

import numpy as np
import torch

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

buf_size = 256000
# Allocate memory and register with NIXL

print("Using NIXL Plugins from:")
print(os.environ["NIXL_PLUGIN_DIR"])

# Example using nixl_agent_config
agent_config = nixl_agent_config(backends=["UCX"])
dst_nixl_agent = nixl_agent("target", agent_config)  # agent 1
src_nixl_agent = nixl_agent("initiator", None)

# register memory
import torch
from torch import tensor

# size = 256*256 tensor on cuda:0
src_tensor = torch.ones(256, 256, device="cuda:0") 

# size = 256*256 tensor on cuda:1
dst_tensor = torch.zeros(256, 256, device="cuda:1") + 5

print("src_tensor: ", src_tensor)
print("dst_tensor: ", dst_tensor)

src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True) # agent 1

src_xfer_descs = src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
dst_xfer_descs = dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True) # agent 1

# Exchange metadata
meta = dst_nixl_agent.get_agent_metadata()
remote_name = src_nixl_agent.add_remote_agent(meta)
print("Loaded name from metadata:", remote_name, flush=True) 

# transfer data
notif = b"UUID1"
xfer_handle = src_nixl_agent.initialize_xfer(
    "READ",
    src_xfer_descs,
    dst_xfer_descs,
    remote_name,
    notif,
)
if not xfer_handle:
    print("Creating transfer failed.")
    exit()

state = src_nixl_agent.transfer(xfer_handle)
assert state != "ERR"

target_done = False
init_done = False

while (not init_done) or (not target_done):
    if not init_done:
        state = src_nixl_agent.check_xfer_state(xfer_handle)
        if state == "ERR":
            print("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            init_done = True
            print("Initiator done")

    if not target_done:
        if dst_nixl_agent.check_remote_xfer_done("initiator", notif):
            target_done = True
            print("Target done")

print("src_tensor: ", src_tensor)
print("dst_tensor: ", dst_tensor)