#!/usr/bin/env python3
"""
Mixture-of-Experts (MoE) Layer Computation Example using NiXL

This example demonstrates:
1. Two experts (E₁ and E₂) with weight matrices 2048x512 and 512x2048
2. GeLU activation function
3. Cross-GPU token transfer using NiXL
4. Complete MoE computation workflow

Based on the technical specifications:
- Expert structure: Two weight matrices (2048x512, 512x2048)
- Activation: GeLU
- Experts placed on different GPUs (E₁ on GPU₁, E₂ on GPU₂)
- Token transfer from GPU₁ to GPU₂ for computation
- Result transfer back to GPU₁
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


class Expert:
    """Expert implementation with two weight matrices and GeLU activation"""
    
    def __init__(self, input_dim=512, hidden_dim=2048, device="cuda:0"):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # First weight matrix: hidden_dim x input_dim (2048 x 512) - for F.linear
        self.weight1 = torch.randn(hidden_dim, input_dim, device=device)
        # Second weight matrix: input_dim x hidden_dim (512 x 2048) - for F.linear
        self.weight2 = torch.randn(input_dim, hidden_dim, device=device)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
    
    def forward(self, x):
        """Forward pass through the expert"""
        # First linear transformation: x @ weight1
        hidden = F.linear(x, self.weight1)
        # GeLU activation
        hidden = F.gelu(hidden)
        # Second linear transformation: hidden @ weight2
        output = F.linear(hidden, self.weight2)
        return output


class MoELayer:
    """Mixture-of-Experts layer with cross-GPU computation"""
    
    def __init__(self, num_experts=2, input_dim=512, hidden_dim=2048):
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create experts on different GPUs
        self.experts = []
        for i in range(num_experts):
            device = f"cuda:{i}"
            expert = Expert(input_dim, hidden_dim, device)
            self.experts.append(expert)
        
        # Initialize NiXL agents
        self._init_nixl_agents()
    
    def _init_nixl_agents(self):
        """Initialize NiXL agents for cross-GPU communication"""
        print("Using NIXL Plugins from:")
        print(os.environ.get("NIXL_PLUGIN_DIR", "Not set"))
        
        # Configure NiXL agents
        agent_config = nixl_agent_config(backends=["UCX"])
        self.dst_nixl_agent = nixl_agent("target", agent_config)
        self.src_nixl_agent = nixl_agent("initiator", None)
        
        print("NiXL agents initialized successfully")
    
    def _transfer_token_to_gpu(self, token, source_gpu, target_gpu):
        """Transfer token from source GPU to target GPU using NiXL"""
        print(f"Transferring token from {source_gpu} to {target_gpu}")
        
        # Create tensors on respective GPUs
        src_tensor = token.to(source_gpu)
        dst_tensor = torch.zeros_like(token, device=target_gpu)
        
        # Register memory with NiXL
        self.src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
        self.dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
        
        # Get transfer descriptors
        src_xfer_descs = self.src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
        dst_xfer_descs = self.dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
        
        # Exchange metadata
        meta = self.dst_nixl_agent.get_agent_metadata()
        remote_name = self.src_nixl_agent.add_remote_agent(meta)
        print(f"Loaded name from metadata: {remote_name}")
        
        # Perform transfer
        notif = b"MOE_TOKEN_TRANSFER"
        xfer_handle = self.src_nixl_agent.initialize_xfer(
            "READ",
            src_xfer_descs,
            dst_xfer_descs,
            remote_name,
            notif,
        )
        
        if not xfer_handle:
            print("Creating transfer failed.")
            return None
        
        state = self.src_nixl_agent.transfer(xfer_handle)
        assert state != "ERR"
        
        # Wait for transfer completion
        target_done = False
        init_done = False
        
        while (not init_done) or (not target_done):
            if not init_done:
                state = self.src_nixl_agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    print("Transfer got to Error state.")
                    return None
                elif state == "DONE":
                    init_done = True
                    print("Initiator done")
            
            if not target_done:
                if self.dst_nixl_agent.check_remote_xfer_done("initiator", notif):
                    target_done = True
                    print("Target done")
        
        return dst_tensor
    
    def _transfer_result_back(self, result, source_gpu, target_gpu):
        """Transfer computation result back to original GPU"""
        print(f"Transferring result from {source_gpu} back to {target_gpu}")
        
        # Create tensors on respective GPUs
        src_tensor = result.to(source_gpu)
        dst_tensor = torch.zeros_like(result, device=target_gpu)
        
        # Register memory with NiXL
        self.src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
        self.dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
        
        # Get transfer descriptors
        src_xfer_descs = self.src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
        dst_xfer_descs = self.dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
        
        # Exchange metadata
        meta = self.dst_nixl_agent.get_agent_metadata()
        remote_name = self.src_nixl_agent.add_remote_agent(meta)
        
        # Perform transfer
        notif = b"MOE_RESULT_TRANSFER"
        xfer_handle = self.src_nixl_agent.initialize_xfer(
            "READ",
            src_xfer_descs,
            dst_xfer_descs,
            remote_name,
            notif,
        )
        
        if not xfer_handle:
            print("Creating result transfer failed.")
            return None
        
        state = self.src_nixl_agent.transfer(xfer_handle)
        assert state != "ERR"
        
        # Wait for transfer completion
        target_done = False
        init_done = False
        
        while (not init_done) or (not target_done):
            if not init_done:
                state = self.src_nixl_agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    print("Result transfer got to Error state.")
                    return None
                elif state == "DONE":
                    init_done = True
                    print("Result transfer initiator done")
            
            if not target_done:
                if self.dst_nixl_agent.check_remote_xfer_done("initiator", notif):
                    target_done = True
                    print("Result transfer target done")
        
        return dst_tensor
    
    def forward(self, token, expert_idx=1):
        """
        Perform MoE layer computation
        
        Args:
            token: Input token tensor
            expert_idx: Index of expert to use (0 for E₁, 1 for E₂)
        
        Returns:
            Computed result tensor
        """
        print(f"\n=== MoE Layer Computation ===")
        print(f"Input token shape: {token.shape}")
        print(f"Token device: {token.device}")
        print(f"Using Expert E{expert_idx + 1}")
        
        # Determine source and target GPUs
        source_gpu = token.device
        target_gpu = f"cuda:{expert_idx}"
        
        print(f"Token is on {source_gpu} but needs to be fed to Expert E{expert_idx + 1} on {target_gpu}")
        
        # Step 1: Transfer token from source GPU to target GPU
        transferred_token = self._transfer_token_to_gpu(token, source_gpu, target_gpu)
        if transferred_token is None:
            print("Token transfer failed")
            return None
        
        print(f"Token transferred to {target_gpu}")
        print(f"Transferred token shape: {transferred_token.shape}")
        
        # Step 2: Perform expert computation
        print(f"Performing Expert E{expert_idx + 1} computation on {target_gpu}")
        expert = self.experts[expert_idx]
        result = expert.forward(transferred_token)
        
        print(f"Expert computation completed")
        print(f"Result shape: {result.shape}")
        print(f"Result device: {result.device}")
        
        # Step 3: Transfer result back to original GPU
        print(f"Returning result from {target_gpu} back to {source_gpu}")
        final_result = self._transfer_result_back(result, target_gpu, source_gpu)
        
        if final_result is None:
            print("Result transfer failed")
            return None
        
        print(f"MoE computation completed successfully")
        print(f"Final result shape: {final_result.shape}")
        print(f"Final result device: {final_result.device}")
        
        return final_result


def main():
    """Main function demonstrating MoE layer computation"""
    print("=== Mixture-of-Experts (MoE) Layer Computation Example ===")
    print("Based on technical specifications:")
    print("- Expert structure: Two weight matrices (2048x512, 512x2048)")
    print("- Activation function: GeLU")
    print("- Experts: E₁ on GPU₁, E₂ on GPU₂")
    print("- Cross-GPU token transfer using NiXL")
    print()
    
    # Check available GPUs
    if torch.cuda.device_count() < 2:
        print("Warning: Need at least 2 GPUs for this example")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        return
    
    # Create MoE layer
    moe_layer = MoELayer(num_experts=2, input_dim=512, hidden_dim=2048)
    
    # Create sample token (batch_size=1, sequence_length=1, hidden_dim=512)
    token = torch.randn(1, 1, 512, device="cuda:0")
    print(f"Created sample token on cuda:0")
    print(f"Token shape: {token.shape}")
    print(f"Token device: {token.device}")
    
    # Perform MoE computation with Expert E₂ (index 1)
    result = moe_layer.forward(token, expert_idx=1)
    
    if result is not None:
        print(f"\n=== Final Results ===")
        print(f"Input token shape: {token.shape}")
        print(f"Output result shape: {result.shape}")
        print(f"Input device: {token.device}")
        print(f"Output device: {result.device}")
        
        # Verify the computation
        print(f"\n=== Verification ===")
        print(f"Input token sum: {token.sum().item():.4f}")
        print(f"Output result sum: {result.sum().item():.4f}")
        print(f"Input token mean: {token.mean().item():.4f}")
        print(f"Output result mean: {result.mean().item():.4f}")
        
        print("\nMoE layer computation completed successfully!")
    else:
        print("MoE computation failed")


if __name__ == "__main__":
    main()
