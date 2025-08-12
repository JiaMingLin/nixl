#!/usr/bin/env python3
"""
MoE Profiling Example for Nsight Systems/Nsight Compute

這個腳本專門設計用於 Nsight profiling，包含：
1. GPU 計算 profiling
2. GPU 間數據傳輸 profiling  
3. GPU-CPU 數據傳輸 profiling
4. 詳細的 profiling markers 和 annotations

使用方式：
1. Nsight Systems: nsys profile python moe_profiling_example.py
2. Nsight Compute: ncu python moe_profiling_example.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.profiler
import numpy as np

# 添加當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moe_simple_example import Expert, init_nixl_agents, transfer_tensor

# 檢查是否有 NVTX (NVIDIA Tools Extension) 支援
try:
    import nvtx
    NVTX_AVAILABLE = True
    print("✓ NVTX 可用於 profiling annotations")
except ImportError:
    NVTX_AVAILABLE = False
    print("⚠️  NVTX 不可用，將使用基本的 profiling")


def nvtx_range(name):
    """NVTX 範圍裝飾器，如果 NVTX 可用則使用，否則使用空操作"""
    def decorator(func):
        if NVTX_AVAILABLE:
            def wrapper(*args, **kwargs):
                nvtx.range_push(name)
                try:
                    result = func(*args, **kwargs)
                finally:
                    nvtx.range_pop()
                return result
            return wrapper
        else:
            return func
    return decorator


class ProfilingExpert(Expert):
    """帶有詳細 profiling 標記的專家實現"""
    
    @nvtx_range("Expert_Forward")
    def forward(self, x):
        """前向傳播，包含詳細的 profiling 標記"""
        
        # 標記第一個線性變換
        if NVTX_AVAILABLE:
            nvtx.range_push("Linear1_Computation")
        
        hidden = F.linear(x, self.weight1)
        torch.cuda.synchronize()  # 確保計算完成
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        # 標記 GeLU 激活
        if NVTX_AVAILABLE:
            nvtx.range_push("GeLU_Activation")
        
        hidden = F.gelu(hidden)
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        # 標記第二個線性變換
        if NVTX_AVAILABLE:
            nvtx.range_push("Linear2_Computation")
        
        output = F.linear(hidden, self.weight2)
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        return output


@nvtx_range("GPU_Memory_Operations")
def profile_gpu_memory_operations():
    """測試和 profile GPU 記憶體操作"""
    print("\n=== GPU 記憶體操作 Profiling ===")
    
    # 測試不同大小的記憶體分配
    sizes = [
        (256, 256),      # 小張量
        (1024, 1024),    # 中等張量
        (2048, 2048),    # 大張量
    ]
    
    for size in sizes:
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Memory_Alloc_{size[0]}x{size[1]}")
        
        # GPU 記憶體分配
        tensor = torch.randn(size, device="cuda:0")
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        print(f"分配 GPU 張量：{size}")


@nvtx_range("GPU_to_GPU_Transfer")
def profile_gpu_to_gpu_transfer():
    """測試和 profile GPU 間數據傳輸"""
    print("\n=== GPU 間數據傳輸 Profiling ===")
    
    if torch.cuda.device_count() < 2:
        print("需要至少 2 個 GPU 來測試 GPU 間傳輸")
        return
    
    # 初始化 NiXL agents
    src_agent, dst_agent = init_nixl_agents()
    
    # 測試不同大小的張量傳輸
    transfer_sizes = [
        (512, 512),      # 1MB
        (1024, 1024),    # 4MB  
        (2048, 2048),    # 16MB
    ]
    
    for size in transfer_sizes:
        if NVTX_AVAILABLE:
            nvtx.range_push(f"GPU_Transfer_{size[0]}x{size[1]}")
        
        # 創建源張量和目標張量
        src_tensor = torch.randn(size, device="cuda:0")
        dst_tensor = torch.zeros(size, device="cuda:1")
        
        # 執行 NiXL 傳輸
        success = transfer_tensor(
            src_agent, dst_agent,
            src_tensor, dst_tensor,
            f"PROFILE_TRANSFER_{size[0]}x{size[1]}"
        )
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        print(f"GPU 間傳輸 {size}: {'成功' if success else '失敗'}")
        
        # 同時測試標準 PyTorch 的 GPU 間複製
        if NVTX_AVAILABLE:
            nvtx.range_push(f"PyTorch_GPU_Copy_{size[0]}x{size[1]}")
        
        pytorch_dst = src_tensor.to("cuda:1")
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()


@nvtx_range("GPU_to_CPU_Transfer")
def profile_gpu_to_cpu_transfer():
    """測試和 profile GPU-CPU 數據傳輸"""
    print("\n=== GPU-CPU 數據傳輸 Profiling ===")
    
    # 測試不同大小的 GPU 到 CPU 傳輸
    transfer_sizes = [
        (256, 256),      # 256KB
        (1024, 1024),    # 4MB
        (2048, 2048),    # 16MB
    ]
    
    for size in transfer_sizes:
        # GPU 到 CPU
        if NVTX_AVAILABLE:
            nvtx.range_push(f"GPU_to_CPU_{size[0]}x{size[1]}")
        
        gpu_tensor = torch.randn(size, device="cuda:0")
        cpu_tensor = gpu_tensor.cpu()
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        # CPU 到 GPU
        if NVTX_AVAILABLE:
            nvtx.range_push(f"CPU_to_GPU_{size[0]}x{size[1]}")
        
        gpu_tensor_back = cpu_tensor.cuda()
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        print(f"GPU-CPU 傳輸 {size}: 完成")


@nvtx_range("Expert_Computation_Profiling")
def profile_expert_computation():
    """測試和 profile 專家計算"""
    print("\n=== 專家計算 Profiling ===")
    
    # 創建專家
    expert_gpu0 = ProfilingExpert(input_dim=512, hidden_dim=2048, device="cuda:0")
    
    if torch.cuda.device_count() >= 2:
        expert_gpu1 = ProfilingExpert(input_dim=512, hidden_dim=2048, device="cuda:1")
    
    # 測試不同批次大小的計算
    batch_sizes = [1, 8, 32, 128]
    sequence_length = 1
    hidden_dim = 512
    
    for batch_size in batch_sizes:
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Expert_Computation_Batch_{batch_size}")
        
        # 創建輸入張量
        input_tensor = torch.randn(batch_size, sequence_length, hidden_dim, device="cuda:0")
        
        # GPU 0 上的專家計算
        if NVTX_AVAILABLE:
            nvtx.range_push(f"Expert_GPU0_Batch_{batch_size}")
        
        output_gpu0 = expert_gpu0.forward(input_tensor)
        torch.cuda.synchronize()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        # 如果有第二個 GPU，在 GPU 1 上進行計算
        if torch.cuda.device_count() >= 2:
            if NVTX_AVAILABLE:
                nvtx.range_push(f"Expert_GPU1_Batch_{batch_size}")
            
            input_gpu1 = input_tensor.to("cuda:1")
            output_gpu1 = expert_gpu1.forward(input_gpu1)
            torch.cuda.synchronize()
            
            if NVTX_AVAILABLE:
                nvtx.range_pop()
        
        if NVTX_AVAILABLE:
            nvtx.range_pop()
        
        print(f"專家計算 (batch_size={batch_size}): 完成")


@nvtx_range("Complete_MoE_Workflow")
def profile_complete_moe_workflow():
    """測試和 profile 完整的 MoE 工作流程"""
    print("\n=== 完整 MoE 工作流程 Profiling ===")
    
    if torch.cuda.device_count() < 2:
        print("需要至少 2 個 GPU 來執行完整的 MoE 工作流程")
        return
    
    # 初始化組件
    src_agent, dst_agent = init_nixl_agents()
    expert_e1 = ProfilingExpert(input_dim=512, hidden_dim=2048, device="cuda:0")
    expert_e2 = ProfilingExpert(input_dim=512, hidden_dim=2048, device="cuda:1")
    
    # 創建輸入 token
    token = torch.randn(1, 1, 512, device="cuda:0")
    
    # 步驟 1: Token 傳輸到 GPU 1
    if NVTX_AVAILABLE:
        nvtx.range_push("MoE_Token_Transfer")
    
    dst_tensor = torch.zeros_like(token, device="cuda:1")
    success = transfer_tensor(
        src_agent, dst_agent,
        token, dst_tensor,
        "MOE_WORKFLOW_TOKEN"
    )
    
    if NVTX_AVAILABLE:
        nvtx.range_pop()
    
    if not success:
        print("Token 傳輸失敗")
        return
    
    # 步驟 2: 在 GPU 1 執行專家計算
    if NVTX_AVAILABLE:
        nvtx.range_push("MoE_Expert_Computation")
    
    result = expert_e2.forward(dst_tensor)
    
    if NVTX_AVAILABLE:
        nvtx.range_pop()
    
    # 步驟 3: 結果傳輸回 GPU 0
    if NVTX_AVAILABLE:
        nvtx.range_push("MoE_Result_Transfer")
    
    final_tensor = torch.zeros_like(result, device="cuda:0")
    success = transfer_tensor(
        src_agent, dst_agent,
        result, final_tensor,
        "MOE_WORKFLOW_RESULT"
    )
    
    if NVTX_AVAILABLE:
        nvtx.range_pop()
    
    if success:
        print("完整 MoE 工作流程: 成功")
    else:
        print("完整 MoE 工作流程: 失敗")


def main():
    """主函數，執行所有 profiling 測試"""
    print("=== MoE Profiling Example ===")
    print("專為 Nsight Systems/Compute profiling 設計")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print()
    
    # 熱身運行（避免初始化開銷影響 profiling）
    if NVTX_AVAILABLE:
        nvtx.range_push("Warmup")
    
    warmup_tensor = torch.randn(100, 100, device="cuda:0")
    warmup_result = torch.mm(warmup_tensor, warmup_tensor)
    torch.cuda.synchronize()
    
    if NVTX_AVAILABLE:
        nvtx.range_pop()
    
    print("熱身完成\n")
    
    # 執行各種 profiling 測試
    try:
        profile_gpu_memory_operations()
        profile_gpu_to_cpu_transfer()
        profile_expert_computation()
        
        if torch.cuda.device_count() >= 2:
            profile_gpu_to_gpu_transfer()
            profile_complete_moe_workflow()
        else:
            print("\n⚠️  跳過 GPU 間傳輸測試（需要至少 2 個 GPU）")
        
        print("\n🎉 所有 profiling 測試完成！")
        
    except Exception as e:
        print(f"\n❌ Profiling 測試過程中發生錯誤: {e}")


if __name__ == "__main__":
    main()
