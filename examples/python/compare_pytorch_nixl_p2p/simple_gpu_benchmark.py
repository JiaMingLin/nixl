#!/usr/bin/env python3
"""
簡化版 NIXL GPU-to-GPU 傳輸效能測試

這是一個簡化版本的效能測試工具，專注於核心功能測試。
適合快速驗證 NIXL 的傳輸效能。
"""

import os
import time
import torch
import statistics
from typing import Tuple

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


def test_gpu_transfer(tensor_size: Tuple[int, int] = (1024, 1024), 
                     dtype: torch.dtype = torch.float32,
                     iterations: int = 5) -> dict:
    """
    執行單一 GPU-to-GPU 傳輸測試
    
    Args:
        tensor_size: 張量大小 (height, width)
        dtype: 資料型別
        iterations: 測試迭代次數
    
    Returns:
        包含測試結果的字典
    """
    print(f"測試張量大小: {tensor_size}, 型別: {dtype}")
    
    # 檢查 GPU 可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    
    gpu_count = torch.cuda.device_count()
    src_device = "cuda:0"
    dst_device = "cuda:1" if gpu_count > 1 else "cuda:0"
    
    print(f"來源裝置: {src_device}, 目標裝置: {dst_device}")
    
    # 初始化 NIXL agents
    print("初始化 NIXL agents...")
    agent_config = nixl_agent_config(backends=["UCX"])
    dst_nixl_agent = nixl_agent("target", agent_config)
    src_nixl_agent = nixl_agent("initiator", None)
    
    # 創建測試張量
    src_tensor = torch.randn(tensor_size, dtype=dtype, device=src_device)
    dst_tensor = torch.zeros(tensor_size, dtype=dtype, device=dst_device)
    
    data_size = src_tensor.numel() * src_tensor.element_size()
    print(f"資料大小: {data_size / (1024**2):.2f} MB")
    
    # 註冊記憶體
    src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
    dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
    
    # 獲取傳輸描述符
    src_xfer_descs = src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
    dst_xfer_descs = dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
    
    # 設定遠端連接
    meta = dst_nixl_agent.get_agent_metadata()
    remote_name = src_nixl_agent.add_remote_agent(meta)
    
    transfer_times = []
    
    print(f"執行 {iterations} 次傳輸測試...")
    
    for i in range(iterations):
        # 重置目標張量
        dst_tensor.zero_()
        
        # 執行傳輸
        notif = f"test_{i}".encode()
        
        xfer_handle = src_nixl_agent.initialize_xfer(
            "READ",
            src_xfer_descs,
            dst_xfer_descs,
            remote_name,
            notif,
        )
        
        if not xfer_handle:
            print(f"迭代 {i}: 創建傳輸失敗")
            continue
        
        # 開始計時
        start_time = time.perf_counter()
        
        state = src_nixl_agent.transfer(xfer_handle)
        if state == "ERR":
            print(f"迭代 {i}: 傳輸初始化失敗")
            continue
        
        # 等待完成
        target_done = False
        init_done = False
        
        while (not init_done) or (not target_done):
            if not init_done:
                state = src_nixl_agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    print(f"迭代 {i}: 傳輸錯誤")
                    break
                elif state == "DONE":
                    init_done = True
            
            if not target_done:
                if dst_nixl_agent.check_remote_xfer_done("initiator", notif):
                    target_done = True
        
        end_time = time.perf_counter()
        
        if init_done and target_done:
            transfer_time = end_time - start_time
            transfer_times.append(transfer_time)
            print(f"  迭代 {i+1}: {transfer_time*1000:.2f} ms")
    
    # 驗證傳輸正確性
    verification_passed = torch.allclose(src_tensor.cpu(), dst_tensor.cpu(), rtol=1e-5)
    print(f"傳輸驗證: {'通過' if verification_passed else '失敗'}")
    
    # 計算統計數據
    if transfer_times:
        avg_time = statistics.mean(transfer_times)
        min_time = min(transfer_times)
        max_time = max(transfer_times)
        avg_throughput = (data_size / (1024**2)) / avg_time
        max_throughput = (data_size / (1024**2)) / min_time
        
        results = {
            'tensor_size': tensor_size,
            'dtype': str(dtype),
            'data_size_mb': data_size / (1024**2),
            'successful_iterations': len(transfer_times),
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'avg_throughput_mbps': avg_throughput,
            'max_throughput_mbps': max_throughput,
            'verification_passed': verification_passed
        }
        
        return results
    else:
        return {'error': '沒有成功的傳輸'}


def run_benchmark_suite():
    """執行一系列基準測試"""
    print("=" * 60)
    print("NIXL GPU-to-GPU 簡化效能測試")
    print("=" * 60)
    
    # 測試案例
    test_cases = [
        ((256, 256), torch.float32),
        ((512, 512), torch.float32),
        ((1024, 1024), torch.float32),
        ((2048, 2048), torch.float32),
        ((1024, 1024), torch.float16),
        ((1024, 1024), torch.int32),
    ]
    
    all_results = []
    
    for tensor_size, dtype in test_cases:
        try:
            result = test_gpu_transfer(tensor_size, dtype, iterations=5)
            all_results.append(result)
            
            if 'error' not in result:
                print(f"平均時間: {result['avg_time_ms']:.2f} ms")
                print(f"平均吞吐量: {result['avg_throughput_mbps']:.2f} MB/s")
                print(f"最大吞吐量: {result['max_throughput_mbps']:.2f} MB/s")
            else:
                print(f"測試失敗: {result['error']}")
                
        except Exception as e:
            print(f"測試出現異常: {e}")
            all_results.append({'error': str(e)})
        
        print("-" * 40)
    
    # 顯示摘要
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        print("\n測試摘要:")
        print(f"成功測試: {len(successful_results)}/{len(all_results)}")
        
        best_throughput = max(successful_results, key=lambda x: x['avg_throughput_mbps'])
        fastest_transfer = min(successful_results, key=lambda x: x['avg_time_ms'])
        
        print(f"最佳吞吐量: {best_throughput['avg_throughput_mbps']:.2f} MB/s")
        print(f"  張量大小: {best_throughput['tensor_size']}")
        print(f"  資料型別: {best_throughput['dtype']}")
        
        print(f"最快傳輸: {fastest_transfer['avg_time_ms']:.2f} ms")
        print(f"  張量大小: {fastest_transfer['tensor_size']}")
        print(f"  資料型別: {fastest_transfer['dtype']}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # 單一測試模式
        result = test_gpu_transfer((1024, 1024), torch.float32, 3)
        print("\n測試結果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        # 完整測試套件
        run_benchmark_suite()
