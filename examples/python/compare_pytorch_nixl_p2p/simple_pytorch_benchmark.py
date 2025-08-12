#!/usr/bin/env python3
"""
簡化版 PyTorch GPU-to-GPU 傳輸效能測試

這是一個簡化版本的 PyTorch GPU 傳輸測試工具，
專注於核心功能測試和與 NIXL 結果的直接比較。
"""

import time
import torch
import statistics
from typing import Tuple, Dict


def test_pytorch_transfer(tensor_size: Tuple[int, int] = (1024, 1024), 
                         dtype: torch.dtype = torch.float32,
                         method: str = "copy_",
                         iterations: int = 5) -> Dict:
    """
    執行單一 PyTorch GPU-to-GPU 傳輸測試
    
    Args:
        tensor_size: 張量大小 (height, width)
        dtype: 資料型別
        method: 傳輸方法 ("copy_", "to", "clone_to")
        iterations: 測試迭代次數
    
    Returns:
        包含測試結果的字典
    """
    print(f"測試張量大小: {tensor_size}, 型別: {dtype}, 方法: {method}")
    
    # 檢查 GPU 可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    
    gpu_count = torch.cuda.device_count()
    src_device = "cuda:0"
    dst_device = "cuda:1" if gpu_count > 1 else "cuda:0"
    
    print(f"來源裝置: {src_device}, 目標裝置: {dst_device}")
    
    # 創建測試張量
    src_tensor = torch.randn(tensor_size, dtype=dtype, device=src_device)
    dst_tensor = torch.zeros(tensor_size, dtype=dtype, device=dst_device)
    
    data_size = src_tensor.numel() * src_tensor.element_size()
    print(f"資料大小: {data_size / (1024**2):.2f} MB")
    
    transfer_times = []
    
    print(f"執行 {iterations} 次傳輸測試...")
    
    for i in range(iterations):
        # 重置目標張量
        dst_tensor.zero_()
        
        # 選擇傳輸方法並執行
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        if method == "copy_":
            dst_tensor.copy_(src_tensor)
        elif method == "to":
            result = src_tensor.to(dst_device)
            dst_tensor.copy_(result)
        elif method == "clone_to":
            result = src_tensor.clone().to(dst_device)
            dst_tensor.copy_(result)
        elif method == "non_blocking":
            dst_tensor.copy_(src_tensor, non_blocking=True)
        else:
            # 預設使用 copy_
            dst_tensor.copy_(src_tensor)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
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
            'framework': 'PyTorch',
            'tensor_size': tensor_size,
            'dtype': str(dtype),
            'method': method,
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


def compare_transfer_methods():
    """比較不同 PyTorch 傳輸方法"""
    print("比較不同 PyTorch 傳輸方法")
    print("=" * 50)
    
    methods = ["copy_", "to", "clone_to", "non_blocking"]
    tensor_size = (1024, 1024)
    
    results = []
    
    for method in methods:
        try:
            result = test_pytorch_transfer(tensor_size, torch.float32, method, 5)
            results.append(result)
            
            if 'error' not in result:
                print(f"{method:<12}: {result['avg_time_ms']:.2f} ms, {result['avg_throughput_mbps']:.2f} MB/s")
            else:
                print(f"{method:<12}: 測試失敗")
                
        except Exception as e:
            print(f"{method:<12}: 異常 - {e}")
        
        print("-" * 30)
    
    # 找出最佳方法
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        best_throughput = max(successful_results, key=lambda x: x['avg_throughput_mbps'])
        fastest_method = min(successful_results, key=lambda x: x['avg_time_ms'])
        
        print(f"\n最佳吞吐量: {best_throughput['method']} ({best_throughput['avg_throughput_mbps']:.2f} MB/s)")
        print(f"最快方法: {fastest_method['method']} ({fastest_method['avg_time_ms']:.2f} ms)")
    
    return results


def run_pytorch_benchmark_suite():
    """執行 PyTorch 基準測試套件"""
    print("=" * 60)
    print("PyTorch GPU-to-GPU 傳輸效能測試")
    print("=" * 60)
    
    # 測試案例
    test_cases = [
        ((256, 256), torch.float32, "copy_"),
        ((512, 512), torch.float32, "copy_"),
        ((1024, 1024), torch.float32, "copy_"),
        ((2048, 2048), torch.float32, "copy_"),
        ((1024, 1024), torch.float16, "copy_"),
        ((1024, 1024), torch.int32, "copy_"),
    ]
    
    all_results = []
    
    for tensor_size, dtype, method in test_cases:
        try:
            result = test_pytorch_transfer(tensor_size, dtype, method, iterations=5)
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
        print(f"\n測試摘要:")
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


def benchmark_memory_patterns():
    """測試不同記憶體存取模式的效能"""
    print("\n測試不同記憶體存取模式")
    print("=" * 40)
    
    # 測試連續 vs 非連續記憶體
    size = (1024, 1024)
    
    # 連續記憶體
    print("測試連續記憶體...")
    contiguous_tensor = torch.randn(size, device="cuda:0")
    dst_tensor = torch.zeros(size, device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    dst_tensor.copy_(contiguous_tensor)
    torch.cuda.synchronize()
    contiguous_time = time.perf_counter() - start
    
    # 非連續記憶體 (轉置)
    print("測試非連續記憶體...")
    non_contiguous_tensor = contiguous_tensor.t()  # 轉置使其非連續
    dst_tensor.zero_()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    dst_tensor.copy_(non_contiguous_tensor.contiguous())  # 需要先使其連續
    torch.cuda.synchronize()
    non_contiguous_time = time.perf_counter() - start
    
    data_size_mb = contiguous_tensor.numel() * contiguous_tensor.element_size() / (1024**2)
    
    print(f"連續記憶體: {contiguous_time*1000:.2f} ms ({data_size_mb/contiguous_time:.2f} MB/s)")
    print(f"非連續記憶體: {non_contiguous_time*1000:.2f} ms ({data_size_mb/non_contiguous_time:.2f} MB/s)")
    print(f"效能差異: {(non_contiguous_time/contiguous_time):.2f}x")


def main():
    """主程式"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # 單一測試模式
            result = test_pytorch_transfer((1024, 1024), torch.float32, "copy_", 3)
            print("\n測試結果:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        elif sys.argv[1] == "--methods":
            # 方法比較模式
            compare_transfer_methods()
        elif sys.argv[1] == "--memory":
            # 記憶體模式測試
            benchmark_memory_patterns()
        else:
            print("可用選項: --single, --methods, --memory")
    else:
        # 完整測試套件
        results = run_pytorch_benchmark_suite()
        
        # 額外測試
        print("\n" + "=" * 60)
        compare_transfer_methods()
        benchmark_memory_patterns()


if __name__ == "__main__":
    main()
