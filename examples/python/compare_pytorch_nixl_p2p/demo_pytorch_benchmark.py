#!/usr/bin/env python3
"""
PyTorch GPU-to-GPU 傳輸效能測試 - 展示腳本

這個腳本演示如何使用 PyTorch 進行 GPU 間資料傳輸的效能測試，
作為與 NIXL 測試結果的對比基準。
"""

import time
import torch
from typing import Tuple


def check_pytorch_environment():
    """檢查 PyTorch 執行環境"""
    print("檢查 PyTorch 執行環境...")
    
    # 檢查 PyTorch 版本
    print(f"✅ PyTorch 版本: {torch.__version__}")
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"✅ CUDA 版本: {torch.version.cuda}")
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 偵測到 {gpu_count} 個 GPU")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        memory = props.total_memory / (1024**3)
        compute = f"{props.major}.{props.minor}"
        print(f"  GPU {i}: {name}")
        print(f"    記憶體: {memory:.1f} GB")
        print(f"    計算能力: {compute}")
    
    return True


def basic_pytorch_transfer_test():
    """基本 PyTorch 傳輸測試"""
    print("\n執行基本 PyTorch 傳輸測試...")
    
    # 設定裝置
    src_device = "cuda:0"
    dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    
    print(f"來源裝置: {src_device}")
    print(f"目標裝置: {dst_device}")
    
    # 創建測試張量 (與 NIXL 範例相同大小)
    tensor_shape = (256, 256)
    src_tensor = torch.ones(tensor_shape, device=src_device)
    dst_tensor = torch.zeros(tensor_shape, device=dst_device) + 5
    
    data_size = src_tensor.numel() * src_tensor.element_size()
    print(f"資料大小: {data_size / 1024:.2f} KB")
    
    # 顯示初始值
    print(f"傳輸前 src_tensor 樣本: {src_tensor[0, :5]}")
    print(f"傳輸前 dst_tensor 樣本: {dst_tensor[0, :5]}")
    
    # 執行傳輸
    print("開始傳輸...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    dst_tensor.copy_(src_tensor)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    transfer_time = end_time - start_time
    throughput = (data_size / 1024 / 1024) / transfer_time
    
    # 顯示結果
    print(f"✅ 傳輸完成!")
    print(f"  傳輸時間: {transfer_time*1000:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} MB/s")
    
    # 驗證結果
    print(f"傳輸後 src_tensor 樣本: {src_tensor[0, :5]}")
    print(f"傳輸後 dst_tensor 樣本: {dst_tensor[0, :5]}")
    
    verification_passed = torch.equal(src_tensor.cpu(), dst_tensor.cpu())
    print(f"  資料驗證: {'✅ 通過' if verification_passed else '❌ 失敗'}")
    
    return verification_passed


def compare_pytorch_transfer_methods():
    """比較不同 PyTorch 傳輸方法的效能"""
    print("\n比較不同 PyTorch 傳輸方法...")
    
    # 測試設定
    tensor_shape = (1024, 1024)
    src_device = "cuda:0"
    dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    
    # 創建測試張量
    src_tensor = torch.randn(tensor_shape, device=src_device)
    data_size_mb = src_tensor.numel() * src_tensor.element_size() / (1024**2)
    
    print(f"測試張量大小: {tensor_shape}")
    print(f"資料大小: {data_size_mb:.2f} MB")
    
    # 測試不同方法
    methods = {
        "copy_": lambda src, dst: dst.copy_(src),
        "to + copy": lambda src, dst: dst.copy_(src.to(dst.device)),
        "clone + to": lambda src, dst: dst.copy_(src.clone().to(dst.device)),
        "non_blocking": lambda src, dst: dst.copy_(src, non_blocking=True)
    }
    
    print(f"\n{'方法':<15} {'時間(ms)':<12} {'吞吐量(MB/s)':<15}")
    print("-" * 45)
    
    results = []
    
    for method_name, method_func in methods.items():
        try:
            # 創建目標張量
            dst_tensor = torch.zeros_like(src_tensor, device=dst_device)
            
            # 熱身
            for _ in range(3):
                method_func(src_tensor, dst_tensor)
            
            # 測試
            times = []
            for _ in range(5):
                dst_tensor.zero_()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                method_func(src_tensor, dst_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            throughput = data_size_mb / avg_time
            
            print(f"{method_name:<15} {avg_time*1000:<12.2f} {throughput:<15.2f}")
            
            results.append({
                'method': method_name,
                'time_ms': avg_time * 1000,
                'throughput_mbps': throughput
            })
            
        except Exception as e:
            print(f"{method_name:<15} 失敗: {e}")
    
    # 找出最佳方法
    if results:
        best_speed = min(results, key=lambda x: x['time_ms'])
        best_throughput = max(results, key=lambda x: x['throughput_mbps'])
        
        print(f"\n最快方法: {best_speed['method']} ({best_speed['time_ms']:.2f} ms)")
        print(f"最高吞吐量: {best_throughput['method']} ({best_throughput['throughput_mbps']:.2f} MB/s)")
    
    return results


def test_different_data_types():
    """測試不同資料型別的傳輸效能"""
    print("\n測試不同資料型別的傳輸效能...")
    
    data_types = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.int32, "int32"),
        (torch.int8, "int8")
    ]
    
    tensor_shape = (1024, 1024)
    src_device = "cuda:0"
    dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    
    print(f"{'型別':<10} {'大小(MB)':<10} {'時間(ms)':<12} {'吞吐量(MB/s)':<15}")
    print("-" * 50)
    
    for dtype, type_name in data_types:
        try:
            # 創建張量
            if dtype.is_floating_point:
                src_tensor = torch.randn(tensor_shape, dtype=dtype, device=src_device)
            else:
                src_tensor = torch.randint(0, 100, tensor_shape, dtype=dtype, device=src_device)
            
            dst_tensor = torch.zeros_like(src_tensor, device=dst_device)
            
            data_size_mb = src_tensor.numel() * src_tensor.element_size() / (1024**2)
            
            # 測試傳輸
            times = []
            for _ in range(5):
                dst_tensor.zero_()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            throughput = data_size_mb / avg_time
            
            print(f"{type_name:<10} {data_size_mb:<10.2f} {avg_time*1000:<12.2f} {throughput:<15.2f}")
            
        except Exception as e:
            print(f"{type_name:<10} 失敗: {e}")


def benchmark_memory_bandwidth():
    """測試記憶體頻寬"""
    print("\n測試記憶體頻寬...")
    
    sizes = [
        (256, 256, "64KB"),
        (512, 512, "1MB"),
        (1024, 1024, "4MB"),
        (2048, 2048, "16MB"),
        (4096, 4096, "64MB")
    ]
    
    src_device = "cuda:0"
    dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    
    print(f"{'大小':<10} {'實際大小':<12} {'時間(ms)':<12} {'頻寬(GB/s)':<15}")
    print("-" * 52)
    
    for height, width, size_name in sizes:
        try:
            src_tensor = torch.randn(height, width, device=src_device)
            dst_tensor = torch.zeros_like(src_tensor, device=dst_device)
            
            actual_size_mb = src_tensor.numel() * src_tensor.element_size() / (1024**2)
            
            # 多次測試取平均
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            bandwidth_gbps = (actual_size_mb / 1024) / avg_time
            
            print(f"{size_name:<10} {actual_size_mb:<12.2f} {avg_time*1000:<12.2f} {bandwidth_gbps:<15.2f}")
            
        except Exception as e:
            print(f"{size_name:<10} 失敗: {e}")


def main():
    """主程式"""
    print("PyTorch GPU-to-GPU 傳輸效能測試展示")
    print("=" * 50)
    
    # 檢查環境
    if not check_pytorch_environment():
        print("\n❌ 環境檢查失敗，請確認 PyTorch 和 CUDA 安裝")
        return
    
    try:
        # 執行基本測試
        if basic_pytorch_transfer_test():
            print("\n✅ 基本傳輸測試成功!")
        else:
            print("\n❌ 基本傳輸測試失敗!")
            return
        
        # 比較不同傳輸方法
        compare_pytorch_transfer_methods()
        
        # 測試不同資料型別
        test_different_data_types()
        
        # 測試記憶體頻寬
        benchmark_memory_bandwidth()
        
        print("\n" + "=" * 50)
        print("PyTorch 測試完成! 您可以:")
        print("1. 執行 ./run_benchmark.sh 選擇不同測試模式")
        print("2. 使用 python3 pytorch_gpu_benchmark.py 進行完整測試")
        print("3. 使用 python3 simple_pytorch_benchmark.py 進行快速測試")
        print("4. 使用 python3 compare_nixl_pytorch.py 比較 NIXL 和 PyTorch")
        
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        print("請檢查 GPU 狀態和 PyTorch 安裝")


if __name__ == "__main__":
    main()
