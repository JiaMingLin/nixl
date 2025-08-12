#!/usr/bin/env python3
"""
PyTorch GPU-to-GPU 資料傳輸效能測試程式

此程式使用純 PyTorch 實作 GPU 間資料傳輸的速度與吞吐量測試，
作為 NIXL 測試結果的對比基準。支援多種傳輸方法和詳細的效能分析。
"""

import os
import sys
import time
import argparse
import statistics
from typing import List, Tuple, Dict, Optional, Callable
import json

import torch
import numpy as np


class PyTorchBenchmarkConfig:
    """PyTorch 測試配置類別"""
    
    def __init__(self):
        # 測試配置
        self.tensor_shapes = [
            (32, 32),       # 小張量
            (512, 512),     # 中等張量
            (2048, 2048),   # 大張量
            (4096, 4096),   # 超大張量
        ]
        self.data_types = [torch.float32, torch.float16, torch.int32, torch.int8]
        
        # 測試的傳輸方法
        self.transfer_methods = [
            "copy_",           # tensor.copy_(src)
            "clone_to",        # src.clone().to(device)
            "direct_to",       # src.to(device)
            "cuda_memcpy",     # 使用 CUDA memcpy
        ]
        
        self.iterations = 10
        self.warmup_iterations = 3
        self.src_device = "cuda:0"
        self.dst_device = "cuda:1"


class PyTorchTransferBenchmark:
    """PyTorch GPU 傳輸效能測試主類別"""
    
    def __init__(self, config: PyTorchBenchmarkConfig):
        self.config = config
        self.results = []
        
        # 檢查 GPU 可用性
        self._check_gpu_availability()
        
        # 預熱 GPU
        self._warmup_gpus()
    
    def _check_gpu_availability(self):
        """檢查 GPU 可用性"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，無法執行 GPU 測試")
        
        gpu_count = torch.cuda.device_count()
        print(f"偵測到 {gpu_count} 個 GPU 裝置")
        
        # 顯示 GPU 資訊
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        if gpu_count < 2:
            print("警告: 只偵測到一個 GPU，將在同一個 GPU 上執行測試")
            self.config.dst_device = self.config.src_device
    
    def _warmup_gpus(self):
        """預熱 GPU 以確保穩定的效能測量"""
        print("預熱 GPU...")
        
        for device in [self.config.src_device, self.config.dst_device]:
            # 創建小張量並執行一些操作
            dummy = torch.randn(100, 100, device=device)
            for _ in range(10):
                dummy = dummy @ dummy
            del dummy
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def create_test_tensors(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """創建測試用的張量"""
        # 在來源 GPU 上創建隨機資料
        src_tensor = torch.randn(shape, dtype=dtype, device=self.config.src_device)
        
        # 在目標 GPU 上創建零張量
        dst_tensor = torch.zeros(shape, dtype=dtype, device=self.config.dst_device)
        
        return src_tensor, dst_tensor
    
    def transfer_copy_(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
        """使用 copy_ 方法進行傳輸"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        dst_tensor.copy_(src_tensor)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def transfer_clone_to(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
        """使用 clone().to() 方法進行傳輸"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = src_tensor.clone().to(dst_tensor.device)
        dst_tensor.copy_(result)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def transfer_direct_to(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
        """使用直接 to() 方法進行傳輸"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = src_tensor.to(dst_tensor.device)
        dst_tensor.copy_(result)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def transfer_cuda_memcpy(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> float:
        """使用 CUDA memcpy 進行傳輸（如果可用）"""
        try:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 使用 PyTorch 的底層 CUDA 操作
            with torch.cuda.device(dst_tensor.device):
                dst_tensor.copy_(src_tensor, non_blocking=True)
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            return end_time - start_time
            
        except Exception as e:
            print(f"CUDA memcpy 失敗: {e}")
            # 回退到 copy_ 方法
            return self.transfer_copy_(src_tensor, dst_tensor)
    
    def get_transfer_function(self, method: str) -> Callable:
        """獲取對應的傳輸函數"""
        transfer_functions = {
            "copy_": self.transfer_copy_,
            "clone_to": self.transfer_clone_to,
            "direct_to": self.transfer_direct_to,
            "cuda_memcpy": self.transfer_cuda_memcpy,
        }
        
        return transfer_functions.get(method, self.transfer_copy_)
    
    def verify_transfer(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> bool:
        """驗證傳輸正確性"""
        try:
            # 將張量移到 CPU 進行比較
            src_cpu = src_tensor.cpu()
            dst_cpu = dst_tensor.cpu()
            
            # 檢查是否相等（允許小的浮點誤差）
            if src_tensor.dtype.is_floating_point:
                return torch.allclose(src_cpu, dst_cpu, rtol=1e-5, atol=1e-6)
            else:
                return torch.equal(src_cpu, dst_cpu)
                
        except Exception as e:
            print(f"傳輸驗證失敗: {e}")
            return False
    
    def run_single_test(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                       method: str) -> Dict:
        """執行單一測試案例"""
        print(f"\n執行測試: 形狀={shape}, 型別={dtype}, 方法={method}")
        
        # 創建測試張量
        src_tensor, dst_tensor = self.create_test_tensors(shape, dtype)
        data_size = src_tensor.numel() * src_tensor.element_size()
        
        print(f"  資料大小: {data_size / (1024**2):.2f} MB")
        
        # 獲取傳輸函數
        transfer_func = self.get_transfer_function(method)
        
        # 執行熱身迭代
        print(f"  執行 {self.config.warmup_iterations} 次熱身...")
        for i in range(self.config.warmup_iterations):
            try:
                dst_tensor.zero_()
                transfer_func(src_tensor, dst_tensor)
            except Exception as e:
                print(f"  熱身 {i} 失敗: {e}")
        
        # 執行實際測試
        print(f"  執行 {self.config.iterations} 次測試...")
        transfer_times = []
        
        for i in range(self.config.iterations):
            try:
                # 重置目標張量
                dst_tensor.zero_()
                
                # 清理 GPU 快取以獲得更一致的結果
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                
                transfer_time = transfer_func(src_tensor, dst_tensor)
                transfer_times.append(transfer_time)
                
                # 每5次迭代顯示進度
                if (i + 1) % 5 == 0:
                    print(f"    完成 {i + 1}/{self.config.iterations} 次迭代")
                    
            except Exception as e:
                print(f"  測試迭代 {i} 失敗: {e}")
                continue
        
        # 驗證最後一次傳輸的正確性
        if transfer_times:
            verification_passed = self.verify_transfer(src_tensor, dst_tensor)
            print(f"  傳輸驗證: {'通過' if verification_passed else '失敗'}")
        else:
            verification_passed = False
            print("  無有效的傳輸時間記錄")
        
        # 計算統計數據
        if transfer_times:
            avg_time = statistics.mean(transfer_times)
            min_time = min(transfer_times)
            max_time = max(transfer_times)
            std_time = statistics.stdev(transfer_times) if len(transfer_times) > 1 else 0
            
            # 計算吞吐量 (MB/s)
            avg_throughput = (data_size / (1024**2)) / avg_time
            max_throughput = (data_size / (1024**2)) / min_time
            
            result = {
                'shape': shape,
                'dtype': str(dtype),
                'method': method,
                'data_size_bytes': data_size,
                'data_size_mb': data_size / (1024**2),
                'iterations': len(transfer_times),
                'avg_time_sec': avg_time,
                'min_time_sec': min_time,
                'max_time_sec': max_time,
                'std_time_sec': std_time,
                'avg_throughput_mbps': avg_throughput,
                'max_throughput_mbps': max_throughput,
                'verification_passed': verification_passed,
                'all_times': transfer_times
            }
            
            print(f"  平均時間: {avg_time*1000:.2f} ms")
            print(f"  平均吞吐量: {avg_throughput:.2f} MB/s")
            print(f"  最大吞吐量: {max_throughput:.2f} MB/s")
            
        else:
            result = {
                'shape': shape,
                'dtype': str(dtype),
                'method': method,
                'data_size_bytes': data_size,
                'data_size_mb': data_size / (1024**2),
                'iterations': 0,
                'error': 'No successful transfers',
                'verification_passed': False
            }
        
        return result
    
    def run_all_tests(self) -> List[Dict]:
        """執行所有測試案例"""
        print("=" * 60)
        print("開始執行 PyTorch GPU-to-GPU 傳輸效能測試")
        print("=" * 60)
        
        all_results = []
        total_tests = len(self.config.tensor_shapes) * len(self.config.data_types) * len(self.config.transfer_methods)
        current_test = 0
        
        for method in self.config.transfer_methods:
            for shape in self.config.tensor_shapes:
                for dtype in self.config.data_types:
                    current_test += 1
                    print(f"\n進度: {current_test}/{total_tests}")
                    
                    try:
                        result = self.run_single_test(shape, dtype, method)
                        all_results.append(result)
                        self.results.append(result)
                        
                    except Exception as e:
                        print(f"測試失敗: {e}")
                        error_result = {
                            'shape': shape,
                            'dtype': str(dtype),
                            'method': method,
                            'error': str(e),
                            'verification_passed': False
                        }
                        all_results.append(error_result)
        
        return all_results
    
    def generate_summary_report(self, results: List[Dict]) -> str:
        """生成測試摘要報告"""
        report = []
        report.append("=" * 80)
        report.append("PyTorch GPU-to-GPU 傳輸效能測試摘要報告")
        report.append("=" * 80)
        
        # 成功測試統計
        successful_tests = [r for r in results if 'error' not in r and r.get('verification_passed', False)]
        failed_tests = [r for r in results if 'error' in r or not r.get('verification_passed', False)]
        
        report.append(f"\n總測試數量: {len(results)}")
        report.append(f"成功測試: {len(successful_tests)}")
        report.append(f"失敗測試: {len(failed_tests)}")
        
        if successful_tests:
            # 按方法分組統計
            methods = set(r['method'] for r in successful_tests)
            
            report.append(f"\n{'方法':<12} {'形狀':<15} {'型別':<12} {'大小(MB)':<10} {'平均時間(ms)':<15} {'吞吐量(MB/s)':<15}")
            report.append("-" * 90)
            
            # 按方法和資料大小排序
            successful_tests.sort(key=lambda x: (x['method'], x['data_size_bytes']))
            
            for result in successful_tests:
                report.append(
                    f"{result['method']:<12} "
                    f"{str(result['shape']):<15} "
                    f"{result['dtype']:<12} "
                    f"{result['data_size_mb']:<10.2f} "
                    f"{result['avg_time_sec']*1000:<15.2f} "
                    f"{result['avg_throughput_mbps']:<15.2f}"
                )
            
            # 各方法的最佳效能
            report.append(f"\n各傳輸方法的最佳效能:")
            for method in methods:
                method_results = [r for r in successful_tests if r['method'] == method]
                if method_results:
                    best = max(method_results, key=lambda x: x['avg_throughput_mbps'])
                    report.append(f"  {method}: {best['avg_throughput_mbps']:.2f} MB/s")
            
            # 全局最佳效能
            best_throughput = max(successful_tests, key=lambda x: x['avg_throughput_mbps'])
            lowest_latency = min(successful_tests, key=lambda x: x['avg_time_sec'])
            
            report.append(f"\n最佳吞吐量: {best_throughput['avg_throughput_mbps']:.2f} MB/s")
            report.append(f"  方法: {best_throughput['method']}, 形狀: {best_throughput['shape']}")
            
            report.append(f"\n最低延遲: {lowest_latency['avg_time_sec']*1000:.2f} ms")
            report.append(f"  方法: {lowest_latency['method']}, 形狀: {lowest_latency['shape']}")
        
        if failed_tests:
            report.append(f"\n失敗的測試:")
            for result in failed_tests:
                error_msg = result.get('error', 'Unknown error')
                report.append(f"  {result['shape']} ({result['dtype']}, {result['method']}): {error_msg}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_results(self, results: List[Dict], filename: str = "pytorch_gpu_benchmark_results.json"):
        """儲存詳細測試結果到 JSON 檔案"""
        output_data = {
            'framework': 'PyTorch',
            'config': {
                'tensor_shapes': self.config.tensor_shapes,
                'data_types': [str(dt) for dt in self.config.data_types],
                'transfer_methods': self.config.transfer_methods,
                'iterations': self.config.iterations,
                'warmup_iterations': self.config.warmup_iterations,
                'src_device': self.config.src_device,
                'dst_device': self.config.dst_device
            },
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            },
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n詳細結果已儲存至: {filename}")
        except Exception as e:
            print(f"儲存結果時發生錯誤: {e}")


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="PyTorch GPU-to-GPU 資料傳輸效能測試程式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--iterations', type=int, default=10,
        help='每個測試案例的迭代次數 (預設: 10)'
    )
    
    parser.add_argument(
        '--warmup', type=int, default=3,
        help='熱身迭代次數 (預設: 3)'
    )
    
    parser.add_argument(
        '--src-device', type=str, default='cuda:0',
        help='來源 GPU 裝置 (預設: cuda:0)'
    )
    
    parser.add_argument(
        '--dst-device', type=str, default='cuda:1',
        help='目標 GPU 裝置 (預設: cuda:1)'
    )
    
    parser.add_argument(
        '--methods', nargs='+', 
        default=['copy_', 'clone_to', 'direct_to', 'cuda_memcpy'],
        choices=['copy_', 'clone_to', 'direct_to', 'cuda_memcpy'],
        help='要測試的傳輸方法'
    )
    
    parser.add_argument(
        '--output', type=str, default='pytorch_gpu_benchmark_results.json',
        help='輸出檔案名稱'
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='快速測試模式 (減少測試案例)'
    )
    
    return parser.parse_args()


def main():
    """主程式進入點"""
    args = parse_arguments()
    
    # 建立測試配置
    config = PyTorchBenchmarkConfig()
    config.iterations = args.iterations
    config.warmup_iterations = args.warmup
    config.src_device = args.src_device
    config.dst_device = args.dst_device
    config.transfer_methods = args.methods
    
    # 快速測試模式
    if args.quick:
        config.tensor_shapes = [(512, 512), (2048, 2048)]
        config.data_types = [torch.float32, torch.int32]
        config.transfer_methods = ['copy_', 'direct_to']
        config.iterations = 5
        config.warmup_iterations = 2
        print("使用快速測試模式")
    
    try:
        # 創建並執行測試
        benchmark = PyTorchTransferBenchmark(config)
        results = benchmark.run_all_tests()
        
        # 生成並顯示報告
        summary = benchmark.generate_summary_report(results)
        print(summary)
        
        # 儲存詳細結果
        benchmark.save_results(results, args.output)
        
    except Exception as e:
        print(f"測試執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
