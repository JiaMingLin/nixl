#!/usr/bin/env python3
"""
NIXL GPU 效能測試自訂配置範例

此檔案展示如何建立自訂的測試配置，以滿足特定的測試需求。
"""

import torch
from nixl_gpu_benchmark import GPUBenchmarkConfig, GPUTransferBenchmark


class QuickTestConfig(GPUBenchmarkConfig):
    """快速測試配置 - 適合開發和除錯"""
    
    def __init__(self):
        super().__init__()
        self.tensor_shapes = [(256, 256), (512, 512)]
        self.data_types = [torch.float32]
        self.backends = ["UCX"]
        self.iterations = 3
        self.warmup_iterations = 1


class MemoryIntensiveConfig(GPUBenchmarkConfig):
    """記憶體密集型測試配置 - 測試大資料傳輸"""
    
    def __init__(self):
        super().__init__()
        self.tensor_shapes = [
            (2048, 2048),    # 16MB (float32)
            (4096, 4096),    # 64MB (float32)
            (8192, 8192),    # 256MB (float32)
        ]
        self.data_types = [torch.float32, torch.float16]
        self.backends = ["UCX"]
        self.iterations = 5
        self.warmup_iterations = 2


class PrecisionTestConfig(GPUBenchmarkConfig):
    """精度測試配置 - 比較不同資料型別"""
    
    def __init__(self):
        super().__init__()
        self.tensor_shapes = [(1024, 1024)]  # 固定大小
        self.data_types = [
            torch.float64,   # 64-bit 雙精度
            torch.float32,   # 32-bit 單精度
            torch.float16,   # 16-bit 半精度
            torch.bfloat16,  # Brain Float 16
            torch.int64,     # 64-bit 整數
            torch.int32,     # 32-bit 整數
            torch.int16,     # 16-bit 整數
            torch.int8,      # 8-bit 整數
        ]
        self.backends = ["UCX"]
        self.iterations = 10
        self.warmup_iterations = 3


class LatencyTestConfig(GPUBenchmarkConfig):
    """延遲測試配置 - 專注於小資料塊的傳輸延遲"""
    
    def __init__(self):
        super().__init__()
        self.tensor_shapes = [
            (1, 1),          # 單一元素
            (8, 8),          # 64 元素
            (32, 32),        # 1K 元素
            (64, 64),        # 4K 元素
            (128, 128),      # 16K 元素
        ]
        self.data_types = [torch.float32]
        self.backends = ["UCX"]
        self.iterations = 20  # 更多迭代以獲得準確的延遲測量
        self.warmup_iterations = 5


class ThroughputTestConfig(GPUBenchmarkConfig):
    """吞吐量測試配置 - 專注於大資料塊的傳輸吞吐量"""
    
    def __init__(self):
        super().__init__()
        self.tensor_shapes = [
            (1024, 1024),    # 4MB (float32)
            (2048, 2048),    # 16MB (float32)
            (4096, 4096),    # 64MB (float32)
            (5792, 5792),    # ~128MB (float32)
        ]
        self.data_types = [torch.float32]
        self.backends = ["UCX"]
        self.iterations = 10
        self.warmup_iterations = 3


def run_custom_benchmark(config_class, test_name):
    """執行自訂配置的測試"""
    print(f"\n{'='*60}")
    print(f"執行 {test_name}")
    print(f"{'='*60}")
    
    try:
        config = config_class()
        benchmark = GPUTransferBenchmark(config)
        results = benchmark.run_all_tests()
        
        # 顯示摘要報告
        summary = benchmark.generate_summary_report(results)
        print(summary)
        
        # 儲存結果
        filename = f"{test_name.lower().replace(' ', '_')}_results.json"
        benchmark.save_results(results, filename)
        
        return results
        
    except Exception as e:
        print(f"測試 '{test_name}' 失敗: {e}")
        return None


def main():
    """主程式 - 執行多種自訂測試配置"""
    
    # 定義要執行的測試配置
    test_configs = [
        (QuickTestConfig, "快速測試"),
        (PrecisionTestConfig, "精度比較測試"),
        (LatencyTestConfig, "延遲測試"),
        (ThroughputTestConfig, "吞吐量測試"),
        # (MemoryIntensiveConfig, "記憶體密集測試"),  # 取消註解以啟用
    ]
    
    all_results = {}
    
    for config_class, test_name in test_configs:
        results = run_custom_benchmark(config_class, test_name)
        if results:
            all_results[test_name] = results
    
    # 產生綜合比較報告
    if all_results:
        print("\n" + "="*80)
        print("綜合測試比較報告")
        print("="*80)
        
        for test_name, results in all_results.items():
            successful_results = [r for r in results if 'error' not in r and r.get('verification_passed', False)]
            
            if successful_results:
                avg_throughput = sum(r['avg_throughput_mbps'] for r in successful_results) / len(successful_results)
                avg_latency = sum(r['avg_time_sec'] for r in successful_results) / len(successful_results)
                
                print(f"\n{test_name}:")
                print(f"  成功測試數: {len(successful_results)}")
                print(f"  平均吞吐量: {avg_throughput:.2f} MB/s")
                print(f"  平均延遲: {avg_latency*1000:.2f} ms")
        
        print("\n測試建議:")
        print("- 比較不同配置的結果以找出最適合您應用的設定")
        print("- 注意延遲和吞吐量之間的權衡")
        print("- 考慮資料型別對效能的影響")


if __name__ == "__main__":
    main()
