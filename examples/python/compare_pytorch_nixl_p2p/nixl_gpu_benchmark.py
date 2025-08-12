#!/usr/bin/env python3
"""
NIXL GPU-to-GPU 資料傳輸效能測試程式

此程式基於 NIXL 函式庫，測試不同條件下 GPU 之間的資料傳輸速度與吞吐量。
支援多種資料大小、不同後端、以及詳細的效能統計分析。
"""

import os
import sys
import time
import argparse
import statistics
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import torch

# NIXL imports
import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


class GPUBenchmarkConfig:
    """測試配置類別"""
    
    def __init__(self):
        # 預設測試配置
        self.data_sizes = [
            1024,           # 1KB
            1024 * 1024,    # 1MB  
            16 * 1024 * 1024,  # 16MB
            64 * 1024 * 1024,  # 64MB
            256 * 1024 * 1024, # 256MB
            1024 * 1024 * 1024 # 1GB
        ]
        self.tensor_shapes = [
            (32, 32),       # 小張量
            (512, 512),     # 中等張量
            (2048, 2048),   # 大張量
            (4096, 4096),   # 超大張量
        ]
        self.data_types = [torch.float32, torch.float16, torch.int32, torch.int8]
        self.backends = ["UCX"]  # 可擴展支援更多後端
        self.iterations = 10
        self.warmup_iterations = 3
        self.src_device = "cuda:0"
        self.dst_device = "cuda:1"


class GPUTransferBenchmark:
    """GPU 傳輸效能測試主類別"""
    
    def __init__(self, config: GPUBenchmarkConfig):
        self.config = config
        self.results = []
        self.src_nixl_agent = None
        self.dst_nixl_agent = None
        
        # 檢查 GPU 可用性
        self._check_gpu_availability()
        
        # 初始化 NIXL agents
        self._initialize_nixl_agents()
    
    def _check_gpu_availability(self):
        """檢查 GPU 可用性"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，無法執行 GPU 測試")
        
        gpu_count = torch.cuda.device_count()
        print(f"偵測到 {gpu_count} 個 GPU 裝置")
        
        if gpu_count < 2:
            print("警告: 只偵測到一個 GPU，將在同一個 GPU 上執行測試")
            self.config.dst_device = self.config.src_device
    
    def _initialize_nixl_agents(self):
        """初始化 NIXL agents"""
        try:
            print("正在初始化 NIXL agents...")
            print(f"使用 NIXL Plugins 從: {os.environ.get('NIXL_PLUGIN_DIR', 'Not set')}")
            
            # 配置並創建 agents
            agent_config = nixl_agent_config(backends=self.config.backends)
            self.dst_nixl_agent = nixl_agent("target", agent_config)
            self.src_nixl_agent = nixl_agent("initiator", None)
            
            print("NIXL agents 初始化成功")
            
        except Exception as e:
            raise RuntimeError(f"NIXL agents 初始化失敗: {e}")
    
    def create_test_tensors(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """創建測試用的張量"""
        # 在來源 GPU 上創建隨機資料
        src_tensor = torch.randn(shape, dtype=dtype, device=self.config.src_device)
        
        # 在目標 GPU 上創建零張量
        dst_tensor = torch.zeros(shape, dtype=dtype, device=self.config.dst_device)
        
        return src_tensor, dst_tensor
    
    def register_tensors_with_nixl(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor):
        """向 NIXL 註冊張量記憶體"""
        try:
            # 註冊來源和目標張量
            self.src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
            self.dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
            
            # 獲取傳輸描述符
            src_xfer_descs = self.src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
            dst_xfer_descs = self.dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
            
            return src_xfer_descs, dst_xfer_descs
            
        except Exception as e:
            raise RuntimeError(f"張量註冊失敗: {e}")
    
    def setup_remote_connection(self):
        """設定遠端連接"""
        try:
            # 交換元數據
            meta = self.dst_nixl_agent.get_agent_metadata()
            remote_name = self.src_nixl_agent.add_remote_agent(meta)
            return remote_name
            
        except Exception as e:
            raise RuntimeError(f"遠端連接設定失敗: {e}")
    
    def execute_transfer(self, src_xfer_descs, dst_xfer_descs, remote_name: str, 
                        test_id: str) -> float:
        """執行單次傳輸並測量時間"""
        notif = test_id.encode()
        
        # 創建傳輸請求
        xfer_handle = self.src_nixl_agent.initialize_xfer(
            "READ",
            src_xfer_descs,
            dst_xfer_descs,
            remote_name,
            notif,
        )
        
        if not xfer_handle:
            raise RuntimeError("創建傳輸請求失敗")
        
        # 記錄開始時間
        start_time = time.perf_counter()
        
        # 開始傳輸
        state = self.src_nixl_agent.transfer(xfer_handle)
        if state == "ERR":
            raise RuntimeError("傳輸初始化失敗")
        
        # 等待傳輸完成
        target_done = False
        init_done = False
        
        while (not init_done) or (not target_done):
            if not init_done:
                state = self.src_nixl_agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    raise RuntimeError("傳輸過程中發生錯誤")
                elif state == "DONE":
                    init_done = True
            
            if not target_done:
                if self.dst_nixl_agent.check_remote_xfer_done("initiator", notif):
                    target_done = True
        
        # 記錄結束時間
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def verify_transfer(self, src_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> bool:
        """驗證傳輸正確性"""
        try:
            # 將目標張量移到 CPU 進行比較
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
                       backend: str) -> Dict:
        """執行單一測試案例"""
        print(f"\n執行測試: 形狀={shape}, 型別={dtype}, 後端={backend}")
        
        # 創建測試張量
        src_tensor, dst_tensor = self.create_test_tensors(shape, dtype)
        data_size = src_tensor.numel() * src_tensor.element_size()
        
        print(f"  資料大小: {data_size / (1024**2):.2f} MB")
        
        # 註冊張量
        src_xfer_descs, dst_xfer_descs = self.register_tensors_with_nixl(src_tensor, dst_tensor)
        
        # 設定遠端連接
        remote_name = self.setup_remote_connection()
        
        # 執行熱身迭代
        print(f"  執行 {self.config.warmup_iterations} 次熱身...")
        for i in range(self.config.warmup_iterations):
            try:
                self.execute_transfer(src_xfer_descs, dst_xfer_descs, remote_name, f"warmup_{i}")
            except Exception as e:
                print(f"  熱身 {i} 失敗: {e}")
        
        # 執行實際測試
        print(f"  執行 {self.config.iterations} 次測試...")
        transfer_times = []
        
        for i in range(self.config.iterations):
            try:
                # 重新創建目標張量以確保每次測試的一致性
                dst_tensor.zero_()
                
                transfer_time = self.execute_transfer(
                    src_xfer_descs, dst_xfer_descs, remote_name, f"test_{i}"
                )
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
                'backend': backend,
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
                'backend': backend,
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
        print("開始執行 NIXL GPU-to-GPU 傳輸效能測試")
        print("=" * 60)
        
        all_results = []
        total_tests = len(self.config.tensor_shapes) * len(self.config.data_types) * len(self.config.backends)
        current_test = 0
        
        for backend in self.config.backends:
            for shape in self.config.tensor_shapes:
                for dtype in self.config.data_types:
                    current_test += 1
                    print(f"\n進度: {current_test}/{total_tests}")
                    
                    try:
                        result = self.run_single_test(shape, dtype, backend)
                        all_results.append(result)
                        self.results.append(result)
                        
                    except Exception as e:
                        print(f"測試失敗: {e}")
                        error_result = {
                            'shape': shape,
                            'dtype': str(dtype),
                            'backend': backend,
                            'error': str(e),
                            'verification_passed': False
                        }
                        all_results.append(error_result)
        
        return all_results
    
    def generate_summary_report(self, results: List[Dict]) -> str:
        """生成測試摘要報告"""
        report = []
        report.append("=" * 80)
        report.append("NIXL GPU-to-GPU 傳輸效能測試摘要報告")
        report.append("=" * 80)
        
        # 成功測試統計
        successful_tests = [r for r in results if 'error' not in r and r.get('verification_passed', False)]
        failed_tests = [r for r in results if 'error' in r or not r.get('verification_passed', False)]
        
        report.append(f"\n總測試數量: {len(results)}")
        report.append(f"成功測試: {len(successful_tests)}")
        report.append(f"失敗測試: {len(failed_tests)}")
        
        if successful_tests:
            # 按資料大小排序結果
            successful_tests.sort(key=lambda x: x['data_size_bytes'])
            
            report.append(f"\n{'形狀':<15} {'型別':<12} {'大小(MB)':<10} {'平均時間(ms)':<15} {'吞吐量(MB/s)':<15} {'驗證':<8}")
            report.append("-" * 80)
            
            for result in successful_tests:
                report.append(
                    f"{str(result['shape']):<15} "
                    f"{result['dtype']:<12} "
                    f"{result['data_size_mb']:<10.2f} "
                    f"{result['avg_time_sec']*1000:<15.2f} "
                    f"{result['avg_throughput_mbps']:<15.2f} "
                    f"{'通過' if result['verification_passed'] else '失敗':<8}"
                )
            
            # 最佳效能統計
            best_throughput = max(successful_tests, key=lambda x: x['avg_throughput_mbps'])
            lowest_latency = min(successful_tests, key=lambda x: x['avg_time_sec'])
            
            report.append(f"\n最佳吞吐量: {best_throughput['avg_throughput_mbps']:.2f} MB/s")
            report.append(f"  形狀: {best_throughput['shape']}, 型別: {best_throughput['dtype']}")
            
            report.append(f"\n最低延遲: {lowest_latency['avg_time_sec']*1000:.2f} ms")
            report.append(f"  形狀: {lowest_latency['shape']}, 型別: {lowest_latency['dtype']}")
        
        if failed_tests:
            report.append(f"\n失敗的測試:")
            for result in failed_tests:
                error_msg = result.get('error', 'Unknown error')
                report.append(f"  {result['shape']} ({result['dtype']}): {error_msg}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_results(self, results: List[Dict], filename: str = "nixl_gpu_benchmark_results.json"):
        """儲存詳細測試結果到 JSON 檔案"""
        output_data = {
            'config': {
                'tensor_shapes': self.config.tensor_shapes,
                'data_types': [str(dt) for dt in self.config.data_types],
                'backends': self.config.backends,
                'iterations': self.config.iterations,
                'warmup_iterations': self.config.warmup_iterations,
                'src_device': self.config.src_device,
                'dst_device': self.config.dst_device
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
        description="NIXL GPU-to-GPU 資料傳輸效能測試程式",
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
        '--backends', nargs='+', default=['UCX'],
        help='要測試的 NIXL 後端 (預設: UCX)'
    )
    
    parser.add_argument(
        '--output', type=str, default='nixl_gpu_benchmark_results.json',
        help='輸出檔案名稱 (預設: nixl_gpu_benchmark_results.json)'
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
    config = GPUBenchmarkConfig()
    config.iterations = args.iterations
    config.warmup_iterations = args.warmup
    config.src_device = args.src_device
    config.dst_device = args.dst_device
    config.backends = args.backends
    
    # 快速測試模式
    if args.quick:
        config.tensor_shapes = [(512, 512), (2048, 2048)]
        config.data_types = [torch.float32, torch.int32]
        config.iterations = 5
        config.warmup_iterations = 2
        print("使用快速測試模式")
    
    try:
        # 創建並執行測試
        benchmark = GPUTransferBenchmark(config)
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
