#!/usr/bin/env python3
"""
NIXL vs PyTorch GPU-to-GPU 傳輸效能比較工具

此工具同時執行 NIXL 和 PyTorch 的 GPU 傳輸測試，
並生成詳細的效能比較報告。
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional
import statistics

import torch

# 檢查 NIXL 可用性
NIXL_AVAILABLE = False
try:
    import nixl._utils as nixl_utils
    from nixl._api import nixl_agent, nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError:
    print("警告: NIXL 模組不可用，將只執行 PyTorch 測試")


class ComparisonBenchmark:
    """NIXL vs PyTorch 效能比較測試"""
    
    def __init__(self, iterations: int = 5, warmup: int = 2):
        self.iterations = iterations
        self.warmup = warmup
        self.results = {
            'nixl': [],
            'pytorch': []
        }
        
        # 檢查環境
        self._check_environment()
    
    def _check_environment(self):
        """檢查測試環境"""
        print("檢查測試環境...")
        
        # 檢查 CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ 偵測到 {gpu_count} 個 GPU")
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
        
        # 檢查 NIXL
        if NIXL_AVAILABLE:
            nixl_dir = os.environ.get('NIXL_PLUGIN_DIR')
            if nixl_dir:
                print(f"✅ NIXL 可用，插件目錄: {nixl_dir}")
            else:
                print("⚠️  NIXL 模組可用但 NIXL_PLUGIN_DIR 未設定")
        else:
            print("❌ NIXL 不可用")
    
    def run_pytorch_test(self, tensor_size: tuple, dtype: torch.dtype) -> Optional[Dict]:
        """執行 PyTorch 測試"""
        try:
            print(f"執行 PyTorch 測試: {tensor_size}, {dtype}")
            
            # 設定裝置
            src_device = "cuda:0"
            dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            
            # 創建張量
            src_tensor = torch.randn(tensor_size, dtype=dtype, device=src_device)
            dst_tensor = torch.zeros(tensor_size, dtype=dtype, device=dst_device)
            
            data_size = src_tensor.numel() * src_tensor.element_size()
            
            # 熱身
            for _ in range(self.warmup):
                dst_tensor.zero_()
                dst_tensor.copy_(src_tensor)
            
            # 測試
            times = []
            for i in range(self.iterations):
                dst_tensor.zero_()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
            
            # 驗證
            verification = torch.allclose(src_tensor.cpu(), dst_tensor.cpu(), rtol=1e-5)
            
            result = {
                'framework': 'PyTorch',
                'tensor_size': tensor_size,
                'dtype': str(dtype),
                'data_size_mb': data_size / (1024**2),
                'avg_time_ms': statistics.mean(times) * 1000,
                'min_time_ms': min(times) * 1000,
                'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                'avg_throughput_mbps': (data_size / (1024**2)) / statistics.mean(times),
                'max_throughput_mbps': (data_size / (1024**2)) / min(times),
                'verification_passed': verification,
                'all_times_ms': [t * 1000 for t in times]
            }
            
            print(f"  PyTorch 平均時間: {result['avg_time_ms']:.2f} ms")
            print(f"  PyTorch 平均吞吐量: {result['avg_throughput_mbps']:.2f} MB/s")
            
            return result
            
        except Exception as e:
            print(f"PyTorch 測試失敗: {e}")
            return None
    
    def run_nixl_test(self, tensor_size: tuple, dtype: torch.dtype) -> Optional[Dict]:
        """執行 NIXL 測試"""
        if not NIXL_AVAILABLE:
            return None
        
        try:
            print(f"執行 NIXL 測試: {tensor_size}, {dtype}")
            
            # 設定裝置
            src_device = "cuda:0"
            dst_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            
            # 初始化 NIXL agents
            agent_config = nixl_agent_config(backends=["UCX"])
            dst_nixl_agent = nixl_agent("target", agent_config)
            src_nixl_agent = nixl_agent("initiator", None)
            
            # 創建張量
            src_tensor = torch.randn(tensor_size, dtype=dtype, device=src_device)
            dst_tensor = torch.zeros(tensor_size, dtype=dtype, device=dst_device)
            
            data_size = src_tensor.numel() * src_tensor.element_size()
            
            # 註冊記憶體
            src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
            dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
            
            src_xfer_descs = src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
            dst_xfer_descs = dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
            
            # 設定遠端連接
            meta = dst_nixl_agent.get_agent_metadata()
            remote_name = src_nixl_agent.add_remote_agent(meta)
            
            # 熱身
            for i in range(self.warmup):
                self._execute_nixl_transfer(
                    src_nixl_agent, dst_nixl_agent, 
                    src_xfer_descs, dst_xfer_descs, 
                    remote_name, f"warmup_{i}",
                    dst_tensor
                )
            
            # 測試
            times = []
            for i in range(self.iterations):
                dst_tensor.zero_()
                
                transfer_time = self._execute_nixl_transfer(
                    src_nixl_agent, dst_nixl_agent,
                    src_xfer_descs, dst_xfer_descs,
                    remote_name, f"test_{i}",
                    dst_tensor
                )
                
                if transfer_time is not None:
                    times.append(transfer_time)
            
            if not times:
                print("  NIXL 測試無有效結果")
                return None
            
            # 驗證
            verification = torch.allclose(src_tensor.cpu(), dst_tensor.cpu(), rtol=1e-5)
            
            result = {
                'framework': 'NIXL',
                'tensor_size': tensor_size,
                'dtype': str(dtype),
                'data_size_mb': data_size / (1024**2),
                'avg_time_ms': statistics.mean(times) * 1000,
                'min_time_ms': min(times) * 1000,
                'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                'avg_throughput_mbps': (data_size / (1024**2)) / statistics.mean(times),
                'max_throughput_mbps': (data_size / (1024**2)) / min(times),
                'verification_passed': verification,
                'all_times_ms': [t * 1000 for t in times]
            }
            
            print(f"  NIXL 平均時間: {result['avg_time_ms']:.2f} ms")
            print(f"  NIXL 平均吞吐量: {result['avg_throughput_mbps']:.2f} MB/s")
            
            return result
            
        except Exception as e:
            print(f"NIXL 測試失敗: {e}")
            return None
    
    def _execute_nixl_transfer(self, src_agent, dst_agent, src_descs, dst_descs, 
                              remote_name: str, test_id: str, dst_tensor) -> Optional[float]:
        """執行單次 NIXL 傳輸"""
        try:
            notif = test_id.encode()
            
            xfer_handle = src_agent.initialize_xfer(
                "READ", src_descs, dst_descs, remote_name, notif
            )
            
            if not xfer_handle:
                return None
            
            start_time = time.perf_counter()
            
            state = src_agent.transfer(xfer_handle)
            if state == "ERR":
                return None
            
            # 等待完成
            target_done = False
            init_done = False
            
            while (not init_done) or (not target_done):
                if not init_done:
                    state = src_agent.check_xfer_state(xfer_handle)
                    if state == "ERR":
                        return None
                    elif state == "DONE":
                        init_done = True
                
                if not target_done:
                    if dst_agent.check_remote_xfer_done("initiator", notif):
                        target_done = True
            
            end_time = time.perf_counter()
            return end_time - start_time
            
        except Exception as e:
            print(f"NIXL 傳輸執行失敗: {e}")
            return None
    
    def run_comparison_test(self, tensor_size: tuple, dtype: torch.dtype) -> Dict:
        """執行比較測試"""
        print(f"\n{'='*60}")
        print(f"比較測試: 張量大小 {tensor_size}, 型別 {dtype}")
        print(f"{'='*60}")
        
        # 執行 PyTorch 測試
        pytorch_result = self.run_pytorch_test(tensor_size, dtype)
        
        # 執行 NIXL 測試
        nixl_result = self.run_nixl_test(tensor_size, dtype)
        
        # 比較結果
        comparison = {
            'tensor_size': tensor_size,
            'dtype': str(dtype),
            'pytorch': pytorch_result,
            'nixl': nixl_result
        }
        
        if pytorch_result and nixl_result:
            # 計算效能比較
            speedup_time = pytorch_result['avg_time_ms'] / nixl_result['avg_time_ms']
            speedup_throughput = nixl_result['avg_throughput_mbps'] / pytorch_result['avg_throughput_mbps']
            
            comparison['performance'] = {
                'nixl_speedup_time': speedup_time,
                'nixl_speedup_throughput': speedup_throughput,
                'pytorch_faster': speedup_time < 1.0,
                'nixl_faster': speedup_time > 1.0
            }
            
            print(f"\n效能比較:")
            print(f"  NIXL vs PyTorch 時間比: {speedup_time:.2f}x")
            print(f"  NIXL vs PyTorch 吞吐量比: {speedup_throughput:.2f}x")
            print(f"  {'NIXL 更快' if speedup_time > 1.0 else 'PyTorch 更快'}")
        
        return comparison
    
    def run_full_comparison(self) -> List[Dict]:
        """執行完整的比較測試"""
        print("開始執行 NIXL vs PyTorch 完整比較測試")
        print("=" * 60)
        
        # 測試案例
        test_cases = [
            ((512, 512), torch.float32),
            ((1024, 1024), torch.float32),
            ((2048, 2048), torch.float32),
            ((1024, 1024), torch.float16),
            ((1024, 1024), torch.int32),
        ]
        
        all_comparisons = []
        
        for tensor_size, dtype in test_cases:
            try:
                comparison = self.run_comparison_test(tensor_size, dtype)
                all_comparisons.append(comparison)
                
            except Exception as e:
                print(f"比較測試失敗: {e}")
                error_comparison = {
                    'tensor_size': tensor_size,
                    'dtype': str(dtype),
                    'error': str(e)
                }
                all_comparisons.append(error_comparison)
        
        return all_comparisons
    
    def generate_comparison_report(self, comparisons: List[Dict]) -> str:
        """生成比較報告"""
        report = []
        report.append("=" * 80)
        report.append("NIXL vs PyTorch GPU-to-GPU 傳輸效能比較報告")
        report.append("=" * 80)
        
        # 成功比較統計
        successful_comparisons = [c for c in comparisons if 'error' not in c and 
                                 c.get('pytorch') and c.get('nixl')]
        
        report.append(f"\n總比較測試: {len(comparisons)}")
        report.append(f"成功比較: {len(successful_comparisons)}")
        
        if successful_comparisons:
            report.append(f"\n{'張量大小':<15} {'型別':<12} {'PyTorch(ms)':<12} {'NIXL(ms)':<12} {'NIXL倍數':<10} {'更快者':<8}")
            report.append("-" * 80)
            
            nixl_wins = 0
            pytorch_wins = 0
            
            for comp in successful_comparisons:
                pytorch_time = comp['pytorch']['avg_time_ms']
                nixl_time = comp['nixl']['avg_time_ms']
                speedup = pytorch_time / nixl_time
                winner = "NIXL" if speedup > 1.0 else "PyTorch"
                
                if speedup > 1.0:
                    nixl_wins += 1
                else:
                    pytorch_wins += 1
                
                report.append(
                    f"{str(comp['tensor_size']):<15} "
                    f"{comp['dtype']:<12} "
                    f"{pytorch_time:<12.2f} "
                    f"{nixl_time:<12.2f} "
                    f"{speedup:<10.2f} "
                    f"{winner:<8}"
                )
            
            # 統計摘要
            report.append(f"\n勝負統計:")
            report.append(f"  NIXL 勝出: {nixl_wins} 次")
            report.append(f"  PyTorch 勝出: {pytorch_wins} 次")
            
            # 平均效能
            avg_speedup = statistics.mean([
                comp['pytorch']['avg_time_ms'] / comp['nixl']['avg_time_ms']
                for comp in successful_comparisons
            ])
            
            avg_throughput_ratio = statistics.mean([
                comp['nixl']['avg_throughput_mbps'] / comp['pytorch']['avg_throughput_mbps']
                for comp in successful_comparisons
            ])
            
            report.append(f"\n平均效能:")
            report.append(f"  NIXL 平均加速比: {avg_speedup:.2f}x")
            report.append(f"  NIXL 平均吞吐量比: {avg_throughput_ratio:.2f}x")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_comparison_results(self, comparisons: List[Dict], filename: str = "nixl_pytorch_comparison.json"):
        """儲存比較結果"""
        output_data = {
            'comparison_type': 'NIXL vs PyTorch',
            'config': {
                'iterations': self.iterations,
                'warmup': self.warmup
            },
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'nixl_available': NIXL_AVAILABLE,
                'nixl_plugin_dir': os.environ.get('NIXL_PLUGIN_DIR', 'Not set')
            },
            'comparisons': comparisons,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n比較結果已儲存至: {filename}")
        except Exception as e:
            print(f"儲存結果時發生錯誤: {e}")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="NIXL vs PyTorch GPU 傳輸效能比較")
    parser.add_argument('--iterations', type=int, default=5, help='測試迭代次數')
    parser.add_argument('--warmup', type=int, default=2, help='熱身迭代次數')
    parser.add_argument('--output', type=str, default='nixl_pytorch_comparison.json', help='輸出檔案名稱')
    parser.add_argument('--single', action='store_true', help='執行單一測試')
    
    args = parser.parse_args()
    
    try:
        benchmark = ComparisonBenchmark(args.iterations, args.warmup)
        
        if args.single:
            # 單一測試
            comparison = benchmark.run_comparison_test((1024, 1024), torch.float32)
            print("\n單一測試結果:")
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
        else:
            # 完整比較測試
            comparisons = benchmark.run_full_comparison()
            
            # 生成並顯示報告
            report = benchmark.generate_comparison_report(comparisons)
            print(report)
            
            # 儲存結果
            benchmark.save_comparison_results(comparisons, args.output)
        
    except Exception as e:
        print(f"比較測試執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
