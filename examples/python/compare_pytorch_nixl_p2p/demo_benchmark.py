#!/usr/bin/env python3
"""
NIXL GPU-to-GPU 效能測試 - 展示腳本

這個腳本演示如何使用 NIXL 進行 GPU 間資料傳輸的效能測試。
基於原始的 nixl_torch_tensor.py，擴展為完整的效能分析工具。
"""

import os
import time
import torch
from typing import Tuple

# NIXL imports (需要正確設定 NIXL_PLUGIN_DIR)
try:
    import nixl._utils as nixl_utils
    from nixl._api import nixl_agent, nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError as e:
    print(f"警告: 無法匯入 NIXL 模組: {e}")
    NIXL_AVAILABLE = False


def check_environment():
    """檢查執行環境"""
    print("檢查執行環境...")
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 偵測到 {gpu_count} 個 GPU")
    
    # 檢查 NIXL 環境
    if not NIXL_AVAILABLE:
        print("❌ NIXL 模組不可用")
        return False
    
    nixl_plugin_dir = os.environ.get('NIXL_PLUGIN_DIR')
    if not nixl_plugin_dir:
        print("❌ NIXL_PLUGIN_DIR 環境變數未設定")
        return False
    
    print(f"✅ NIXL 插件目錄: {nixl_plugin_dir}")
    return True


def basic_transfer_test():
    """基本傳輸測試 - 基於原始腳本"""
    print("\n執行基本傳輸測試...")
    
    # 使用原始腳本的設定
    buf_size = 256000
    
    # 初始化 NIXL agents (與原始腳本相同)
    agent_config = nixl_agent_config(backends=["UCX"])
    dst_nixl_agent = nixl_agent("target", agent_config)
    src_nixl_agent = nixl_agent("initiator", None)
    
    # 創建張量 (與原始腳本相同，但加入計時)
    print("創建測試張量...")
    src_tensor = torch.ones(256, 256, device="cuda:0") 
    dst_tensor = torch.zeros(256, 256, device="cuda:1") + 5
    
    data_size = src_tensor.numel() * src_tensor.element_size()
    print(f"資料大小: {data_size / 1024:.2f} KB")
    
    # 註冊記憶體 (與原始腳本相同)
    src_nixl_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
    dst_nixl_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
    
    src_xfer_descs = src_nixl_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
    dst_xfer_descs = dst_nixl_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
    
    # 設定遠端連接 (與原始腳本相同)
    meta = dst_nixl_agent.get_agent_metadata()
    remote_name = src_nixl_agent.add_remote_agent(meta)
    print(f"遠端名稱: {remote_name}")
    
    # 執行傳輸 (與原始腳本相同，但加入計時)
    print("開始傳輸...")
    start_time = time.perf_counter()
    
    notif = b"DEMO_TEST"
    xfer_handle = src_nixl_agent.initialize_xfer(
        "READ",
        src_xfer_descs,
        dst_xfer_descs,
        remote_name,
        notif,
    )
    
    if not xfer_handle:
        print("❌ 創建傳輸失敗")
        return False
    
    state = src_nixl_agent.transfer(xfer_handle)
    if state == "ERR":
        print("❌ 傳輸初始化失敗")
        return False
    
    # 等待完成 (與原始腳本相同)
    target_done = False
    init_done = False
    
    while (not init_done) or (not target_done):
        if not init_done:
            state = src_nixl_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("❌ 傳輸過程發生錯誤")
                return False
            elif state == "DONE":
                init_done = True
        
        if not target_done:
            if dst_nixl_agent.check_remote_xfer_done("initiator", notif):
                target_done = True
    
    end_time = time.perf_counter()
    transfer_time = end_time - start_time
    
    # 計算效能指標
    throughput = (data_size / 1024 / 1024) / transfer_time  # MB/s
    
    print(f"✅ 傳輸完成!")
    print(f"  傳輸時間: {transfer_time*1000:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} MB/s")
    
    # 驗證結果
    verification_passed = torch.allclose(src_tensor.cpu(), dst_tensor.cpu())
    print(f"  資料驗證: {'✅ 通過' if verification_passed else '❌ 失敗'}")
    
    return verification_passed


def performance_comparison():
    """效能比較測試"""
    print("\n執行效能比較測試...")
    
    test_sizes = [
        (128, 128, "小型"),
        (512, 512, "中型"), 
        (1024, 1024, "大型"),
        (2048, 2048, "超大型")
    ]
    
    print(f"{'大小':<10} {'資料量':<10} {'時間(ms)':<12} {'吞吐量(MB/s)':<15}")
    print("-" * 50)
    
    for height, width, size_name in test_sizes:
        try:
            # 簡化的效能測試
            src_tensor = torch.randn(height, width, device="cuda:0")
            dst_tensor = torch.zeros(height, width, device="cuda:1")
            
            data_size = src_tensor.numel() * src_tensor.element_size()
            
            # 使用 PyTorch 的 copy_ 作為基準比較
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 簡單的 GPU-to-GPU 複製
            dst_tensor.copy_(src_tensor)
            torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            copy_time = end_time - start_time
            throughput = (data_size / 1024 / 1024) / copy_time
            
            print(f"{size_name:<10} {data_size/1024:.1f}KB {copy_time*1000:<12.2f} {throughput:<15.2f}")
            
        except Exception as e:
            print(f"{size_name:<10} 測試失敗: {e}")


def demo_advanced_features():
    """展示進階功能"""
    print("\n展示進階功能...")
    
    print("1. 不同資料型別的影響:")
    data_types = [torch.float32, torch.float16, torch.int32, torch.int8]
    
    for dtype in data_types:
        try:
            tensor = torch.ones(512, 512, dtype=dtype, device="cuda:0")
            size_bytes = tensor.numel() * tensor.element_size()
            print(f"  {str(dtype):<15}: {size_bytes/1024:.1f} KB")
        except Exception as e:
            print(f"  {str(dtype):<15}: 不支援")
    
    print("\n2. 記憶體使用情況:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            cached = torch.cuda.memory_reserved(i) / 1024 / 1024
            print(f"  GPU {i}: 已分配 {allocated:.1f}MB, 快取 {cached:.1f}MB")


def main():
    """主程式"""
    print("NIXL GPU-to-GPU 效能測試展示")
    print("=" * 50)
    
    # 檢查環境
    if not check_environment():
        print("\n❌ 環境檢查失敗，請確認:")
        print("1. CUDA 已正確安裝")
        print("2. PyTorch 支援 CUDA")
        print("3. NIXL 模組可用")
        print("4. NIXL_PLUGIN_DIR 環境變數已設定")
        return
    
    try:
        # 執行基本測試
        if basic_transfer_test():
            print("\n✅ 基本傳輸測試成功!")
        else:
            print("\n❌ 基本傳輸測試失敗!")
            return
        
        # 效能比較
        performance_comparison()
        
        # 進階功能展示
        demo_advanced_features()
        
        print("\n" + "=" * 50)
        print("展示完成! 您可以:")
        print("1. 執行 ./run_benchmark.sh 進行完整測試")
        print("2. 使用 python3 nixl_gpu_benchmark.py --quick 快速測試")
        print("3. 使用 python3 simple_gpu_benchmark.py 簡單測試")
        print("4. 參考 README_BENCHMARK.md 了解更多功能")
        
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        print("請檢查 NIXL 設定和 GPU 狀態")


if __name__ == "__main__":
    main()
