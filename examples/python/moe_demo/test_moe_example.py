#!/usr/bin/env python3
"""
測試 MoE 範例的腳本

這個腳本會測試：
1. 專家結構是否正確
2. NiXL 傳輸是否正常工作
3. 跨 GPU 計算是否成功
"""

import sys
import os
import torch
import torch.nn.functional as F

# 添加當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moe_simple_example import Expert, init_nixl_agents, transfer_tensor


def test_expert_structure():
    """測試專家結構"""
    print("=== 測試專家結構 ===")
    
    # 創建專家
    expert = Expert(input_dim=512, hidden_dim=2048, device="cuda:0")
    
    # 檢查權重形狀 (適配 F.linear 的要求)
    assert expert.weight1.shape == (2048, 512), f"權重1形狀錯誤：{expert.weight1.shape}"
    assert expert.weight2.shape == (512, 2048), f"權重2形狀錯誤：{expert.weight2.shape}"
    
    # 創建測試輸入
    test_input = torch.randn(1, 1, 512, device="cuda:0")
    
    # 執行前向傳播
    output = expert.forward(test_input)
    
    # 檢查輸出形狀
    assert output.shape == (1, 1, 512), f"輸出形狀錯誤：{output.shape}"
    assert output.device == torch.device("cuda:0"), f"輸出設備錯誤：{output.device}"
    
    print("✓ 專家結構測試通過")
    return True


def test_nixl_agents():
    """測試 NiXL agents 初始化"""
    print("=== 測試 NiXL Agents ===")
    
    try:
        src_agent, dst_agent = init_nixl_agents()
        print("✓ NiXL agents 初始化成功")
        return True
    except Exception as e:
        print(f"✗ NiXL agents 初始化失敗：{e}")
        return False


def test_gpu_availability():
    """測試 GPU 可用性"""
    print("=== 測試 GPU 可用性 ===")
    
    gpu_count = torch.cuda.device_count()
    print(f"可用 GPU 數量：{gpu_count}")
    
    if gpu_count < 2:
        print("✗ 需要至少 2 個 GPU 來執行完整測試")
        return False
    
    # 檢查每個 GPU
    for i in range(gpu_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
    
    print("✓ GPU 可用性測試通過")
    return True


def test_cross_gpu_transfer():
    """測試跨 GPU 傳輸"""
    print("=== 測試跨 GPU 傳輸 ===")
    
    if torch.cuda.device_count() < 2:
        print("跳過跨 GPU 傳輸測試（需要至少 2 個 GPU）")
        return True
    
    try:
        # 初始化 NiXL agents
        src_agent, dst_agent = init_nixl_agents()
        
        # 創建測試張量
        src_tensor = torch.randn(256, 256, device="cuda:0")
        dst_tensor = torch.zeros_like(src_tensor, device="cuda:1")
        
        # 執行傳輸
        success = transfer_tensor(
            src_agent, dst_agent,
            src_tensor, dst_tensor,
            "TEST_TRANSFER"
        )
        
        if success:
            print("✓ 跨 GPU 傳輸測試通過")
            return True
        else:
            print("✗ 跨 GPU 傳輸測試失敗")
            return False
            
    except Exception as e:
        print(f"✗ 跨 GPU 傳輸測試失敗：{e}")
        return False


def test_expert_computation():
    """測試專家計算"""
    print("=== 測試專家計算 ===")
    
    try:
        # 創建專家
        expert = Expert(input_dim=512, hidden_dim=2048, device="cuda:0")
        
        # 創建測試輸入
        test_input = torch.randn(1, 1, 512, device="cuda:0")
        
        # 執行計算
        output = expert.forward(test_input)
        
        # 驗證輸出
        assert output.shape == test_input.shape, "輸出形狀與輸入不匹配"
        assert not torch.allclose(output, test_input), "輸出與輸入相同，可能計算有問題"
        
        print("✓ 專家計算測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 專家計算測試失敗：{e}")
        return False


def test_gelu_activation():
    """測試 GeLU 激活函數"""
    print("=== 測試 GeLU 激活函數 ===")
    
    try:
        # 創建測試輸入
        x = torch.randn(10, 10)
        
        # 手動實現 GeLU
        expected = x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
        # 使用 PyTorch 的 GeLU
        actual = F.gelu(x)
        
        # 比較結果
        assert torch.allclose(expected, actual, atol=1e-6), "GeLU 實現不正確"
        
        print("✓ GeLU 激活函數測試通過")
        return True
        
    except Exception as e:
        print(f"✗ GeLU 激活函數測試失敗：{e}")
        return False


def run_all_tests():
    """執行所有測試"""
    print("開始執行 MoE 範例測試...")
    print("=" * 50)
    
    tests = [
        ("GPU 可用性", test_gpu_availability),
        ("專家結構", test_expert_structure),
        ("GeLU 激活函數", test_gelu_activation),
        ("專家計算", test_expert_computation),
        ("NiXL Agents", test_nixl_agents),
        ("跨 GPU 傳輸", test_cross_gpu_transfer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n執行測試：{test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"測試失敗：{test_name}")
        except Exception as e:
            print(f"測試異常：{test_name} - {e}")
    
    print("\n" + "=" * 50)
    print(f"測試結果：{passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！MoE 範例可以正常使用。")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查環境設定。")
        return False


def main():
    """主函數"""
    print("MoE 範例測試腳本")
    print("=" * 50)
    
    # 檢查 PyTorch 和 CUDA
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    print(f"CUDA 版本：{torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU 數量：{torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}：{torch.cuda.get_device_name(i)}")
    else:
        print("警告：CUDA 不可用，某些測試將被跳過")
    
    print()
    
    # 執行測試
    success = run_all_tests()
    
    if success:
        print("\n✅ 測試完成，範例可以正常使用")
        sys.exit(0)
    else:
        print("\n❌ 測試失敗，請檢查環境設定")
        sys.exit(1)


if __name__ == "__main__":
    main()
