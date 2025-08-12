#!/usr/bin/env python3
"""
簡化的 Mixture-of-Experts (MoE) 層計算範例

基於圖片中的技術規格：
- 專家結構：兩個權重矩陣 (2048x512, 512x2048)
- 激活函數：GeLU
- 專家放置：E₁ 在 GPU₁，E₂ 在 GPU₂
- 計算流程：
  1. 輸入 token T 在 GPU₁ 但需要傳送到 GPU₂ 的專家 E₂
  2. 使用 NiXL 將 T 從 GPU₁ 傳送到 GPU₂
  3. 在 GPU₂ 執行專家和 token 計算
  4. 將結果 R 從 GPU₂ 傳回 GPU₁
"""

import os
import torch
import torch.nn.functional as F

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config


class Expert:
    """專家實現，包含兩個權重矩陣和 GeLU 激活函數"""
    
    def __init__(self, input_dim=512, hidden_dim=2048, device="cuda:0"):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 第一個權重矩陣：hidden_dim x input_dim (2048 x 512) - 用於 F.linear
        self.weight1 = torch.randn(hidden_dim, input_dim, device=device)
        # 第二個權重矩陣：input_dim x hidden_dim (512 x 2048) - 用於 F.linear  
        self.weight2 = torch.randn(input_dim, hidden_dim, device=device)
        
        # 正確初始化權重
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
    
    def forward(self, x):
        """前向傳播通過專家"""
        # 第一個線性變換：x @ weight1
        hidden = F.linear(x, self.weight1)
        # GeLU 激活
        hidden = F.gelu(hidden)
        # 第二個線性變換：hidden @ weight2
        output = F.linear(hidden, self.weight2)
        return output


def init_nixl_agents():
    """初始化 NiXL agents 用於跨 GPU 通信"""
    print("使用 NIXL Plugins 來自：")
    print(os.environ.get("NIXL_PLUGIN_DIR", "未設定"))
    
    # 配置 NiXL agents
    agent_config = nixl_agent_config(backends=["UCX"])
    dst_nixl_agent = nixl_agent("target", agent_config)
    src_nixl_agent = nixl_agent("initiator", None)
    
    print("NiXL agents 初始化成功")
    return src_nixl_agent, dst_nixl_agent


def transfer_tensor(src_agent, dst_agent, src_tensor, dst_tensor, transfer_name):
    """使用 NiXL 傳輸張量"""
    print(f"傳輸 {transfer_name}...")
    
    # 註冊記憶體
    src_agent.register_memory(src_tensor, "VRAM", is_sorted=True)
    dst_agent.register_memory(dst_tensor, "VRAM", is_sorted=True)
    
    # 獲取傳輸描述符
    src_xfer_descs = src_agent.get_xfer_descs(src_tensor, "VRAM", is_sorted=True)
    dst_xfer_descs = dst_agent.get_xfer_descs(dst_tensor, "VRAM", is_sorted=True)
    
    # 交換元數據
    meta = dst_agent.get_agent_metadata()
    remote_name = src_agent.add_remote_agent(meta)
    print(f"從元數據載入名稱: {remote_name}")
    
    # 執行傳輸
    notif = transfer_name.encode()
    xfer_handle = src_agent.initialize_xfer(
        "READ",
        src_xfer_descs,
        dst_xfer_descs,
        remote_name,
        notif,
    )
    
    if not xfer_handle:
        print("創建傳輸失敗。")
        return False
    
    state = src_agent.transfer(xfer_handle)
    assert state != "ERR"
    
    # 等待傳輸完成
    target_done = False
    init_done = False
    
    while (not init_done) or (not target_done):
        if not init_done:
            state = src_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("傳輸進入錯誤狀態。")
                return False
            elif state == "DONE":
                init_done = True
                print("發起者完成")
        
        if not target_done:
            if dst_agent.check_remote_xfer_done("initiator", notif):
                target_done = True
                print("目標完成")
    
    return True


def moe_computation():
    """執行 MoE 層計算"""
    print("=== Mixture-of-Experts (MoE) 層計算範例 ===")
    print("基於技術規格：")
    print("- 專家結構：兩個權重矩陣 (2048x512, 512x2048)")
    print("- 激活函數：GeLU")
    print("- 專家放置：E₁ 在 GPU₁，E₂ 在 GPU₂")
    print("- 使用 NiXL 進行跨 GPU token 傳輸")
    print()
    
    # 檢查可用 GPU
    if torch.cuda.device_count() < 2:
        print("警告：需要至少 2 個 GPU 來執行此範例")
        print(f"可用 GPU：{torch.cuda.device_count()}")
        return
    
    # 初始化 NiXL agents
    src_agent, dst_agent = init_nixl_agents()
    
    # 創建專家
    print("\n=== 創建專家 ===")
    expert_e1 = Expert(input_dim=512, hidden_dim=2048, device="cuda:0")
    expert_e2 = Expert(input_dim=512, hidden_dim=2048, device="cuda:1")
    
    print(f"專家 E₁ 在 {expert_e1.device}")
    print(f"專家 E₂ 在 {expert_e2.device}")
    print(f"權重矩陣 1 形狀：{expert_e1.weight1.shape}")
    print(f"權重矩陣 2 形狀：{expert_e1.weight2.shape}")
    
    # 創建樣本 token
    print("\n=== 創建輸入 Token ===")
    token = torch.randn(1, 1, 512, device="cuda:0")
    print(f"Token 形狀：{token.shape}")
    print(f"Token 設備：{token.device}")
    print(f"Token 值（前 5 個）：{token.flatten()[:5]}")
    
    # 步驟 1：將 token 從 GPU₁ 傳送到 GPU₂
    print("\n=== 步驟 1：Token 傳輸 ===")
    print("Token 在 GPU₁ 但需要傳送到 GPU₂ 的專家 E₂")
    
    # 在目標 GPU 創建接收張量
    dst_tensor = torch.zeros_like(token, device="cuda:1")
    
    # 使用 NiXL 傳輸
    success = transfer_tensor(
        src_agent, dst_agent, 
        token, dst_tensor, 
        "MOE_TOKEN_TRANSFER"
    )
    
    if not success:
        print("Token 傳輸失敗")
        return
    
    print("Token 成功傳輸到 GPU₂")
    
    # 步驟 2：在 GPU₂ 執行專家計算
    print("\n=== 步驟 2：專家計算 ===")
    print("在 GPU₂ 執行專家 E₂ 計算")
    
    result = expert_e2.forward(dst_tensor)
    
    print(f"專家計算完成")
    print(f"結果形狀：{result.shape}")
    print(f"結果設備：{result.device}")
    
    # 步驟 3：將結果從 GPU₂ 傳回 GPU₁
    print("\n=== 步驟 3：結果傳輸 ===")
    print("將結果從 GPU₂ 傳回 GPU₁")
    
    # 在原始 GPU 創建接收張量
    final_tensor = torch.zeros_like(result, device="cuda:0")
    
    # 使用 NiXL 傳輸結果
    success = transfer_tensor(
        src_agent, dst_agent, 
        result, final_tensor, 
        "MOE_RESULT_TRANSFER"
    )
    
    if not success:
        print("結果傳輸失敗")
        return
    
    print("結果成功傳輸回 GPU₁")
    
    # 最終結果
    print("\n=== 最終結果 ===")
    print(f"輸入 token 形狀：{token.shape}")
    print(f"輸出結果形狀：{final_tensor.shape}")
    print(f"輸入設備：{token.device}")
    print(f"輸出設備：{final_tensor.device}")
    
    # 驗證計算
    print(f"\n=== 驗證 ===")
    print(f"輸入 token 總和：{token.sum().item():.4f}")
    print(f"輸出結果總和：{final_tensor.sum().item():.4f}")
    print(f"輸入 token 平均值：{token.mean().item():.4f}")
    print(f"輸出結果平均值：{final_tensor.mean().item():.4f}")
    
    print("\nMoE 層計算成功完成！")


if __name__ == "__main__":
    moe_computation()
