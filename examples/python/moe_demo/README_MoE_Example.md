# Mixture-of-Experts (MoE) 層計算範例

這個範例展示了如何使用 NiXL 來實現跨 GPU 的 Mixture-of-Experts 層計算。

## 技術規格

基於提供的圖片規格：

- **專家結構**：每個專家包含兩個權重矩陣
  - 第一個矩陣：2048 x 512
  - 第二個矩陣：512 x 2048
- **激活函數**：GeLU (Gaussian Error Linear Unit)
- **專家放置**：
  - E₁ 放置在 GPU₁
  - E₂ 放置在 GPU₂
- **計算流程**：
  1. 輸入 token T 在 GPU₁ 但需要傳送到 GPU₂ 的專家 E₂
  2. 使用 NiXL 將 T 從 GPU₁ 傳送到 GPU₂
  3. 在 GPU₂ 執行專家和 token 計算
  4. 將結果 R 從 GPU₂ 傳回 GPU₁

## 檔案說明

### 1. `moe_layer_example.py`
完整的 MoE 層實現，包含：
- `Expert` 類別：實現專家結構（兩個權重矩陣 + GeLU）
- `MoELayer` 類別：管理多個專家和跨 GPU 通信
- 完整的 NiXL 傳輸流程
- 錯誤處理和驗證

### 2. `moe_simple_example.py`
簡化版本的 MoE 範例，更容易理解：
- 清晰的步驟說明
- 中文註釋
- 基本的專家實現
- 直接的 NiXL 傳輸流程

### 3. `moe_layer_example.ipynb`
Jupyter notebook 版本，適合互動式學習：
- 分步驟執行
- 即時結果顯示
- 詳細的輸出說明

## 執行需求

### 硬體需求
- 至少 2 個 GPU
- 支援 CUDA 的 GPU

### 軟體需求
```bash
# 安裝必要的套件
pip install torch numpy nixl
```

### 環境設定
```bash
# 設定 NiXL 環境變數
export NIXL_PLUGIN_DIR=/path/to/nixl/plugins
```

## 執行範例

### 執行完整版本
```bash
cd examples/python
python moe_layer_example.py
```

### 執行簡化版本
```bash
cd examples/python
python moe_simple_example.py
```

### 執行 Jupyter notebook
```bash
cd examples/python/notebooks
jupyter notebook moe_layer_example.ipynb
```

## 預期輸出

成功執行後，您應該看到類似以下的輸出：

```
=== Mixture-of-Experts (MoE) 層計算範例 ===
基於技術規格：
- 專家結構：兩個權重矩陣 (2048x512, 512x2048)
- 激活函數：GeLU
- 專家放置：E₁ 在 GPU₁，E₂ 在 GPU₂
- 使用 NiXL 進行跨 GPU token 傳輸

=== 創建專家 ===
專家 E₁ 在 cuda:0
專家 E₂ 在 cuda:1
權重矩陣 1 形狀：torch.Size([512, 2048])
權重矩陣 2 形狀：torch.Size([2048, 512])

=== 創建輸入 Token ===
Token 形狀：torch.Size([1, 1, 512])
Token 設備：cuda:0

=== 步驟 1：Token 傳輸 ===
傳輸 MOE_TOKEN_TRANSFER...
從元數據載入名稱: b'target'
發起者完成
目標完成
Token 成功傳輸到 GPU₂

=== 步驟 2：專家計算 ===
在 GPU₂ 執行專家 E₂ 計算
專家計算完成
結果形狀：torch.Size([1, 1, 512])
結果設備：cuda:1

=== 步驟 3：結果傳輸 ===
將結果從 GPU₂ 傳回 GPU₁
傳輸 MOE_RESULT_TRANSFER...
發起者完成
目標完成
結果成功傳輸回 GPU₁

=== 最終結果 ===
輸入 token 形狀：torch.Size([1, 1, 512])
輸出結果形狀：torch.Size([1, 1, 512])
輸入設備：cuda:0
輸出設備：cuda:0

=== 驗證 ===
輸入 token 總和：-12.3456
輸出結果總和：23.4567
輸入 token 平均值：-0.0241
輸出結果平均值：0.0458

MoE 層計算成功完成！
```

## 關鍵概念

### 1. 專家結構
每個專家包含兩個線性層和一個 GeLU 激活函數：
```
輸入 (512) → 線性層 (512→2048) → GeLU → 線性層 (2048→512) → 輸出 (512)
```

### 2. 跨 GPU 通信
使用 NiXL 進行高效的 GPU 間數據傳輸：
- 記憶體註冊
- 傳輸描述符獲取
- 元數據交換
- 非同步傳輸

### 3. 計算流程
1. **Token 傳輸**：將輸入從源 GPU 傳送到目標 GPU
2. **專家計算**：在目標 GPU 執行專家前向傳播
3. **結果傳輸**：將計算結果傳回源 GPU

## 故障排除

### 常見問題

1. **GPU 數量不足**
   ```
   警告：需要至少 2 個 GPU 來執行此範例
   可用 GPU：1
   ```
   解決方案：確保系統有至少 2 個 GPU

2. **NiXL 初始化失敗**
   ```
   創建傳輸失敗。
   ```
   解決方案：檢查 NiXL 環境設定和插件路徑

3. **CUDA 記憶體不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解決方案：減少批次大小或使用較小的模型

### 除錯技巧

1. 檢查 GPU 可用性：
   ```python
   import torch
   print(f"可用 GPU：{torch.cuda.device_count()}")
   ```

2. 檢查 NiXL 環境：
   ```python
   import os
   print(f"NIXL_PLUGIN_DIR：{os.environ.get('NIXL_PLUGIN_DIR')}")
   ```

3. 驗證張量設備：
   ```python
   print(f"張量設備：{tensor.device}")
   ```

## 擴展和修改

### 添加更多專家
```python
# 創建多個專家
experts = []
for i in range(num_experts):
    device = f"cuda:{i % torch.cuda.device_count()}"
    expert = Expert(input_dim=512, hidden_dim=2048, device=device)
    experts.append(expert)
```

### 修改專家結構
```python
class CustomExpert(Expert):
    def forward(self, x):
        # 自定義前向傳播邏輯
        hidden = F.linear(x, self.weight1)
        hidden = F.relu(hidden)  # 使用 ReLU 而不是 GeLU
        output = F.linear(hidden, self.weight2)
        return output
```

### 添加路由機制
```python
def route_to_expert(token, experts, routing_weights):
    """根據路由權重選擇專家"""
    expert_idx = torch.argmax(routing_weights).item()
    return experts[expert_idx].forward(token)
```

## 參考資料

- [NiXL 文檔](https://github.com/your-org/nixl)
- [PyTorch 文檔](https://pytorch.org/docs/)
- [Mixture of Experts 論文](https://arxiv.org/abs/1701.06538)

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這些範例！
