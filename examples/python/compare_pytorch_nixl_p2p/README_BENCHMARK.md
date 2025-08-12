# NIXL GPU-to-GPU 效能測試工具

本目錄包含基於 NIXL 函式庫的 GPU 間資料傳輸效能測試工具，用於評估不同條件下的傳輸速度與吞吐量。

## 檔案說明

### 主要測試程式

#### NIXL 測試工具

1. **`nixl_gpu_benchmark.py`** - 完整的 NIXL 效能測試程式
   - 支援多種張量大小、資料型別和後端配置
   - 詳細的統計分析和結果報告
   - 可匯出 JSON 格式的測試結果
   - 支援自訂測試參數

2. **`simple_gpu_benchmark.py`** - 簡化版 NIXL 測試程式
   - 專注於核心功能測試
   - 適合快速驗證和基本效能評估
   - 更易於理解的程式結構

3. **`nixl_torch_tensor.py`** - 原始 NIXL 範例程式
   - 基本的 NIXL 使用示範
   - 單次傳輸測試

#### PyTorch 測試工具

4. **`pytorch_gpu_benchmark.py`** - 完整的 PyTorch 效能測試程式
   - 測試多種 PyTorch 傳輸方法 (copy_, to, clone_to, non_blocking)
   - 詳細的效能統計和分析
   - 作為 NIXL 測試的對比基準

5. **`simple_pytorch_benchmark.py`** - 簡化版 PyTorch 測試程式
   - 快速 PyTorch 傳輸效能測試
   - 包含不同傳輸方法的比較
   - 記憶體存取模式測試

#### 比較工具

6. **`compare_nixl_pytorch.py`** - NIXL vs PyTorch 比較測試
   - 同時執行 NIXL 和 PyTorch 測試
   - 直接效能比較和分析
   - 生成詳細的比較報告

### 輔助工具

- **`run_benchmark.sh`** - 自動化執行腳本
- **`README_BENCHMARK.md`** - 本說明文件

## 環境需求

### 硬體需求
- 至少一個 NVIDIA GPU (建議兩個或以上)
- 支援 CUDA 的系統

### 軟體需求
- Python 3.7+
- PyTorch (支援 CUDA)
- NIXL 函式庫及其依賴
- NVIDIA CUDA Toolkit

### 環境設定
```bash
# 設定 NIXL 插件目錄 (必要)
export NIXL_PLUGIN_DIR=/path/to/nixl/plugins

# 確認 CUDA 路徑
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 使用方法

### 方法一：使用自動化腳本 (推薦)

```bash
# 執行自動化測試腳本
./run_benchmark.sh
```

腳本會提供七種測試模式：
1. 快速測試 (簡化版 NIXL)
2. 單一測試 (簡化版 NIXL)
3. 完整效能測試 (NIXL)
4. 快速完整測試 (NIXL)
5. PyTorch 效能測試
6. PyTorch 簡化測試
7. NIXL vs PyTorch 比較測試

### 方法二：直接執行 Python 腳本

#### 簡化版測試

```bash
# 執行基本測試套件
python3 simple_gpu_benchmark.py

# 執行單一測試
python3 simple_gpu_benchmark.py --single
```

#### 完整效能測試

```bash
# 使用預設參數
python3 nixl_gpu_benchmark.py

# 自訂測試參數
python3 nixl_gpu_benchmark.py \
    --iterations 20 \
    --warmup 5 \
    --src-device cuda:0 \
    --dst-device cuda:1 \
    --backends UCX \
    --output my_benchmark_results.json

# 快速測試模式
python3 nixl_gpu_benchmark.py --quick
```

#### PyTorch 測試

```bash
# 執行完整 PyTorch 測試
python3 pytorch_gpu_benchmark.py

# 執行簡化 PyTorch 測試
python3 simple_pytorch_benchmark.py

# PyTorch 單一測試
python3 simple_pytorch_benchmark.py --single

# PyTorch 方法比較
python3 simple_pytorch_benchmark.py --methods

# PyTorch 記憶體模式測試
python3 simple_pytorch_benchmark.py --memory

# 自訂 PyTorch 測試參數
python3 pytorch_gpu_benchmark.py \
    --iterations 15 \
    --methods copy_ direct_to \
    --output pytorch_results.json
```

#### 比較測試

```bash
# 執行 NIXL vs PyTorch 比較
python3 compare_nixl_pytorch.py

# 單一比較測試
python3 compare_nixl_pytorch.py --single

# 自訂比較測試
python3 compare_nixl_pytorch.py \
    --iterations 10 \
    --output comparison_results.json
```

### 命令列參數說明

完整測試程式支援以下參數：

- `--iterations N`: 每個測試案例的迭代次數 (預設: 10)
- `--warmup N`: 熱身迭代次數 (預設: 3)
- `--src-device DEVICE`: 來源 GPU 裝置 (預設: cuda:0)
- `--dst-device DEVICE`: 目標 GPU 裝置 (預設: cuda:1)
- `--backends LIST`: 要測試的 NIXL 後端 (預設: UCX)
- `--output FILE`: 輸出檔案名稱 (預設: nixl_gpu_benchmark_results.json)
- `--quick`: 快速測試模式，減少測試案例

## 測試項目

### 張量大小測試
- 小張量: 32x32
- 中等張量: 512x512  
- 大張量: 2048x2048
- 超大張量: 4096x4096

### 資料型別測試
- `torch.float32` (32-bit 浮點數)
- `torch.float16` (16-bit 浮點數)
- `torch.int32` (32-bit 整數)
- `torch.int8` (8-bit 整數)

### 效能指標
- **傳輸時間**: 平均、最小、最大、標準差
- **吞吐量**: 平均和最大吞吐量 (MB/s)
- **資料驗證**: 確保傳輸正確性
- **成功率**: 統計成功的傳輸次數

## 輸出結果

### 控制台輸出
測試過程中會顯示：
- 即時的測試進度
- 每個測試案例的結果摘要
- 最終的效能統計報告

### JSON 結果檔案
詳細的測試結果會儲存為 JSON 格式，包含：
- 測試配置資訊
- 每個測試案例的詳細數據
- 時間戳記和中繼資料

範例 JSON 結構：
```json
{
  "config": {
    "tensor_shapes": [[512, 512], [2048, 2048]],
    "data_types": ["torch.float32", "torch.int32"],
    "backends": ["UCX"],
    "iterations": 10
  },
  "results": [
    {
      "shape": [512, 512],
      "dtype": "torch.float32",
      "data_size_mb": 1.0,
      "avg_time_sec": 0.001234,
      "avg_throughput_mbps": 810.37,
      "verification_passed": true
    }
  ]
}
```

## 效能分析建議

### 基準比較
1. **不同張量大小**: 觀察大小對吞吐量的影響
2. **不同資料型別**: 比較不同精度的傳輸效率
3. **GPU 配置**: 測試同一 GPU vs 不同 GPU 的效能差異

### 最佳化方向
1. **資料對齊**: 確保張量記憶體對齊
2. **批次大小**: 測試最適合的資料塊大小
3. **記憶體配置**: 預先分配記憶體以減少動態分配開銷

### 故障排除

#### 常見錯誤

1. **NIXL_PLUGIN_DIR 未設定**
   ```
   錯誤: NIXL_PLUGIN_DIR 環境變數未設定
   解決方法: export NIXL_PLUGIN_DIR=/path/to/nixl/plugins
   ```

2. **CUDA 不可用**
   ```
   錯誤: CUDA 不可用，無法執行 GPU 測試
   解決方法: 檢查 NVIDIA 驅動程式和 CUDA 安裝
   ```

3. **記憶體不足**
   ```
   錯誤: CUDA out of memory
   解決方法: 減少張量大小或使用 --quick 模式
   ```

4. **傳輸驗證失敗**
   ```
   警告: 傳輸驗證失敗
   可能原因: 硬體問題或 NIXL 配置錯誤
   ```

#### 除錯技巧

1. **啟用詳細日誌**
   ```bash
   export NIXL_LOG_LEVEL=DEBUG
   ```

2. **檢查 GPU 狀態**
   ```bash
   nvidia-smi
   ```

3. **測試 PyTorch CUDA**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

## 進階使用

### 自訂測試案例

可以修改 `GPUBenchmarkConfig` 類別來自訂測試參數：

```python
class CustomConfig(GPUBenchmarkConfig):
    def __init__(self):
        super().__init__()
        # 自訂張量大小
        self.tensor_shapes = [(128, 128), (1024, 1024)]
        # 自訂資料型別
        self.data_types = [torch.float32]
        # 自訂迭代次數
        self.iterations = 5
```

### 批次測試

可以建立腳本來自動執行多組測試：

```bash
#!/bin/bash
for backend in UCX; do
    for device_pair in "cuda:0,cuda:1" "cuda:0,cuda:0"; do
        IFS=',' read -r src dst <<< "$device_pair"
        python3 nixl_gpu_benchmark.py \
            --src-device $src \
            --dst-device $dst \
            --backends $backend \
            --output "results_${backend}_${src//:/}_${dst//:/}.json"
    done
done
```

## 貢獻與反饋

如果您發現錯誤或有改進建議，請：
1. 檢查現有的問題報告
2. 提供詳細的錯誤資訊和環境配置
3. 包含重現步驟和預期行為描述

## 授權條款

本工具遵循 NIXL 專案的授權條款。
