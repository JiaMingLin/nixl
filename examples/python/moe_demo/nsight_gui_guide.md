# Nsight GUI Timeline Trace 使用指南

這個指南將教您如何使用 Nsight Systems 和 Nsight Compute 的 GUI 來查看和分析 timeline trace。

## 1. Nsight Systems GUI - Timeline 分析

### 啟動 Nsight Systems GUI

```bash
# 方法 1: 直接啟動 GUI
nsight-sys

# 方法 2: 打開現有的 profile 檔案
nsight-sys moe_basic_profile.nsys-rep

# 方法 3: 如果檔案在其他位置
nsight-sys /path/to/your/profile.nsys-rep
```

### Timeline 視圖主要組件

#### 🕐 Timeline 面板
- **位置**: 視窗中央主要區域
- **功能**: 顯示時間軸上的所有 GPU/CPU 活動
- **操作**:
  - 滾輪縮放時間軸
  - 拖拽平移時間軸
  - 點擊選擇特定事件

#### 📊 軌道 (Tracks) 說明

1. **CUDA API Track**
   - 顯示 `cudaMemcpy`, `cudaLaunchKernel` 等 API 呼叫
   - 顏色代表不同的 API 類型
   - 長度代表執行時間

2. **GPU Kernels Track**
   - 顯示實際在 GPU 上執行的核心
   - 每個核心顯示為一個矩形塊
   - 高度表示 GPU 使用率

3. **Memory Transfer Track**
   - Host-to-Device 傳輸（通常是藍色）
   - Device-to-Host 傳輸（通常是綠色）
   - Device-to-Device 傳輸（通常是橙色）

4. **CPU Threads Track**
   - 顯示 CPU 線程活動
   - Python 主線程執行

### 🎯 關鍵功能使用

#### 查看 MoE 專家計算
1. **找到專家計算區域**
   ```
   在 Timeline 中找到以下模式：
   - API 呼叫: cudaLaunchKernel (linear transformation)
   - GPU 核心: sgemm, gemm_kernel 等
   - 記憶體傳輸: memcpy 操作
   ```

2. **分析計算重疊**
   - 查看 GPU 核心是否與記憶體傳輸重疊
   - 檢查是否有 GPU 空閒時間

#### 查看 NiXL 跨 GPU 傳輸
1. **識別 NiXL 傳輸**
   ```
   尋找以下模式：
   - CPU 活動: NiXL agent 初始化
   - GPU 記憶體操作: register_memory, transfer
   - 設備間通信: UCX 相關活動
   ```

2. **測量傳輸時間**
   - 點擊傳輸事件查看詳細信息
   - 測量從開始到完成的總時間

### 🔍 詳細分析步驟

#### 步驟 1: 概覽分析
```
1. 打開 profile 檔案
2. 使用 Ctrl+F 縮放到全視圖
3. 觀察整體執行模式：
   - GPU 使用率是否持續
   - 是否有明顯的空閒期
   - 記憶體傳輸與計算的比例
```

#### 步驟 2: 專家計算分析
```
1. 縮放到專家計算區域
2. 檢查每個專家的執行時間
3. 查看專家間的依賴關係
4. 分析是否有並行機會
```

#### 步驟 3: 記憶體傳輸分析
```
1. 查看 Memory Transfer Track
2. 識別傳輸大小和方向
3. 檢查傳輸是否與計算重疊
4. 找出傳輸瓶頸
```

### 🎨 GUI 操作技巧

#### 導航技巧
- **W/A/S/D**: 縮放和平移
- **滾輪**: 水平縮放
- **Shift+滾輪**: 垂直縮放
- **空格**: 重置視圖
- **F**: 適配選中區域

#### 測量技巧
```
1. 選擇區域測量:
   - 按住左鍵拖拽選擇時間範圍
   - 底部顯示選中區域的持續時間

2. 事件詳細信息:
   - 點擊任何事件查看詳細信息
   - 右側面板顯示參數和性能數據

3. 比較功能:
   - 選擇多個相似事件
   - 使用統計面板比較性能
```

### 📈 關鍵指標查看

#### GPU 使用率分析
```
1. 查看 GPU Utilization Track
2. 尋找使用率低於 80% 的區域
3. 識別可能的優化機會
```

#### 記憶體頻寬分析
```
1. 查看 Memory Bandwidth Track
2. 檢查是否接近硬體極限
3. 分析傳輸效率
```

## 2. Nsight Compute GUI - 核心詳細分析

### 啟動 Nsight Compute GUI

```bash
# 打開已生成的核心 profile
ncu-ui moe_kernel_profile.ncu-rep

# 或直接啟動 GUI
ncu-ui
```

### 🔬 核心分析視圖

#### Source View
```
功能: 顯示與性能相關的源代碼
用途: 
- 查看哪些代碼行消耗最多時間
- 識別熱點代碼
- 理解核心執行流程
```

#### Details View
```
功能: 顯示詳細的性能指標
重要指標:
- SM Efficiency: 流式多處理器效率
- Memory Throughput: 記憶體吞吐量
- Occupancy: 佔用率
- Warp Execution Efficiency: Warp 執行效率
```

#### Summary View
```
功能: 高層次的性能概覽
包含:
- 執行時間
- 主要瓶頸識別
- 優化建議
```

## 3. 實際示範步驟

### 步驟 1: 生成適合 GUI 分析的 Profile

```bash
# 生成包含完整信息的 profile
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    --output=moe_gui_analysis \
    python moe_profiling_example.py
```

### 步驟 2: 打開 Nsight Systems GUI

```bash
# 啟動 GUI 並載入檔案
nsight-sys moe_gui_analysis.nsys-rep &
```

### 步驟 3: GUI 分析工作流程

#### 在 Timeline 中分析 MoE 流程
1. **找到專家計算模式**
   ```
   查找模式:
   Token 準備 → 專家選擇 → 跨 GPU 傳輸 → 專家計算 → 結果傳輸
   ```

2. **測量關鍵時間**
   ```
   - 專家計算時間
   - 傳輸時間
   - 等待時間
   - 總延遲
   ```

3. **識別瓶頸**
   ```
   查看是否有:
   - GPU 空閒時間
   - 記憶體傳輸排隊
   - CPU-GPU 同步點
   ```

### 步驟 4: 使用 Statistics 面板

```
位置: 視窗底部
功能:
1. 選擇感興趣的事件類型
2. 查看統計摘要
3. 比較不同執行的性能
4. 導出統計數據
```

## 4. 進階技巧

### 🎯 專門針對 MoE 的分析

#### 專家負載平衡分析
```
1. 選擇所有專家計算核心
2. 比較執行時間分布
3. 檢查是否有明顯的不平衡
```

#### NiXL 傳輸效率分析
```
1. 找到 NiXL 相關的 CPU 活動
2. 測量傳輸建立時間
3. 比較實際傳輸時間與理論值
```

### 📊 自定義視圖

#### 創建專門的 Track
```
1. 右鍵點擊 Track 區域
2. 選擇 "Add Track"
3. 選擇特定的 API 或核心類型
4. 創建專門的 MoE 分析視圖
```

#### 設置過濾器
```
1. 使用 Filter 功能
2. 只顯示特定的事件類型
3. 隱藏不相關的信息
```

## 5. 常見問題解決

### GUI 無法啟動
```bash
# 檢查顯示環境
echo $DISPLAY

# 如果是遠程連接，啟用 X11 轉發
ssh -X username@hostname

# 或使用 VNC/遠程桌面
```

### Profile 檔案太大
```bash
# 使用更短的 profiling 時間
timeout 30s nsys profile python moe_profiling_example.py

# 或減少追蹤的事件類型
nsys profile --trace=cuda python moe_profiling_example.py
```

### 權限問題
```bash
# 確保有讀取權限
chmod 644 *.nsys-rep *.ncu-rep

# 確保在正確的目錄
ls -la *.nsys-rep
```

## 6. 快速啟動指令

```bash
# 生成 GUI 友好的 profile 並打開
nsys profile --trace=cuda,nvtx,osrt --output=gui_analysis python moe_profiling_example.py && nsight-sys gui_analysis.nsys-rep &

# 核心級分析
sudo ncu --set basic -o kernel_analysis python moe_profiling_example.py && ncu-ui kernel_analysis.ncu-rep &
```

現在您可以使用這些工具來詳細分析您的 MoE 系統的時間軸性能！Timeline trace 將為您提供直觀的視覺化，幫助您理解程式的執行流程和性能瓶頸。


