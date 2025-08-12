# MoE Profiling 指南

本指南說明如何使用 NVIDIA Nsight 工具對 MoE (Mixture-of-Experts) 程式進行詳細的 profiling。

## 前置需求

### 1. 安裝 NVIDIA Nsight 工具

```bash
# 安裝 Nsight Systems (系統級 profiling)
# 通常隨 CUDA Toolkit 一起安裝，或從 NVIDIA 開發者網站下載

# 安裝 Nsight Compute (核心級 profiling)  
# 同樣隨 CUDA Toolkit 安裝，或單獨下載

# 檢查安裝
nsys --version
ncu --version
```

### 2. 安裝 NVTX (可選但推薦)

```bash
# 在 nixl 環境中安裝
conda activate nixl
pip install nvtx-plugins
```

## Profiling 類型和工具選擇

### 1. Nsight Systems - 系統級 profiling
適用於：
- 整體程式執行流程分析
- GPU 使用率時間線
- GPU 間數據傳輸
- GPU-CPU 數據傳輸
- 記憶體分配/釋放
- CUDA API 呼叫追蹤

### 2. Nsight Compute - 核心級 profiling
適用於：
- 個別 CUDA kernel 詳細分析
- 記憶體頻寬使用
- 計算單元使用率
- Warp 執行效率
- Cache 命中率

## 使用方式

### 1. 基本 Nsight Systems Profiling

```bash
# 激活環境
source ~/miniconda3/bin/activate nixl
cd /home/jiaming/workplace/nixl/examples/python

# 基本 profiling
nsys profile --trace=cuda,nvtx,osrt --output=moe_profile python moe_profiling_example.py

# 詳細 profiling（包含記憶體追蹤）
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=moe_detailed_profile \
    python moe_profiling_example.py
```

### 2. Nsight Compute Profiling

```bash
# 基本核心分析
ncu --set full --output=moe_kernel_profile python moe_profiling_example.py

# 針對特定核心類型
ncu --kernel-regex="(?i)gemm|conv|linear" --set full --output=moe_compute_profile python moe_profiling_example.py

# 記憶體分析
ncu --set memory --output=moe_memory_profile python moe_profiling_example.py
```

### 3. 組合使用 - 完整分析

```bash
# 步驟 1: 使用 Nsight Systems 獲得整體視圖
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    --output=moe_systems_profile \
    python moe_profiling_example.py

# 步驟 2: 使用 Nsight Compute 深入分析特定核心
ncu \
    --set full \
    --kernel-regex="(?i)gemm|linear" \
    --launch-skip-before-match=0 \
    --launch-count=10 \
    --output=moe_compute_profile \
    python moe_profiling_example.py
```

## 分析重點

### 1. GPU 計算性能分析

查看指標：
- **SM 使用率** - 流式多處理器使用效率
- **記憶體頻寬使用率** - 全域記憶體頻寬使用
- **計算/記憶體比率** - 計算密集 vs 記憶體密集
- **Warp 執行效率** - 分支分化和佔用率

重點關注：
- Linear transformation kernels (GEMM 操作)
- GeLU activation 核心
- 記憶體分配/複製操作

### 2. GPU 間數據傳輸分析

查看指標：
- **P2P 傳輸頻寬** - GPU 間直接傳輸
- **PCIe 頻寬使用** - 通過 PCIe 的傳輸
- **傳輸延遲** - 傳輸開始到完成的時間
- **重疊程度** - 計算與傳輸的重疊

NiXL 特有分析：
- UCX 後端性能
- agent 初始化開銷
- 記憶體註冊開銷

### 3. GPU-CPU 數據傳輸分析

查看指標：
- **Host-Device 頻寬** - CPU-GPU 傳輸速度
- **Pinned vs Pageable** - 記憶體類型對性能的影響
- **非同步傳輸** - 傳輸與計算的重疊
- **記憶體複製開銷** - 複製操作的開銷

## 最佳化建議

### 1. 基於 Profiling 結果的最佳化

#### 如果 GPU 使用率低：
- 增加批次大小
- 減少 GPU-CPU 同步點
- 使用非同步操作

#### 如果記憶體頻寬受限：
- 最佳化記憶體存取模式
- 使用記憶體融合操作
- 減少記憶體分配/釋放

#### 如果傳輸開銷高：
- 重疊計算與傳輸
- 批次多個小傳輸
- 使用 pinned memory

### 2. MoE 特定最佳化

- **專家放置策略** - 基於通信成本最佳化專家分布
- **Token routing 最佳化** - 減少跨 GPU 傳輸
- **批次處理** - 批次多個 token 以攤銷傳輸成本

## 查看 Profiling 結果

### 1. Nsight Systems 結果
```bash
# 在 GUI 中打開（如果有桌面環境）
nsight-sys moe_profile.nsys-rep

# 命令列分析
nsys stats moe_profile.nsys-rep
nsys export --type=sqlite moe_profile.nsys-rep
```

### 2. Nsight Compute 結果
```bash
# 在 GUI 中打開
ncu-ui moe_kernel_profile.ncu-rep

# 命令列報告
ncu --import moe_kernel_profile.ncu-rep --print-summary
```

## 進階 Profiling 技巧

### 1. 自定義 NVTX 標記
程式中已包含 NVTX 標記，可以在 Nsight Systems 中看到：
- Expert_Forward
- Linear1_Computation  
- GeLU_Activation
- Linear2_Computation
- GPU_to_GPU_Transfer
- Complete_MoE_Workflow

### 2. 多執行比較
```bash
# 執行多次並比較
for i in {1..3}; do
    nsys profile --output=moe_run_$i python moe_profiling_example.py
done

# 比較結果
nsys stats moe_run_1.nsys-rep moe_run_2.nsys-rep moe_run_3.nsys-rep
```

### 3. 針對性分析
```bash
# 只分析 GPU 間傳輸
nsys profile --trace=cuda --cuda-memory-usage=true python -c "
from moe_profiling_example import profile_gpu_to_gpu_transfer
profile_gpu_to_gpu_transfer()
"

# 只分析專家計算  
ncu --set full python -c "
from moe_profiling_example import profile_expert_computation
profile_expert_computation()
"
```

## 常見問題和解決方案

### 1. Permission 錯誤
```bash
# 如果遇到 permission 錯誤，嘗試：
sudo nsys profile --trace=cuda,nvtx python moe_profiling_example.py
```

### 2. 記憶體不足
```bash
# 減少 profiling 的詳細程度
nsys profile --trace=cuda --sample=none python moe_profiling_example.py
```

### 3. NVTX 不可用
如果 NVTX 不可用，程式仍會正常執行，只是缺少自定義標記。

## 範例指令總結

```bash
# 進入正確的環境和目錄
source ~/miniconda3/bin/activate nixl
cd /home/jiaming/workplace/nixl/examples/python

# 完整的 profiling 流程
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=moe_complete_profile \
    python moe_profiling_example.py

# 查看結果摘要
nsys stats moe_complete_profile.nsys-rep
```
