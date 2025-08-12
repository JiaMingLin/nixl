# 完整的 Timeline Trace 分析指南

本指南涵蓋了在各種環境下查看 Nsight timeline trace 的方法。

## 🖥️ 方法 1: 本地桌面環境（推薦）

### 如果您有本地桌面環境

```bash
# 1. 直接啟動 Nsight Systems GUI
nsight-sys moe_basic_profile.nsys-rep

# 2. 或使用我們的啟動腳本
./launch_gui.sh
```

### Timeline 查看重點

1. **整體執行流程**
   ```
   查看順序：
   Python 主線程 → CUDA API 呼叫 → GPU 核心執行 → 記憶體傳輸
   ```

2. **MoE 專家計算模式**
   ```
   典型模式：
   Token 準備 → Expert 選擇 → 跨 GPU 傳輸 → 專家計算 → 結果回傳
   ```

3. **性能瓶頸識別**
   ```
   尋找：
   - GPU 空閒時間（藍色區域中的空白）
   - 記憶體傳輸排隊（橙色/綠色條重疊）
   - CPU-GPU 同步點（垂直線對齊）
   ```

## 🔧 方法 2: 命令列分析（無 GUI 環境）

### 詳細統計分析

```bash
# 1. 基本統計摘要
nsys stats moe_basic_profile.nsys-rep

# 2. 導出到 SQLite 進行詳細分析
nsys export --type=sqlite moe_basic_profile.nsys-rep

# 3. 查看特定類型的統計
nsys stats --report cuda_api_sum moe_basic_profile.nsys-rep
nsys stats --report cuda_gpu_kern_sum moe_basic_profile.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum moe_basic_profile.nsys-rep
```

### 關鍵指標分析

```bash
# GPU 核心執行時間分析
echo "=== GPU 核心性能 ==="
nsys stats --report cuda_gpu_kern_sum moe_basic_profile.nsys-rep | head -20

# 記憶體傳輸分析  
echo "=== 記憶體傳輸性能 ==="
nsys stats --report cuda_gpu_mem_time_sum moe_basic_profile.nsys-rep

# CUDA API 開銷分析
echo "=== CUDA API 開銷 ==="
nsys stats --report cuda_api_sum moe_basic_profile.nsys-rep | head -15
```

### 自定義分析腳本

```python
# timeline_analyzer.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def analyze_timeline(nsys_sqlite_file):
    """分析 timeline 數據"""
    conn = sqlite3.connect(nsys_sqlite_file)
    
    # 查詢 GPU 核心執行時間
    kernels_df = pd.read_sql_query("""
        SELECT name, duration, start, end 
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
    """, conn)
    
    # 查詢記憶體傳輸
    memcpy_df = pd.read_sql_query("""
        SELECT copyKind, bytes, duration, start, end
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        ORDER BY start  
    """, conn)
    
    print("=== 核心執行統計 ===")
    print(f"總核心數: {len(kernels_df)}")
    print(f"總執行時間: {kernels_df['duration'].sum()/1e6:.2f} ms")
    print(f"平均核心時間: {kernels_df['duration'].mean()/1e3:.2f} μs")
    
    print("\n=== 記憶體傳輸統計 ===")
    print(f"總傳輸次數: {len(memcpy_df)}")
    print(f"總傳輸量: {memcpy_df['bytes'].sum()/1e6:.2f} MB")
    print(f"總傳輸時間: {memcpy_df['duration'].sum()/1e6:.2f} ms")
    
    conn.close()
    return kernels_df, memcpy_df

# 使用方法
kernels, memcpy = analyze_timeline('moe_basic_profile.sqlite')
```

## 🌐 方法 3: 遠程桌面/VNC（遠程伺服器）

### 設置 VNC 伺服器

```bash
# 1. 安裝 VNC 伺服器
sudo apt update
sudo apt install tigervnc-standalone-server

# 2. 設置 VNC 密碼
vncpasswd

# 3. 啟動 VNC 伺服器
vncserver :1 -geometry 1920x1080 -depth 24

# 4. 在 VNC 會話中啟動 Nsight
export DISPLAY=:1
nsight-sys moe_basic_profile.nsys-rep
```

### 使用 VNC 客戶端連接

```
連接地址: your_server_ip:5901
密碼: 您設置的 VNC 密碼
```

## 📱 方法 4: X11 轉發（SSH）

### 啟用 X11 轉發

```bash
# 1. 從本地連接到遠程伺服器
ssh -X username@server_ip

# 2. 或使用 trusted X11 轉發
ssh -Y username@server_ip

# 3. 測試 X11 轉發
xeyes  # 如果出現眼睛圖標，表示 X11 轉發工作正常

# 4. 啟動 Nsight
nsight-sys moe_basic_profile.nsys-rep
```

## 🔍 Timeline 分析重點（適用於所有方法）

### 1. MoE 專家計算分析

#### 尋找專家計算模式
```
Timeline 模式識別：
1. Token 輸入處理
   - 小的 GPU 核心（elementwise operations）
   - 記憶體分配操作
   
2. 專家選擇
   - 路由計算核心
   - 條件分支操作
   
3. 跨 GPU 傳輸（NiXL）
   - CPU 線程活動（NiXL agents）
   - Device-to-Device 記憶體傳輸
   
4. 專家計算
   - GEMM/SGEMM 核心（主要計算）
   - GeLU 激活函數核心
   
5. 結果聚合
   - 反向傳輸
   - 結果合併操作
```

#### 性能指標
```
關鍵測量：
- 專家計算時間 vs 傳輸時間比例
- GPU 使用率（核心執行密度）
- 記憶體頻寬使用率
- 不同專家的負載平衡
```

### 2. 記憶體傳輸模式分析

#### Host-Device 傳輸
```
查看項目：
- 藍色條：Host to Device（輸入數據）
- 綠色條：Device to Host（輸出結果）
- 條的長度：傳輸時間
- 條的高度：傳輸大小
```

#### Device-Device 傳輸（NiXL）
```
特殊模式：
- 橙色條：GPU 間直接傳輸
- 與 CPU 活動的時間對應
- 可能有初始化開銷
```

### 3. GPU 核心執行分析

#### 核心類型識別
```
常見核心：
- ampere_sgemm_*：矩陣乘法（專家的線性層）
- elementwise_*：元素級操作（激活函數）
- vectorized_*：向量化操作（記憶體操作）
- memcpy_*：記憶體複製
```

#### 執行效率指標
```
查看：
- 核心執行密度（是否有空隙）
- 相同類型核心的執行時間一致性
- 核心之間的依賴關係
```

## 📊 實際分析範例

### 基於我們的 Profile 數據

根據之前的 profiling 結果，您應該在 timeline 中看到：

1. **CUDA API 模式**
   ```
   29.0% - cuMemcpyAsync（頻繁的小傳輸）
   25.5% - cudaLaunchKernel（專家計算）
   13.2% - cudaDeviceSynchronize（同步開銷）
   ```

2. **GPU 核心模式**
   ```
   35.0% - distribution_elementwise（數據準備）
   17.9% - ampere_sgemm_128x64（主要計算）
   9.9% - ampere_sgemm_128x32（輔助計算）
   ```

3. **記憶體傳輸模式**
   ```
   64.7% - Device-to-Host（結果傳輸）
   35.3% - Host-to-Device（輸入傳輸）
   高頻率小批次傳輸（2,708 次，平均 24KB）
   ```

### 優化機會識別

在 Timeline 中查找這些模式：

1. **記憶體傳輸優化**
   ```
   問題：頻繁的小傳輸
   Timeline 表現：多個短的藍色/綠色條
   解決方案：批次化傳輸
   ```

2. **GPU 空閒時間**
   ```
   問題：GPU 核心軌道有空白
   Timeline 表現：GPU Kernels track 有間隙
   解決方案：流水線化或異步執行
   ```

3. **同步開銷**
   ```
   問題：過多的 CPU-GPU 同步
   Timeline 表現：垂直對齊的結束點
   解決方案：減少不必要的同步
   ```

## 🛠️ 快速啟動指令

```bash
# 1. 生成新的 timeline profile
nsys profile --trace=cuda,nvtx,osrt --output=new_timeline python moe_profiling_example.py

# 2. 如果有 GUI 環境
./launch_gui.sh

# 3. 如果只有命令列
nsys stats new_timeline.nsys-rep

# 4. 導出詳細數據進行自定義分析
nsys export --type=sqlite new_timeline.nsys-rep
```

現在您有了完整的工具集來分析 MoE 系統的 timeline trace，無論在什麼環境下都可以進行深入的性能分析！


