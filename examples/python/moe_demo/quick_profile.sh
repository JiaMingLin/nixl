#!/bin/bash
# 快速 MoE Profiling 腳本
# 用於日常性能監控和基準測試

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MoE 快速 Profiling 工具 ===${NC}"
echo "自動執行系統級和核心級 profiling"
echo

# 檢查環境
echo -e "${YELLOW}檢查環境...${NC}"
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}錯誤: Nsight Systems (nsys) 未安裝${NC}"
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    echo -e "${RED}錯誤: Nsight Compute (ncu) 未安裝${NC}"
    exit 1
fi

# 激活 conda 環境
echo -e "${YELLOW}激活 nixl 環境...${NC}"
if [ -f "~/miniconda3/bin/activate" ]; then
    source ~/miniconda3/bin/activate nixl
else
    echo -e "${RED}警告: 找不到 conda 環境，繼續使用當前環境${NC}"
fi

# 創建輸出目錄
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="profiling_results_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

echo -e "${GREEN}結果將保存到: $OUTPUT_DIR${NC}"
echo

# 1. 基本功能測試
echo -e "${YELLOW}步驟 1: 執行基本功能測試...${NC}"
if python test_moe_example.py > $OUTPUT_DIR/basic_test.log 2>&1; then
    echo -e "${GREEN}✓ 基本功能測試通過${NC}"
else
    echo -e "${RED}✗ 基本功能測試失敗，請檢查 $OUTPUT_DIR/basic_test.log${NC}"
    exit 1
fi

# 2. Nsight Systems Profiling
echo -e "${YELLOW}步驟 2: 執行系統級 profiling (Nsight Systems)...${NC}"
nsys profile \
    --trace=cuda,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=$OUTPUT_DIR/moe_systems_profile \
    python moe_profiling_example.py > $OUTPUT_DIR/nsys_output.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 系統級 profiling 完成${NC}"
    
    # 生成統計報告
    echo "  生成統計報告..."
    nsys stats $OUTPUT_DIR/moe_systems_profile.nsys-rep > $OUTPUT_DIR/nsys_stats.txt 2>&1
    echo -e "${GREEN}  ✓ 統計報告已保存到 nsys_stats.txt${NC}"
else
    echo -e "${RED}✗ 系統級 profiling 失敗${NC}"
fi

# 3. Nsight Compute Profiling (如果有 sudo 權限)
echo -e "${YELLOW}步驟 3: 執行核心級 profiling (Nsight Compute)...${NC}"
echo "注意: 這需要 sudo 權限來訪問 GPU 性能計數器"

# 詢問是否執行 Nsight Compute
read -p "是否執行 Nsight Compute profiling? (需要 sudo) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "執行核心級 profiling..."
    sudo env PATH=$PATH ncu \
        --set basic \
        -k regex:gemm \
        -c 3 \
        --export=$OUTPUT_DIR/moe_kernel_profile \
        python -c "
from moe_profiling_example import profile_expert_computation
profile_expert_computation()
" > $OUTPUT_DIR/ncu_output.log 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 核心級 profiling 完成${NC}"
    else
        echo -e "${YELLOW}⚠ 核心級 profiling 可能受到權限限制${NC}"
    fi
else
    echo "跳過核心級 profiling"
fi

# 4. 生成簡化報告
echo -e "${YELLOW}步驟 4: 生成性能報告...${NC}"

cat > $OUTPUT_DIR/performance_summary.txt << EOF
MoE Profiling 摘要報告
生成時間: $(date)
========================================

1. 基本測試結果:
$(tail -n 5 $OUTPUT_DIR/basic_test.log 2>/dev/null || echo "無可用數據")

2. GPU 信息:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "無法獲取 GPU 信息")

3. CUDA API 統計 (前 5 項):
$(grep -A 10 "CUDA API Summary" $OUTPUT_DIR/nsys_stats.txt 2>/dev/null | head -15 || echo "無可用數據")

4. GPU 核心統計 (前 5 項):  
$(grep -A 10 "CUDA GPU Kernel Summary" $OUTPUT_DIR/nsys_stats.txt 2>/dev/null | head -15 || echo "無可用數據")

5. 記憶體傳輸統計:
$(grep -A 10 "CUDA GPU MemOps Summary" $OUTPUT_DIR/nsys_stats.txt 2>/dev/null | head -15 || echo "無可用數據")

檔案列表:
$(ls -la $OUTPUT_DIR/)

建議:
- 查看 nsys_stats.txt 了解詳細性能數據
- 如果執行了 ncu profiling，查看 moe_kernel_profile.ncu-rep
- 使用 nsight-sys 或 ncu-ui 打開 .nsys-rep 和 .ncu-rep 檔案進行詳細分析

EOF

echo -e "${GREEN}✓ 性能報告已生成${NC}"

# 5. 顯示摘要
echo
echo -e "${GREEN}=== Profiling 完成 ===${NC}"
echo -e "結果保存在: ${GREEN}$OUTPUT_DIR${NC}"
echo
echo "主要檔案:"
echo "  - performance_summary.txt  : 性能摘要報告"
echo "  - moe_systems_profile.nsys-rep : Nsight Systems 分析檔案"
echo "  - nsys_stats.txt          : 詳細統計數據"
if [ -f "$OUTPUT_DIR/moe_kernel_profile.ncu-rep" ]; then
    echo "  - moe_kernel_profile.ncu-rep : Nsight Compute 分析檔案"
fi
echo

# 顯示快速統計
echo -e "${YELLOW}快速統計:${NC}"
if [ -f "$OUTPUT_DIR/nsys_stats.txt" ]; then
    echo "GPU 核心數量: $(grep -c "Context.*Stream.*Device" $OUTPUT_DIR/nsys_stats.txt 2>/dev/null || echo "N/A")"
    echo "CUDA API 呼叫總數: $(grep "Num Calls" $OUTPUT_DIR/nsys_stats.txt | head -1 | awk '{print $3}' 2>/dev/null || echo "N/A")"
    echo "記憶體傳輸總量: $(grep -A 5 "CUDA GPU MemOps Summary (by Size)" $OUTPUT_DIR/nsys_stats.txt | grep "Total (MB)" -A 3 | tail -n +2 | awk '{sum+=$1} END{printf "%.2f MB\n", sum}' 2>/dev/null || echo "N/A")"
fi

echo
echo -e "${GREEN}使用以下指令查看詳細結果:${NC}"
echo "  cat $OUTPUT_DIR/performance_summary.txt"
echo "  nsight-sys $OUTPUT_DIR/moe_systems_profile.nsys-rep"
if [ -f "$OUTPUT_DIR/moe_kernel_profile.ncu-rep" ]; then
    echo "  ncu-ui $OUTPUT_DIR/moe_kernel_profile.ncu-rep"
fi

echo
echo -e "${YELLOW}如需持續監控，可設置定期執行此腳本${NC}"


