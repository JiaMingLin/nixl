#!/bin/bash
# Nsight GUI 啟動腳本

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Nsight GUI 啟動工具 ===${NC}"
echo

# 檢查 DISPLAY 環境
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}設置 DISPLAY 環境變數...${NC}"
    export DISPLAY=:0
fi

echo -e "DISPLAY: ${GREEN}$DISPLAY${NC}"

# 檢查可用的 profile 檔案
echo -e "${YELLOW}檢查可用的 profile 檔案...${NC}"

NSYS_FILES=(*.nsys-rep)
NCU_FILES=(*.ncu-rep)

if [ -f "${NSYS_FILES[0]}" ]; then
    echo -e "${GREEN}找到 Nsight Systems 檔案:${NC}"
    for file in "${NSYS_FILES[@]}"; do
        echo "  - $file"
    done
else
    echo -e "${RED}未找到 .nsys-rep 檔案${NC}"
fi

if [ -f "${NCU_FILES[0]}" ]; then
    echo -e "${GREEN}找到 Nsight Compute 檔案:${NC}"
    for file in "${NCU_FILES[@]}"; do
        echo "  - $file"
    done
else
    echo -e "${YELLOW}未找到 .ncu-rep 檔案${NC}"
fi

echo

# 選擇要打開的檔案類型
echo -e "${YELLOW}選擇要啟動的 GUI:${NC}"
echo "1) Nsight Systems (Timeline 分析)"
echo "2) Nsight Compute (核心分析)" 
echo "3) 兩者都打開"
echo "4) 退出"

read -p "請選擇 [1-4]: " choice

case $choice in
    1)
        if [ -f "${NSYS_FILES[0]}" ]; then
            echo -e "${GREEN}啟動 Nsight Systems...${NC}"
            echo "檔案: ${NSYS_FILES[0]}"
            
            # 檢查是否可以啟動 GUI
            if xhost &>/dev/null; then
                nsight-sys "${NSYS_FILES[0]}" &
                echo -e "${GREEN}✓ Nsight Systems 已在背景啟動${NC}"
            else
                echo -e "${RED}無法啟動 GUI，可能需要桌面環境${NC}"
                echo "替代方案：使用命令列查看統計："
                echo "nsys stats ${NSYS_FILES[0]}"
            fi
        else
            echo -e "${RED}未找到 .nsys-rep 檔案，請先執行 profiling${NC}"
        fi
        ;;
    2)
        if [ -f "${NCU_FILES[0]}" ]; then
            echo -e "${GREEN}啟動 Nsight Compute...${NC}"
            echo "檔案: ${NCU_FILES[0]}"
            
            if xhost &>/dev/null; then
                ncu-ui "${NCU_FILES[0]}" &
                echo -e "${GREEN}✓ Nsight Compute 已在背景啟動${NC}"
            else
                echo -e "${RED}無法啟動 GUI，可能需要桌面環境${NC}"
                echo "替代方案：使用命令列查看："
                echo "ncu --import ${NCU_FILES[0]} --print-summary"
            fi
        else
            echo -e "${RED}未找到 .ncu-rep 檔案，請先執行核心級 profiling${NC}"
            echo "執行: sudo ncu --set basic -o kernel_profile python moe_profiling_example.py"
        fi
        ;;
    3)
        echo -e "${GREEN}啟動兩個 GUI...${NC}"
        
        if [ -f "${NSYS_FILES[0]}" ]; then
            if xhost &>/dev/null; then
                nsight-sys "${NSYS_FILES[0]}" &
                echo -e "${GREEN}✓ Nsight Systems 已啟動${NC}"
            fi
        fi
        
        if [ -f "${NCU_FILES[0]}" ]; then
            if xhost &>/dev/null; then
                ncu-ui "${NCU_FILES[0]}" &
                echo -e "${GREEN}✓ Nsight Compute 已啟動${NC}"
            fi
        fi
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}無效選擇${NC}"
        exit 1
        ;;
esac

echo
echo -e "${YELLOW}GUI 使用提示：${NC}"
echo "📊 Nsight Systems Timeline 查看要點："
echo "  - 使用滾輪縮放時間軸"
echo "  - 點擊事件查看詳細信息"
echo "  - 查看 GPU Kernels track 了解核心執行"
echo "  - 查看 Memory Transfer track 了解傳輸模式"
echo

echo "🔬 Nsight Compute 核心分析要點："
echo "  - 查看 Details 頁面的性能指標"
echo "  - 注意 Occupancy 和 Memory Throughput"
echo "  - 檢查 Source 頁面的熱點代碼"
echo

echo -e "${GREEN}更多詳細說明請查看: nsight_gui_guide.md${NC}"

# 如果是在遠程連接，提供額外建議
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
    echo
    echo -e "${YELLOW}⚠️  檢測到遠程連接${NC}"
    echo "如果 GUI 無法顯示，請嘗試："
    echo "1. 使用 ssh -X 連接以啟用 X11 轉發"
    echo "2. 或使用 VNC/遠程桌面"
    echo "3. 或在本地機器上安裝 Nsight 並複製 profile 檔案"
fi


