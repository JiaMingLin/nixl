#!/bin/bash

# NIXL GPU-to-GPU 效能測試執行腳本

set -e

echo "NIXL GPU-to-GPU 效能測試"
echo "========================"

# 檢查 NIXL_PLUGIN_DIR 是否設定
if [ -z "$NIXL_PLUGIN_DIR" ]; then
    echo "錯誤: NIXL_PLUGIN_DIR 環境變數未設定"
    echo "請設定 NIXL_PLUGIN_DIR 指向 NIXL 插件目錄"
    exit 1
fi

echo "NIXL 插件目錄: $NIXL_PLUGIN_DIR"

# 檢查 GPU 可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: nvidia-smi 未找到，可能沒有 NVIDIA GPU"
else
    echo "GPU 資訊:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
fi

echo ""

# 檢查 Python 和必要套件
if ! python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 數量: {torch.cuda.device_count()}')" 2>/dev/null; then
    echo "錯誤: PyTorch 未正確安裝或 CUDA 不可用"
    exit 1
fi

echo ""

# 執行測試
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 提供不同的測試選項
echo "選擇測試模式:"
echo "1. 快速測試 (簡化版 NIXL)"
echo "2. 單一測試 (簡化版 NIXL)"
echo "3. 完整效能測試 (NIXL)"
echo "4. 快速完整測試 (NIXL)"
echo "5. PyTorch 效能測試"
echo "6. PyTorch 簡化測試"
echo "7. NIXL vs PyTorch 比較測試"

read -p "請選擇 (1-7): " choice

case $choice in
    1)
        echo "執行簡化版效能測試..."
        python3 "$SCRIPT_DIR/simple_gpu_benchmark.py"
        ;;
    2)
        echo "執行單一測試..."
        python3 "$SCRIPT_DIR/simple_gpu_benchmark.py" --single
        ;;
    3)
        echo "執行完整效能測試..."
        python3 "$SCRIPT_DIR/nixl_gpu_benchmark.py" --iterations 10
        ;;
    4)
        echo "執行快速完整測試..."
        python3 "$SCRIPT_DIR/nixl_gpu_benchmark.py" --quick
        ;;
    5)
        echo "執行 PyTorch 效能測試..."
        python3 "$SCRIPT_DIR/pytorch_gpu_benchmark.py"
        ;;
    6)
        echo "執行 PyTorch 簡化測試..."
        python3 "$SCRIPT_DIR/simple_pytorch_benchmark.py"
        ;;
    7)
        echo "執行 NIXL vs PyTorch 比較測試..."
        python3 "$SCRIPT_DIR/compare_nixl_pytorch.py"
        ;;
    *)
        echo "無效的選擇，使用預設的簡化測試"
        python3 "$SCRIPT_DIR/simple_gpu_benchmark.py"
        ;;
esac

echo ""
echo "測試完成!"
