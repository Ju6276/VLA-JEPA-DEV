#!/bin/bash
# Quick training test to verify the full pipeline
# This script runs a few training steps to ensure everything works

echo "=========================================="
echo "Quick Training Test"
echo "=========================================="
echo ""

# Set the config path
CONFIG_PATH="config/lam_lerobot.yaml"

echo "1. Testing configuration loading..."
python -c "
import yaml
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
print('✓ Configuration loaded successfully')
print(f\"  Data mix: {config['data']['data_mix']}\")
print(f\"  Batch size: {config['data']['batch_size']}\")
print(f\"  Num frames: {config['data']['num_frames']}\")
"

echo ""
echo "2. Running quick training test (10 steps)..."
echo "   This will verify:"
echo "   - Data loading works"
echo "   - Model forward pass works"
echo "   - Loss computation works"
echo "   - Backward pass works"
echo ""

python main.py fit \
    --config $CONFIG_PATH \
    --trainer.max_steps=10 \
    --trainer.log_every_n_steps=1 \
    --trainer.enable_checkpointing=false \
    --model.enable_wandb=false

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Quick training test PASSED!"
    echo "=========================================="
    echo ""
    echo "The training pipeline is working correctly."
    echo "You can now start full training with:"
    echo "  bash train.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Quick training test FAILED"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    exit 1
fi


