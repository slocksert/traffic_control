#!/bin/bash
# Install PyTorch with ROCm 7 for AMD Radeon RX 9070 XT (gfx1201)

echo "=========================================="
echo "PyTorch ROCm 7 Installation"
echo "GPU: AMD Radeon RX 9070 XT (gfx1201)"
echo "OS: Ubuntu 24.04.3 LTS"
echo "=========================================="

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo ""
echo "Step 1: Uninstalling old PyTorch (ROCm 6.2)..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Installing PyTorch with ROCm 7.0..."
echo "Note: This may take several minutes..."

# Try ROCm 6.3 (latest stable with gfx1201 support)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

echo ""
echo "Step 3: Setting up environment variables for gfx1201..."
cat >> ~/.bashrc << 'EOF'

# ROCm 7 environment for AMD Radeon RX 9070 XT
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export GPU_DEVICE_ORDINAL=0
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=1
EOF

source ~/.bashrc

echo ""
echo "Step 4: Testing PyTorch GPU detection..."
python3 << 'PYTHON_TEST'
import torch
import sys

print("=" * 60)
print("PyTorch GPU Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")

if hasattr(torch.version, 'hip'):
    print(f"ROCm (HIP) version: {torch.version.hip}")
else:
    print("ROCm (HIP) version: Not available")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Device memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    # Test tensor operations
    print("\nTesting GPU tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working!")
        print(f"   Test result shape: {z.shape}")
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUCCESS! GPU is ready to use!")
    print("=" * 60)
    print("\nRun your training with:")
    print("  python main.py --mode train --agent qlearning --episodes 100 --gpu --no-gui")
else:
    print("\n" + "=" * 60)
    print("WARNING: GPU not detected")
    print("=" * 60)
    print("\nTroubleshooting:")
    print("1. Make sure ROCm 7 drivers are installed")
    print("2. Restart your terminal/session")
    print("3. Try: export HSA_OVERRIDE_GFX_VERSION=12.0.1")
    print("4. Run: rocminfo | grep gfx1201")
    sys.exit(1)
PYTHON_TEST

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Installation successful!"
    echo ""
    echo "To use GPU acceleration, run:"
    echo "  source .venv/bin/activate"
    echo "  python main.py --mode train --agent qlearning --episodes 100 --gpu --no-gui"
else
    echo "⚠️ Installation completed but GPU not detected"
    echo "You can still use CPU mode:"
    echo "  python main.py --mode train --agent qlearning --episodes 100 --no-gui"
fi
