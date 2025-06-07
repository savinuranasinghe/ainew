#!/bin/bash
# build-all.sh - Cross-platform build script for macOS/Linux

set -e

echo "🚀 Building AI Image Upscaler for distribution..."
echo "Platform: $(uname)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "electron-app/package.json" ] || [ ! -f "python-backend/main.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf dist/
rm -rf electron-app/dist/
rm -rf electron-app/backend/
rm -rf python-backend/dist/
rm -rf python-backend/build/
print_success "Cleanup complete"

# Check Python and dependencies
print_status "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    print_error "pip is required but not installed"
    exit 1
fi

# Install Python dependencies if needed
if [ ! -f "python-backend/venv/bin/activate" ] && [ ! -f "python-backend/venv/Scripts/activate" ]; then
    print_status "Creating Python virtual environment..."
    cd python-backend
    python3 -m venv venv
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
    
    pip install --upgrade pip
    pip install pyinstaller
    pip install -r requirements.txt || {
        print_warning "requirements.txt not found, installing essential packages..."
        pip install fastapi uvicorn pillow torch torchvision numpy requests timm
    }
    cd ..
else
    print_status "Activating existing Python environment..."
    cd python-backend
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
    cd ..
fi

# Build Python backend
print_status "Building Python backend with PyInstaller..."
cd python-backend

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_status "Creating requirements.txt..."
    cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
pillow==10.1.0
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
requests==2.31.0
timm==0.9.12
EOF
fi

# Create backend.spec if it doesn't exist
if [ ! -f "backend.spec" ]; then
    print_status "Creating PyInstaller spec file..."
    cat > backend.spec << 'EOF'
import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('swinir_arch.py', '.'),
        ('../models', 'models'),
    ],
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'torch',
        'torchvision',
        'timm',
        'PIL',
        'numpy',
        'fastapi',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ai-upscaler-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF
fi

# Install PyInstaller if not already installed
pip show pyinstaller > /dev/null 2>&1 || {
    print_status "Installing PyInstaller..."
    pip install pyinstaller
}

# Build with PyInstaller
print_status "Running PyInstaller..."
pyinstaller backend.spec --clean --noconfirm

if [ ! -f "dist/ai-upscaler-backend" ]; then
    print_error "PyInstaller build failed - executable not found"
    exit 1
fi

print_success "Python backend built successfully"
cd ..

# Prepare Electron app
print_status "Preparing Electron application..."
mkdir -p electron-app/backend

# Copy backend executable
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp python-backend/dist/ai-upscaler-backend electron-app/backend/
    chmod +x electron-app/backend/ai-upscaler-backend
elif [[ "$OSTYPE" == "linux"* ]]; then
    cp python-backend/dist/ai-upscaler-backend electron-app/backend/
    chmod +x electron-app/backend/ai-upscaler-backend
else
    # Windows/MinGW
    cp python-backend/dist/ai-upscaler-backend.exe electron-app/backend/
fi

print_success "Backend copied to Electron app"

# Install Node.js dependencies and build Electron app
print_status "Building Electron application..."
cd electron-app

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    print_error "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is required but not installed"
    exit 1
fi

# Install dependencies
print_status "Installing Node.js dependencies..."
npm install

# Install electron-builder if not present
npm list electron-builder > /dev/null 2>&1 || {
    print_status "Installing electron-builder..."
    npm install --save-dev electron-builder
}

# Build the application
print_status "Building Electron app for current platform..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    npm run build-mac
elif [[ "$OSTYPE" == "linux"* ]]; then
    npm run build-win  # Cross-compile for Windows on Linux (if configured)
    print_warning "Linux build completed. For native Linux builds, update package.json build config."
else
    npm run build-win
fi

cd ..

# Create release directory
print_status "Organizing release files..."
mkdir -p releases/v1.0.0

# Copy built files
if [ -d "electron-app/dist" ]; then
    cp -r electron-app/dist/* releases/v1.0.0/
    print_success "Files copied to releases/v1.0.0/"
fi

# Create user documentation
print_status "Creating user documentation..."
cat > releases/v1.0.0/README.md << 'EOF'
# AI Image Upscaler v1.0.0

## What's New
- Professional 4x image upscaling using SwinIR AI
- Smart tiling for large images
- Drag & drop interface
- Cross-platform support (macOS & Windows)

## Installation

### macOS
1. Download the `.dmg` file
2. Open it and drag the app to Applications folder
3. Launch from Applications (you may need to right-click and "Open" the first time)

### Windows
1. Download the `-Setup.exe` file for installer version
2. Or download the `-portable.exe` for portable version
3. Run the installer or portable executable
4. Launch the application

## System Requirements
- **macOS**: 10.14 or later
- **Windows**: Windows 10 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **CPU**: 64-bit processor

## Usage
1. Launch AI Image Upscaler
2. Drag an image into the window or click to browse
3. Click "Upscale Image" 
4. Wait for processing (may take 30 seconds to 2 minutes)
5. Download your enhanced 4x resolution image

## Supported Formats
- Input: JPG, JPEG, PNG, BMP, TIFF
- Output: PNG (highest quality)

## Troubleshooting
- If the app won't start, try running as administrator (Windows) or check Security & Privacy settings (macOS)
- For slow processing, close other applications to free up memory
- Large images (>2000px) may take longer to process

## Support
For issues or questions, please check the GitHub repository or contact the developer.

---
Built with ❤️ using Electron and SwinIR AI technology
EOF

print_success "Build completed successfully!"
print_success "Distribution files are available in: releases/v1.0.0/"

# Show summary
echo ""
echo "📦 Build Summary:"
echo "=================="
ls -la releases/v1.0.0/
echo ""
print_success "Your AI Image Upscaler is ready for distribution!"
print_status "Share the files in releases/v1.0.0/ with your friends"

# Optional: Create ZIP archive
read -p "📦 Create ZIP archive for easy sharing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating ZIP archive..."
    cd releases
    zip -r "AI-Image-Upscaler-v1.0.0.zip" v1.0.0/
    print_success "ZIP archive created: releases/AI-Image-Upscaler-v1.0.0.zip"
    cd ..
fi