#!/bin/bash
# build-professional.sh - Professional installer build script

set -e

echo "🚀 Building Professional AI Image Upscaler Installers..."
echo "Platform: $(uname)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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

print_header() {
    echo -e "${PURPLE}🎯 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "electron-app/package.json" ] || [ ! -f "python-backend/main.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_header "STEP 1: CLEANUP & PREPARATION"
print_status "Cleaning previous builds..."
rm -rf dist/
rm -rf electron-app/dist/
rm -rf electron-app/backend/
rm -rf python-backend/dist/
rm -rf python-backend/build/
rm -rf releases/
print_success "Cleanup complete"

print_header "STEP 2: PYTHON ENVIRONMENT SETUP"
print_status "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    print_error "pip is required but not installed"
    exit 1
fi

# Setup Python virtual environment
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

print_header "STEP 3: PYTHON BACKEND BUILD"
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
print_status "Building Python backend with PyInstaller..."
pyinstaller backend.spec --clean --noconfirm

if [ ! -f "dist/ai-upscaler-backend" ]; then
    print_error "PyInstaller build failed - executable not found"
    exit 1
fi

print_success "Python backend built successfully"
cd ..

print_header "STEP 4: ELECTRON APP PREPARATION"
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

# Create assets if they don't exist
print_status "Setting up installer assets..."
if [ ! -d "electron-app/assets" ]; then
    mkdir -p electron-app/assets
    print_warning "Assets folder created. Please add icon files before next build!"
fi

# Create basic license if it doesn't exist
if [ ! -f "electron-app/assets/license.txt" ]; then
    cat > electron-app/assets/license.txt << 'EOF'
AI Image Upscaler - End User License Agreement (EULA)

Copyright (c) 2024 Your Name. All rights reserved.

This software is provided "as is" without warranty of any kind, express or implied.

GRANT OF LICENSE
You are granted a non-exclusive license to use this software for personal and commercial purposes.

RESTRICTIONS
- You may not reverse engineer, decompile, or disassemble the software
- You may not distribute or sell copies of this software
- You may not remove any copyright notices

By installing this software, you agree to these terms.
EOF
    print_warning "Basic license created. Please customize it with your information!"
fi

print_header "STEP 5: NODE.JS & ELECTRON BUILD"
cd electron-app

# Check Node.js installation
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    print_error "Please install Node.js from https://nodejs.org/"
    exit 1
fi

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

# Build the professional installers
print_status "Building professional installers..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Building for macOS (Intel + Apple Silicon)..."
    npm run build-mac
    
    print_status "Cross-compiling for Windows..."
    npm run build-win || print_warning "Windows cross-compile failed. Build on Windows for best results."
elif [[ "$OSTYPE" == "linux"* ]]; then
    print_status "Building for Linux and cross-compiling for Windows..."
    npm run build-win
else
    print_status "Building for Windows..."
    npm run build-win
fi

cd ..

print_header "STEP 6: ORGANIZING PROFESSIONAL RELEASE"
print_status "Creating professional release structure..."
mkdir -p releases/v1.0.0/installers
mkdir -p releases/v1.0.0/portable
mkdir -p releases/v1.0.0/docs

# Copy built files with better organization
if [ -d "electron-app/dist" ]; then
    # Organize by type
    find electron-app/dist -name "*.dmg" -exec cp {} releases/v1.0.0/installers/ \; 2>/dev/null || true
    find electron-app/dist -name "*Setup*.exe" -exec cp {} releases/v1.0.0/installers/ \; 2>/dev/null || true
    find electron-app/dist -name "*.pkg" -exec cp {} releases/v1.0.0/installers/ \; 2>/dev/null || true
    find electron-app/dist -name "*portable*.exe" -exec cp {} releases/v1.0.0/portable/ \; 2>/dev/null || true
    find electron-app/dist -name "*.zip" -exec cp {} releases/v1.0.0/portable/ \; 2>/dev/null || true
    
    print_success "Installers organized by type"
fi

# Create comprehensive user documentation
print_status "Creating user documentation..."
cat > releases/v1.0.0/README.md << 'EOF'
# 🚀 AI Image Upscaler v1.0.0

Professional AI-powered 4x image upscaling using SwinIR technology.

## 📦 Installation Options

### Windows Users

#### Option 1: Full Installer (Recommended)
- **File**: `AI Image Upscaler Setup 1.0.0.exe` (in `installers/` folder)
- **Features**: 
  - Professional installation wizard
  - Desktop shortcut creation
  - Start Menu integration
  - File associations (right-click images → "Upscale with AI")
  - Automatic uninstaller
- **Installation**: Run the installer and follow the wizard

#### Option 2: Portable Version
- **File**: `AI Image Upscaler-1.0.0-portable.exe` (in `portable/` folder)
- **Features**: 
  - No installation required
  - Run from anywhere (USB drive, Desktop, etc.)
  - Smaller download
- **Usage**: Download and double-click to run

### macOS Users

#### Option 1: DMG Installer (Recommended)
- **File**: `AI Image Upscaler-1.0.0.dmg` (in `installers/` folder)
- **Installation**: 
  1. Double-click the DMG file
  2. Drag the app to Applications folder
  3. If security warning appears: Right-click app → Open

#### Option 2: PKG Installer
- **File**: `AI Image Upscaler-1.0.0.pkg` (in `installers/` folder)
- **Features**: System-level installation with automatic shortcuts

## 🎯 Quick Start

1. **Install** using your preferred method above
2. **Launch** AI Image Upscaler
3. **Drag & drop** an image or click "Browse Files"
4. **Click "Upscale Image"** and wait for processing
5. **Download** your enhanced 4x resolution image

## 🖼️ Supported Formats

- **Input**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Output**: PNG (highest quality preservation)

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10+ (64-bit) or macOS 10.14+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: 64-bit processor

### Recommended
- **RAM**: 8GB or more
- **SSD**: For faster processing
- **GPU**: Any modern graphics card (optional, uses CPU by default)

## 🔧 Troubleshooting

### Windows Issues
- **"Windows protected your PC"**: Click "More info" → "Run anyway"
- **App won't start**: Right-click → "Run as administrator"
- **Slow processing**: Close other applications to free RAM

### macOS Issues
- **"App can't be opened"**: Right-click → Open → Open again
- **Security warning**: System Preferences → Security & Privacy → "Allow anyway"
- **Can't move app**: Run in Terminal: `sudo xattr -rd com.apple.quarantine "/Applications/AI Image Upscaler.app"`

### Performance Tips
- **Close other apps** during processing
- **Use smaller images** for faster results (app handles large images automatically)
- **Ensure adequate free disk space** (at least 1GB)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-image-upscaler/issues)
- **Documentation**: [GitHub Wiki](https://github.com/yourusername/ai-image-upscaler/wiki)
- **Updates**: Check GitHub for new releases

## 📄 License

Licensed under MIT License. See `license.txt` for details.

---
**Built with ❤️ using Electron and SwinIR AI technology**
EOF

# Create installation guide
cat > releases/v1.0.0/INSTALLATION-GUIDE.txt << 'EOF'
INSTALLATION GUIDE - AI Image Upscaler v1.0.0
==============================================

WINDOWS INSTALLATION:
1. Download "AI Image Upscaler Setup 1.0.0.exe" from installers/ folder
2. Double-click to run the installer
3. If Windows shows security warning:
   - Click "More info"
   - Click "Run anyway"
4. Follow the installation wizard:
   - Accept license agreement
   - Choose installation folder (or keep default)
   - Select components:
     ✓ Desktop shortcut (recommended)
     ✓ Start Menu shortcut (recommended)
     ✓ File associations (recommended)
   - Click Install
5. Launch from Desktop shortcut or Start Menu

MACOS INSTALLATION:
1. Download "AI Image Upscaler-1.0.0.dmg" from installers/ folder
2. Double-click the DMG file to open it
3. Drag "AI Image Upscaler" to the Applications folder
4. Launch from Applications folder
5. If security warning appears:
   - Right-click the app
   - Select "Open"
   - Click "Open" in the dialog

PORTABLE VERSION (Windows):
1. Download "AI Image Upscaler-1.0.0-portable.exe" from portable/ folder
2. Save to desired location (Desktop, USB drive, etc.)
3. Double-click to run - no installation needed!

POST-INSTALLATION:
- First launch may take 30-60 seconds to initialize
- Try upscaling a small test image to verify everything works
- Check the README.md for usage instructions and troubleshooting

For support: https://github.com/yourusername/ai-image-upscaler
EOF

# Copy license to docs
cp electron-app/assets/license.txt releases/v1.0.0/docs/ 2>/dev/null || true

print_success "Professional documentation created"

print_header "STEP 7: FINAL VERIFICATION"
print_status "Verifying build results..."

echo ""
echo "📦 Professional Release Contents:"
echo "================================="
echo "📁 releases/v1.0.0/"
echo "├── 📁 installers/          (Full installers)"
ls -la releases/v1.0.0/installers/ 2>/dev/null || echo "   └── (no installer files found)"
echo "├── 📁 portable/            (Portable versions)"
ls -la releases/v1.0.0/portable/ 2>/dev/null || echo "   └── (no portable files found)"
echo "├── 📁 docs/                (Documentation)"
ls -la releases/v1.0.0/docs/ 2>/dev/null || echo "   └── (no docs found)"
echo "├── 📄 README.md            (User guide)"
echo "└── 📄 INSTALLATION-GUIDE.txt (Setup instructions)"
echo ""

print_success "Professional AI Image Upscaler build completed!"
print_success "Distribution files are in: releases/v1.0.0/"

echo ""
echo "🎉 WHAT YOUR FRIENDS GET:"
echo "========================"
echo "✅ Professional installer with wizard"
echo "✅ Desktop shortcuts automatically created"
echo "✅ Start Menu integration (Windows)"
echo "✅ File associations (right-click images)"
echo "✅ Clean uninstaller"
echo "✅ Comprehensive user documentation"
echo "✅ Multiple installation options"
echo ""
echo "🚀 Ready for professional distribution!"

# Optional: Create distribution archive
read -p "📦 Create ZIP archive for easy distribution? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating distribution archive..."
    cd releases
    zip -r "AI-Image-Upscaler-v1.0.0-Professional.zip" v1.0.0/
    print_success "Archive created: releases/AI-Image-Upscaler-v1.0.0-Professional.zip"
    cd ..
    echo ""
    echo "📤 Share this ZIP file with your friends for the complete package!"
fi