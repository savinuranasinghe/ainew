#!/bin/bash
# create-assets.sh - Create all required asset files

echo "🎨 Creating professional installer assets..."

# Create assets directory
mkdir -p electron-app/assets

# Create license.txt
echo "📄 Creating license file..."
cat > electron-app/assets/license.txt << 'EOF'
AI Image Upscaler - End User License Agreement (EULA)

Copyright (c) 2024 Your Name. All rights reserved.

LICENSE AGREEMENT

This software is provided "as is" without warranty of any kind, express or implied, 
including but not limited to the warranties of merchantability, fitness for a 
particular purpose and noninfringement.

GRANT OF LICENSE
You are granted a non-exclusive license to use this software for personal and 
commercial purposes subject to the following conditions:

PERMITTED USES
- Use the software on any number of devices you own or control
- Create backup copies for personal use
- Use for both personal and commercial image enhancement

RESTRICTIONS
- You may not reverse engineer, decompile, or disassemble the software
- You may not distribute or sell copies of this software without permission
- You may not remove any copyright notices or proprietary markings
- You may not use this software for illegal purposes

DISCLAIMER
This software uses AI technology for image enhancement. Results may vary based on 
input image quality and content. The software is provided without any guarantee 
of specific results or performance.

PRIVACY
This software processes images locally on your device. No images are sent to 
external servers unless explicitly stated.

TERMINATION
This license is effective until terminated. Your rights under this license will 
terminate automatically without notice if you fail to comply with any term(s) 
of this agreement.

SUPPORT
For support, updates, and more information, please visit:
https://github.com/yourusername/ai-image-upscaler

By installing and using this software, you acknowledge that you have read this 
agreement, understand it, and agree to be bound by its terms and conditions.

Last updated: December 2024
EOF

# Create simple icon files (placeholder - replace with real icons)
echo "🖼️ Creating placeholder icon files..."

# Create a simple colored square as placeholder icon (requires ImageMagick)
if command -v convert &> /dev/null; then
    echo "Creating icons with ImageMagick..."
    # Create base PNG icon (512x512)
    convert -size 512x512 gradient:blue-purple electron-app/assets/icon.png
    
    # Create ICO for Windows (multiple sizes)
    convert electron-app/assets/icon.png -resize 256x256 -resize 128x128 -resize 64x64 -resize 48x48 -resize 32x32 -resize 16x16 electron-app/assets/icon.ico
    
    # Create ICNS for macOS (requires additional tools)
    if command -v iconutil &> /dev/null; then
        mkdir -p electron-app/assets/icon.iconset
        convert electron-app/assets/icon.png -resize 16x16 electron-app/assets/icon.iconset/icon_16x16.png
        convert electron-app/assets/icon.png -resize 32x32 electron-app/assets/icon.iconset/icon_16x16@2x.png
        convert electron-app/assets/icon.png -resize 32x32 electron-app/assets/icon.iconset/icon_32x32.png
        convert electron-app/assets/icon.png -resize 64x64 electron-app/assets/icon.iconset/icon_32x32@2x.png
        convert electron-app/assets/icon.png -resize 128x128 electron-app/assets/icon.iconset/icon_128x128.png
        convert electron-app/assets/icon.png -resize 256x256 electron-app/assets/icon.iconset/icon_128x128@2x.png
        convert electron-app/assets/icon.png -resize 256x256 electron-app/assets/icon.iconset/icon_256x256.png
        convert electron-app/assets/icon.png -resize 512x512 electron-app/assets/icon.iconset/icon_256x256@2x.png
        convert electron-app/assets/icon.png -resize 512x512 electron-app/assets/icon.iconset/icon_512x512.png
        convert electron-app/assets/icon.png -resize 1024x1024 electron-app/assets/icon.iconset/icon_512x512@2x.png
        iconutil -c icns electron-app/assets/icon.iconset
        rm -rf electron-app/assets/icon.iconset
    fi
else
    echo "⚠️  ImageMagick not found. Please add your own icon files:"
    echo "   - electron-app/assets/icon.png (512x512)"
    echo "   - electron-app/assets/icon.ico (for Windows)"
    echo "   - electron-app/assets/icon.icns (for macOS)"
fi

# Create README for users
echo "📖 Creating user README..."
cat > electron-app/assets/README.txt << 'EOF'
AI Image Upscaler v1.0.0
========================

Professional 4x image upscaling using SwinIR AI technology.

INSTALLATION INSTRUCTIONS:

Windows:
1. Run the installer (.exe file)
2. Follow the setup wizard
3. Choose installation options:
   - Desktop shortcut (recommended)
   - Start Menu shortcut (recommended)
   - File associations (recommended)
4. Click Install and wait for completion
5. Launch from Desktop or Start Menu

macOS:
1. Open the .dmg file
2. Drag AI Image Upscaler to Applications folder
3. If you get a security warning:
   - Right-click the app → Open
   - Or go to System Preferences → Security & Privacy → Allow

USAGE:
1. Launch AI Image Upscaler
2. Drag and drop an image into the window
3. Or click "Browse" to select an image
4. Click "Upscale Image"
5. Wait for processing (30 seconds to 2 minutes)
6. Download your enhanced 4x resolution image

SUPPORTED FORMATS:
Input: JPG, JPEG, PNG, BMP, TIFF
Output: PNG (highest quality)

SYSTEM REQUIREMENTS:
- Windows 10+ or macOS 10.14+
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- 64-bit processor

TROUBLESHOOTING:
- Slow processing: Close other apps, ensure adequate RAM
- Won't start: Try running as administrator (Windows) or check security settings (macOS)
- Large images take longer to process

For support: https://github.com/savinuranasinghe/ai-image-upscaler
EOF

echo "✅ Assets created successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Replace placeholder icons with your own designs"
echo "2. Update the GitHub URL in license.txt and README.txt"
echo "3. Update 'Your Name' in license.txt with your actual name"
echo "4. Run the build script to create professional installers"