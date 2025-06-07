const { app, BrowserWindow, shell, dialog, Menu, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { existsSync } = require('fs');
const os = require('os');

let mainWindow;
let backendProcess = null;
let isQuitting = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: getAssetPath('icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false,
    autoHideMenuBar: true
  });

  createMenu();
  mainWindow.loadFile('index.html');
  
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    startBackend();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Image',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            openFileDialog();
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About AI Image Upscaler',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About AI Image Upscaler',
              message: 'AI Image Upscaler v1.0.0',
              detail: 'Professional 4x image upscaling using SwinIR AI technology.\n\nBuilt with Electron and Python.\n\nCopyright © 2024 Your Name',
              buttons: ['OK']
            });
          }
        },
        {
          label: 'Visit Website',
          click: () => {
            shell.openExternal('https://github.com/yourusername/ai-image-upscaler');
          }
        }
      ]
    }
  ];

  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    template[4].submenu = [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function openFileDialog() {
  dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      {
        name: 'Images',
        extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
      }
    ]
  }).then(result => {
    if (!result.canceled && result.filePaths.length > 0) {
      mainWindow.webContents.send('file-selected', result.filePaths[0]);
    }
  });
}

function getAssetPath(filename) {
  return path.join(__dirname, 'assets', filename);
}

function getBackendPath() {
  let backendName = 'ai-upscaler-backend';
  
  if (process.platform === 'win32') {
    backendName += '.exe';
  }

  console.log('=== BACKEND PATH DEBUG ===');
  console.log('App is packaged:', app.isPackaged);
  console.log('Platform:', process.platform);
  console.log('Process execPath:', process.execPath);
  console.log('Process resourcesPath:', process.resourcesPath);
  console.log('__dirname:', __dirname);
  
  if (app.isPackaged) {
    // In packaged app
    const backendPath = path.join(process.resourcesPath, 'backend', backendName);
    console.log('Packaged app backend path:', backendPath);
    console.log('Backend exists:', existsSync(backendPath));
    return backendPath;
  } else {
    // Development mode - try multiple locations
    const locations = [
      path.join(__dirname, 'backend', backendName),
      path.join(__dirname, '..', 'python-backend', 'dist', backendName),
      path.join(process.cwd(), 'python-backend', 'dist', backendName),
      path.join(process.cwd(), 'electron-app', 'backend', backendName)
    ];
    
    console.log('Development mode - searching:');
    for (const location of locations) {
      console.log(`  Checking: ${location} - exists: ${existsSync(location)}`);
      if (existsSync(location)) {
        console.log(`  ✅ Found backend at: ${location}`);
        return location;
      }
    }
    
    console.log('  ❌ Backend not found in any location');
    return locations[0];
  }
}

function startBackend() {
  const backendPath = getBackendPath();
  
  console.log('=== STARTING BACKEND ===');
  console.log('Backend path:', backendPath);
  console.log('Backend exists:', existsSync(backendPath));
  
  if (!existsSync(backendPath)) {
    console.error('Backend executable not found:', backendPath);
    showBackendError('Backend not found. Please ensure the application was installed correctly.');
    return;
  }

  console.log('Starting backend:', backendPath);
  
  try {
    backendProcess = spawn(backendPath, [], {
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: false,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'
      }
    });

    backendProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      console.log('Backend:', output);
    });

    backendProcess.stderr.on('data', (data) => {
      const message = data.toString().trim();
      console.error('Backend Error:', message);
      
      if (message.includes('Permission denied') || message.includes('cannot execute')) {
        showBackendError('Permission error: Please ensure the application has proper permissions.');
      }
    });

    backendProcess.on('close', (code) => {
      console.log('Backend process exited with code', code);
      if (code !== 0 && !isQuitting) {
        showBackendError(`Backend crashed with code ${code}. Try restarting the application.`);
      }
    });

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend:', error);
      showBackendError(`Failed to start backend: ${error.message}`);
    });

    // Wait 3 seconds before checking backend health
    setTimeout(checkBackendHealth, 3000);

  } catch (error) {
    console.error('Error spawning backend:', error);
    showBackendError(`Error starting backend: ${error.message}`);
  }
}

function checkBackendHealth() {
  console.log('=== CHECKING BACKEND HEALTH ===');
  const axios = require('axios');
  
  axios.get('http://127.0.0.1:8000/')
    .then(response => {
      console.log('✅ Backend is healthy:', response.data);
      console.log('Backend status:', response.data.model_status);
    })
    .catch(error => {
      console.warn('❌ Backend health check failed:', error.message);
      console.log('Will retry in 3 seconds...');
      
      // Retry once more after 3 seconds
      setTimeout(() => {
        axios.get('http://127.0.0.1:8000/')
          .then(response => {
            console.log('✅ Backend is healthy (retry):', response.data);
          })
          .catch(retryError => {
            console.error('❌ Backend still not responding:', retryError.message);
          });
      }, 3000);
    });
}

function showBackendError(message) {
  dialog.showErrorBox('Backend Error', message);
}

function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend...');
    
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', backendProcess.pid, '/f', '/t']);
    } else {
      backendProcess.kill('SIGTERM');
    }
    
    backendProcess = null;
  }
}

// Handle file associations (when user double-clicks image files)
app.setAsDefaultProtocolClient('ai-upscaler');

// Handle file opening on startup
if (process.platform === 'win32' && process.argv.length >= 2) {
  handleFileOpen(process.argv[process.argv.length - 1]);
}

app.on('open-file', (event, filePath) => {
  event.preventDefault();
  handleFileOpen(filePath);
});

function handleFileOpen(filePath) {
  if (mainWindow && filePath && filePath.match(/\.(jpg|jpeg|png|bmp|tiff|webp)$/i)) {
    mainWindow.webContents.send('file-opened', filePath);
  }
}

// App event handlers
app.whenReady().then(() => {
  console.log('=== APP STARTING ===');
  console.log('Electron version:', process.versions.electron);
  console.log('Node version:', process.versions.node);
  console.log('Platform:', process.platform);
  console.log('App path:', app.getAppPath());
  
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  isQuitting = true;
  stopBackend();
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', (event) => {
  isQuitting = true;
  
  if (backendProcess) {
    event.preventDefault();
    stopBackend();
    
    setTimeout(() => {
      app.quit();
    }, 1000);
  }
});

app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  if (url.startsWith('http://127.0.0.1:8000')) {
    event.preventDefault();
    callback(true);
  } else {
    callback(false);
  }
});

app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});