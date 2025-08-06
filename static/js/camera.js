// Camera toggle functions
function switchCamera(mode) {
    if (isDetectionActive) {
        showAlert('warning', 'Cannot switch camera while detection is active');
        return;
    }
    
    // For IP camera mode, show settings first instead of immediately switching
    if (mode === 'ip') {
        showIpCameraSetup();
        return;
    }
    
    // For local camera, switch immediately
    performCameraSwitch(mode);
}

function showIpCameraSetup() {
    // Show IP camera settings without switching yet
    const ipSettings = document.getElementById('ipCameraSettings');
    const localBtn = document.getElementById('localCameraBtn');
    const ipBtn = document.getElementById('ipCameraBtn');
    
    ipSettings.style.display = 'block';
    ipBtn.classList.add('active');
    localBtn.classList.remove('active');
    
    // Don't actually switch camera mode yet - just show the setup
    showAlert('info', 'Please configure your IP camera URL and test the connection before switching.');
}

function performCameraSwitch(mode) {
    fetch('/api/camera/switch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode: mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentCameraMode = data.mode;
            updateCameraUI();
            showAlert('success', `Switched to ${data.mode} camera`);
            // Refresh video feed
            const videoFeed = document.getElementById('videoFeed');
            if (videoFeed) {
                videoFeed.src = '/video_feed?' + new Date().getTime();
            }
        } else {
            showAlert('warning', data.message);
        }
    })
    .catch(error => {
        console.error('Error switching camera:', error);
        showAlert('danger', 'Error switching camera');
    });
}

function setIpCameraUrl() {
    const url = document.getElementById('ipCameraUrl').value;
    
    if (!url) {
        showAlert('danger', 'Please enter an IP camera URL');
        return;
    }
    
    fetch('/api/camera/set-ip', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ip_url: url })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'IP camera URL saved. Click "Test & Switch" to verify connection and switch to IP camera mode.');
        } else {
            showAlert('danger', data.error);
        }
    })
    .catch(error => {
        console.error('Error setting IP camera URL:', error);
        showAlert('danger', 'Error updating IP camera URL');
    });
}

function updateCameraUI() {
    const localBtn = document.getElementById('localCameraBtn');
    const ipBtn = document.getElementById('ipCameraBtn');
    const ipSettings = document.getElementById('ipCameraSettings');
    
    // Update button states
    localBtn.classList.toggle('active', currentCameraMode === 'local');
    ipBtn.classList.toggle('active', currentCameraMode === 'ip');
    
    // Show/hide IP settings only if we're actually in IP mode
    ipSettings.style.display = currentCameraMode === 'ip' ? 'block' : 'none';
}

function updateCameraStatus() {
    fetch('/api/camera/status')
    .then(response => response.json())
    .then(data => {
        const statusIndicator = document.getElementById('cameraStatus');
        const statusText = document.getElementById('cameraStatusText');
        
        if (data.connected) {
            statusIndicator.className = 'status-indicator status-active';
            statusText.textContent = `${data.mode.charAt(0).toUpperCase() + data.mode.slice(1)} Camera Connected`;
        } else {
            statusIndicator.className = 'status-indicator status-inactive';
            statusText.textContent = 'Camera Disconnected';
        }
        
        currentCameraMode = data.mode;
        updateCameraUI();
    })
    .catch(error => {
        console.error('Error getting camera status:', error);
    });
}

// Test IP camera connection
function testIpCamera() {
    const url = document.getElementById('ipCameraUrl').value;
    const testBtn = event.target;
    const originalText = testBtn.innerHTML;
    
    testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    testBtn.disabled = true;
    
    fetch('/api/camera/test-ip', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ip_url: url })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'IP camera connection successful!');
        } else {
            showAlert('danger', data.error || 'Failed to connect to IP camera');
        }
    })
    .catch(error => {
        console.error('Error testing IP camera:', error);
        showAlert('danger', 'Error testing IP camera connection');
    })
    .finally(() => {
        testBtn.innerHTML = originalText;
        testBtn.disabled = false;
    });
}

function testAndSwitchToIpCamera() {
    const url = document.getElementById('ipCameraUrl').value;
    const testBtn = event.target;
    const originalText = testBtn.innerHTML;
    
    if (!url) {
        showAlert('danger', 'Please enter an IP camera URL first');
        return;
    }
    
    testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    testBtn.disabled = true;
    
    fetch('/api/camera/test-ip', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ip_url: url })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'IP camera connection successful! Switching to IP camera mode...');
            // Now switch to IP camera mode
            performCameraSwitch('ip');
        } else {
            showAlert('danger', data.error || 'Failed to connect to IP camera');
        }
    })
    .catch(error => {
        console.error('Error testing IP camera:', error);
        showAlert('danger', 'Error testing IP camera connection');
    })
    .finally(() => {
        testBtn.innerHTML = originalText;
        testBtn.disabled = false;
    });
} 