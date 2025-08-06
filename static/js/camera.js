// Camera toggle functions
function switchCamera(mode) {
    if (isDetectionActive) {
        showAlert('warning', 'Cannot switch camera while detection is active');
        return;
    }
    
    performCameraSwitch(mode);
}

async function performCameraSwitch(mode) {
    try {
        const response = await fetch('/api/camera/switch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ mode: mode })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentCameraMode = mode;
            
            if (mode === 'ip') {
                // For IP mode, show the configuration panel
                showIpCameraSetup();
                showAlert('info', 'Please configure your IP camera address');
            } else {
                // For local mode, hide IP settings and update UI
                document.getElementById('ipCameraSettings').style.display = 'none';
                updateCameraUI();
                showAlert('success', 'Switched to local camera');
            }
        } else {
            showAlert('danger', result.message || 'Failed to switch camera');
        }
    } catch (error) {
        showAlert('danger', 'Error switching camera: ' + error.message);
    }
}

function showIpCameraSetup() {
    // Show IP camera settings panel
    const ipSettings = document.getElementById('ipCameraSettings');
    ipSettings.style.display = 'block';
    
    // Focus on the IP input field
    const ipInput = document.getElementById('ipCameraUrl');
    ipInput.focus();
    
    // Update camera status to show "needs configuration"
    updateCameraStatus();
}

async function setIpCameraUrl() {
    const ipUrl = document.getElementById('ipCameraUrl').value.trim();
    
    if (!ipUrl) {
        showAlert('danger', 'Please enter an IP camera URL');
        return;
    }
    
    try {
        const response = await fetch('/api/camera/set-ip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ip_url: ipUrl })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', result.message);
            document.getElementById('ipCameraSettings').style.display = 'none';
            updateCameraUI();
        } else {
            showAlert('danger', result.message || 'Failed to configure IP camera');
        }
    } catch (error) {
        showAlert('danger', 'Error configuring IP camera: ' + error.message);
    }
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

async function updateCameraStatus() {
    try {
        const response = await fetch('/api/camera/status');
        const status = await response.json();
        
        const statusIndicator = document.getElementById('cameraStatus');
        const statusText = document.getElementById('cameraStatusText');
        
        if (status.mode === 'ip') {
            if (status.needs_configuration) {
                statusIndicator.className = 'status-indicator status-warning';
                statusText.textContent = 'IP Camera - Needs Configuration';
            } else if (status.connected) {
                statusIndicator.className = 'status-indicator status-active';
                statusText.textContent = 'IP Camera - Connected';
            } else {
                statusIndicator.className = 'status-indicator status-inactive';
                statusText.textContent = 'IP Camera - Disconnected';
            }
        } else {
            if (status.connected) {
                statusIndicator.className = 'status-indicator status-active';
                statusText.textContent = 'Local Camera - Connected';
            } else {
                statusIndicator.className = 'status-indicator status-inactive';
                statusText.textContent = 'Local Camera - Disconnected';
            }
        }
    } catch (error) {
        console.error('Error updating camera status:', error);
    }
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