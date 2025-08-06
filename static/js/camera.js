/**
 * Camera Control Module
 * 
 * This module handles camera-related functionality including camera
 * switching, IP camera configuration, and camera status monitoring.
 * Currently supports local camera with IP camera functionality
 * marked as "Feature TBC" (To Be Confirmed).
 * 
 * Key Features:
 * - Camera mode switching (Local/IP)
 * - IP camera configuration
 * - Camera status monitoring
 * - Connection testing
 * 
 * Camera Modes:
 * - Local: Built-in or USB webcam
 * - IP Camera: Network camera (currently disabled)
 * 
 * Status Monitoring:
 * - Connection status
 * - Camera availability
 * - Configuration status
 */

/**
 * Switch Camera Mode
 * 
 * Switches between local and IP camera modes. Currently only
 * local camera is fully functional, with IP camera marked as
 * a future feature.
 * 
 * Process:
 * 1. Validate camera mode
 * 2. Send switch request to backend
 * 3. Update UI state
 * 4. Handle configuration requirements
 * 
 * @param {string} mode - Camera mode ('local' or 'ip')
 * @async
 * @throws {Error} If switch fails
 */
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

/**
 * Show IP Camera Setup
 * 
 * Displays the IP camera configuration panel when IP camera
 * mode is selected. Currently disabled as IP camera is TBC.
 * 
 * Process:
 * 1. Show configuration panel
 * 2. Focus on input field
 * 3. Update camera status
 */
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

/**
 * Set IP Camera URL
 * 
 * Configures the IP camera URL and tests the connection.
 * Currently disabled as IP camera functionality is TBC.
 * 
 * Process:
 * 1. Validate URL format
 * 2. Send configuration to backend
 * 3. Test connection
 * 4. Update status
 * 
 * @async
 * @throws {Error} If configuration fails
 */
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

/**
 * Test IP Camera Connection
 * 
 * Tests the connection to an IP camera URL to verify
 * it's accessible and working. Currently disabled.
 * 
 * Process:
 * 1. Validate URL
 * 2. Test connection
 * 3. Report results
 * 
 * @async
 * @throws {Error} If test fails
 */
async function testIpCamera() {
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

/**
 * Test And Switch To IP Camera
 * 
 * Tests the connection to an IP camera URL and, if successful,
 * switches the camera mode to IP camera.
 * 
 * Process:
 * 1. Validate URL
 * 2. Test connection
 * 3. If successful, switch to IP camera mode
 * 4. Show success message
 * 
 * @async
 * @throws {Error} If test fails
 */
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

/**
 * Update Camera UI
 * 
 * Synchronizes the UI buttons and settings based on the current
 * camera mode. Updates button states and visibility of IP settings.
 */
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

/**
 * Update Camera Status
 * 
 * Updates the camera status display to show current
 * connection state and camera mode.
 * 
 * Process:
 * 1. Fetch camera status from backend
 * 2. Update status indicators
 * 3. Show connection state
 * 
 * @async
 * @throws {Error} If status update fails
 */
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