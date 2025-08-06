// Start detection
async function startDetection() {
    const classSelect = document.getElementById('classSelect');
    const sessionName = document.getElementById('sessionName').value.trim();
    const sessionStartTime = document.getElementById('sessionStartTime').value;
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (!classSelect.value) {
        showAlert('danger', 'Please select a class!');
        return;
    }
    if (!sessionName) {
        showAlert('danger', 'Please enter a session name!');
        return;
    }
    if (!sessionStartTime) {
        showAlert('danger', 'Please set the class start time!');
        return;
    }

    startBtn.disabled = true;
    startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    
    try {
        const response = await fetch('/api/start-detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                session_name: sessionName,
                session_start_time: sessionStartTime,
                class_name: classSelect.value
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentSessionId = result.session_id;
            isDetectionActive = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            showAlert('success', 'Detection started successfully!');
            
            // Start attendance updates now that detection is active
            startAttendanceUpdates();
            
        } else {
            showAlert('danger', result.error || 'Failed to start detection');
        }
    } catch (error) {
        showAlert('danger', 'Error starting detection: ' + error.message);
    } finally {
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    }
}

// Stop detection
async function stopDetection() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    stopBtn.disabled = true;
    stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
    
    try {
        const response = await fetch('/api/stop-detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentSessionId = null;
            isDetectionActive = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            showAlert('success', 'Detection stopped successfully!');
            
            // Stop attendance updates now that detection is inactive
            stopAttendanceUpdates();
            
            // Clear attendance display
            updateLiveAttendanceFeed({});
            updateAbsentStudentsList({});
            
        } else {
            showAlert('danger', result.error || 'Failed to stop detection');
        }
    } catch (error) {
        showAlert('danger', 'Error stopping detection: ' + error.message);
    } finally {
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
    }
}

// Update attendance only (simplified - no need to check detection_active)
async function updateAttendanceOnly() {
    try {
        const response = await fetch('/api/attendance');
        const data = await response.json();
        
        const presentCount = document.getElementById('presentCount');
        const totalStudents = document.getElementById('totalStudents');
        
        // Update counts if elements exist
        if (presentCount) {
            presentCount.textContent = data.attendance?.present_students || 0;
        }
        if (totalStudents) {
            totalStudents.textContent = data.attendance?.all_students?.length || 0;
        }
        
        // Update attendance display
        updateLiveAttendanceFeed(data.attendance);
        updateAbsentStudentsList(data.attendance);
        
        // Debug logging
        console.log('Attendance update:', {
            detection_active: data.detection_active,
            present_students: data.attendance?.present_students,
            attendance_data: data.attendance?.attendance,
            all_students: data.attendance?.all_students
        });
        
    } catch (error) {
        console.error('Error updating attendance:', error);
    }
}

// Update system status (keep this as is - it's useful for system health)
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        const statusIndicator = document.getElementById('systemStatus');
        const statusText = document.getElementById('statusText');
        const presentCount = document.getElementById('presentCount');
        const totalStudents = document.getElementById('totalStudents');
        
        // Update status indicator
        if (status.face_system_ready) {
            statusIndicator.className = 'status-indicator status-active';
            statusText.textContent = 'System Ready';
        } else {
            statusIndicator.className = 'status-indicator status-inactive';
            statusText.textContent = 'System Error';
        }
        
        // Update counts if elements exist
        if (presentCount) {
            presentCount.textContent = status.current_session?.present_students || 0;
        }
        if (totalStudents) {
            totalStudents.textContent = status.students_count || 0;
        }
        
        // Debug logging
        console.log('Status update:', {
            detection_active: status.detection_active,
            present_students: status.current_session?.present_students,
            attendance_data: status.current_session?.attendance,
            all_students: status.current_session?.all_students
        });
        
        // Update live attendance feed only if detection is active
        if (status.detection_active) {
            updateLiveAttendanceFeed(status.current_session);
            updateAbsentStudentsList(status.current_session);
        }
        
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Check detection status for standby/idle states
function checkDetectionStatus() {
    fetch('/api/detection-status')
        .then(response => response.json())
        .then(data => {
            const idleOverlay = document.getElementById('idleOverlay');
            const standbyIndicator = document.getElementById('standbyIndicator');
            const standbyOverlay = document.getElementById('standbyOverlay');
            
            console.log('Detection status:', data); // Debug logging
            
            // Only show overlays if detection is actually active
            if (!isDetectionActive) {
                // Hide all overlays if detection is not active
                idleOverlay.style.display = 'none';
                standbyIndicator.style.display = 'none';
                standbyOverlay.style.display = 'none';
                return;
            }
            
            if (data.state === 'idle' && data.idle_overlay_active) {
                // Show idle overlay (highest priority)
                idleOverlay.style.display = 'flex';
                standbyIndicator.style.display = 'none';
                standbyOverlay.style.display = 'none';
                console.log('Showing idle overlay');
            } else if (data.state === 'standby') {
                // Show standby overlay
                idleOverlay.style.display = 'none';
                standbyIndicator.style.display = 'block';
                standbyOverlay.style.display = 'flex';
                console.log('Showing standby overlay');
            } else {
                // Normal active state - hide all overlays
                idleOverlay.style.display = 'none';
                standbyIndicator.style.display = 'none';
                standbyOverlay.style.display = 'none';
                console.log('Hiding all overlays - active state');
            }
        })
        .catch(error => {
            console.error('Error checking detection status:', error);
        });
}

// Resume detection from idle state - using new dedicated endpoint
async function resumeDetection() {
    const resumeBtn = event.target;
    const originalText = resumeBtn.innerHTML;
    
    try {
        // Show loading state
        resumeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resuming...';
        resumeBtn.disabled = true;

        // Use the new dedicated endpoint for resuming from idle
        const response = await fetch('/api/resume-from-idle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Update local state
            isDetectionActive = true;
            
            // Update UI to show resumed session
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            // Update button states - detection is now active
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Hide ALL overlays immediately and prevent them from showing again
            document.getElementById('idleOverlay').style.display = 'none';
            document.getElementById('standbyOverlay').style.display = 'none';
            document.getElementById('standbyIndicator').style.display = 'none';
            
            // Start attendance updates
            startAttendanceUpdates();
            
            showAlert('success', 'Detection resumed successfully!');
            
            // Temporarily disable detection status checks to prevent overlay from reappearing
            if (statusUpdateInterval) {
                clearInterval(statusUpdateInterval);
                // Restart status updates after a delay to ensure backend has reset
                setTimeout(() => {
                    startStatusUpdates();
                }, 3000); // Wait 3 seconds before resuming status checks
            }
            
        } else {
            throw new Error(result.error || 'Failed to resume detection');
        }
    } catch (error) {
        console.error('Error resuming detection:', error);
        showAlert('danger', 'Error resuming detection: ' + error.message);
    } finally {
        // Always reset button state, even if there was an error
        resumeBtn.innerHTML = originalText;
        resumeBtn.disabled = false;
    }
}

// Capture current frame and detect unknown faces
async function captureCurrentFrame() {
    try {
        const response = await fetch('/api/capture-screenshot', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            if (result.unknown_faces.length > 0) {
                showUnknownFacesModal(result.unknown_faces, result.screenshot_path);
            } else {
                showAlert('info', 'No unknown faces detected in the current frame.');
            }
        } else {
            showAlert('danger', result.error || 'Failed to capture frame.');
        }
    } catch (error) {
        showAlert('danger', 'Error capturing frame: ' + error.message);
    }
} 