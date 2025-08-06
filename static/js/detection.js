/**
 * Detection Control Module
 * 
 * This module handles all face detection session management including
 * starting, stopping, and resuming detection sessions. It manages the
 * communication with the backend detection system and handles the
 * detection state transitions.
 * 
 * Key Features:
 * - Session start/stop control
 * - Detection state management
 * - Idle mode handling
 * - Resume functionality
 * - Unknown face capture
 * 
 * State Management:
 * - Detection active/inactive states
 * - Session ID tracking
 * - Button state synchronization
 * - Real-time status updates
 * 
 * API Endpoints Used:
 * - POST /api/start-detection: Start new detection session
 * - POST /api/stop-detection: Stop current detection session
 * - POST /api/resume-from-idle: Resume from idle state
 * - POST /api/capture-screenshot: Capture unknown faces
 */

/**
 * Start Detection Session
 * 
 * Initiates a new face detection session with the specified parameters.
 * Validates input data, communicates with the backend to start detection,
 * and updates the UI state accordingly.
 * 
 * Required Parameters:
 * - Class selection (from dropdown)
 * - Session name (user input)
 * - Session start time (for punctuality tracking)
 * 
 * Process:
 * 1. Validate all required inputs
 * 2. Send start request to backend
 * 3. Update UI state (buttons, status)
 * 4. Start attendance updates
 * 5. Show success/error feedback
 * 
 * @async
 * @throws {Error} If detection fails to start
 */
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

/**
 * Stop Detection Session
 * 
 * Terminates the current detection session and saves the attendance data.
 * Cleans up the session state and resets the UI to the initial state.
 * 
 * Process:
 * 1. Send stop request to backend
 * 2. Clear session ID
 * 3. Reset UI state
 * 4. Stop attendance updates
 * 5. Clear attendance displays
 * 
 * @async
 * @throws {Error} If detection fails to stop
 */
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

/**
 * Update attendance only (simplified - no need to check detection_active)
 * 
 * Fetches and displays attendance data from the backend.
 * Updates the present count, total students, and attendance feed.
 * 
 * @async
 * @throws {Error} If attendance data cannot be fetched
 */
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

/**
 * Update system status (keep this as is - it's useful for system health)
 * 
 * Fetches and displays system status from the backend.
 * Updates status indicators and counts.
 * 
 * @async
 * @throws {Error} If system status cannot be fetched
 */
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

/**
 * Check detection status for standby/idle states
 * 
 * Polls the backend to determine if the system is in an idle or standby
 * state and displays appropriate overlays.
 * 
 * @async
 * @throws {Error} If detection status cannot be fetched
 */
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

/**
 * Resume Detection from Idle State
 * 
 * Resumes detection after the system has entered idle mode due to
 * no face detection for an extended period. Restarts the detection
 * loop and updates the UI state.
 * 
 * Idle State Triggers:
 * - No faces detected for 120 seconds (configurable)
 * - System automatically enters idle monitoring mode
 * 
 * Process:
 * 1. Send resume request to backend
 * 2. Update detection state
 * 3. Hide idle overlay
 * 4. Restart detection loop
 * 
 * @async
 * @throws {Error} If resume fails
 */
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

/**
 * Capture Unknown Faces
 * 
 * Captures the current video frame and identifies unknown faces for
 * potential addition to the student database. Opens a modal for
 * managing unrecognized faces.
 * 
 * Process:
 * 1. Capture current video frame
 * 2. Detect unknown faces in frame
 * 3. Display faces in modal
 * 4. Allow user to add as new students
 * 
 * @async
 * @throws {Error} If capture fails
 */
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