// Load sessions
async function loadSessions() {
    const sessionsList = document.getElementById('sessionsList');
    sessionsList.innerHTML = '<div class="loading"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Loading sessions...</p></div>';
    
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        if (data.sessions && data.sessions.length > 0) {
            let html = '';
            data.sessions.forEach(session => {
                const startTime = new Date(session.start_time).toLocaleString();
                const duration = session.duration_minutes ? `${session.duration_minutes.toFixed(1)} min` : 'N/A';
                
                html += `
                    <div class="session-item" onclick="showSessionDetails('${session.session_id}')">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="mb-1">${session.session_id}</h6>
                                <small class="text-muted">Started: ${startTime}</small>
                            </div>
                            <div class="text-end">
                                <div class="fw-bold">${session.present_students}/${session.total_students}</div>
                                <small class="text-muted">${duration}</small>
                            </div>
                        </div>
                    </div>
                `;
            });
            sessionsList.innerHTML = html;
        } else {
            sessionsList.innerHTML = '<p class="text-muted">No sessions found.</p>';
        }
    } catch (error) {
        sessionsList.innerHTML = '<p class="text-danger">Error loading sessions: ' + error.message + '</p>';
    }
}

// Show session details
async function showSessionDetails(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`);
        const session = await response.json();
        
        const modalBody = document.getElementById('sessionModalBody');
        const exportBtn = document.getElementById('exportBtn');
        
        exportBtn.onclick = () => exportSession(sessionId);
        
        let html = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Session Information</h6>
                    <p><strong>ID:</strong> ${session.session_id}</p>
                    <p><strong>Start Time:</strong> ${new Date(session.start_time).toLocaleString()}</p>
                    <p><strong>End Time:</strong> ${session.end_time ? new Date(session.end_time).toLocaleString() : 'N/A'}</p>
                    <p><strong>Duration:</strong> ${session.duration_minutes ? session.duration_minutes.toFixed(1) + ' minutes' : 'N/A'}</p>
                </div>
                <div class="col-md-6">
                    <h6>Statistics</h6>
                    <p><strong>Total Students:</strong> ${session.total_students}</p>
                    <p><strong>Present:</strong> ${session.present_students}</p>
                    <p><strong>Attendance Rate:</strong> ${(session.attendance_rate * 100).toFixed(1)}%</p>
                    <p><strong>Total Detections:</strong> ${session.total_detections}</p>
                </div>
            </div>
        `;
        
        if (session.attendance && Object.keys(session.attendance).length > 0) {
            html += `
                <hr>
                <h6>Attendance Details</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Student</th>
                                <th>First Seen</th>
                                <th>Last Seen</th>
                                <th>Confidence</th>
                                <th>Detections</th>
                                <th>Lateness</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
        
            // Sort students by prefix number
            function extractStudentNumber(studentName) {
                const match = studentName.match(/^(\d+)_/);
                return match ? parseInt(match[1]) : Infinity;
            }
            
            const sortedEntries = Object.entries(session.attendance).sort((a, b) => {
                return extractStudentNumber(a[0]) - extractStudentNumber(b[0]);
            });
            
            sortedEntries.forEach(([student, data]) => {
                html += `
                    <tr>
                        <td><strong>${student}</strong></td>
                        <td>${new Date(data.first_seen).toLocaleTimeString()}</td>
                        <td>${new Date(data.last_seen).toLocaleTimeString()}</td>
                        <td>${(data.confidence * 100).toFixed(1)}%</td>
                        <td>${data.detection_count}</td>
                        <td>
                            <span class="badge bg-${data.lateness?.color || 'secondary'}">
                                ${data.lateness?.status || 'Unknown'}
                            </span>
                        </td>
                    </tr>
                `;
            });
            
            html += `
                        </tbody>
                    </table>
                </div>
            `;
        }
        
        modalBody.innerHTML = html;
        
        const modal = new bootstrap.Modal(document.getElementById('sessionModal'));
        modal.show();
        
    } catch (error) {
        showAlert('danger', 'Error loading session details: ' + error.message);
    }
}

// Export session
async function exportSession(sessionId) {
    try {
        // This will trigger the CSV generation and download
        window.open(`/api/export/${sessionId}`, '_blank');
    } catch (error) {
        showAlert('danger', 'Error exporting session: ' + error.message);
    }
}

// Show resume session modal with available sessions
function showResumeSessionModal(sessions) {
    const resumeSessionList = document.getElementById('resumeSessionList');
    
    if (!sessions || sessions.length === 0) {
        resumeSessionList.innerHTML = '<p class="text-muted">No sessions found for this class.</p>';
    } else {
        let html = '';
        sessions.forEach(session => {
            const startTime = new Date(session.start_time).toLocaleString();
            const presentCount = session.present_students || 0;
            const totalCount = session.total_students || 0;
            
            html += `
                <div class="session-item" onclick="resumeSession('${session.session_id}')">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${session.session_name || session.session_id}</h6>
                            <small class="text-muted">Started: ${startTime}</small>
                        </div>
                        <div class="text-end">
                            <div class="fw-bold">${presentCount}/${totalCount}</div>
                            <small class="text-muted">Click to resume</small>
                        </div>
                    </div>
                </div>
            `;
        });
        resumeSessionList.innerHTML = html;
    }
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('resumeSessionModal'));
    modal.show();
}

// Resume a specific session
async function resumeSession(sessionId) {
    try {
        // Show loading state
        const resumeSessionList = document.getElementById('resumeSessionList');
        resumeSessionList.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Resuming session...</p></div>';
        
        // Call the resume session endpoint
        const response = await fetch('/api/resume-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('resumeSessionModal'));
            modal.hide();
            
            // Set the session as active
            currentSessionId = sessionId;
            isDetectionActive = true;
            
            // Update UI to show resumed session
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const sessionNameInput = document.getElementById('sessionName');
            const sessionStartTimeInput = document.getElementById('sessionStartTime');
            
            // Update form fields with session data
            if (result.session) {
                sessionNameInput.value = result.session.session_name || result.session.session_id;
                if (result.session.session_start_time) {
                    // Convert ISO string to datetime-local format
                    const sessionStartTime = new Date(result.session.session_start_time);
                    const localDateTime = sessionStartTime.toISOString().slice(0, 16);
                    sessionStartTimeInput.value = localDateTime;
                }
            }
            
            // Update button states - detection is now active
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Start attendance updates
            startAttendanceUpdates();
            
            // Update attendance display with resumed session data
            if (result.session && result.session.attendance) {
                updateLiveAttendanceFeed(result.session);
                updateAbsentStudentsList(result.session);
            }
            
            showAlert('success', `Session "${sessionId}" resumed successfully! Detection is now active.`);
            
        } else {
            resumeSessionList.innerHTML = '<p class="text-danger">Error: ' + (result.error || 'Failed to resume session') + '</p>';
        }
        
    } catch (error) {
        const resumeSessionList = document.getElementById('resumeSessionList');
        resumeSessionList.innerHTML = '<p class="text-danger">Error resuming session: ' + error.message + '</p>';
    }
} 