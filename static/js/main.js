// Global variables
let currentSessionId = null;
let statusUpdateInterval = null;
let attendanceUpdateInterval = null;
let isDetectionActive = false;
let lastPresentStudents = new Set();
let currentCameraMode = 'local';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Show loading screen
    const loadingScreen = document.getElementById('loadingScreen');
    const loadingSubtext = document.getElementById('loadingSubtext');
    
    // Initialize systems step by step
    initializeApp();
    
    async function initializeApp() {
        try {
            // Step 1: Load classes
            loadingSubtext.textContent = 'Loading available classes...';
            await loadClasses();

            // Step 2: Update status
            loadingSubtext.textContent = 'Checking system status...';
            await updateStatus();
            
            // Step 3: Load students
            loadingSubtext.textContent = 'Loading students...';
            await loadStudents();
            
            // Step 4: Load sessions
            loadingSubtext.textContent = 'Loading session history...';
            await loadSessions();
            
            // Step 5: Hide loading screen
            setTimeout(() => {
                loadingScreen.classList.add('hidden');
            }, 500);
            
            // Start only status updates (not attendance updates)
            startStatusUpdates();
            
        } catch (error) {
            console.error('Error during initialization:', error);
            loadingSubtext.textContent = 'Error during initialization. Please refresh the page.';
        }
    }

    // Show/hide sections based on method
    document.querySelectorAll('input[name="addMethod"]').forEach(radio => {
        radio.addEventListener('change', function() {
            document.getElementById('singleImageSection').classList.toggle('d-none', this.value !== 'single');
            document.getElementById('folderUploadSection').classList.toggle('d-none', this.value !== 'folder');
            document.getElementById('existingFolderSection').classList.toggle('d-none', this.value !== 'existing');
        });
    });

    // Fix tab navigation - ensure video feed doesn't interfere
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('click', function(e) {
            // Pause video feed temporarily during tab switch
            const videoFeed = document.getElementById('videoFeed');
            if (videoFeed) {
                videoFeed.style.display = 'none';
                setTimeout(() => {
                    videoFeed.style.display = 'block';
                }, 100);
            }
        });
    });

    // Add event listeners for class and session name
    const classSelect = document.getElementById('classSelect');
    const sessionNameInput = document.getElementById('sessionName');
    const startBtn = document.getElementById('startBtn');
    const resumeSessionBtn = document.getElementById('resumeSessionBtn');

    function checkStartReady() {
        const classSelected = classSelect && classSelect.value;
        const sessionName = sessionNameInput && sessionNameInput.value.trim();
        startBtn.disabled = !(classSelected && sessionName);
        resumeSessionBtn.style.display = classSelected && !isDetectionActive ? '' : 'none';
    }
        
    if (classSelect) classSelect.addEventListener('change', checkStartReady);
    if (sessionNameInput) sessionNameInput.addEventListener('input', checkStartReady);

    // Initial check
    checkStartReady();

    // Handle resume session button click
    document.getElementById('resumeSessionBtn').onclick = async function() {
        try {
            // Fetch available sessions for the selected class
            const className = document.getElementById('classSelect').value;
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            // Filter sessions by class if needed
            let sessions = data.sessions || [];
            if (className) {
                sessions = sessions.filter(session => session.class_name === className);
            }
            
            // Show modal with available sessions
            showResumeSessionModal(sessions);
            
        } catch (error) {
            showAlert('danger', 'Error loading sessions: ' + error.message);
        }
    };

    // Add camera event listeners
    document.getElementById('localCameraBtn').addEventListener('click', () => switchCamera('local'));
    document.getElementById('ipCameraBtn').addEventListener('click', () => switchCamera('ip'));
    
    // Update camera status periodically
    updateCameraStatus();
    setInterval(updateCameraStatus, 5000); // Update every 5 seconds
});

// Start only status updates (not attendance updates)
function startStatusUpdates() {
    // Update status every 5 seconds
    statusUpdateInterval = setInterval(() => {
        updateStatus();
        // Also check detection status for standby/idle states
        if (isDetectionActive) {
            checkDetectionStatus();
        }
    }, 5000);
}

// Start attendance updates when detection starts
function startAttendanceUpdates() {
    if (attendanceUpdateInterval) {
        clearInterval(attendanceUpdateInterval);
    }
    
    attendanceUpdateInterval = setInterval(() => {
        // Only update if we're on the live tab and detection is active
        const liveTab = document.getElementById('live-tab');
        const liveTabPane = document.getElementById('live');
        const isLiveTabActive = (
            (liveTab && liveTab.classList.contains('active')) ||
            (liveTabPane && liveTabPane.classList.contains('active')) ||
            (liveTabPane && liveTabPane.classList.contains('show'))
        );
        
        if (isLiveTabActive && isDetectionActive) {
            updateAttendanceOnly();
        }
    }, 2000);
}

// Stop attendance updates when detection stops
function stopAttendanceUpdates() {
    if (attendanceUpdateInterval) {
        clearInterval(attendanceUpdateInterval);
        attendanceUpdateInterval = null;
    }
}

// Show alert function
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    
    // Calculate position to stack notifications vertically
    const existingAlerts = document.querySelectorAll('.alert.position-fixed');
    const topPosition = 20 + (existingAlerts.length * 80); // 80px spacing between alerts
    
    alertDiv.style.cssText = `top: ${topPosition}px; right: 20px; z-index: 9999; min-width: 300px; max-width: 400px;`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Tab change handler
document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
    tab.addEventListener('shown.bs.tab', function (event) {
        if (event.target.id === 'live-tab') {
            // updateAttendanceOnly(); // Refresh attendance data - now handled by startAttendanceUpdates
        }
    });
});

// Improve tab detection logic
setInterval(() => {
    // Check if live tab is active using multiple methods
    const liveTab = document.getElementById('live-tab');
    const liveTabPane = document.getElementById('live');
    const isLiveTabActive = (
        (liveTab && liveTab.classList.contains('active')) ||
        (liveTabPane && liveTabPane.classList.contains('active')) ||
        (liveTabPane && liveTabPane.classList.contains('show'))
    );
    
    if (isLiveTabActive) {
        // updateAttendanceOnly(); // Refresh attendance data - now handled by startAttendanceUpdates
    }
}, 2000);

// Example: Folder selection (for Electron or desktop app, use file dialog; for web, use backend API to list folders)
function selectFolder() {
    // Implement folder selection logic here (e.g., open a modal or fetch folder list from backend)
    alert('Folder selection not implemented in this demo.');
} 