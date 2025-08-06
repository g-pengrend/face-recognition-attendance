/**
 * Main Application Controller
 * 
 * This file handles the core application initialization, global state management,
 * and coordination between different modules. It serves as the entry point for
 * the application and manages the overall application lifecycle.
 * 
 * Key Responsibilities:
 * - Application initialization and startup sequence
 * - Global variable management (session ID, detection state, camera mode)
 * - Real-time status updates and polling
 * - Event listener setup and management
 * - UI state synchronization
 * 
 * Global Variables:
 * - currentSessionId: Tracks the active session ID
 * - statusUpdateInterval: Interval for system status updates
 * - attendanceUpdateInterval: Interval for attendance data updates
 * - isDetectionActive: Boolean flag for detection state
 * - lastPresentStudents: Set of students present in last update
 * - currentCameraMode: Current camera mode ('local' or 'ip')
 * 
 * Dependencies:
 * - detection.js: For detection control functions
 * - attendance.js: For attendance data management
 * - camera.js: For camera control functions
 * - classes.js: For class management
 * - sessions.js: For session history
 * - utils.js: For utility functions
 */

// Global variables
let currentSessionId = null;
let statusUpdateInterval = null;
let attendanceUpdateInterval = null;
let isDetectionActive = false;
let lastPresentStudents = new Set();
let currentCameraMode = 'local';

/**
 * Application Initialization
 * 
 * Sets up the application when the DOM is loaded. Performs a step-by-step
 * initialization sequence to ensure all systems are ready before showing
 * the main interface.
 * 
 * Initialization Steps:
 * 1. Load available classes
 * 2. Check system status
 * 3. Load student data
 * 4. Load session history
 * 5. Start status updates
 * 6. Hide loading screen
 */
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
    // document.getElementById('ipCameraBtn').addEventListener('click', () => switchCamera('ip')); // This line is commented out to disable the feature
    
    // Update camera status periodically
    updateCameraStatus();
    setInterval(updateCameraStatus, 5000); // Update every 5 seconds
});

/**
 * Start System Status Updates
 * 
 * Initiates periodic polling of system status to keep the UI synchronized
 * with the backend state. Updates system status indicators and manages
 * button states based on current system status.
 * 
 * Polling Frequency: Every 2 seconds
 * Updates: System status, detection state, camera status
 */
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

/**
 * Start Attendance Updates
 * 
 * Begins real-time polling of attendance data to provide live updates
 * of student presence. Updates the attendance feed and absent students
 * list in real-time.
 * 
 * Polling Frequency: Every 1 second
 * Updates: Live attendance feed, absent students list
 */
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

/**
 * Stop Attendance Updates
 * 
 * Halts the real-time attendance polling to conserve resources when
 * detection is not active. Clears intervals and resets attendance displays.
 */
function stopAttendanceUpdates() {
    if (attendanceUpdateInterval) {
        clearInterval(attendanceUpdateInterval);
        attendanceUpdateInterval = null;
    }
}

/**
 * Display Alert Messages
 * 
 * Shows user-friendly alert messages using Bootstrap toast notifications.
 * Supports different alert types (success, danger, warning, info) with
 * appropriate styling and auto-dismiss functionality.
 * 
 * @param {string} type - Alert type ('success', 'danger', 'warning', 'info')
 * @param {string} message - Alert message to display
 */
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