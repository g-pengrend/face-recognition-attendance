/**
 * Attendance Management Module
 * 
 * This module handles all attendance-related functionality including
 * real-time attendance updates, student management, and attendance
 * data processing. It manages the live attendance feed and provides
 * tools for adding and managing students.
 * 
 * Key Features:
 * - Live attendance feed updates
 * - Student presence tracking
 * - Unknown face management
 * - Student addition and removal
 * - Attendance toast notifications
 * 
 * Data Flow:
 * - Receives attendance data from backend
 * - Updates UI components in real-time
 * - Manages student database operations
 * - Handles attendance notifications
 * 
 * UI Components Managed:
 * - Live attendance list
 * - Absent students list
 * - Attendance toast notifications
 * - Student management modals
 */

/**
 * Update Live Attendance Feed
 * 
 * Updates the real-time display of currently present students.
 * Compares current attendance with previous state to show changes
 * and trigger notifications for new arrivals.
 * 
 * Process:
 * 1. Receive attendance data from backend
 * 2. Compare with previous state
 * 3. Update attendance list display
 * 4. Show notifications for new students
 * 5. Update present student count
 * 
 * @param {Object} attendance - Attendance data from backend
 * @param {Object} attendance.attendance - Object with student attendance records
 * @param {Array} attendance.all_students - List of all registered students
 */
function updateLiveAttendanceFeed(attendance) {
    const feedDiv = document.getElementById('liveAttendanceList');
    if (!attendance || !attendance.attendance || Object.keys(attendance.attendance).length === 0) {
        feedDiv.innerHTML = '<p class="text-muted">No students detected yet.</p>';
        lastPresentStudents = new Set(); // Reset
        return;
    }
    
    let html = '';
    const currentPresent = new Set();

    // Sort students by most recent attendance (last_seen descending)
    const sortedEntries = Object.entries(attendance.attendance).sort((a, b) => {
        const aTime = new Date(a[1].last_seen).getTime();
        const bTime = new Date(b[1].last_seen).getTime();
        return bTime - aTime;
    });

    sortedEntries.forEach(([student, data]) => {
        currentPresent.add(student);
        let badgeClass = 'badge-on-time';
        let itemClass = '';
        let lateness = data.lateness || {};
        if (lateness.category === 'On Time') {
            badgeClass = 'badge-on-time';
            itemClass = '';
        } else if (lateness.category === '30 min late') {
            badgeClass = 'badge-late-0-30';
            itemClass = 'late-0-30';
        } else if (lateness.category === '1 hour late') {
            badgeClass = 'badge-late-30-60';
            itemClass = 'late-30-60';
        } else if (lateness.category === '1.5 hours late' || 
                   lateness.category === '2 hours late' || 
                   lateness.category === '2.5 hours late' || 
                   lateness.category === '3 hours late') {
            badgeClass = 'badge-late-60-plus';
            itemClass = 'late-60-plus';
        } else if (lateness.category === 'Absent') {
            badgeClass = 'badge-absent';
            itemClass = 'absent';
        } else {
            badgeClass = 'badge-on-time';
            itemClass = 'unknown';
        }
        
        html += `
            <div class="student-item ${itemClass}">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-user-check text-success"></i>
                        <strong>${student}</strong>
                        <span class="attendance-badge ${badgeClass}">
                            <i class="fas fa-check-circle text-success"></i>
                            ${lateness.status || 'On Time'}
                        </span>
                    </div>
                    <div class="text-end">
                        <small class="text-muted">${(data.confidence * 100).toFixed(1)}% confidence</small><br>
                        <small class="text-muted">${data.detection_count} detections</small>
                    </div>
                </div>
                <small class="text-muted">
                    First seen: ${new Date(data.first_seen).toLocaleTimeString()} | 
                    Last seen: ${new Date(data.last_seen).toLocaleTimeString()}
                </small>
            </div>
        `;
    });
    feedDiv.innerHTML = html;

    // Show toast for any new student(s)
    currentPresent.forEach(student => {
        if (!lastPresentStudents.has(student)) {
            showAttendanceToast(student);
        }
    });
    lastPresentStudents = currentPresent;
}

/**
 * Update Absent Students List
 * 
 * Displays the list of students who are not yet present in the
 * current session. Calculates absent students by comparing registered
 * students with present students.
 * 
 * Process:
 * 1. Get list of all registered students
 * 2. Compare with present students
 * 3. Display absent students list
 * 4. Update absent count
 * 
 * @param {Object} attendance - Attendance data from backend
 */
function updateAbsentStudentsList(attendance) {
    const absentDiv = document.getElementById('absentStudentsList');
    if (!attendance) {
        absentDiv.innerHTML = '<p class="text-muted">No data.</p>';
        return;
    }
    
    // Get all students from the system
    const allStudents = attendance.all_students || [];
    const present = attendance.attendance ? Object.keys(attendance.attendance) : [];
    const absent = allStudents.filter(s => !present.includes(s));
    
    console.log('Absent students calculation:', {
        all_students: allStudents,
        present: present,
        absent: absent
    });
    
    if (absent.length === 0) {
        if (allStudents.length > 0) {
        absentDiv.innerHTML = '<p class="text-success">All students present!</p>';
        } else {
            absentDiv.innerHTML = '<p class="text-muted">No students registered.</p>';
        }
    } else {
        let html = '';
        absent.forEach(student => {
            html += `
                <div class="student-item unknown">
                    <i class="fas fa-user-times text-danger"></i>
                    <strong>${student}</strong>
                </div>
            `;
        });
        absentDiv.innerHTML = html;
    }
}

/**
 * Show Attendance Toast Notification
 * 
 * Displays a temporary notification when a student is marked present.
 * Shows the student name and automatically dismisses after a few seconds.
 * 
 * @param {string} studentName - Name of the student marked present
 */
function showAttendanceToast(studentName) {
    const toast = document.getElementById('attendanceToast');
    const toastText = document.getElementById('attendanceToastText');
    toastText.textContent = studentName;  // Just show the name, no "Attendance recorded:" prefix
    toastText.title = studentName;
    toast.style.display = 'flex';
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => { toast.style.display = 'none'; }, 300);
    }, 1500);  // Show for 1.5 seconds instead of 2
}

/**
 * Load Students List
 * 
 * Fetches and displays the list of all registered students in the
 * current class. Updates the students tab with current student data.
 * 
 * Process:
 * 1. Fetch students from backend
 * 2. Display in students list
 * 3. Show student count and details
 * 
 * @async
 * @throws {Error} If loading fails
 */
async function loadStudents() {
    const studentsList = document.getElementById('studentsList');
    studentsList.innerHTML = '<div class="loading"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Loading students...</p></div>';
    
    try {
        const response = await fetch('/api/students');
        const data = await response.json();
        
        if (data.students && data.students.length > 0) {
            let html = '';
            data.students.forEach(student => {
                html += `
                    <div class="student-item">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-user"></i>
                                <strong>${student}</strong>
                            </div>
                            <button class="btn btn-sm btn-outline-danger" onclick="removeStudent('${student}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                `;
            });
            studentsList.innerHTML = html;
        } else {
            studentsList.innerHTML = '<p class="text-muted">No students registered. Add student photos to the "students" folder.</p>';
        }
    } catch (error) {
        studentsList.innerHTML = '<p class="text-danger">Error loading students: ' + error.message + '</p>';
    }
}

/**
 * Remove Student
 * 
 * Removes a student from the current class and updates the student
 * database. Requires confirmation and updates the UI accordingly.
 * 
 * Process:
 * 1. Confirm deletion with user
 * 2. Send removal request to backend
 * 3. Update local student list
 * 4. Refresh attendance displays
 * 
 * @param {string} studentName - Name of student to remove
 * @async
 * @throws {Error} If removal fails
 */
async function removeStudent(studentName) {
    if (!confirm(`Are you sure you want to remove ${studentName}?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/remove-student', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: studentName })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', `Student ${studentName} removed successfully`);
            loadStudents();
        } else {
            showAlert('danger', result.error || 'Failed to remove student');
        }
    } catch (error) {
        showAlert('danger', 'Error removing student: ' + error.message);
    }
} 