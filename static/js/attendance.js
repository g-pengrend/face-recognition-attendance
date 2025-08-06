// Update live attendance feed
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
        if (lateness.category === '0-30 min late') {
            badgeClass = 'badge-late-0-30';
            itemClass = 'late-0-30';
        } else if (lateness.category === '30-60 min late') {
            badgeClass = 'badge-late-30-60';
            itemClass = 'late-30-60';
        } else if (lateness.category === '60+ min late') {
            badgeClass = 'badge-late-60-plus';
            itemClass = 'late-60-plus';
        } else if (lateness.category === 'On Time') {
            badgeClass = 'badge-on-time';
            itemClass = '';
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

// Update absent students list
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

// Load students
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

// Remove student
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