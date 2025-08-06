/**
 * Utility Functions Module
 * 
 * This module contains utility functions used across the application
 * including student management, cache management, and helper functions.
 * Provides common functionality that doesn't fit into other modules.
 * 
 * Key Features:
 * - Student addition utilities
 * - Cache management functions
 * - Performance monitoring
 * - Helper functions
 * 
 * Cache Management:
 * - Cache status monitoring
 * - Cache cleanup operations
 * - Performance statistics
 * 
 * Student Management:
 * - Student addition workflows
 * - Form validation
 * - File upload handling
 */

/**
 * Submit Add Student Form
 * 
 * Processes the student addition form and submits the data
 * to the backend. Handles different addition methods (single
 * image, multiple images, existing folder).
 * 
 * Process:
 * 1. Validate form data
 * 2. Prepare form data
 * 3. Submit to backend
 * 4. Handle response
 * 
 * @async
 * @throws {Error} If submission fails
 */
async function submitAddStudent() {
    const method = document.querySelector('input[name="addMethod"]:checked').value;
    const currentClass = document.getElementById('classSelect').value;
    let formData = new FormData();

    if (!currentClass) {
        showAlert('danger', 'Please select a class first.');
        return;
    }

    formData.append('class_name', currentClass);

    if (method === 'single') {
        const name = document.getElementById('studentName').value;
        const file = document.getElementById('studentImage').files[0];
        const folder = document.getElementById('studentFolder').value;
        if (!name || !file) {
            showAlert('danger', 'Please provide a name and photo.');
            return;
        }
        formData.append('name', name);
        formData.append('image', file);
        formData.append('folder', folder);
    } else if (method === 'folder') {
        const name = document.getElementById('folderStudentName').value;
        const files = document.getElementById('folderImages').files;
        const folder = document.getElementById('folderName').value;
        if (!name || files.length === 0 || !folder) {
            showAlert('danger', 'Please provide a name, folder, and select images.');
            return;
        }
        formData.append('name', name);
        formData.append('folder', folder);
        for (let i = 0; i < files.length; i++) {
            formData.append('images', files[i]);
        }
    } else if (method === 'existing') {
        const name = document.getElementById('existingStudentName').value;
        const folderPath = document.getElementById('existingFolderPath').value;
        if (!name || !folderPath) {
            showAlert('danger', 'Please provide a name and select a folder.');
            return;
        }
        formData.append('name', name);
        formData.append('folder_path', folderPath);
    }

    // Send to backend
    try {
        const response = await fetch('/api/add-student', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (result.success) {
            showAlert('success', result.message || 'Student added!');
            // Optionally refresh student list
            loadStudents();
            // Close modal
            var modal = bootstrap.Modal.getInstance(document.getElementById('addStudentModal'));
            modal.hide();
        } else {
            showAlert('danger', result.error || 'Failed to add student.');
        }
    } catch (error) {
        showAlert('danger', 'Error adding student: ' + error.message);
    }
}

/**
 * Show Cache Status
 * 
 * Displays information about the current cache status including
 * cache size, hit rates, and performance statistics.
 * 
 * Process:
 * 1. Fetch cache status from backend
 * 2. Display statistics
 * 3. Show performance metrics
 * 
 * @async
 * @throws {Error} If status fetch fails
 */
async function showCacheStatus() {
    try {
        const response = await fetch('/api/cache/status');
        const data = await response.json();
        
        if (data.success) {
            console.log('Cache status:', data.cache_info);
            const currentClass = document.getElementById('classSelect').value;
            if (currentClass && data.cache_info[currentClass]) {
                const cacheInfo = data.cache_info[currentClass];
                if (cacheInfo.students && cacheInfo.students.valid) {
                    console.log(`Class "${currentClass}" data is cached`);
                }
            }
        }
    } catch (error) {
        console.error('Error getting cache status:', error);
    }
}

/**
 * Clear All Cache
 * 
 * Removes all cached data to free up space and resolve
 * potential cache-related issues.
 * 
 * Process:
 * 1. Send clear request to backend
 * 2. Confirm operation
 * 3. Update cache status
 * 
 * @async
 * @throws {Error} If clear operation fails
 */
async function clearCache() {
    try {
        const response = await fetch('/api/cache/clear', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', 'Cache cleared successfully');
        } else {
            showAlert('danger', result.error || 'Failed to clear cache');
        }
    } catch (error) {
        showAlert('danger', 'Error clearing cache: ' + error.message);
    }
}

/**
 * Cleanup Old Cache
 * 
 * Removes old cache files to free up space while preserving
 * recent cache data for performance.
 * 
 * Process:
 * 1. Identify old cache files
 * 2. Remove expired data
 * 3. Update cache status
 * 
 * @async
 * @throws {Error} If cleanup fails
 */
async function cleanupCache() {
    try {
        const maxAge = prompt('Enter maximum age in hours (default: 24):', '24');
        if (maxAge === null) return; // User cancelled
        
        const response = await fetch('/api/cache/cleanup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ max_age_hours: parseInt(maxAge) || 24 })
        });
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', result.message);
        } else {
            showAlert('danger', result.error || 'Failed to cleanup cache');
        }
    } catch (error) {
        showAlert('danger', 'Error cleaning up cache: ' + error.message);
    }
}

/**
 * Show Cache Statistics
 * 
 * Displays detailed cache performance statistics including
 * hit rates, file counts, and storage usage.
 * 
 * Process:
 * 1. Fetch cache statistics
 * 2. Display performance metrics
 * 3. Show storage information
 * 
 * @async
 * @throws {Error} If statistics fetch fails
 */
async function showCacheStats() {
    try {
        const response = await fetch('/api/cache/stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            const message = `Cache Statistics:\n` +
                          `Total Files: ${stats.total_files}\n` +
                          `Total Size: ${stats.total_size_mb.toFixed(2)} MB\n` +
                          `Cached Classes: ${Object.keys(stats.cache_info).length}`;
            
            alert(message);
        } else {
            showAlert('danger', data.error || 'Failed to get cache stats');
        }
    } catch (error) {
        showAlert('danger', 'Error getting cache stats: ' + error.message);
    }
}

// Show modal for unknown faces with thumbnails
function showUnknownFacesModal(unknownFaces, screenshotPath) {
    // First, load existing students for the dropdown
    loadStudentsForDropdown().then(existingStudents => {
        let modalHtml = `
            <div class="modal fade" id="unknownFacesModal" tabindex="-1" data-bs-backdrop="static">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Unknown Faces Detected</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row mb-3">
                                <div class="col-12">
                                    <h6>Screenshot:</h6>
                                    <img src="/${screenshotPath}" class="img-fluid border" style="max-height: 300px;" 
                                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                    <div style="display:none;" class="alert alert-warning">
                                        Screenshot not available. Please try again.
                                    </div>
                                </div>
                            </div>
                            <p>Found ${unknownFaces.length} unknown face(s). Each face is labeled on the screenshot above.</p>
                            <div class="row">
        `;
        
        unknownFaces.forEach((face, index) => {
            modalHtml += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-body">
                            <h6>Face #${face.unknown_index + 1}</h6>
                            <div class="text-center mb-2">
                                <img src="/${face.thumbnail_path}" class="img-thumbnail" 
                                     style="max-width: 150px; max-height: 150px;"
                                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div style="display:none;" class="alert alert-warning">
                                    Thumbnail not available
                                </div>
                            </div>
                            <p class="text-muted">Confidence: ${(face.confidence * 100).toFixed(1)}%</p>
                            <div class="mb-2">
                                <label class="form-label">Student:</label>
                                <div class="input-group">
                                    <input type="text" class="form-control" id="studentSearch_${face.index}" 
                                           placeholder="Search students..." onkeyup="filterStudentDropdown(${face.index})">
                                    <button class="btn btn-outline-secondary" type="button" onclick="clearStudentSearch(${face.index})">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                                <div id="studentDropdown_${face.index}" class="student-dropdown" style="display:none;">
                                    <select class="form-select mt-1" id="studentSelect_${face.index}" size="5" onchange="selectExistingStudent(${face.index}, this.value)">
                                        <option value="">-- Select existing student --</option>
                                        ${existingStudents.map(student => 
                                            `<option value="${student}" class="student-option">${student}</option>`
                                        ).join('')}
                                    </select>
                                </div>
                                <div id="selectedStudent_${face.index}" class="selected-student-display" style="display:none;">
                                    <div class="alert alert-info d-flex justify-content-between align-items-center">
                                        <span><i class="fas fa-user"></i> <strong id="selectedStudentName_${face.index}"></strong></span>
                                        <button class="btn btn-sm btn-outline-secondary" onclick="changeStudentSelection(${face.index})">
                                            <i class="fas fa-edit"></i> Change
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="mb-2" id="newStudentSection_${face.index}">
                                <label class="form-label">Or enter new student name:</label>
                                <input type="text" class="form-control" id="newStudentName_${face.index}" 
                                       placeholder="Enter new student name">
                            </div>
                            <button class="btn btn-primary btn-sm" onclick="addFaceAsStudent(${face.index})">
                                <i class="fas fa-user-plus"></i> Add Face
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        modalHtml += `
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        `;

        // Remove any existing modal and append the new one
        const existingModal = document.getElementById('unknownFacesModal');
        if (existingModal) existingModal.remove();
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('unknownFacesModal'));
        modal.show();

        // Ensure modal is fully removed from DOM on close to prevent tab freeze
        document.getElementById('unknownFacesModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });

        // Initialize dropdown behavior for each face
        unknownFaces.forEach((face, index) => {
            toggleStudentDropdown(face.index);
        });
    });
}

// Helper to load students for dropdown with sorting and search
async function loadStudentsForDropdown() {
    try {
        const response = await fetch('/api/students');
        const data = await response.json();
        const students = data.students || [];
        
        // Sort students numerically (1_XXX, 2_XXX, 10_XXX, 11_XXX)
        return students.sort((a, b) => {
            // Extract numbers from student names (e.g., "1_John" -> 1, "10_Mary" -> 10)
            const numA = parseInt(a.match(/^(\d+)_/)?.[1] || '0');
            const numB = parseInt(b.match(/^(\d+)_/)?.[1] || '0');
            return numA - numB;
        });
    } catch (error) {
        return [];
    }
}

// Enhanced handleStudentSelection function
function handleStudentSelection(faceIndex, selectedStudent) {
    const select = document.getElementById(`studentSelect_${faceIndex}`);
    const input = document.getElementById(`newStudentName_${faceIndex}`);
    
    if (selectedStudent) {
        // Update the input field with the selected student name
        input.value = selectedStudent;
        
        // Add visual feedback that a student was selected
        input.classList.add('student-selected');
        
        // Clear the dropdown selection
        select.value = "";
        
        // Focus on the input field
        input.focus();
        
        // Remove the visual feedback after a short delay
        setTimeout(() => {
            input.classList.remove('student-selected');
        }, 2000);
    }
}

// Add a face as a student (handles both existing and new students)
async function addFaceAsStudent(faceIndex) {
    const selectedDisplay = document.getElementById(`selectedStudent_${faceIndex}`);
    const newStudentInput = document.getElementById(`newStudentName_${faceIndex}`);
    
    let studentName = '';
    let isNewStudent = false;
    
    // Check if an existing student is selected
    if (selectedDisplay.style.display !== 'none') {
        studentName = document.getElementById(`selectedStudentName_${faceIndex}`).textContent;
        isNewStudent = false;
    } else {
        // Check if a new student name is entered
        studentName = newStudentInput.value.trim();
        isNewStudent = true;
    }
    
    if (!studentName) {
        showAlert('danger', 'Please select an existing student or enter a new student name.');
        return;
    }

    const currentClass = document.getElementById('classSelect').value;

    try {
        // Show loading state
        const addButton = event.target;
        const originalText = addButton.innerHTML;
        addButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
        addButton.disabled = true;

        const response = await fetch('/api/save-face-photo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                face_index: faceIndex,
                student_name: studentName,
                is_new_student: isNewStudent,
                class_name: currentClass
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('success', result.message || 'Face photo saved!');
            loadStudents();
            
            // Mark this face as completed
            const faceCard = addButton.closest('.card');
            faceCard.style.opacity = '0.5';
            addButton.innerHTML = '<i class="fas fa-check"></i> Added';
            addButton.classList.remove('btn-primary');
            addButton.classList.add('btn-success');
            addButton.disabled = true;
            
        } else {
            showAlert('danger', result.error || 'Failed to save face photo.');
            // Reset button state
            addButton.innerHTML = originalText;
            addButton.disabled = false;
        }
    } catch (error) {
        showAlert('danger', 'Error saving face photo: ' + error.message);
        // Reset button state
        const addButton = event.target;
        addButton.innerHTML = originalText;
        addButton.disabled = false;
    }
}

// Filter dropdown options based on search input
function filterStudentDropdown(faceIndex) {
    const searchInput = document.getElementById(`studentSearch_${faceIndex}`);
    const filter = searchInput.value.toLowerCase();
    const select = document.getElementById(`studentSelect_${faceIndex}`);
    const dropdown = document.getElementById(`studentDropdown_${faceIndex}`);
    
    // Show dropdown when typing
    if (filter.length > 0) {
        dropdown.style.display = 'block';
    }
    
    Array.from(select.options).forEach(option => {
        if (option.value === "") return; // Always show the placeholder
        option.style.display = option.text.toLowerCase().includes(filter) ? "" : "none";
    });
}

// Clear search and reset dropdown
function clearStudentSearch(faceIndex) {
    const searchInput = document.getElementById(`studentSearch_${faceIndex}`);
    const dropdown = document.getElementById(`studentDropdown_${faceIndex}`);
    
    searchInput.value = "";
    filterStudentDropdown(faceIndex);
    searchInput.focus();
}

// Show/hide dropdown when search input is focused
function toggleStudentDropdown(faceIndex) {
    const searchInput = document.getElementById(`studentSearch_${faceIndex}`);
    const dropdown = document.getElementById(`studentDropdown_${faceIndex}`);
    const selectedDisplay = document.getElementById(`selectedStudent_${faceIndex}`);
    
    // Show dropdown when search input is focused
    searchInput.addEventListener('focus', function() {
        if (!selectedDisplay.style.display || selectedDisplay.style.display === 'none') {
            dropdown.style.display = 'block';
        }
    });
    
    // Hide dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });
}

// Select an existing student from dropdown
function selectExistingStudent(faceIndex, selectedStudent) {
    if (!selectedStudent) return;
    
    const searchInput = document.getElementById(`studentSearch_${faceIndex}`);
    const dropdown = document.getElementById(`studentDropdown_${faceIndex}`);
    const selectedDisplay = document.getElementById(`selectedStudent_${faceIndex}`);
    const selectedName = document.getElementById(`selectedStudentName_${faceIndex}`);
    const newStudentSection = document.getElementById(`newStudentSection_${faceIndex}`);
    
    // Hide dropdown and search input
    dropdown.style.display = 'none';
    searchInput.style.display = 'none';
    
    // Show selected student display
    selectedName.textContent = selectedStudent;
    selectedDisplay.style.display = 'block';
    
    // Hide the "new student" section since we selected an existing one
    newStudentSection.style.display = 'none';
    
    // Clear the new student input
    document.getElementById(`newStudentName_${faceIndex}`).value = '';
}

// Change student selection (go back to search)
function changeStudentSelection(faceIndex) {
    const searchInput = document.getElementById(`studentSearch_${faceIndex}`);
    const dropdown = document.getElementById(`studentDropdown_${faceIndex}`);
    const selectedDisplay = document.getElementById(`selectedStudent_${faceIndex}`);
    const newStudentSection = document.getElementById(`newStudentSection_${faceIndex}`);
    
    // Show search input and dropdown
    searchInput.style.display = 'block';
    searchInput.value = '';
    dropdown.style.display = 'block';
    
    // Hide selected student display
    selectedDisplay.style.display = 'none';
    
    // Show the "new student" section
    newStudentSection.style.display = 'block';
    
    // Focus on search input
    searchInput.focus();
} 