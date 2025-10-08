/**
 * Class Management Module
 * 
 * This module handles all class-related functionality including
 * class loading, class switching, and class creation. It manages
 * the class selection interface and provides tools for creating
 * new classes from CSV files.
 * 
 * Key Features:
 * - Class loading and display
 * - Class switching
 * - CSV-based class creation
 * - Student folder management
 * 
 * Data Flow:
 * - Fetches available classes from backend
 * - Manages class selection state
 * - Handles CSV file processing
 * - Updates student counts
 * 
 * UI Components:
 * - Class selector dropdown
 * - Create class modal
 * - Class creation progress
 */

/**
 * Load Available Classes
 * 
 * Fetches and displays the list of available classes in the
 * class selector dropdown. Updates the UI with current class
 * information and student counts.
 * 
 * Process:
 * 1. Fetch classes from backend
 * 2. Populate dropdown
 * 3. Set current class
 * 4. Update student counts
 * 
 * @async
 * @throws {Error} If loading fails
 */
async function loadClasses() {
    try {
        const response = await fetch('/api/classes');
        const data = await response.json();
        
        const classSelect = document.getElementById('classSelect');
        
        if (data.classes && data.classes.length > 0) {
            let html = '<option value="">Select a class...</option>';
            data.classes.forEach(className => {
                const selected = className === data.current_class ? 'selected' : '';
                html += `<option value="${className}" ${selected}>${className}</option>`;
            });
            classSelect.innerHTML = html;
        } else {
            classSelect.innerHTML = '<option value="">No classes found</option>';
        }
        
        // If there's a current class, load its students
        if (data.current_class) {
            await loadStudents();
        }
        
    } catch (error) {
        console.error('Error loading classes:', error);
        document.getElementById('classSelect').innerHTML = '<option value="">Error loading classes</option>';
    }
}

/**
 * Change Active Class
 * 
 * Switches to a different class and loads its associated
 * student data. Updates the UI to reflect the new class
 * selection and student information.
 * 
 * Process:
 * 1. Validate class selection
 * 2. Send class change request to backend
 * 3. Update UI state
 * 4. Load class-specific data
 * 
 * @async
 * @throws {Error} If class change fails
 */
async function changeClass() {
    const classSelect = document.getElementById('classSelect');
    const selectedClass = classSelect.value;
    
    if (!selectedClass) {
        return;
    }
    
    // Show loading for class change
    const loadingScreen = document.getElementById('loadingScreen');
    const loadingSubtext = document.getElementById('loadingSubtext');
    loadingScreen.classList.remove('hidden');
    loadingSubtext.textContent = `Loading class: ${selectedClass}...`;
    
    try {
        const response = await fetch('/api/set-class', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ class_name: selectedClass })
        });
        
        const result = await response.json();
        
        if (result.success) {
            const message = result.cached ? 
                `Class "${selectedClass}" loaded from cache!` : 
                `Class "${selectedClass}" loaded and cached!`;
            showAlert('success', message);
            await loadStudents();
            await updateStatus();
            await showCacheStatus();
        } else {
            showAlert('danger', result.error || 'Failed to change class');
        }
    } catch (error) {
        showAlert('danger', 'Error changing class: ' + error.message);
    } finally {
        // Hide loading screen
        setTimeout(() => {
            loadingScreen.classList.add('hidden');
        }, 500);
    }
}

/**
 * Show Create Class Modal
 * 
 * Displays the modal for creating a new class from a CSV file.
 * Provides interface for class name input and CSV file upload.
 * 
 * Process:
 * 1. Show modal
 * 2. Reset form
 * 3. Prepare for file upload
 */
function showCreateClassModal() {
    // Clear previous form data
    document.getElementById('newClassName').value = '';
    document.getElementById('csvFile').value = '';
    document.getElementById('createClassProgress').style.display = 'none';
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('createClassModal'));
    modal.show();
}

/**
 * Create New Class from CSV
 * 
 * Creates a new class by processing a CSV file containing
 * student information. Automatically creates student folders
 * and sets up the class structure.
 * 
 * Process:
 * 1. Validate CSV file
 * 2. Process student data
 * 3. Create class folders
 * 4. Update class list
 * 
 * @async
 * @throws {Error} If creation fails
 */
async function createNewClass() {
    const className = document.getElementById('newClassName').value.trim();
    const csvFile = document.getElementById('csvFile').files[0];
    
    // Validation
    if (!className) {
        showAlert('danger', 'Please enter a class name.');
        return;
    }
    
    if (!csvFile) {
        showAlert('danger', 'Please select a CSV file.');
        return;
    }
    
    // Check file extension
    if (!csvFile.name.toLowerCase().endsWith('.csv')) {
        showAlert('danger', 'Please select a valid CSV file.');
        return;
    }
    
    // Show progress
    document.getElementById('createClassProgress').style.display = 'block';
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('class_name', className);
        formData.append('csv_file', csvFile);
        
        // Send to backend
        const response = await fetch('/api/create-class', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const message = result.skipped_rows > 0 
                ? `Class "${className}" created successfully with ${result.students_count} students! (Processed ${result.processed_rows} rows, skipped ${result.skipped_rows} invalid rows)`
                : `Class "${className}" created successfully with ${result.students_count} students!`;
            
            showAlert('success', message);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createClassModal'));
            modal.hide();
            
            // Refresh classes list
            await loadClasses();
            
            // Select the new class
            const classSelect = document.getElementById('classSelect');
            classSelect.value = className;
            await changeClass();
            
        } else {
            showAlert('danger', result.error || 'Failed to create class.');
        }
        
    } catch (error) {
        showAlert('danger', 'Error creating class: ' + error.message);
    } finally {
        // Hide progress
        document.getElementById('createClassProgress').style.display = 'none';
    }
} 