// Load available classes
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

// Change class
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

// Show create class modal
function showCreateClassModal() {
    // Clear previous form data
    document.getElementById('newClassName').value = '';
    document.getElementById('csvFile').value = '';
    document.getElementById('createClassProgress').style.display = 'none';
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('createClassModal'));
    modal.show();
}

// Create new class from CSV
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
            showAlert('success', `Class "${className}" created successfully with ${result.students_count} students!`);
            
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