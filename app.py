from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from datetime import datetime
import threading
import time
import logging
import random # Added for smart sampling
import csv
import tempfile
import shutil

from face_recognition import FaceRecognitionSystem
from attendance_manager import AttendanceManager

# Set ONNX Runtime to prefer CoreML before importing any other modules
os.environ['ONNXRUNTIME_PROVIDER_NAMES'] = 'CoreMLExecutionProvider,CPUExecutionProvider'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
face_system = None
attendance_manager = None
camera = None
detection_active = False
detection_thread = None
current_frame = None
current_faces = []
captured_frame = None  # Store the frame when screenshot is taken
captured_faces = []    # Store faces from the captured frame

class FaceSamplingManager:
    def __init__(self, batch_size=5, batch_interval=2.0):
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.last_batch_time = time.time()
        self.face_history = {}  # Track when each face was last processed
        self.current_batch = []
        
    def get_next_batch(self, detected_faces):
        """
        Get the next batch of faces to process, ensuring fair coverage
        """
        current_time = time.time()
        
        if not detected_faces:
            return []
        
        # Check if it's time to rotate batches
        if current_time - self.last_batch_time >= self.batch_interval:
            self._update_batch(detected_faces, current_time)
            self.last_batch_time = current_time
        
        return self.current_batch
    
    def _update_batch(self, detected_faces, current_time):
        """
        Update the current batch with smart sampling
        """
        num_faces = len(detected_faces)
        
        if num_faces <= self.batch_size:
            # Process all faces if we have fewer than batch size
            self.current_batch = detected_faces
            return
        
        # Calculate priority scores for each face
        face_scores = []
        for i, face in enumerate(detected_faces):
            # Base score: time since last processed (higher = more priority)
            last_processed = self.face_history.get(i, 0)
            time_since_last = current_time - last_processed
            
            # Bonus score: confidence level
            confidence_bonus = face.get('confidence', 0) * 10
            
            # Bonus score: face size (larger faces get priority)
            bbox = face.get('bbox', [0, 0, 0, 0])
            face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            size_bonus = min(face_size / 1000, 5)  # Cap size bonus
            
            total_score = time_since_last + confidence_bonus + size_bonus
            face_scores.append((total_score, i, face))
        
        # Sort by priority score and select top batch_size
        face_scores.sort(reverse=True)
        selected_indices = [idx for _, idx, _ in face_scores[:self.batch_size]]
        
        # Update current batch and mark faces as processed
        self.current_batch = [detected_faces[i] for i in selected_indices]
        for idx in selected_indices:
            self.face_history[idx] = current_time
        
        logger.info(f"Smart batch rotation: {len(self.current_batch)}/{num_faces} faces selected")

# Initialize the sampling manager
face_sampling_manager = FaceSamplingManager(batch_size=5, batch_interval=2.0)

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def initialize_systems():
    """Initialize face recognition and attendance systems"""
    global face_system, attendance_manager
    
    try:
        face_system = FaceRecognitionSystem(students_folder="students", threshold=0.6)
        attendance_manager = AttendanceManager(logs_folder="attendance_logs")
        logger.info("Systems initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        return False

def get_camera():
    """Get camera instance with optimized settings"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Reduce resolution for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def detection_loop():
    """Detection loop with smart rotating face sampling and performance monitoring"""
    global detection_active, face_system, attendance_manager, face_sampling_manager, current_frame, current_faces
    
    camera = get_camera()
    if not camera.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Starting detection loop with performance monitoring")
    
    while detection_active:
        loop_start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        try:
            # Store current frame for photo capture
            current_frame = frame.copy()
            
            # Detect all faces in the frame with timing
            detection_start = time.time()
            all_faces = face_system.detect_faces(frame)
            detection_time = (time.time() - detection_start) * 1000  # Convert to milliseconds
            current_faces = all_faces  # Store for photo capture
            
            # Log performance if detection is slow
            if detection_time > 100:  # More than 100ms
                logger.warning(f"Slow face detection: {detection_time:.1f}ms for {len(all_faces)} faces")
            
            # Process faces immediately for faster attendance recording
            for face in all_faces:
                if face['student_name'] and face['confidence'] >= 0.6:
                    # Known face - mark attendance immediately
                    success = attendance_manager.mark_attendance(
                        face['student_name'], 
                        face['confidence']
                    )
                    if success:
                        logger.info(f"Marked attendance for {face['student_name']} (confidence: {face['confidence']:.2f})")
                    else:
                        logger.warning(f"Failed to mark attendance for {face['student_name']}")
            
            # Calculate total loop time
            total_loop_time = (time.time() - loop_start_time) * 1000
            if total_loop_time > 200:  # More than 200ms total
                logger.warning(f"Slow detection loop: {total_loop_time:.1f}ms total")
            
            time.sleep(0.1)  # Reduced sleep time for faster response
            
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            time.sleep(0.1)
    
    logger.info("Detection loop stopped")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    global face_system, attendance_manager, detection_active
    
    # Get current attendance data
    current_attendance = {}
    if attendance_manager:
        current_attendance = attendance_manager.get_current_attendance()
    
    # Get all students list
    all_students = []
    if face_system:
        all_students = face_system.get_students_list()
    
    # Add all_students to current_attendance
    if current_attendance:
        current_attendance['all_students'] = all_students
    
    status = {
        'face_system_ready': face_system is not None and face_system.initialized,
        'detection_active': detection_active,
        'students_count': len(all_students),
        'current_session': convert_numpy_types(current_attendance)
    }
    
    logger.info(f"Status: detection_active={detection_active}, present_students={current_attendance.get('present_students', 0)}")
    return jsonify(status)

@app.route('/api/students')
def get_students():
    """Get list of registered students"""
    global face_system
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    students = face_system.get_students_list()
    return jsonify({'students': students})

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    """Start face detection and attendance tracking"""
    global detection_active, detection_thread, attendance_manager
    
    if detection_active:
        return jsonify({'error': 'Detection already active'}), 400
    
    try:
        # Get session details from request
        session_name = request.json.get('session_name')
        session_start_time = request.json.get('session_start_time')
        class_name = request.json.get('class_name')
        
        # Start attendance session with start time
        session_id = attendance_manager.start_session(session_name, session_start_time, class_name)
        
        # Set total students count
        total_students = len(face_system.get_students_list())
        attendance_manager.set_total_students(total_students)
        
        # Start detection thread
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        logger.info(f"Started detection session: {session_id} with class start time: {session_start_time}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Detection started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-detection', methods=['POST'])
def stop_detection():
    """Stop face detection and attendance tracking"""
    global detection_active, attendance_manager
    
    if not detection_active:
        return jsonify({'error': 'No active detection session'}), 400
    
    try:
        # Stop detection
        detection_active = False
        
        # End attendance session
        attendance_manager.end_session()
        
        logger.info("Stopped detection session")
        return jsonify({
            'success': True,
            'message': 'Detection stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance')
def get_attendance():
    """Get current attendance status"""
    global attendance_manager, detection_active, face_system
    
    if not attendance_manager:
        return jsonify({'detection_active': detection_active, 'attendance': {}}), 200
    
    attendance = attendance_manager.get_current_attendance()
    
    # Get all students list
    all_students = []
    if face_system:
        all_students = face_system.get_students_list()
    
    # Add all_students to attendance
    if attendance:
        attendance['all_students'] = all_students
    
    # Add detection status to response
    response = {
        'detection_active': detection_active,
        'attendance': convert_numpy_types(attendance)
    }
    
    logger.info(f"Attendance API: detection_active={detection_active}, present_students={attendance.get('present_students', 0)}")
    return jsonify(response)

@app.route('/api/sessions')
def get_sessions():
    """List all saved sessions (JSON files)"""
    sessions = []
    for fname in os.listdir('attendance_logs'):
        if fname.endswith('.json'):
            with open(os.path.join('attendance_logs', fname)) as f:
                session = json.load(f)
                sessions.append({
                    'session_id': session.get('session_id', fname.replace('.json', '')),
                    'session_name': session.get('session_name', ''),
                    'class_name': session.get('class_name', ''),
                    'start_time': session.get('session_start_time', ''),
                    'active': session.get('active', False),
                    'filename': fname,
                    'present_students': len(session.get('attendance', {})),
                    'total_students': session.get('total_students', 0)
                })
    # Sort by start_time descending
    sessions.sort(key=lambda s: s['start_time'], reverse=True)
    return jsonify({'sessions': sessions})

@app.route('/api/sessions/<session_id>')
def get_session(session_id):
    """Get specific session details"""
    global attendance_manager
    
    if not attendance_manager:
        return jsonify({'error': 'Attendance manager not initialized'}), 500
    
    session = attendance_manager.get_session_summary(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(convert_numpy_types(session))

@app.route('/api/export/<session_id>')
def export_session(session_id):
    """Export session to CSV"""
    global attendance_manager
    
    if not attendance_manager:
        return jsonify({'error': 'Attendance manager not initialized'}), 500
    
    csv_path = attendance_manager.export_to_csv(session_id)
    if not csv_path:
        return jsonify({'error': 'Failed to export session'}), 500
    
    return send_file(csv_path, as_attachment=True, download_name=f"{session_id}_attendance.csv")

@app.route('/api/classes')
def get_classes():
    """Get list of available classes"""
    global face_system
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    classes = face_system.get_available_classes()
    current_class = face_system.get_current_class()
    
    return jsonify({
        'classes': classes,
        'current_class': current_class
    })

@app.route('/api/set-class', methods=['POST'])
def set_class():
    """Set the current class and load its students"""
    global face_system, attendance_manager
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    try:
        data = request.json
        class_name = data.get('class_name')
        
        if not class_name:
            return jsonify({'error': 'Class name is required'}), 400
        
        success = face_system.set_current_class(class_name)
        if success:
            # Reset attendance manager for new class
            if attendance_manager:
                attendance_manager.end_session()  # End any existing session
            
            return jsonify({
                'success': True,
                'message': f'Switched to class: {class_name}',
                'students_count': len(face_system.get_students_list())
            })
        else:
            return jsonify({'error': f'Class {class_name} not found'}), 404
            
    except Exception as e:
        logger.error(f"Error setting class: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-class')
def get_current_class():
    """Get the currently selected class"""
    global face_system
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    current_class = face_system.get_current_class()
    return jsonify({'current_class': current_class})

@app.route('/api/add-student', methods=['POST'])
def add_student():
    """
    Add a new student via UI (single image, multiple images, or existing folder)
    """
    name = request.form.get('name')
    folder = request.form.get('folder')
    folder_path = request.form.get('folder_path')
    images = request.files.getlist('images')
    image = request.files.get('image')
    class_name = request.form.get('class_name')  # Add class name parameter

    # Handle single image upload
    if image:
        if class_name:
            save_path = os.path.join('students', class_name, folder or '', f"{name}.jpg")
        else:
            save_path = os.path.join('students', folder or '', f"{name}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        # Add to face system
        face_system.add_student(name, save_path, class_name)
        return jsonify({'success': True, 'message': f'Student {name} added with single image.'})

    # Handle multiple images (folder)
    elif images and folder:
        if class_name:
            folder_dir = os.path.join('students', class_name, folder)
        else:
            folder_dir = os.path.join('students', folder)
        os.makedirs(folder_dir, exist_ok=True)
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(folder_dir, f"{name}_{i}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
        # Add to face system (implement add_student_multiple_images)
        face_system.add_student_multiple_images(name, image_paths)
        return jsonify({'success': True, 'message': f'Student {name} added with {len(image_paths)} images.'})

    # Handle existing folder
    elif folder_path:
        # List all images in the folder
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not image_paths:
            return jsonify({'success': False, 'error': 'No images found in folder.'})
        face_system.add_student_multiple_images(name, image_paths)
        return jsonify({'success': True, 'message': f'Student {name} added from existing folder.'})

    return jsonify({'success': False, 'error': 'Invalid request.'})

@app.route('/api/remove-student', methods=['POST'])
def remove_student():
    """Remove a student"""
    global face_system
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    try:
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        success = face_system.remove_student(name)
        if success:
            return jsonify({'success': True, 'message': f'Student {name} removed successfully'})
        else:
            return jsonify({'error': 'Student not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing student: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/daily-summary')
def get_daily_summary():
    """Get daily attendance summary"""
    global attendance_manager
    
    if not attendance_manager:
        return jsonify({'error': 'Attendance manager not initialized'}), 500
    
    try:
        date_str = request.args.get('date')
        target_date = None
        
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        summary = attendance_manager.get_daily_summary(target_date)
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        logger.error(f"Error getting daily summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        camera = get_camera()
        if not camera.isOpened():
            logger.error("Failed to open camera for streaming")
            return
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Draw detection results on frame
            if detection_active and face_system:
                faces = face_system.detect_faces(frame)
                
                for i, face in enumerate(faces):
                    bbox = face['bbox']
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Draw bounding box
                    color = (0, 255, 0) if face['student_name'] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw student name and confidence
                    if face['student_name']:
                        label = f"{face['student_name']} ({face['confidence']:.2f})"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        # Show face index for unknown faces
                        label = f"Unknown #{i+1}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture-screenshot', methods=['POST'])
def capture_screenshot():
    """Capture a screenshot and return detected faces with image"""
    global current_frame, face_system, captured_frame, captured_faces
    
    try:
        if current_frame is None:
            return jsonify({'error': 'No frame available'}), 400
        
        # Store the captured frame and its faces
        captured_frame = current_frame.copy()
        captured_faces = face_system.detect_faces(captured_frame)
        
        # Create a copy of the frame for drawing
        screenshot = captured_frame.copy()
        
        # Filter only unknown faces and draw them on screenshot
        unknown_faces = []
        unknown_face_index = 0  # Counter for unknown faces only
        
        for i, face in enumerate(captured_faces):
            if not face['student_name']:
                # Draw bounding box and label on screenshot
                bbox = face['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Calculate expanded bounding box for visualization
                face_width = x2 - x1
                face_height = y2 - y1
                expand_x = int(face_width * 0.5)
                expand_y = int(face_height * 0.5)
                
                height, width = captured_frame.shape[:2]
                new_x1 = max(0, x1 - expand_x)
                new_y1 = max(0, y1 - expand_y)
                new_x2 = min(width, x2 + expand_x)
                new_y2 = min(height, y2 + expand_y)
                
                # Draw expanded bounding box (dashed line)
                cv2.rectangle(screenshot, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 0), 2)
                
                # Draw original bounding box (solid line)
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Draw face number
                label = f"Unknown #{unknown_face_index + 1}"
                cv2.putText(screenshot, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Crop the expanded face for thumbnail
                face_crop = captured_frame[new_y1:new_y2, new_x1:new_x2]
                if face_crop.size > 0:
                    # Save thumbnail
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    thumbnail_filename = f"thumbnail_{timestamp}_{unknown_face_index}.jpg"
                    thumbnail_path = os.path.join("temp", thumbnail_filename)
                    
                    # Ensure temp directory exists
                    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
                    
                    # Save the thumbnail
                    success = cv2.imwrite(thumbnail_path, face_crop)
                    if not success:
                        logger.error(f"Failed to save thumbnail: {thumbnail_path}")
                    
                    unknown_faces.append({
                        'index': i,  # Original face index in captured frame
                        'unknown_index': unknown_face_index,  # Unknown face number (1, 2, 3...)
                        'bbox': convert_numpy_types(face['bbox']),
                        'confidence': face.get('confidence', 0),
                        'embedding': convert_numpy_types(face['embedding']),
                        'thumbnail_path': thumbnail_path.replace('\\', '/')
                    })
                    
                    unknown_face_index += 1
        
        # Save screenshot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_filename = f"screenshot_{timestamp}.jpg"
        screenshot_path = os.path.join("temp", screenshot_filename)
        
        # Ensure temp directory exists
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        
        # Save the screenshot
        success = cv2.imwrite(screenshot_path, screenshot)
        if not success:
            logger.error(f"Failed to save screenshot: {screenshot_path}")
        
        logger.info(f"Captured {len(unknown_faces)} unknown faces")
        
        return jsonify({
            'success': True,
            'unknown_faces': unknown_faces,
            'total_faces': len(captured_faces),
            'screenshot_path': screenshot_path.replace('\\', '/'),
            'screenshot_filename': screenshot_filename
        })
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-face-photo', methods=['POST'])
def save_face_photo():
    """Save a cropped face photo and add to student database"""
    global captured_frame, captured_faces, face_system
    
    try:
        data = request.json
        face_index = data.get('face_index')
        student_name = data.get('student_name')
        is_new_student = data.get('is_new_student', True)
        class_name = data.get('class_name')  # Add class name parameter
        
        if captured_frame is None:
            return jsonify({'error': 'No captured frame available'}), 400
        
        if face_index >= len(captured_faces):
            return jsonify({'error': f'Face index {face_index} out of range (max: {len(captured_faces)-1})'}), 400
        
        # Get the face data from captured frame
        face = captured_faces[face_index]
        bbox = face['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Expand the bounding box to include more context
        height, width = captured_frame.shape[:2]
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Expand by 50% on each side
        expand_x = int(face_width * 0.5)
        expand_y = int(face_height * 0.5)
        
        # Calculate expanded coordinates with bounds checking
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(width, x2 + expand_x)
        new_y2 = min(height, y2 + expand_y)
        
        # Crop the expanded face from the captured frame
        face_crop = captured_frame[new_y1:new_y2, new_x1:new_x2]
        
        if face_crop.size == 0:
            return jsonify({'error': 'Invalid face crop'}), 400
        
        logger.info(f"Original bbox: ({x1}, {y1}, {x2}, {y2})")
        logger.info(f"Expanded bbox: ({new_x1}, {new_y1}, {new_x2}, {new_y2})")
        logger.info(f"Face crop dimensions: {face_crop.shape}")
        
        # Create student folder in the appropriate class
        if class_name:
            student_folder = os.path.join('students', class_name, student_name)
        else:
            student_folder = os.path.join('students', student_name)
        
        os.makedirs(student_folder, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_name}_{timestamp}.jpg"
        filepath = os.path.join(student_folder, filename)
        
        # Save the face crop with proper encoding
        success = cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return jsonify({'error': 'Failed to save face crop'}), 500
        
        logger.info(f"Saved face crop to: {filepath}")
        
        # Verify the saved image can be read and has a face
        test_img = cv2.imread(filepath)
        if test_img is None:
            logger.error(f"Saved image cannot be read back: {filepath}")
            return jsonify({'error': 'Saved image is corrupted'}), 500
        
        logger.info(f"Test read image dimensions: {test_img.shape}")
        
        # Test face detection on the saved image
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_faces = face_system.app.get(test_img_rgb)
        logger.info(f"Face detection test on saved image found {len(test_faces)} faces")
        
        if len(test_faces) == 0:
            logger.error(f"No faces detected in saved image: {filepath}")
            # Don't delete the file, just return error
            return jsonify({'error': 'No faces detected in the cropped image'}), 500
        
        # Add to face system with detailed logging
        db_success = False
        try:
            if is_new_student:
                # Add as new student
                logger.info(f"Attempting to add new student: {student_name} with file: {filepath}")
                db_success = face_system.add_student(student_name, filepath, class_name)
                logger.info(f"Result of add_student for {student_name}: {db_success}")
            else:
                # Add to existing student (multiple images)
                logger.info(f"Attempting to add image to existing student: {student_name} with file: {filepath}")
                db_success = face_system.add_image_to_student(student_name, filepath, class_name)
                logger.info(f"Result of add_image_to_student for {student_name}: {db_success}")
        except Exception as e:
            logger.error(f"Exception in face system add: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Clean up the file if face system addition failed
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Exception in face system: {str(e)}'}), 500
        
        if db_success:
            logger.info(f"Successfully added face for {student_name}")
            return jsonify({
                'success': True,
                'message': f'Face photo saved for {student_name}',
                'filepath': filepath
            })
        else:
            logger.error(f"Face system returned False for {student_name}")
            # Clean up the file if face system addition failed
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Face system failed to add student'}), 500
            
    except Exception as e:
        logger.error(f"Error saving face photo: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-session', methods=['POST'])
def load_session():
    session_id = request.json.get('session_id')
    path = os.path.join('attendance_logs', f"{session_id}.json")
    if not os.path.exists(path):
        return jsonify({'error': 'Session not found'}), 404
    with open(path) as f:
        session = json.load(f)
    return jsonify({'session': session})

@app.route('/api/resume-session', methods=['POST'])
def resume_session():
    """Resume a previous session by session_id"""
    global attendance_manager, detection_active, detection_thread
    try:
        session_id = request.json.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        path = os.path.join('attendance_logs', f"{session_id}.json")
        if not os.path.exists(path):
            return jsonify({'error': 'Session not found'}), 404
        with open(path) as f:
            session = json.load(f)
        
        # Set as current session in attendance_manager
        attendance_manager.current_session = session
        
        # Start detection loop if not already running
        if not detection_active:
            detection_active = True
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()
            logger.info(f"Started detection loop for resumed session: {session_id}")
        
        return jsonify({'success': True, 'session': session})
    except Exception as e:
        logger.error(f"Error resuming session: {e}")
        return jsonify({'error': str(e)}), 500

# Update the static file serving route
@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """Serve temporary files (screenshots and thumbnails)"""
    return send_file(os.path.join('temp', filename))

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize systems
    if not initialize_systems():
        logger.error("Failed to initialize systems. Exiting.")
        exit(1)
    
    try:
        # Create necessary directories
        os.makedirs("students", exist_ok=True)
        os.makedirs("temp", exist_ok=True)  # Changed to root temp
        os.makedirs("attendance_logs", exist_ok=True)
        
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5155, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        release_camera() 