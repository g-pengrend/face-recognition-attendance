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
import pickle
import hashlib
from pathlib import Path

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

# Add this with other global variables at the top
camera_mode = "local"  # "local" or "ip"
ip_camera_url = "http://192.168.1.100:8080/video"  # Update with your phone's IP

# Add a new global variable to track the actual detection state
detection_state = "stopped"  # "active", "standby", "idle", "stopped"

# Adaptive detection variables
last_face_detection_time = time.time()
idle_timeout = 120  # 30 seconds for testing
is_idle = False
is_standby = False  # New standby state
idle_overlay_active = False
detection_cycle_time = 0.1  # Default cycle time (100ms)
# Update the standby cycle time
standby_cycle_time = 2.0  # Slower cycle when in standby mode (2 seconds)
standby_timeout = 10  # Seconds without faces before entering standby
consecutive_no_faces_count = 0  # Track consecutive frames with no faces
last_standby_check_time = time.time()  # Track when we last checked for standby

class_cache = {}
student_cache = {}
cache_dir = "cache"
cache_metadata_file = os.path.join(cache_dir, "cache_metadata.json")

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

def ensure_cache_directory():
    """Ensure cache directory exists"""
    os.makedirs(cache_dir, exist_ok=True)

def get_folder_hash(folder_path):
    """Calculate hash of folder contents for cache invalidation"""
    if not os.path.exists(folder_path):
        return None
    
    hash_md5 = hashlib.md5()
    max_file_size = 10 * 1024 * 1024  # 10MB limit
    
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > max_file_size:
                    logger.warning(f"Skipping large file {file_path} ({file_size} bytes)")
                    continue
                    
                with open(file_path, 'rb') as f:
                    stat = os.stat(file_path)
                    content = f.read()
                    hash_md5.update(content)
                    hash_md5.update(str(stat.st_mtime).encode())
            except Exception as e:
                logger.warning(f"Error hashing file {file_path}: {e}")
    
    return hash_md5.hexdigest()

def load_cache_metadata():
    """Load cache metadata from file"""
    if os.path.exists(cache_metadata_file):
        try:
            with open(cache_metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
    return {}

def save_cache_metadata(metadata):
    """Save cache metadata to file"""
    try:
        with open(cache_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache metadata: {e}")

def get_cache_key(class_name, cache_type):
    """Generate cache key for class data"""
    return f"{class_name}::{cache_type}"  # Use :: as separator

def get_cache_filename(class_name, cache_type):
    """Generate safe cache filename"""
    safe_class_name = class_name.replace('/', '_').replace('\\', '_')
    return f"{safe_class_name}_{cache_type}.pkl"

def is_cache_valid(class_name, cache_type):
    """Check if cache is valid for a class"""
    metadata = load_cache_metadata()
    cache_key = get_cache_key(class_name, cache_type)
    
    logger.info(f"Checking cache validity for {cache_key}")
    logger.info(f"Available cache keys: {list(metadata.keys())}")
    
    if cache_key not in metadata:
        logger.info(f"Cache key {cache_key} not found in metadata")
        return False
    
    cache_info = metadata[cache_key]
    folder_path = os.path.join("students", class_name)
    current_hash = get_folder_hash(folder_path)
    stored_hash = cache_info.get('hash')
    
    logger.info(f"Current hash: {current_hash}")
    logger.info(f"Stored hash: {stored_hash}")
    
    is_valid = cache_info.get('hash') == current_hash
    logger.info(f"Cache valid: {is_valid}")
    
    return is_valid

def save_to_cache(class_name, cache_type, data):
    """Save data to cache with improved error handling and performance monitoring"""
    start_time = time.time()
    try:
        ensure_cache_directory()
        cache_key = get_cache_key(class_name, cache_type)
        cache_filename = get_cache_filename(class_name, cache_type)
        cache_file = os.path.join(cache_dir, cache_filename)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Update metadata
        metadata = load_cache_metadata()
        folder_path = os.path.join("students", class_name)
        current_hash = get_folder_hash(folder_path)
        
        if current_hash:  # Only save if hash calculation succeeded
            metadata[cache_key] = {
                'hash': current_hash,
                'timestamp': datetime.now().isoformat(),
                'cache_file': cache_file
            }
            save_cache_metadata(metadata)
            elapsed_time = time.time() - start_time
            logger.info(f"Cached {cache_type} data for class {class_name} in {elapsed_time:.2f}s")
        else:
            logger.warning(f"Could not calculate hash for class {class_name}, skipping cache")
            
    except Exception as e:
        logger.error(f"Error saving cache for {class_name}: {e}")

def load_from_cache(class_name, cache_type):
    """Load data from cache with debugging"""
    global cache_hits, cache_misses
    
    logger.info(f"Attempting to load cache for {class_name}::{cache_type}")
    
    if not is_cache_valid(class_name, cache_type):
        cache_misses += 1
        logger.info(f"Cache invalid for {class_name}::{cache_type}")
        return None
    
    cache_key = get_cache_key(class_name, cache_type)
    cache_filename = get_cache_filename(class_name, cache_type)
    cache_file = os.path.join(cache_dir, cache_filename)
    
    logger.info(f"Loading from cache file: {cache_file}")
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        cache_hits += 1
        logger.info(f"Successfully loaded {cache_type} data from cache for class {class_name}")
        return data
    except Exception as e:
        cache_misses += 1
        logger.error(f"Error loading cache for {class_name}: {e}")
        return None

def invalidate_cache(class_name=None):
    """Invalidate cache for a specific class or all classes"""
    metadata = load_cache_metadata()
    
    if class_name:
        # Invalidate specific class
        cache_key = get_cache_key(class_name, "students")
        if cache_key in metadata:
            cache_file = metadata[cache_key].get('cache_file')
            if cache_file and os.path.exists(cache_file):
                os.remove(cache_file)
            del metadata[cache_key]
        
        cache_key = get_cache_key(class_name, "photos")
        if cache_key in metadata:
            cache_file = metadata[cache_key].get('cache_file')
            if cache_file and os.path.exists(cache_file):
                os.remove(cache_file)
            del metadata[cache_key]
    else:
        # Invalidate all cache
        for cache_info in metadata.values():
            cache_file = cache_info.get('cache_file')
            if cache_file and os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except OSError as e:
                    logger.warning(f"Could not remove cache file {cache_file}: {e}")
        metadata.clear()
    
    save_cache_metadata(metadata)
    logger.info(f"Invalidated cache for {'all classes' if class_name is None else class_name}")

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
        # Ensure cache directory exists
        ensure_cache_directory()
        
        # Clean up old cache files on startup (older than 7 days)
        cleaned_count = cleanup_old_cache(max_age_hours=168)  # 7 days
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old cache files on startup")
        
        # Initialize face recognition system
        face_system = FaceRecognitionSystem()
        
        # Initialize attendance manager
        attendance_manager = AttendanceManager()
        attendance_manager.set_face_system(face_system)
        
        logger.info("Systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        return False

def get_camera():
    """Get camera instance with optimized settings"""
    global camera, camera_mode, ip_camera_url
    
    if camera is None:
        if camera_mode == "ip":
            # For IP mode, only try to initialize if we have a valid URL that's not the default
            if ip_camera_url and ip_camera_url != "http://192.168.1.100:8080/video":
                camera = _initialize_camera()
            else:
                # Return None to indicate IP camera needs configuration
                return None
        else:
            camera = _initialize_camera()
    
    return camera

def _initialize_camera():
    """Initialize camera based on current mode"""
    global camera_mode, ip_camera_url
    
    if camera_mode == "ip":
        try:
            logger.info(f"Attempting to connect to IP camera: {ip_camera_url}")
            camera = cv2.VideoCapture(ip_camera_url)
            
            # Give more time for IP camera connection and don't auto-fallback
            # Try to read a frame to test the connection
            ret, test_frame = camera.read()
            if camera.isOpened() and ret:
                logger.info("Connected to IP camera successfully")
                # Set camera properties for IP camera
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 15)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                logger.warning("Failed to connect to IP camera - keeping IP mode for manual configuration")
                # Don't auto-fallback, let user configure the IP
                camera = None
                return camera
        except Exception as e:
            logger.warning(f"Error connecting to IP camera: {e} - keeping IP mode for manual configuration")
            # Don't auto-fallback, let user configure the IP
            camera = None
            return camera
    else:
        logger.info("Using local camera")
        camera = cv2.VideoCapture(0)
        # Set camera properties for local camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return camera

def switch_camera(target_mode=None):
    """Switch between local and IP camera"""
    global camera, camera_mode, detection_active
    
    if detection_active:
        logger.warning("Cannot switch camera while detection is active")
        return False
    
    # Determine target mode
    if target_mode is None:
        target_mode = "ip" if camera_mode == "local" else "local"
    
    if target_mode not in ["local", "ip"]:
        logger.error(f"Invalid camera mode: {target_mode}")
        return False
    
    # Only switch if the mode is different
    if target_mode == camera_mode:
        logger.info(f"Already using {camera_mode} camera")
        return True
    
    # Release current camera
    if camera is not None:
        camera.release()
        camera = None
    
    # Switch mode
    camera_mode = target_mode
    
    # Only initialize camera immediately for local mode
    # For IP mode, let the user configure the IP first
    if target_mode == "local":
        camera = _initialize_camera()
        logger.info(f"Switched to {camera_mode} camera")
    else:
        logger.info(f"Switched to {camera_mode} camera mode - please configure IP address")
    
    return True

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def detection_loop():
    """Adaptive detection loop with proper state management"""
    global detection_active, face_system, attendance_manager, current_frame, current_faces
    global last_face_detection_time, is_idle, is_standby, idle_overlay_active, detection_cycle_time
    global consecutive_no_faces_count, detection_state, last_standby_check_time
    
    # Don't start detection if we're in IP mode but camera isn't configured
    if camera_mode == "ip":
        logger.info("IP camera mode selected - waiting for configuration")
        detection_state = "waiting_for_config"
        
        # Wait for IP camera to be configured
        while detection_active and camera_mode == "ip":
            camera = get_camera()
            if camera is not None and camera.isOpened():
                logger.info("IP camera configured - starting detection")
                break
            time.sleep(1.0)  # Check every second
        
        if not detection_active:
            logger.info("Detection stopped while waiting for IP camera configuration")
            return
    
    # Now get the camera (should be configured by now)
    camera = get_camera()
    if not camera.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Starting adaptive detection loop")
    detection_state = "active"
    
    while detection_active:
        loop_start_time = time.time()
        current_time = time.time()
        
        # Check for manual idle mode first (before processing any frames)
        if is_idle:
            logger.info("Manual idle mode detected - breaking out of detection loop")
            break
        
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            time.sleep(detection_cycle_time)
            continue
        
        try:
            # Store current frame for photo capture
            current_frame = frame.copy()
            
            # Detect all faces in the frame with timing
            detection_start = time.time()
            all_faces = face_system.detect_faces(frame)
            detection_time = (time.time() - detection_start) * 1000  # Convert to milliseconds
            current_faces = all_faces  # Store for photo capture
            
            # Update detection state based on faces found
            if all_faces:
                # Faces detected - reset counters and use normal speed
                last_face_detection_time = current_time
                consecutive_no_faces_count = 0
                last_standby_check_time = current_time  # Reset standby check time
                
                # Exit standby mode if we were in it
                if is_standby:
                    is_standby = False
                    detection_cycle_time = 0.1
                    detection_state = "active"
                    logger.info("Face detected - exiting standby mode, returning to normal speed")
                else:
                    detection_cycle_time = 0.1  # Normal speed when faces are present
                    detection_state = "active"
                    logger.debug("Faces detected - using normal detection speed")
                
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
            else:
                # No faces detected - increment counter
                consecutive_no_faces_count += 1
                
                # Check for standby mode using actual elapsed time
                time_since_last_face = current_time - last_face_detection_time
                
                # Check for standby mode FIRST (before idle)
                if not is_standby and time_since_last_face >= standby_timeout:
                    is_standby = True
                    detection_cycle_time = standby_cycle_time
                    detection_state = "standby"
                    logger.info(f"No faces for {standby_timeout}s - entering standby mode ({standby_cycle_time}s cycle)")
                
                # Check for idle mode AFTER standby (only if not in standby)
                elif not is_idle and time_since_last_face > idle_timeout:
                    is_idle = True
                    is_standby = False  # Exit standby when going idle
                    idle_overlay_active = True
                    detection_state = "idle"
                    logger.info(f"System went idle after {idle_timeout} seconds of no face detection")
                    # When idle, we'll break out of the detection loop
                    break
            
            # Log performance if detection is slow
            if detection_time > 100:  # More than 100ms
                logger.warning(f"Slow face detection: {detection_time:.1f}ms for {len(all_faces)} faces")
            
            # Calculate total loop time
            total_loop_time = (time.time() - loop_start_time) * 1000
            if total_loop_time > 200:  # More than 200ms total
                logger.warning(f"Slow detection loop: {total_loop_time:.1f}ms total")
            
            # Sleep based on current cycle time
            time.sleep(detection_cycle_time)
            
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            time.sleep(detection_cycle_time)
    
    # If we're here and idle, start the idle monitoring loop
    if is_idle:
        idle_monitoring_loop()
        # After idle monitoring loop exits, restart the main detection loop if still active
        if detection_active and not is_idle:
            logger.info("Restarting main detection loop after exiting idle mode")
            detection_loop()  # Recursive call to restart the detection loop
        else:
            logger.info("Detection stopped after idle mode")
    else:
        logger.info("Detection loop stopped")

def idle_monitoring_loop():
    """Separate loop that runs when system is idle - only checks for resume signal"""
    global detection_active, is_idle, idle_overlay_active, last_face_detection_time, detection_state
    
    logger.info("Starting idle monitoring loop")
    
    while detection_active and is_idle:
        # In idle mode, we only need to:
        # 1. Keep the camera frame updated for the overlay
        # 2. Check if we should resume detection
        # 3. Sleep for a longer period since we're not doing face detection
        
        camera = get_camera()
        if camera.isOpened():
            ret, frame = camera.read()
            if ret:
                global current_frame
                current_frame = frame.copy()
        
        # Sleep for 2 seconds - we're not doing any detection work
        time.sleep(2.0)
    
    logger.info("Idle monitoring loop stopped")

def resume_detection_from_idle():
    """Resume detection from idle state"""
    global is_idle, is_standby, idle_overlay_active, detection_active, last_face_detection_time, consecutive_no_faces_count, detection_thread, detection_state
    
    if is_idle:
        is_idle = False
        is_standby = False
        idle_overlay_active = False
        detection_active = True
        detection_state = 'active'  # Explicitly set detection state
        last_face_detection_time = time.time()  # Reset the timer
        consecutive_no_faces_count = 0
        last_standby_check_time = time.time()  # Reset standby check time
        detection_state = "active"
        logger.info("Detection resumed from idle state - all timers reset")
        
        return True
    else:
        return False

# Updated API endpoints to show proper state
@app.route('/api/status')
def get_status():
    """Get system status with proper detection state"""
    global face_system, attendance_manager, detection_active, detection_state
    
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
        'detection_state': detection_state,  # New field showing actual state
        'students_count': len(all_students),
        'current_session': convert_numpy_types(current_attendance)
    }
    
    logger.info(f"Status: detection_active={detection_active}, detection_state={detection_state}, present_students={current_attendance.get('present_students', 0)}")
    return jsonify(status)

@app.route('/api/detection-status')
def detection_status():
    """Get the current detection status (active, idle, standby)"""
    global detection_active, is_idle, is_standby, idle_overlay_active, detection_state
    
    try:
        # Use the actual global variables from your detection system
        return jsonify({
            'detection_active': detection_active,
            'idle_overlay_active': idle_overlay_active,
            'state': detection_state  # This will be 'active', 'idle', 'standby', or 'stopped'
        })
        
    except Exception as e:
        logger.error(f"Error in detection status: {e}")
        return jsonify({'error': str(e)}), 500

# Update the attendance API to show proper state
@app.route('/api/attendance')
def get_attendance():
    """Get current attendance status with proper detection state"""
    global attendance_manager, detection_active, face_system, detection_state
    
    if not attendance_manager:
        return jsonify({'detection_active': detection_active, 'detection_state': detection_state, 'attendance': {}}), 200
    
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
        'detection_state': detection_state,  # New field
        'attendance': convert_numpy_types(attendance)
    }
    
    logger.info(f"Attendance API: detection_active={detection_active}, detection_state={detection_state}, present_students={attendance.get('present_students', 0)}")
    return jsonify(response)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

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

@app.route('/api/sessions')
def get_sessions():
    """List all saved sessions (JSON files)"""
    sessions = []
    for fname in os.listdir('attendance_logs'):
        if fname.endswith('.json'):
            try:
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
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading session file {fname}: {e}")
                # Optionally remove corrupted file
                # os.remove(os.path.join('attendance_logs', fname))
                continue
    
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
    """Set the current class with caching"""
    try:
        data = request.get_json()
        class_name = data.get('class_name')
        
        if not class_name:
            return jsonify({'error': 'Class name is required'}), 400
        
        logger.info(f"Setting class: {class_name}")
        
        # Try to load from cache first
        cached_students = load_from_cache(class_name, "students")
        logger.info(f"Cache load result: {cached_students is not None}")
        
        if cached_students:
            logger.info(f"Using cached data for {class_name}")
            # Use cached data without loading from folder
            success = face_system.set_current_class_from_cache(class_name, cached_students)
            if success:
                logger.info(f"Loaded class {class_name} from cache with {len(face_system.students_db)} students")
        else:
            logger.info(f"No cache found, loading from folder for {class_name}")
            # Load normally and cache
            success = face_system.set_current_class(class_name)
            if success:
                # Cache the loaded data
                cache_data = {
                    'students_db': face_system.students_db,
                    'student_names': face_system.get_students_list(),
                    'timestamp': datetime.now().isoformat()
                }
                save_to_cache(class_name, "students", cache_data)
                logger.info(f"Loaded class {class_name} from folder and cached with {len(face_system.students_db)} students")
        
        if not success:
            return jsonify({'error': f'Failed to load class {class_name}'}), 400
        
        # Update attendance manager
        final_student_count = len(face_system.get_students_list())
        attendance_manager.set_total_students(final_student_count)
        
        logger.info(f"Final student count: {final_student_count}")
        
        return jsonify({
            'success': True,
            'message': f'Class {class_name} loaded successfully',
            'students_count': final_student_count,
            'cached': cached_students is not None
        })
        
    except Exception as e:
        logger.error(f"Error setting class: {e}")
        return jsonify({'error': f'Error setting class: {str(e)}'}), 500

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
    """Add student with cache invalidation"""
    name = request.form.get('name')
    folder = request.form.get('folder')
    folder_path = request.form.get('folder_path')
    images = request.files.getlist('images')
    image = request.files.get('image')
    class_name = request.form.get('class_name')

    try:
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
            
            # Invalidate cache for the class
            if class_name:
                invalidate_cache(class_name)
            
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
            # Add to face system
            face_system.add_student_multiple_images(name, image_paths)
            
            # Invalidate cache for the class
            if class_name:
                invalidate_cache(class_name)
            
            return jsonify({'success': True, 'message': f'Student {name} added with {len(image_paths)} images.'})

        # Handle existing folder
        elif folder_path:
            # List all images in the folder
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_paths:
                return jsonify({'success': False, 'error': 'No images found in folder.'})
            face_system.add_student_multiple_images(name, image_paths)
            
            # Invalidate cache for the current class
            class_name = face_system.get_current_class()
            if class_name:
                invalidate_cache(class_name)
            
            return jsonify({'success': True, 'message': f'Student {name} added from existing folder.'})

        return jsonify({'success': False, 'error': 'Invalid request.'})
        
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        return jsonify({'success': False, 'error': f'Error adding student: {str(e)}'}), 500

@app.route('/api/remove-student', methods=['POST'])
def remove_student():
    """Remove student with cache invalidation"""
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
            # After successfully removing student, invalidate cache
            class_name = face_system.get_current_class()
            if class_name:
                invalidate_cache(class_name)
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
    """Video streaming route - NO overlays, only face detection boxes"""
    def generate():
        global current_frame
        
        camera = get_camera()
        if camera is None:
            # If camera is None (IP mode not configured), return a blank frame or error
            logger.warning("Camera not available - IP camera may need configuration")
            # Create a blank frame with text
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Camera not configured", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank_frame, "Please configure IP camera", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
        
        if not camera.isOpened():
            logger.error("Failed to open camera for streaming")
            return
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Draw detection results on frame (only face boxes and names)
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
            
            # Store current frame for photo capture
            current_frame = frame.copy()
            
            # NO OVERLAYS - let the frontend handle all overlays
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
        test_faces = face_system.detect_faces(test_img_rgb)
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
            # After successfully saving the face photo, invalidate cache
            class_name = data.get('class_name') or face_system.get_current_class()
            if class_name:
                invalidate_cache(class_name)
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

# Add this new endpoint after the existing routes
@app.route('/api/resume-from-idle', methods=['POST'])
def resume_from_idle():
    """Resume detection from idle state"""
    global is_idle, is_standby, idle_overlay_active, detection_active, last_face_detection_time, consecutive_no_faces_count, detection_thread, detection_state
    
    try:
        # Reset all idle-related variables
        is_idle = False
        is_standby = False
        idle_overlay_active = False
        detection_active = True
        detection_state = 'active'  # Explicitly set detection state
        last_face_detection_time = time.time()
        consecutive_no_faces_count = 0
        
        # Force a small delay to ensure the detection loop picks up the changes
        time.sleep(0.1)
        
        # Check if detection thread is still running
        if detection_thread and detection_thread.is_alive():
            logger.info("Detection thread still running, resuming from idle")
        else:
            # If thread is not running, start a new one
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()
            logger.info("Started new detection thread after resuming from idle")
        
        logger.info(f"Detection resumed from idle state. Current state: is_idle={is_idle}, is_standby={is_standby}, detection_state={detection_state}")
        
        return jsonify({
            'success': True,
            'message': 'Detection resumed from idle state',
            'debug_info': {
                'is_idle': is_idle,
                'is_standby': is_standby,
                'detection_active': detection_active,
                'detection_state': detection_state
            }
        })
        
    except Exception as e:
        logger.error(f"Error resuming from idle: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Update the static file serving route
@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """Serve temporary files (screenshots and thumbnails)"""
    return send_file(os.path.join('temp', filename))

@app.route('/api/create-class', methods=['POST'])
def create_class():
    """Create a new class from CSV file"""
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        csv_file = request.files['csv_file']
        class_name = request.form.get('class_name', '').strip()
        
        if not class_name:
            return jsonify({'error': 'Class name is required'}), 400
        
        if not csv_file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Create class folder
        class_folder = os.path.join("students", class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Read CSV and create student folders
        students_count = 0
        try:
            # Read CSV content
            csv_content = csv_file.read().decode('utf-8')
            csv_reader = csv.DictReader(csv_content.splitlines())
            
            for row in csv_reader:
                serial = row.get('Serial', '').strip()
                name = row.get('Name', '').strip()
                
                if serial and name:
                    # Create student folder with format: 01_John Doe
                    student_folder = os.path.join(class_folder, f"{serial}_{name}")
                    os.makedirs(student_folder, exist_ok=True)
                    students_count += 1
                    
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 400
        
        if students_count == 0:
            # Clean up empty class folder
            if os.path.exists(class_folder):
                shutil.rmtree(class_folder)
            return jsonify({'error': 'No valid students found in CSV'}), 400
        
        logger.info(f"Created class '{class_name}' with {students_count} students")
        
        # Invalidate cache for the new class
        invalidate_cache(class_name)
        
        return jsonify({
            'success': True,
            'class_name': class_name,
            'students_count': students_count,
            'message': f'Class "{class_name}" created successfully with {students_count} students'
        })
        
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        return jsonify({'error': f'Error creating class: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cache"""
    try:
        invalidate_cache()
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': f'Error clearing cache: {str(e)}'}), 500

@app.route('/api/cache/status')
def cache_status():
    """Get cache status"""
    try:
        metadata = load_cache_metadata()
        cache_info = {}
        
        for cache_key, info in metadata.items():
            try:
                class_name, cache_type = cache_key.split('::', 1)
                if class_name not in cache_info:
                    cache_info[class_name] = {}
                
                cache_info[class_name][cache_type] = {
                    'timestamp': info.get('timestamp'),
                    'valid': is_cache_valid(class_name, cache_type)
                }
            except ValueError:
                logger.warning(f"Invalid cache key format: {cache_key}")
                continue
        
        return jsonify({
            'success': True,
            'cache_info': cache_info,
            'total_cached_classes': len(cache_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return jsonify({'error': f'Error getting cache status: {str(e)}'}), 500

def get_cache_stats():
    """Get cache statistics"""
    try:
        metadata = load_cache_metadata()
        total_files = 0
        total_size = 0
        cache_info = {}
        
        for cache_key, info in metadata.items():
            try:
                class_name, cache_type = cache_key.split('::', 1)
                cache_file = info.get('cache_file')
                
                if cache_file and os.path.exists(cache_file):
                    file_size = os.path.getsize(cache_file)
                    total_size += file_size
                    total_files += 1
                    
                    if class_name not in cache_info:
                        cache_info[class_name] = {}
                    
                    cache_info[class_name][cache_type] = {
                        'size': file_size,
                        'timestamp': info.get('timestamp'),
                        'valid': is_cache_valid(class_name, cache_type)
                    }
                    
            except ValueError:
                continue
        
        return {
            'total_files': total_files,
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_info': cache_info
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return None

# Add this endpoint
@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    try:
        stats = get_cache_stats()
        if stats:
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({'error': 'Could not retrieve cache statistics'}), 500
            
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': f'Error getting cache stats: {str(e)}'}), 500

def cleanup_old_cache(max_age_hours=24):
    """Clean up cache files older than specified hours"""
    try:
        metadata = load_cache_metadata()
        current_time = datetime.now()
        cleaned_count = 0
        
        for cache_key, info in metadata.items():
            timestamp_str = info.get('timestamp')
            if timestamp_str:
                try:
                    cache_time = datetime.fromisoformat(timestamp_str)
                    age_hours = (current_time - cache_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        cache_file = info.get('cache_file')
                        if cache_file and os.path.exists(cache_file):
                            os.remove(cache_file)
                            cleaned_count += 1
                        
                        # Remove from metadata
                        del metadata[cache_key]
                        
                except ValueError:
                    continue
        
        if cleaned_count > 0:
            save_cache_metadata(metadata)
            logger.info(f"Cleaned up {cleaned_count} old cache files")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        return 0

# Add this endpoint
@app.route('/api/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Clean up old cache files"""
    try:
        max_age = request.json.get('max_age_hours', 24)
        cleaned_count = cleanup_old_cache(max_age)
        
        return jsonify({
            'success': True,
            'cleaned_count': cleaned_count,
            'message': f'Cleaned up {cleaned_count} old cache files'
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        return jsonify({'error': f'Error cleaning up cache: {str(e)}'}), 500

# Add these global variables at the top with other globals
cache_hits = 0
cache_misses = 0

def get_cache_performance():
    """Get cache performance statistics"""
    total_requests = cache_hits + cache_misses
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    return {
        'hits': cache_hits,
        'misses': cache_misses,
        'total_requests': total_requests,
        'hit_rate': hit_rate
    }

# Update the load_from_cache function
def load_from_cache(class_name, cache_type):
    """Load data from cache with debugging"""
    global cache_hits, cache_misses
    
    logger.info(f"Attempting to load cache for {class_name}::{cache_type}")
    
    if not is_cache_valid(class_name, cache_type):
        cache_misses += 1
        logger.info(f"Cache invalid for {class_name}::{cache_type}")
        return None
    
    cache_key = get_cache_key(class_name, cache_type)
    cache_filename = get_cache_filename(class_name, cache_type)
    cache_file = os.path.join(cache_dir, cache_filename)
    
    logger.info(f"Loading from cache file: {cache_file}")
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        cache_hits += 1
        logger.info(f"Successfully loaded {cache_type} data from cache for class {class_name}")
        return data
    except Exception as e:
        cache_misses += 1
        logger.error(f"Error loading cache for {class_name}: {e}")
        return None

# Add this endpoint
@app.route('/api/cache/performance')
def cache_performance():
    """Get cache performance statistics"""
    try:
        performance = get_cache_performance()
        return jsonify({
            'success': True,
            'performance': performance
        })
    except Exception as e:
        logger.error(f"Error getting cache performance: {e}")
        return jsonify({'error': f'Error getting cache performance: {str(e)}'}), 500

@app.route('/api/cache/debug/<class_name>')
def debug_cache(class_name):
    """Debug cache for a specific class"""
    try:
        metadata = load_cache_metadata()
        cache_key = get_cache_key(class_name, "students")
        cache_filename = get_cache_filename(class_name, "students")
        cache_file = os.path.join(cache_dir, cache_filename)
        folder_path = os.path.join("students", class_name)
        
        debug_info = {
            'class_name': class_name,
            'cache_key': cache_key,
            'cache_filename': cache_filename,
            'cache_file_exists': os.path.exists(cache_file),
            'folder_exists': os.path.exists(folder_path),
            'metadata_keys': list(metadata.keys()),
            'cache_in_metadata': cache_key in metadata,
        }
        
        if cache_key in metadata:
            cache_info = metadata[cache_key]
            debug_info.update({
                'stored_hash': cache_info.get('hash'),
                'timestamp': cache_info.get('timestamp'),
                'cache_file_path': cache_info.get('cache_file')
            })
        
        current_hash = get_folder_hash(folder_path)
        debug_info['current_hash'] = current_hash
        
        is_valid = is_cache_valid(class_name, "students")
        debug_info['cache_valid'] = is_valid
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/status')
def get_camera_status():
    """Get current camera status"""
    global camera_mode, camera, ip_camera_url, detection_state
    
    status = {
        'mode': camera_mode,
        'connected': camera is not None and camera.isOpened() if camera else False,
        'ip_url': ip_camera_url if camera_mode == "ip" else None,
        'needs_configuration': camera_mode == "ip" and (camera is None or not camera.isOpened()),
        'detection_state': detection_state
    }
    
    return jsonify(status)

@app.route('/api/camera/switch', methods=['POST'])
def switch_camera_endpoint():
    """Switch camera mode"""
    global camera_mode, detection_active
    
    try:
        if detection_active:
            return jsonify({
                'success': False,
                'message': 'Cannot switch camera while detection is active'
            }), 400
        
        data = request.get_json()
        target_mode = data.get('mode') if data else None
        
        # If no specific mode requested, toggle between local and ip
        if not target_mode:
            target_mode = "ip" if camera_mode == "local" else "local"
        
        if target_mode not in ["local", "ip"]:
            return jsonify({
                'success': False,
                'message': 'Invalid camera mode. Must be "local" or "ip"'
            }), 400
        
        # Only switch if the mode is different
        if target_mode != camera_mode:
            success = switch_camera(target_mode)
            if success:
                if target_mode == "ip":
                    return jsonify({
                        'success': True,
                        'message': 'Switched to IP camera mode. Please configure the IP address.',
                        'mode': camera_mode,
                        'needs_configuration': True
                    })
                else:
                    return jsonify({
                        'success': True,
                        'message': f'Switched to {camera_mode} camera',
                        'mode': camera_mode
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to switch camera'
                }), 500
        else:
            return jsonify({
                'success': True,
                'message': f'Already using {camera_mode} camera',
                'mode': camera_mode
            })
            
    except Exception as e:
        logger.error(f"Error switching camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/set-ip', methods=['POST'])
def set_ip_camera_url():
    """Set IP camera URL and test connection"""
    global ip_camera_url, camera
    
    try:
        data = request.get_json()
        new_url = data.get('ip_url')
        
        if not new_url:
            return jsonify({'error': 'IP URL is required'}), 400
        
        # Validate URL format
        if not new_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400
        
        # Update the IP camera URL
        ip_camera_url = new_url
        logger.info(f"Updated IP camera URL to: {ip_camera_url}")
        
        # Release any existing camera connection
        if camera is not None:
            camera.release()
            camera = None
        
        # Test the new IP camera connection
        test_camera = cv2.VideoCapture(ip_camera_url)
        
        if not test_camera.isOpened():
            test_camera.release()
            return jsonify({
                'success': False,
                'message': 'Failed to connect to IP camera. Please check the URL and try again.',
                'ip_url': ip_camera_url
            }), 400
        
        # Try to read a frame to verify connection
        ret, frame = test_camera.read()
        test_camera.release()
        
        if not ret or frame is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read frame from IP camera. Please check the connection.',
                'ip_url': ip_camera_url
            }), 400
        
        # If we get here, the connection is successful
        # Initialize the actual camera with the new URL
        camera = _initialize_camera()
        
        if camera is None or not camera.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to initialize IP camera after successful test.',
                'ip_url': ip_camera_url
            }), 500
        
        logger.info("IP camera connection test successful and camera initialized")
        return jsonify({
            'success': True,
            'message': f'IP camera connected successfully! URL: {ip_camera_url}',
            'ip_url': ip_camera_url,
            'frame_size': f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "Unknown"
        })
        
    except Exception as e:
        logger.error(f"Error setting IP camera URL: {e}")
        return jsonify({'error': f'Connection test failed: {str(e)}'}), 500

@app.route('/api/camera/test-ip', methods=['POST'])
def test_ip_camera():
    """Test IP camera connection"""
    try:
        data = request.get_json()
        test_url = data.get('ip_url')
        
        if not test_url:
            return jsonify({'error': 'IP URL is required'}), 400
        
        # Validate URL format
        if not test_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400
        
        logger.info(f"Testing IP camera connection: {test_url}")
        
        # Try to connect to the IP camera
        test_camera = cv2.VideoCapture(test_url)
        
        if not test_camera.isOpened():
            return jsonify({'error': 'Failed to open IP camera connection'}), 400
        
        # Try to read a frame
        ret, frame = test_camera.read()
        test_camera.release()
        
        if not ret or frame is None:
            return jsonify({'error': 'Failed to read frame from IP camera'}), 400
        
        logger.info("IP camera test successful")
        return jsonify({
            'success': True,
            'message': 'IP camera connection test successful',
            'frame_size': f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "Unknown"
        })
        
    except Exception as e:
        logger.error(f"Error testing IP camera: {e}")
        return jsonify({'error': f'Connection test failed: {str(e)}'}), 500

@app.route('/api/enter-idle', methods=['POST'])
def enter_idle():
    """Manually enter idle mode"""
    global is_idle, is_standby, idle_overlay_active, detection_state, last_face_detection_time
    
    try:
        # Set the same state variables that trigger idle mode
        is_idle = True
        is_standby = False
        idle_overlay_active = True
        detection_state = 'idle'
        
        # Force the detection loop to break out by setting the timer to trigger idle
        # This ensures the detection loop will exit and enter idle monitoring
        last_face_detection_time = time.time() - (idle_timeout + 1)  # Force idle trigger
        
        logger.info("Manually entered idle mode via API - forcing detection loop to exit")
        return jsonify({'success': True, 'message': 'System entered idle mode'})
    except Exception as e:
        logger.error(f"Error entering idle mode: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        os.makedirs("cache", exist_ok=True)  # Add this line
        
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5155, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        release_camera() 