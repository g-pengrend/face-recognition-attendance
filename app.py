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

from face_recognition import FaceRecognitionSystem
from attendance_manager import AttendanceManager

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
    """Get camera instance"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def detection_loop():
    """Main detection loop for face recognition and attendance tracking"""
    global detection_active, face_system, attendance_manager
    
    camera = get_camera()
    if not camera.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Starting detection loop")
    
    while detection_active:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        try:
            # Detect and recognize faces
            faces = face_system.detect_faces(frame)
            
            # Mark attendance for recognized faces
            for face in faces:
                if face['student_name'] and face['confidence'] >= 0.6:
                    success = attendance_manager.mark_attendance(
                        face['student_name'], 
                        face['confidence']
                    )
                    if success:
                        logger.info(f"Marked attendance for {face['student_name']} (confidence: {face['confidence']:.2f})")
                    else:
                        logger.warning(f"Failed to mark attendance for {face['student_name']}")
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
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
    
    status = {
        'face_system_ready': face_system is not None and face_system.initialized,
        'detection_active': detection_active,
        'students_count': len(face_system.get_students_list()) if face_system else 0,
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
        session_start_time = request.json.get('session_start_time')  # <-- ADD THIS LINE
        
        # Start attendance session with start time
        session_id = attendance_manager.start_session(session_name, session_start_time)  # <-- PASS session_start_time
        
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
    global attendance_manager, detection_active
    
    if not attendance_manager:
        return jsonify({'detection_active': detection_active, 'attendance': {}}), 200
    
    attendance = attendance_manager.get_current_attendance()
    
    # Add detection status to response
    response = {
        'detection_active': detection_active,
        'attendance': convert_numpy_types(attendance)
    }
    
    logger.info(f"Attendance API: detection_active={detection_active}, present_students={attendance.get('present_students', 0)}")
    return jsonify(response)

@app.route('/api/sessions')
def get_sessions():
    """Get list of all sessions"""
    global attendance_manager
    
    if not attendance_manager:
        return jsonify({'error': 'Attendance manager not initialized'}), 500
    
    sessions = attendance_manager.list_sessions()
    return jsonify({'sessions': convert_numpy_types(sessions)})

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

@app.route('/api/add-student', methods=['POST'])
def add_student():
    """Add a new student via API"""
    global face_system
    
    if not face_system:
        return jsonify({'error': 'Face system not initialized'}), 500
    
    try:
        data = request.json
        name = data.get('name')
        image_path = data.get('image_path')
        
        if not name or not image_path:
            return jsonify({'error': 'Name and image_path are required'}), 400
        
        success = face_system.add_student(name, image_path)
        if success:
            return jsonify({'success': True, 'message': f'Student {name} added successfully'})
        else:
            return jsonify({'error': 'Failed to add student'}), 500
            
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        return jsonify({'error': str(e)}), 500

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
                
                for face in faces:
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
                        cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        os.makedirs("attendance_logs", exist_ok=True)
        
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5155, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        release_camera() 