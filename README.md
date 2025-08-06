# Classroom Attendance System

A real-time face recognition attendance system built with Flask, OpenCV, and InsightFace. Automatically tracks student attendance using computer vision and provides a modern web interface for session management.

## 🚀 Features

### Core Functionality
- **Real-time Face Recognition**: Automatic student identification using InsightFace
- **Multi-Class Support**: Manage multiple classes with separate student databases
- **Adaptive Detection**: Smart performance optimization based on activity levels
- **Session Management**: Create, resume, and export attendance sessions
- **Lateness Tracking**: Automatic calculation of student punctuality

### Advanced Features
- **Smart Caching**: Performance optimization with intelligent caching system
- **CSV Import**: Bulk class creation from CSV files
- **Unknown Face Detection**: Capture and add unrecognized faces as new students
- **Session Export**: Generate detailed CSV reports with attendance statistics
- **Cache Management**: Monitor and manage system performance

### User Interface
- **Modern Web UI**: Responsive design with real-time updates
- **Live Attendance Feed**: Real-time display of present students
- **Session History**: Complete audit trail of all attendance sessions
- **Student Management**: Add, remove, and manage student profiles

## 🎓 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Camera**: Webcam or IP camera (local webcam recommended)
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for models and cache

### Hardware Acceleration (Still not working, onnxruntime issue)
- **Apple Silicon**: CoreML acceleration for M1/M2 Macs
- **NVIDIA GPU**: CUDA support for faster processing
- **Intel CPU**: Optimized CPU execution

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/g-pengrend/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Setup Test
```bash
python test_setup.py
```

### 5. Start the Application
```bash
python app.py
```

### 6. Access the System
Open your browser and navigate to: `http://localhost:5155`

## 🎓 Onboarding Guide

### First-Time Setup

#### 1. System Initialization
- The system will automatically initialize face recognition models
- Wait for the "System Ready" status indicator
- Check that your camera is connected and working

#### 2. Create Your First Class
1. Click "Create New Class" in the header
2. Enter a class name (e.g., "CI2504P")
3. Prepare a CSV file with student data:
   ```csv
   Serial,Name
   01,John Doe
   02,Jane Smith
   ```
4. Upload the CSV file and click "Create Class"
5. Student folders will be created automatically

#### 3. Add Student Photos
1. Navigate to the "Students" tab
2. Click "Add New Student"
3. Choose your preferred method:
   - **Single Image**: Upload one photo per student
   - **Multiple Images**: Upload several photos for better recognition
   - **Existing Folder**: Use photos from a local folder
4. Add photos to the created student folders

#### 4. Start Your First Session
1. Select your class from the dropdown
2. Enter a session name (e.g., "Math 101 - Week 1")
3. Set the class start time for punctuality tracking
4. Click "Start Detection"
5. Students will be automatically recognized and marked present

### Daily Usage

#### Starting a Session
1. **Select Class**: Choose the appropriate class from the dropdown
2. **Session Details**: Enter session name and start time
3. **Start Detection**: Click the green "Start Detection" button
4. **Monitor**: Watch the live attendance feed for real-time updates

#### During Session
- **Live Updates**: Present students appear in real-time
- **Unknown Faces**: Use "Capture Unknown Faces" to add new students
- **Session Management**: Stop detection when class ends

#### After Session
1. **Stop Detection**: Click the red "Stop Detection" button
2. **Review**: Check the Sessions tab for attendance summary
3. **Export**: Download CSV reports for record-keeping

### Advanced Features

#### Cache Management
- **Monitor Performance**: Use "Cache Stats" to check system performance
- **Cleanup**: Regularly run "Cleanup Cache" to free up space
- **Clear Cache**: Use "Clear Cache" if experiencing issues

#### Student Management
- **Add Students**: Use the Students tab to add new students
- **Multiple Photos**: Add several photos per student for better recognition
- **Remove Students**: Delete students who are no longer in the class

#### Session Management
- **Resume Sessions**: Continue interrupted sessions
- **Export Data**: Generate detailed CSV reports
- **Session History**: Review all past sessions

## 📁 File Structure

```
insightface-attendance/
├── app.py                 # Main Flask application
├── face_recognition.py    # Face recognition logic
├── attendance_manager.py  # Attendance tracking
├── test_setup.py         # Setup verification
├── requirements.txt      # Python dependencies
├── students/             # Student photos
├── attendance_logs/      # Session records
└── templates/
    └── index.html        # Web interface
```

## 🔧 Technical Details

- **Face Recognition**: InsightFace with ONNX Runtime
- **Backend**: Flask web server
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: JSON-based storage for simplicity
- **Optimization**: Apple Silicon (M3) optimized inference

## 🆘 Troubleshooting

### Camera Not Working
- Ensure your webcam is connected and not in use by other applications
- Try restarting the application
- Check camera permissions in your OS

### Face Recognition Issues
- Use high-quality, well-lit photos for student registration
- Ensure faces are clearly visible and front-facing
- Adjust lighting in your classroom

### Performance Issues
- Close other applications using the camera
- Reduce video resolution if needed
- Ensure adequate lighting

### Attendance Not Updating
- Check browser console for debug messages
- Ensure detection is active (green indicator)
- Verify student photos are properly loaded
- Run `python test_attendance.py` to test the system

### General Issues
1. Run `python test_setup.py` to diagnose problems
2. Run `python test_attendance.py` to test attendance functionality
3. Check the console output for error messages
4. Ensure all dependencies are installed correctly
5. Verify student photos are in the correct format

## 📊 Sample Output

The system generates detailed attendance records including:
- Student name
- First and last detection times
- Recognition confidence
- Number of detections
- Session duration and statistics

## 🎯 Performance Tips

- Use good lighting for better recognition
- Clear, high-quality student photos
- Close other camera-using applications
- Ensure adequate system resources

## 📋 Prerequisites

- Python 3.8+
- Apple M3 Mac (optimized for Apple Silicon)
- Webcam

## 🚀 Ready to Use!

Your attendance system will automatically:
- Detect faces in real-time
- Match them against registered students
- Track attendance with timestamps
- Generate detailed reports
- Export data for record-keeping

Happy teaching! 📚✨ 