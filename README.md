# Classroom Attendance System with Face Recognition

A real-time attendance tracking system using InsightFace for face detection and recognition, optimized for Apple Silicon (M3) Macs.

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your Setup
```bash
python test_setup.py
```

### 3. Add Student Photos
Place clear, front-facing photos in the `students` folder:
```
students/
├── john_doe.jpg
├── jane_smith.jpg
└── mike_wilson.jpg
```

**Photo Requirements:**
- Clear, front-facing photos
- One face per photo
- Supported formats: JPG, PNG, BMP
- Filename becomes the student name (without extension)

### 4. Start the Application
```bash
python app.py
```

### 5. Open Your Browser
Go to: `http://localhost:5155`

## 🎯 Features

- **Real-time Face Detection**: Live webcam feed with face recognition
- **Automatic Attendance Tracking**: Timestamps and confidence scores
- **Session Management**: Multiple sessions with detailed records
- **Modern Web Interface**: Responsive UI with real-time updates
- **Data Export**: CSV export with comprehensive statistics
- **Student Database**: Store student faces and information
- **Apple M3 Optimized**: Optimized for Apple Silicon performance

## 🎓 How to Use

### Starting a Session
1. Click "Start Detection" 
2. Students will be automatically recognized as they appear in the camera
3. Attendance is marked with timestamps
4. Click "Stop Detection" when finished

### Viewing Results
- **Live Tab**: See real-time detection and current attendance
- **Attendance Tab**: View current session details
- **Sessions Tab**: Browse previous sessions and export data
- **Students Tab**: Manage registered students

### Exporting Data
- Click on any session in the Sessions tab
- Click "Export CSV" to download attendance records

### Advanced Usage
- **Custom Session Names**: Enter a custom name when starting detection
- **Multiple Sessions**: Run multiple sessions per day - each is saved separately
- **Data Management**: Attendance logs stored in JSON format with detailed statistics

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