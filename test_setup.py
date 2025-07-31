#!/usr/bin/env python3
"""
Test script to verify the attendance system setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import insightface
        print(f"✓ InsightFace imported successfully")
    except ImportError as e:
        print(f"✗ InsightFace import failed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        
        # Test for available providers (newer API)
        try:
            providers = ort.get_available_providers()
            print(f"  Available providers: {providers}")
        except AttributeError:
            # Fallback for older versions
            print("  Provider information not available")
            
    except ImportError as e:
        print(f"✗ ONNX Runtime import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test if camera can be accessed"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ Camera opened but failed to read frame")
                cap.release()
                return False
        else:
            print("✗ Failed to open camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_face_recognition():
    """Test face recognition system initialization"""
    print("\nTesting face recognition system...")
    
    try:
        from face_recognition import FaceRecognitionSystem
        
        # Initialize with empty students folder
        face_system = FaceRecognitionSystem(students_folder="test_students", threshold=0.6)
        
        if face_system.initialized:
            print("✓ Face recognition system initialized successfully")
            return True
        else:
            print("✗ Face recognition system failed to initialize")
            return False
            
    except Exception as e:
        print(f"✗ Face recognition test failed: {e}")
        return False

def test_attendance_manager():
    """Test attendance manager initialization"""
    print("\nTesting attendance manager...")
    
    try:
        from attendance_manager import AttendanceManager
        
        attendance_manager = AttendanceManager(logs_folder="test_logs")
        print("✓ Attendance manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Attendance manager test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = ['students', 'attendance_logs', 'templates']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Directory missing: {dir_name}")
            return False
    
    return True

def test_files():
    """Test if required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'app.py',
        'face_recognition.py',
        'attendance_manager.py',
        'requirements.txt',
        'templates/index.html'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File exists: {file_name}")
        else:
            print(f"✗ File missing: {file_name}")
            return False
    
    return True

def test_hardware_acceleration():
    """Test hardware acceleration availability"""
    print("\nTesting hardware acceleration...")
    
    try:
        import onnxruntime as ort
        ort.preload_dlls()
        providers = ort.get_available_providers()
        
        print(f"Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDA provider detected")
            
            # Test if CUDA actually works
            try:
                test_providers = [('CUDAExecutionProvider', {})]
                ort.InferenceSession("", providers=test_providers)
                print("✓ CUDA acceleration working")
                return "CUDA"
            except Exception as e:
                print(f"⚠ CUDA detected but failed to load: {e}")
                print("  → Install CUDA Toolkit or Runtime Libraries")
                return "CUDA_FAILED"
                
        elif 'CoreMLExecutionProvider' in providers:
            print("✓ CoreML acceleration available (Apple Silicon)")
            return "CoreML"
        else:
            print("⚠ Using CPU-only execution")
            return "CPU"
            
    except Exception as e:
        print(f"✗ Error testing hardware acceleration: {e}")
        return "Unknown"

def main():
    """Run all tests"""
    print("=" * 50)
    print("Attendance System Setup Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_files),
        ("Directory Structure", test_directories),
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("Face Recognition", test_face_recognition),
        ("Attendance Manager", test_attendance_manager),
        ("Hardware Acceleration", test_hardware_acceleration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to run.")
        print("\nTo start the application:")
        print("1. Add student photos to the 'students' folder")
        print("2. Run: python app.py")
        print("3. Open your browser to: http://localhost:5000")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Ensure your webcam is connected and accessible")
        print("- Check that all files are in the correct locations")
    
    # Cleanup test directories
    import shutil
    if os.path.exists("test_students"):
        shutil.rmtree("test_students")
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")

if __name__ == "__main__":
    main() 