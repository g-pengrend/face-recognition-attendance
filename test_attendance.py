#!/usr/bin/env python3
"""
Test script to verify attendance system functionality
"""

import requests
import time
import json

def test_attendance_system():
    """Test the attendance system"""
    base_url = "http://localhost:5155"
    
    print("🧪 Testing Attendance System")
    print("=" * 50)
    
    # Test 1: Check system status
    print("\n1. Checking system status...")
    try:
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        print(f"✅ System ready: {status['face_system_ready']}")
        print(f"✅ Students loaded: {status['students_count']}")
        print(f"✅ Detection active: {status['detection_active']}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return
    
    # Test 2: Check current attendance
    print("\n2. Checking current attendance...")
    try:
        response = requests.get(f"{base_url}/api/attendance")
        attendance = response.json()
        print(f"✅ Detection active: {attendance['detection_active']}")
        print(f"✅ Present students: {attendance['attendance'].get('present_students', 0)}")
        print(f"✅ Session ID: {attendance['attendance'].get('session_id', 'None')}")
    except Exception as e:
        print(f"❌ Error checking attendance: {e}")
        return
    
    # Test 3: Start detection session
    print("\n3. Starting detection session...")
    try:
        response = requests.post(f"{base_url}/api/start-detection", 
                               json={"session_name": "test_session"})
        result = response.json()
        if result.get('success'):
            print(f"✅ Detection started: {result['session_id']}")
        else:
            print(f"❌ Failed to start detection: {result.get('error')}")
            return
    except Exception as e:
        print(f"❌ Error starting detection: {e}")
        return
    
    # Test 4: Monitor attendance for 10 seconds
    print("\n4. Monitoring attendance for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        try:
            response = requests.get(f"{base_url}/api/attendance")
            attendance = response.json()
            present_count = attendance['attendance'].get('present_students', 0)
            print(f"⏱️  Present students: {present_count}")
            
            if present_count > 0:
                print("🎉 Students detected! Attendance system is working!")
                break
                
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error monitoring attendance: {e}")
            break
    
    # Test 5: Stop detection
    print("\n5. Stopping detection...")
    try:
        response = requests.post(f"{base_url}/api/stop-detection")
        result = response.json()
        if result.get('success'):
            print("✅ Detection stopped successfully")
        else:
            print(f"❌ Failed to stop detection: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error stopping detection: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_attendance_system() 