#!/usr/bin/env python3
"""
IP Camera Setup Test Script
This script helps you test your IP camera connection before using it with the attendance system.
"""

import cv2
import sys
import time

def test_ip_camera(url):
    """Test IP camera connection and display a frame"""
    print(f"Testing IP camera connection: {url}")
    print("=" * 50)
    
    try:
        # Try to connect to the IP camera
        print("1. Attempting to connect...")
        camera = cv2.VideoCapture(url)
        
        if not camera.isOpened():
            print("❌ Failed to open IP camera connection")
            return False
        
        print("✅ Connection opened successfully")
        
        # Try to read a frame
        print("2. Attempting to read frame...")
        ret, frame = camera.read()
        
        if not ret or frame is None:
            print("❌ Failed to read frame from IP camera")
            camera.release()
            return False
        
        print("✅ Frame read successfully")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Display the frame for 3 seconds
        print("3. Displaying frame for 3 seconds...")
        cv2.imshow('IP Camera Test', frame)
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        
        # Release the camera
        camera.release()
        
        print("✅ IP camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    print("IP Camera Setup Test")
    print("=" * 50)
    print()
    
    # Common IP camera URLs
    common_urls = [
        "http://192.168.1.100:8080/video",  # IP Webcam default
        "http://192.168.1.100:4747/video",  # DroidCam default
        "http://192.168.1.101:8080/video",  # Alternative IP
        "http://192.168.1.102:8080/video",  # Another alternative
    ]
    
    print("Common IP camera URLs to try:")
    for i, url in enumerate(common_urls, 1):
        print(f"{i}. {url}")
    print()
    
    # Get user input
    while True:
        choice = input("Enter a number to test a common URL, or type 'custom' to enter your own URL: ").strip()
        
        if choice.lower() == 'custom':
            url = input("Enter your IP camera URL: ").strip()
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(common_urls):
            url = common_urls[int(choice) - 1]
            break
        else:
            print("Invalid choice. Please try again.")
    
    print()
    
    # Test the camera
    success = test_ip_camera(url)
    
    print()
    if success:
        print("🎉 Your IP camera is working! You can now use it in the attendance system.")
        print("Make sure to:")
        print("1. Use the same URL in the web interface")
        print("2. Keep your phone's IP camera app running")
        print("3. Ensure both devices are on the same network")
    else:
        print("❌ IP camera test failed. Please check:")
        print("1. Your phone's IP address (use 'ipconfig' on Windows or 'ifconfig' on Mac/Linux)")
        print("2. Your IP camera app is running and accessible")
        print("3. Both devices are on the same network")
        print("4. No firewall is blocking the connection")
        print("5. The URL format is correct")

if __name__ == "__main__":
    main() 