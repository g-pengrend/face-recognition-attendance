# IP Camera Setup Guide

This guide will help you set up your phone as an IP camera for the face recognition attendance system.

## Prerequisites

1. **Both devices on the same network**: Your computer and phone must be connected to the same WiFi network
2. **IP camera app installed**: Install one of the recommended apps below
3. **Network access**: Ensure no firewall is blocking the connection

## Recommended IP Camera Apps

### Android Apps
1. **IP Webcam** (Free) - Most popular and reliable
   - Download from Google Play Store
   - Default URL: `http://[PHONE_IP]:8080/video`
   - Features: High quality, low latency, easy setup

2. **DroidCam** (Free/Paid)
   - Download from Google Play Store
   - Default URL: `http://[PHONE_IP]:4747/video`
   - Features: Good quality, multiple connection options

3. **EpocCam** (Free/Paid)
   - Download from Google Play Store
   - Default URL: `http://[PHONE_IP]:8080/video`
   - Features: Cross-platform, good quality

### iOS Apps
1. **IP Camera Lite** (Free)
   - Download from App Store
   - Default URL: `http://[PHONE_IP]:8080/video`
   - Features: Simple setup, good quality

2. **EpocCam** (Free/Paid)
   - Download from App Store
   - Default URL: `http://[PHONE_IP]:8080/video`
   - Features: Cross-platform, professional features

## Step-by-Step Setup

### Step 1: Find Your Phone's IP Address

**Android:**
1. Go to Settings → Network & Internet → WiFi
2. Tap on your connected network
3. Note the IP address (e.g., 192.168.1.100)

**iOS:**
1. Go to Settings → WiFi
2. Tap the (i) icon next to your connected network
3. Note the IP address (e.g., 192.168.1.100)

### Step 2: Install and Configure IP Camera App

**Using IP Webcam (Recommended):**

1. Install "IP Webcam" from Google Play Store
2. Open the app
3. Scroll down and tap "Start server"
4. The app will show a URL like: `http://192.168.1.100:8080`
5. Open this URL in your computer's browser to test
6. You should see your phone's camera feed

### Step 3: Test the Connection

**Option 1: Use the test script**
```bash
python test_camera_setup.py
```

**Option 2: Test in browser**
1. Open your browser
2. Go to: `http://[PHONE_IP]:8080/video`
3. You should see your camera feed

### Step 4: Configure in Attendance System

1. Start the attendance system
2. Go to the camera controls section
3. Click the "Phone" button to switch to IP camera mode
4. Enter your IP camera URL: `http://[PHONE_IP]:8080/video`
5. Click "Save"
6. Click "Test" to verify the connection
7. If successful, you can now use your phone as a camera!

## Troubleshooting

### Common Issues

**1. "Failed to connect to IP camera"**
- Check that both devices are on the same network
- Verify the IP address is correct
- Ensure the IP camera app is running
- Try restarting the IP camera app

**2. "Failed to read frame"**
- Check that the URL format is correct
- Ensure the IP camera app is actively streaming
- Try a different port (8080, 4747, etc.)
- Check if your firewall is blocking the connection

**3. "Connection timeout"**
- Move devices closer to the WiFi router
- Check WiFi signal strength
- Try using a different network
- Restart both devices

**4. "Invalid URL format"**
- Make sure the URL starts with `http://` or `https://`
- Check for typos in the IP address
- Ensure the port number is included

### Network Troubleshooting

**Check network connectivity:**
```bash
# On Windows
ipconfig

# On Mac/Linux
ifconfig

# Test connectivity
ping [PHONE_IP]
```

**Common IP ranges:**
- Home networks: 192.168.1.x, 192.168.0.x, 10.0.0.x
- Mobile hotspots: 192.168.43.x, 192.168.42.x

### Advanced Configuration

**Custom Ports:**
If the default port doesn't work, try:
- 8080 (most common)
- 4747 (DroidCam)
- 8081, 8082 (alternatives)

**URL Formats:**
```
http://[IP]:[PORT]/video
http://[IP]:[PORT]/mjpeg
http://[IP]:[PORT]/stream
```

## Tips for Best Performance

1. **Use 5GHz WiFi** when possible for better bandwidth
2. **Keep devices close** to the WiFi router
3. **Close other apps** on your phone to free up resources
4. **Use a phone stand** to keep the camera stable
5. **Ensure good lighting** for better face recognition
6. **Keep the phone plugged in** to avoid battery issues

## Security Considerations

1. **Use on trusted networks** only
2. **Don't expose IP camera** to the internet
3. **Use strong WiFi passwords**
4. **Consider using VPN** for additional security
5. **Close IP camera app** when not in use

## Alternative Solutions

If IP camera doesn't work, consider:
1. **USB webcam** connected to your computer
2. **Built-in laptop camera**
3. **External USB camera**
4. **Network IP camera** (dedicated hardware)

## Support

If you're still having issues:
1. Check the app's documentation
2. Try a different IP camera app
3. Test with the provided test script
4. Check network logs for errors
5. Consider using a local camera instead 