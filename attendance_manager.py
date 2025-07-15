import json
import csv
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import logging

class AttendanceManager:
    def __init__(self, logs_folder="attendance_logs"):
        """
        Initialize the attendance manager
        
        Args:
            logs_folder (str): Folder to store attendance logs
        """
        self.logs_folder = logs_folder
        self.current_session = None
        self.attendance_data = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create logs folder if it doesn't exist
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
            self.logger.info(f"Created attendance logs folder: {logs_folder}")
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _calculate_lateness(self, session_start_time: str, arrival_time: str) -> Dict:
        """
        Calculate lateness information based on blocks of 30 minutes.
        """
        try:
            start_time = datetime.fromisoformat(session_start_time)
            arrival = datetime.fromisoformat(arrival_time)
            time_diff = arrival - start_time
            minutes_late = time_diff.total_seconds() / 60.0

            if minutes_late < 0:
                status = "On Time"
                color = "success"
                category = "On Time"
            elif minutes_late < 30:
                status = "On Time"
                color = "success"
                category = "On Time"
            elif minutes_late < 60:
                status = "30 minutes late"
                color = "warning"
                category = "30 min late"
            elif minutes_late < 90:
                status = "1 hour late"
                color = "warning"
                category = "1 hour late"
            elif minutes_late < 120:
                status = "90 minutes late"
                color = "danger"
                category = "90 min late"
            elif minutes_late < 150:
                status = "2 hours late"
                color = "danger"
                category = "2 hours late"
            elif minutes_late < 180:
                status = "2.5 hours late"
                color = "danger"
                category = "2.5 hours late"
            elif minutes_late < 210:
                status = "3 hours late"
                color = "danger"
                category = "3 hours late"
            else:
                status = "Absent"
                color = "secondary"
                category = "Absent"

            return {
                'minutes_late': minutes_late,
                'status': status,
                'color': color,
                'category': category
            }
        except Exception as e:
            self.logger.error(f"Error calculating lateness: {e}")
            return {
                'minutes_late': 0,
                'status': "Unknown",
                'color': "secondary",
                'category': "Unknown"
            }
    
    def start_session(self, session_name: Optional[str] = None, session_start_time: Optional[str] = None) -> str:
        """
        Start a new attendance session
        
        Args:
            session_name (str, optional): Custom session name
            session_start_time (str, optional): Session start time (ISO format)
            
        Returns:
            str: Session ID
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # If no start time provided, use current time
        if session_start_time is None:
            session_start_time = datetime.now().isoformat()
        
        self.current_session = {
            'id': session_name,
            'start_time': datetime.now().isoformat(),  # Detection start time
            'session_start_time': session_start_time,  # Session start time for punctuality
            'end_time': None,
            'attendance': {},
            'total_students': 0,
            'present_students': 0
        }
        
        self.logger.info(f"Started attendance session: {session_name} with start time: {session_start_time}")
        return session_name
    
    def end_session(self) -> bool:
        """
        End the current attendance session
        
        Returns:
            bool: True if successful, False if no active session
        """
        if self.current_session is None:
            return False
        
        self.current_session['end_time'] = datetime.now().isoformat()
        
        # Save session data
        self._save_session()
        
        session_id = self.current_session['id']
        self.logger.info(f"Ended attendance session: {session_id}")
        
        # Clear current session
        self.current_session = None
        return True
    
    def mark_attendance(self, student_name: str, confidence: float = 1.0) -> bool:
        """
        Mark a student as present
        
        Args:
            student_name (str): Name of the student
            confidence (float): Recognition confidence (0.0 to 1.0)
            
        Returns:
            bool: True if marked successfully, False otherwise
        """
        if self.current_session is None:
            self.logger.warning("No active session to mark attendance")
            return False
        
        # Convert NumPy types to native Python types
        confidence = self._convert_numpy_types(confidence)
        
        timestamp = datetime.now().isoformat()
        
        # Calculate lateness if session start time is available
        lateness_info = {}
        if 'session_start_time' in self.current_session:
            lateness_info = self._calculate_lateness(
                self.current_session['session_start_time'], 
                timestamp
            )
        
        # Check if student already marked present
        if student_name in self.current_session['attendance']:
            # Update existing entry with new timestamp and confidence
            self.current_session['attendance'][student_name].update({
                'last_seen': timestamp,
                'confidence': max(self.current_session['attendance'][student_name]['confidence'], confidence),
                'detection_count': self.current_session['attendance'][student_name]['detection_count'] + 1
            })
        else:
            # Add new attendance entry
            self.current_session['attendance'][student_name] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'confidence': confidence,
                'detection_count': 1,
                'lateness': lateness_info
            }
            self.current_session['present_students'] += 1
        
        self.logger.info(f"Marked attendance for {student_name} (confidence: {confidence:.2f}, lateness: {lateness_info.get('status', 'Unknown')})")
        return True
    
    def get_current_attendance(self) -> Dict:
        """
        Get current attendance status
        
        Returns:
            dict: Current attendance data
        """
        if self.current_session is None:
            return {}
        
        # Calculate lateness for all students
        attendance_with_lateness = {}
        for student_name, data in self.current_session['attendance'].items():
            student_data = data.copy()
            if 'session_start_time' in self.current_session and 'lateness' not in data:
                # Calculate lateness for existing entries
                lateness_info = self._calculate_lateness(
                    self.current_session['session_start_time'], 
                    data['first_seen']
                )
                student_data['lateness'] = lateness_info
            attendance_with_lateness[student_name] = student_data
        
        return {
            'session_id': self.current_session['id'],
            'start_time': self.current_session['start_time'],
            'session_start_time': self.current_session.get('session_start_time'),
            'present_students': self.current_session['present_students'],
            'attendance': attendance_with_lateness
        }
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get summary of a specific session
        
        Args:
            session_id (str): Session ID to retrieve
            
        Returns:
            dict: Session summary or None if not found
        """
        session_file = os.path.join(self.logs_folder, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Get the session start time (class start time)
            session_start_time = session_data.get('session_start_time', session_data['start_time'])
            
            # Recalculate lateness for all students using the correct session start time
            for student_name, data in session_data['attendance'].items():
                if 'first_seen' in data:
                    data['lateness'] = self._calculate_lateness(session_start_time, data['first_seen'])
            
            # Calculate additional statistics
            attendance_count = len(session_data['attendance'])
            total_detections = sum(entry['detection_count'] for entry in session_data['attendance'].values())
            
            # Calculate lateness statistics
            lateness_stats = self._calculate_lateness_statistics(session_data)
            
            summary = {
                'session_id': session_data['id'],
                'start_time': session_data['start_time'],
                'session_start_time': session_start_time,
                'end_time': session_data['end_time'],
                'duration_minutes': self._calculate_duration(session_data['start_time'], session_data['end_time']),
                'total_students': session_data['total_students'],
                'present_students': attendance_count,
                'attendance_rate': attendance_count / session_data['total_students'] if session_data['total_students'] > 0 else 0,
                'total_detections': total_detections,
                'attendance': session_data['attendance'],
                'lateness_stats': lateness_stats
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def _calculate_lateness_statistics(self, session_data: Dict) -> Dict:
        """Calculate lateness statistics for a session"""
        lateness_categories = {
            'On Time': 0,
            '30 min late': 0,
            '1 hour late': 0,
            '90 min late': 0,
            '2 hours late': 0,
            '2.5 hours late': 0,
            '3 hours late': 0,
            'Absent': 0
        }
        
        for student_data in session_data['attendance'].values():
            if 'lateness' in student_data:
                category = student_data['lateness'].get('category', 'Unknown')
                if category in lateness_categories:
                    lateness_categories[category] += 1
        
        return lateness_categories
    
    def list_sessions(self) -> List[Dict]:
        """
        List all available sessions
        
        Returns:
            list: List of session summaries
        """
        sessions = []
        
        for filename in os.listdir(self.logs_folder):
            if filename.endswith('.json'):
                session_id = filename[:-5]  # Remove .json extension
                summary = self.get_session_summary(session_id)
                if summary:
                    sessions.append(summary)
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x['start_time'], reverse=True)
        return sessions
    
    def export_to_csv(self, session_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export attendance data to CSV with lateness information
        
        Args:
            session_id (str): Session ID to export
            output_path (str, optional): Output file path
            
        Returns:
            str: Path to exported CSV file or None if failed
        """
        summary = self.get_session_summary(session_id)
        if not summary:
            return None
        
        if output_path is None:
            output_path = os.path.join(self.logs_folder, f"{session_id}_attendance.csv")
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'Student Name',
                    'First Seen',
                    'Last Seen',
                    'Confidence',
                    'Detection Count',
                    'Duration (minutes)',
                    'Lateness Status',
                    'Minutes Late',
                    'Lateness Category'
                ])
                
                # Write attendance data
                for student_name, data in summary['attendance'].items():
                    duration = self._calculate_duration(data['first_seen'], data['last_seen'])
                    
                    # Get lateness information
                    lateness_status = "Unknown"
                    minutes_late = 0
                    lateness_category = "Unknown"
                    
                    if 'lateness' in data:
                        lateness_status = data['lateness'].get('status', 'Unknown')
                        minutes_late = data['lateness'].get('minutes_late', 0)
                        lateness_category = data['lateness'].get('category', 'Unknown')
                    
                    writer.writerow([
                        student_name,
                        data['first_seen'],
                        data['last_seen'],
                        f"{data['confidence']:.3f}",
                        data['detection_count'],
                        duration,
                        lateness_status,
                        f"{minutes_late:.1f}",
                        lateness_category
                    ])
                
                # Write summary
                writer.writerow([])
                writer.writerow(['Summary'])
                writer.writerow(['Session ID', summary['session_id']])
                writer.writerow(['Detection Start Time', summary['start_time']])
                writer.writerow(['Class Start Time', summary['session_start_time']])  # <-- Added this line
                writer.writerow(['End Time', summary['end_time']])
                writer.writerow(['Duration (minutes)', summary['duration_minutes']])
                writer.writerow(['Total Students', summary['total_students']])
                writer.writerow(['Present Students', summary['present_students']])
                writer.writerow(['Attendance Rate', f"{summary['attendance_rate']:.2%}"])
                writer.writerow(['Total Detections', summary['total_detections']])
                
                # Write lateness statistics
                if 'lateness_stats' in summary:
                    writer.writerow([])
                    writer.writerow(['Lateness Statistics'])
                    for category, count in summary['lateness_stats'].items():
                        writer.writerow([category, count])
            
            self.logger.info(f"Exported attendance data to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return None
    
    def set_total_students(self, count: int) -> bool:
        """
        Set the total number of students for the current session
        
        Args:
            count (int): Total number of students
            
        Returns:
            bool: True if successful, False if no active session
        """
        if self.current_session is None:
            return False
        
        self.current_session['total_students'] = count
        return True
    
    def _save_session(self) -> bool:
        """Save current session to file"""
        if self.current_session is None:
            return False
        
        try:
            # Convert NumPy types before saving
            session_data = self._convert_numpy_types(self.current_session)
            
            session_file = os.path.join(self.logs_folder, f"{self.current_session['id']}.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
            return False
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """
        Calculate duration between two timestamps in minutes
        
        Args:
            start_time (str): Start timestamp (ISO format)
            end_time (str): End timestamp (ISO format)
            
        Returns:
            float: Duration in minutes
        """
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return duration.total_seconds() / 60.0
        except Exception:
            return 0.0
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict:
        """
        Get summary of all sessions for a specific date
        
        Args:
            target_date (date, optional): Target date (defaults to today)
            
        Returns:
            dict: Daily summary
        """
        if target_date is None:
            target_date = date.today()
        
        sessions = self.list_sessions()
        daily_sessions = []
        
        for session in sessions:
            session_date = datetime.fromisoformat(session['start_time']).date()
            if session_date == target_date:
                daily_sessions.append(session)
        
        if not daily_sessions:
            return {
                'date': target_date.isoformat(),
                'total_sessions': 0,
                'total_students': 0,
                'total_present': 0,
                'average_attendance_rate': 0.0
            }
        
        total_students = sum(s['total_students'] for s in daily_sessions)
        total_present = sum(s['present_students'] for s in daily_sessions)
        avg_attendance_rate = total_present / total_students if total_students > 0 else 0.0
        
        return {
            'date': target_date.isoformat(),
            'total_sessions': len(daily_sessions),
            'total_students': total_students,
            'total_present': total_present,
            'average_attendance_rate': avg_attendance_rate,
            'sessions': daily_sessions
        } 