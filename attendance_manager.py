import json
import csv
import os
import numpy as np
from datetime import datetime, date
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
    
    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new attendance session
        
        Args:
            session_name (str, optional): Custom session name
            
        Returns:
            str: Session ID
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'id': session_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'attendance': {},
            'total_students': 0,
            'present_students': 0
        }
        
        self.logger.info(f"Started attendance session: {session_name}")
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
                'detection_count': 1
            }
            self.current_session['present_students'] += 1
        
        self.logger.info(f"Marked attendance for {student_name} (confidence: {confidence:.2f})")
        return True
    
    def get_current_attendance(self) -> Dict:
        """
        Get current attendance status
        
        Returns:
            dict: Current attendance data
        """
        if self.current_session is None:
            return {}
        
        return {
            'session_id': self.current_session['id'],
            'start_time': self.current_session['start_time'],
            'present_students': self.current_session['present_students'],
            'attendance': self.current_session['attendance']
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
            
            # Calculate additional statistics
            attendance_count = len(session_data['attendance'])
            total_detections = sum(entry['detection_count'] for entry in session_data['attendance'].values())
            
            summary = {
                'session_id': session_data['id'],
                'start_time': session_data['start_time'],
                'end_time': session_data['end_time'],
                'duration_minutes': self._calculate_duration(session_data['start_time'], session_data['end_time']),
                'total_students': session_data['total_students'],
                'present_students': attendance_count,
                'attendance_rate': attendance_count / session_data['total_students'] if session_data['total_students'] > 0 else 0,
                'total_detections': total_detections,
                'attendance': session_data['attendance']
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {e}")
            return None
    
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
        Export attendance data to CSV
        
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
                    'Duration (minutes)'
                ])
                
                # Write attendance data
                for student_name, data in summary['attendance'].items():
                    duration = self._calculate_duration(data['first_seen'], data['last_seen'])
                    writer.writerow([
                        student_name,
                        data['first_seen'],
                        data['last_seen'],
                        f"{data['confidence']:.3f}",
                        data['detection_count'],
                        duration
                    ])
                
                # Write summary
                writer.writerow([])
                writer.writerow(['Summary'])
                writer.writerow(['Session ID', summary['session_id']])
                writer.writerow(['Start Time', summary['start_time']])
                writer.writerow(['End Time', summary['end_time']])
                writer.writerow(['Duration (minutes)', summary['duration_minutes']])
                writer.writerow(['Total Students', summary['total_students']])
                writer.writerow(['Present Students', summary['present_students']])
                writer.writerow(['Attendance Rate', f"{summary['attendance_rate']:.2%}"])
                writer.writerow(['Total Detections', summary['total_detections']])
            
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