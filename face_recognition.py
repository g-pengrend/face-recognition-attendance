import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
from datetime import datetime
import logging

class FaceRecognitionSystem:
    def __init__(self, students_folder="students", threshold=0.6):
        """
        Initialize with distance-aware threshold
        """
        self.students_folder = students_folder
        self.base_threshold = threshold
        self.distance_thresholds = {
            'close': threshold - 0.1,    # Easier for close faces
            'medium': threshold,         # Standard threshold
            'far': threshold + 0.1       # Stricter for far faces
        }
        self.students_db = {}
        self.app = None
        self.initialized = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_insightface()
        self._load_students()
    
    def _initialize_insightface(self):
        """Initialize InsightFace with Apple Silicon optimization"""
        try:
            # Initialize InsightFace app
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            # For Apple Silicon, we'll use CPU provider with optimizations
            providers = ['CPUExecutionProvider']
            
            self.logger.info("InsightFace initialized successfully")
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise
    
    def _load_students(self):
        """Load student photos and create embeddings"""
        if not os.path.exists(self.students_folder):
            os.makedirs(self.students_folder)
            self.logger.info(f"Created students folder: {self.students_folder}")
            return
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(self.students_folder):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                student_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.students_folder, filename)
                
                try:
                    # Load and process student image
                    img = cv2.imread(image_path)
                    if img is None:
                        self.logger.warning(f"Could not load image: {image_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the image
                    faces = self.app.get(img_rgb)
                    
                    if len(faces) == 0:
                        self.logger.warning(f"No faces detected in {filename}")
                        continue
                    
                    if len(faces) > 1:
                        self.logger.warning(f"Multiple faces detected in {filename}, using the first one")
                    
                    # Store the face embedding
                    face = faces[0]
                    self.students_db[student_name] = {
                        'embedding': face.embedding,
                        'image_path': image_path,
                        'bbox': face.bbox,
                        'landmarks': face.kps
                    }
                    
                    self.logger.info(f"Loaded student: {student_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
        
        self.logger.info(f"Loaded {len(self.students_db)} students")
    
    def recognize_face(self, face_embedding):
        """
        Recognize a face by comparing embeddings
        
        Args:
            face_embedding (np.ndarray): Face embedding to recognize
            
        Returns:
            tuple: (student_name, confidence) or (None, 0.0) if not recognized
        """
        if not self.initialized or len(self.students_db) == 0:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        for student_name, student_data in self.students_db.items():
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, student_data['embedding']) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(student_data['embedding'])
            )
            
            if similarity > best_confidence and similarity >= self.threshold:
                best_confidence = similarity
                best_match = student_name
        
        return best_match, best_confidence
    
    def detect_faces(self, frame):
        """
        Optimized face detection and recognition
        """
        if not self.initialized:
            return []
        
        # Resize frame for better performance
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces with optimized parameters
        faces = self.app.get(frame_rgb)
        
        results = []
        for face in faces:
            # Recognize the face
            student_name, confidence = self.recognize_face(face.embedding)
            
            result = {
                'bbox': face.bbox,
                'landmarks': face.kps,
                'student_name': student_name,
                'confidence': confidence,
                'embedding': face.embedding
            }
            results.append(result)
        
        return results
    
    def add_student(self, name, image_path):
        """
        Enhanced student addition with multiple angles and distances
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)
            
            if len(faces) == 0:
                return False
            
            # Use the best quality face (largest bounding box)
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            self.students_db[name] = {
                'embedding': best_face.embedding,
                'image_path': image_path,
                'bbox': best_face.bbox,
                'landmarks': best_face.kps
            }
            
            self.logger.info(f"Added new student: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding student {name}: {e}")
            return False
    
    def get_students_list(self):
        """Get list of all registered students"""
        return list(self.students_db.keys())
    
    def remove_student(self, name):
        """Remove a student from the database"""
        if name in self.students_db:
            del self.students_db[name]
            self.logger.info(f"Removed student: {name}")
            return True
        return False
    
    def save_database(self, filepath="students_database.json"):
        """Save the student database to a JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            db_to_save = {}
            for name, data in self.students_db.items():
                db_to_save[name] = {
                    'embedding': data['embedding'].tolist(),
                    'image_path': data['image_path'],
                    'bbox': data['bbox'].tolist(),
                    'landmarks': data['landmarks'].tolist()
                }
            
            with open(filepath, 'w') as f:
                json.dump(db_to_save, f)
            
            self.logger.info(f"Database saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving database: {e}")
            return False
    
    def load_database(self, filepath="students_database.json"):
        """Load the student database from a JSON file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                db_loaded = json.load(f)
            
            # Convert lists back to numpy arrays
            for name, data in db_loaded.items():
                self.students_db[name] = {
                    'embedding': np.array(data['embedding']),
                    'image_path': data['image_path'],
                    'bbox': np.array(data['bbox']),
                    'landmarks': np.array(data['landmarks'])
                }
            
            self.logger.info(f"Database loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading database: {e}")
            return False 