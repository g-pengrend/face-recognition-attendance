import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
from datetime import datetime
import logging
from typing import List

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
        """Load student photos and create embeddings from folder structure"""
        if not os.path.exists(self.students_folder):
            os.makedirs(self.students_folder)
            self.logger.info(f"Created students folder: {self.students_folder}")
            return
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Scan for student folders
        for item in os.listdir(self.students_folder):
            item_path = os.path.join(self.students_folder, item)
            
            if os.path.isdir(item_path):
                # This is a student folder
                student_name = item
                student_images = []
                
                # Find all images in the student folder
                for filename in os.listdir(item_path):
                    if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                        image_path = os.path.join(item_path, filename)
                        student_images.append(image_path)
                
                if not student_images:
                    self.logger.warning(f"No images found in student folder: {student_name}")
                    continue
                
                # Load the student with multiple images
                success = self._load_student_with_multiple_images(student_name, student_images)
                if success:
                    self.logger.info(f"Loaded student: {student_name} with {len(student_images)} images")
                else:
                    self.logger.error(f"Failed to load student: {student_name}")
            
            elif os.path.isfile(item_path):
                # This is a single image file (backward compatibility)
                if any(item.lower().endswith(fmt) for fmt in supported_formats):
                    student_name = os.path.splitext(item)[0]
                    try:
                        self._load_single_student_image(student_name, item_path)
                        self.logger.info(f"Loaded student: {student_name} (single image)")
                    except Exception as e:
                        self.logger.error(f"Error processing {item}: {e}")
        
        self.logger.info(f"Loaded {len(self.students_db)} students")

    def _load_student_with_multiple_images(self, student_name: str, image_paths: List[str]) -> bool:
        """
        Load a student with multiple reference images
        
        Args:
            student_name (str): Name of the student
            image_paths (List[str]): List of image file paths
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            embeddings = []
            valid_images = []
            
            for image_path in image_paths:
                try:
                    # Load and process image
                    img = cv2.imread(image_path)
                    if img is None:
                        self.logger.warning(f"Could not load image: {image_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the image
                    faces = self.app.get(img_rgb)
                    
                    if len(faces) == 0:
                        self.logger.warning(f"No faces detected in {image_path}")
                        continue
                    
                    if len(faces) > 1:
                        self.logger.warning(f"Multiple faces detected in {image_path}, using the first one")
                    
                    # Store the face embedding
                    face = faces[0]
                    embeddings.append(face.embedding)
                    valid_images.append(image_path)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    continue
            
            if not embeddings:
                self.logger.error(f"No valid images found for student: {student_name}")
                return False
            
            # Store multiple embeddings for the student
            self.students_db[student_name] = {
                'embeddings': embeddings,
                'image_paths': valid_images,
                'primary_embedding': embeddings[0]  # Use first as primary
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading student {student_name}: {e}")
            return False

    def _load_single_student_image(self, student_name: str, image_path: str):
        """
        Load a single student image (backward compatibility)
        
        Args:
            student_name (str): Name of the student
            image_path (str): Path to the image file
        """
        # Load and process student image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        faces = self.app.get(img_rgb)
        
        if len(faces) == 0:
            raise ValueError(f"No faces detected in {image_path}")
        
        if len(faces) > 1:
            self.logger.warning(f"Multiple faces detected in {image_path}, using the first one")
        
        # Store the face embedding
        face = faces[0]
        self.students_db[student_name] = {
            'embedding': face.embedding,
            'image_path': image_path,
            'bbox': face.bbox,
            'landmarks': face.kps
        }
    
    def recognize_face(self, face_embedding):
        """
        Recognize a face by comparing embeddings (supports multiple embeddings per student)
        
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
            # Handle multiple embeddings per student
            if 'embeddings' in student_data:
                # Check against all embeddings for this student
                for embedding in student_data['embeddings']:
                    similarity = np.dot(face_embedding, embedding) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(embedding)
                    )
                    
                    if similarity > best_confidence and similarity >= self.base_threshold:
                        best_confidence = similarity
                        best_match = student_name
            else:
                # Single embedding (backward compatibility)
                similarity = np.dot(face_embedding, student_data['embedding']) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(student_data['embedding'])
                )
                
                if similarity > best_confidence and similarity >= self.base_threshold:
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

    def add_student_with_embedding(self, name: str, embedding: np.ndarray):
        """
        Add a student directly with an embedding (for unknown faces)
        
        Args:
            name (str): Name of the student
            embedding (np.ndarray): Face embedding
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create student folder if it doesn't exist
            student_folder = os.path.join(self.students_folder, name)
            os.makedirs(student_folder, exist_ok=True)
            
            # Store the embedding
            self.students_db[name] = {
                'embeddings': [embedding],
                'image_paths': [],  # No image file for now
                'primary_embedding': embedding
            }
            
            self.logger.info(f"Added student {name} with embedding (no image file)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding student {name} with embedding: {e}")
            return False 

    def add_image_to_student(self, student_name: str, image_path: str):
        """
        Add an additional image to an existing student
        
        Args:
            student_name (str): Name of the existing student
            image_path (str): Path to the new image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if student_name not in self.students_db:
                return False
            
            # Load and process the new image
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)
            
            if len(faces) == 0:
                return False
            
            # Use the best quality face
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Add to existing student's embeddings
            if 'embeddings' in self.students_db[student_name]:
                self.students_db[student_name]['embeddings'].append(best_face.embedding)
                self.students_db[student_name]['image_paths'].append(image_path)
            else:
                # Convert single embedding to multiple embeddings format
                old_data = self.students_db[student_name]
                self.students_db[student_name] = {
                    'embeddings': [old_data['embedding'], best_face.embedding],
                    'image_paths': [old_data.get('image_path', ''), image_path],
                    'primary_embedding': old_data['embedding']
                }
            
            self.logger.info(f"Added image to existing student: {student_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding image to student {student_name}: {e}")
            return False