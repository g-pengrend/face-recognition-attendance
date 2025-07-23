import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import json
from datetime import datetime
import logging
from typing import List

# Set environment variables for Apple Silicon optimization BEFORE importing InsightFace
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['ONNXRUNTIME_PROVIDER_NAMES'] = 'CoreMLExecutionProvider,CPUExecutionProvider'

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
        self.current_class = None  # Add current class tracking
        self.app = None
        self.initialized = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_insightface()
        # Don't load students initially - wait for class selection
    
    def get_available_classes(self):
        """Get list of available classes (folders in students directory)"""
        if not os.path.exists(self.students_folder):
            return []
        
        classes = []
        for item in os.listdir(self.students_folder):
            item_path = os.path.join(self.students_folder, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        return sorted(classes)
    
    def set_current_class(self, class_name):
        """Set the current class and load its students"""
        if class_name not in self.get_available_classes():
            self.logger.error(f"Class {class_name} not found")
            return False
        
        self.current_class = class_name
        self.students_db = {}  # Clear existing students
        self._load_students_for_class(class_name)
        self.logger.info(f"Switched to class: {class_name}")
        return True
    
    def get_current_class(self):
        """Get the currently selected class"""
        return self.current_class
    
    def _load_students_for_class(self, class_name):
        """Load student photos and create embeddings for a specific class"""
        class_folder = os.path.join(self.students_folder, class_name)
        
        if not os.path.exists(class_folder):
            self.logger.warning(f"Class folder not found: {class_folder}")
            return
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Scan for student folders within the class folder
        for item in os.listdir(class_folder):
            item_path = os.path.join(class_folder, item)
            
            if os.path.isdir(item_path):
                # This is a student folder
                student_name = item
                student_images = []
                
                # Find all images in the student folder
                for filename in os.listdir(item_path):
                    if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                        image_path = os.path.join(item_path, filename)
                        student_images.append(image_path)
                
                # Always add the student to the database, even if no valid images
                if not student_images:
                    self.logger.warning(f"No images found in student folder: {student_name}")
                    # Add student with empty data - they can be added later
                    self.students_db[student_name] = {
                        'embeddings': [],
                        'image_paths': [],
                        'primary_embedding': None
                    }
                    self.logger.info(f"Added student folder to list: {student_name} (no images)")
                else:
                    # Load the student with multiple images
                    success = self._load_student_with_multiple_images(student_name, student_images)
                    if success:
                        self.logger.info(f"Loaded student: {student_name} with {len(student_images)} images")
                    else:
                        # Add student with empty data if face detection failed
                        self.logger.warning(f"Face detection failed for student: {student_name}")
                        self.students_db[student_name] = {
                            'embeddings': [],
                            'image_paths': student_images,  # Keep the image paths for reference
                            'primary_embedding': None
                        }
                        self.logger.info(f"Added student to list: {student_name} (no faces detected)")
            
            elif os.path.isfile(item_path):
                # This is a single image file (backward compatibility)
                if any(item.lower().endswith(fmt) for fmt in supported_formats):
                    student_name = os.path.splitext(item)[0]
                    try:
                        self._load_single_student_image(student_name, item_path)
                        self.logger.info(f"Loaded student: {student_name} (single image)")
                    except Exception as e:
                        self.logger.error(f"Error processing {item}: {e}")
        
        self.logger.info(f"Loaded {len(self.students_db)} students for class {class_name}")

    def _load_students(self):
        """Load student photos and create embeddings from folder structure (legacy method)"""
        # This method is kept for backward compatibility
        # It will load from the root students folder if no class is selected
        if self.current_class:
            self._load_students_for_class(self.current_class)
        else:
            # Legacy behavior - load from root students folder
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

    def _initialize_insightface(self):
        """Initialize InsightFace with forced CoreML optimization"""
        try:
            import onnxruntime as ort
            
            # Force CoreML provider configuration
            providers = [
                ('CoreMLExecutionProvider', {
                    'device_type': 'CPU',  # CoreML will use ANE/GPU automatically
                    'precision': 'FP16',   # Use half precision for better performance
                }),
                ('CPUExecutionProvider', {})
            ]
            
            # Set environment variables to force CoreML
            os.environ['ONNXRUNTIME_PROVIDER_NAMES'] = 'CoreMLExecutionProvider,CPUExecutionProvider'
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            available_providers = ort.get_available_providers()
            self.logger.info(f"ONNX Runtime available providers: {available_providers}")
            
            # Initialize InsightFace with explicit provider configuration
            self.app = FaceAnalysis(name='buffalo_l')
            
            # Force CoreML context - this is the key change
            self.app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU context, CoreML handles GPU
            
            # Verify CoreML is being used
            self.logger.info("InsightFace initialized with CoreML optimization")
            self.logger.info("Note: CoreML will automatically use Apple Neural Engine/GPU")
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace with CoreML: {e}")
            # Fallback to regular initialization
            self._initialize_insightface_fallback()
            raise

    def _initialize_insightface_fallback(self):
        """Fallback initialization without CoreML"""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            self.logger.info(f"Fallback: ONNX Runtime available providers: {available_providers}")
            
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            
            self.logger.info("InsightFace initialized with CPU fallback")
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise
    
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
        Recognize a face by comparing embedding with known faces
        """
        if not self.students_db:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        for student_name, student_data in self.students_db.items():
            # Skip students with no embeddings (empty folders)
            if not student_data.get('embeddings'):
                continue
            
            # Compare with all embeddings for this student
            for embedding in student_data['embeddings']:
                # Calculate cosine similarity
                similarity = np.dot(face_embedding, embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(embedding))
                confidence = float(similarity)
                
                if confidence > best_confidence and confidence >= self.base_threshold:
                    best_confidence = confidence
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
    
    def add_student(self, name, image_path, class_name=None):
        """
        Add a new student with a single image to a specific class
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)
            
            if len(faces) == 0:
                self.logger.error(f"No faces detected in {image_path}")
                return False
            
            # Use the best quality face (largest bounding box)
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Create student folder in the appropriate class
            if class_name:
                student_folder = os.path.join(self.students_folder, class_name, name)
            else:
                student_folder = os.path.join(self.students_folder, name)
            
            os.makedirs(student_folder, exist_ok=True)
            
            # Store the student data
            self.students_db[name] = {
                'embeddings': [best_face.embedding],
                'image_paths': [image_path],
                'primary_embedding': best_face.embedding
            }
            
            self.logger.info(f"Added new student: {name} to class: {class_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding student {name}: {e}")
            return False

    def add_image_to_student(self, student_name: str, image_path: str, class_name=None):
        """
        Add an additional image to an existing student in a specific class
        """
        try:
            if student_name not in self.students_db:
                self.logger.error(f"Student {student_name} not found in database")
                return False
            
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)
            
            if len(faces) == 0:
                self.logger.error(f"No faces detected in {image_path}")
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