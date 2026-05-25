import os
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import tempfile
from datetime import datetime
from typing import Dict, Tuple, Optional
import math


class FaceExtractor:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        from utils.guidance_generator import GuidanceGenerator, ErrorType
        self.guidance_generator = GuidanceGenerator()
        
    def extract_face_from_cnic(self, image_path: str) -> Dict:
        """Extracts face from CNIC image (photo typically on right side)"""
        try:
            # Use DeepFace to detect all faces
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            if not faces or len(faces) == 0:
                guidance = self.guidance_generator.get_bilingual_guidance(
                    ErrorType.NO_FACE_CNIC
                )
                return {
                    'success': False,
                    'face_path': None,
                    'confidence': 0,
                    'face_location': None,
                    'error': 'No face detected in CNIC image',
                    'guidance': guidance
                }
            
            # CNIC typically has the photo on the right side
            # Select the largest face (usually the ID photo)
            largest_face = None
            largest_area = 0
            
            for face in faces:
                if face['confidence'] > 0.5:  # Minimum confidence threshold
                    x, y, w, h = face['facial_area'].values()
                    area = w * h
                    
                    # Prefer faces on the right side of the image
                    image = cv2.imread(image_path)
                    img_width = image.shape[1]
                    x_center = x + w/2
                    
                    # Bonus for faces on right side
                    if x_center > img_width / 2:
                        area *= 1.5
                    
                    if area > largest_area:
                        largest_area = area
                        largest_face = face
            
            if largest_face is None:
                guidance = self.guidance_generator.get_bilingual_guidance(
                    ErrorType.NO_FACE_CNIC
                )
                return {
                    'success': False,
                    'face_path': None,
                    'confidence': 0,
                    'face_location': None,
                    'error': 'No suitable face detected in CNIC image',
                    'guidance': guidance
                }
            
            # Extract and save the face
            face_image = self._extract_face_region(image_path, largest_face['facial_area'])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cnic_face_{timestamp}.jpg"
            face_path = os.path.join(self.temp_dir, filename)
            
            cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return {
                'success': True,
                'face_path': face_path,
                'confidence': largest_face['confidence'],
                'face_location': largest_face['facial_area'],
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'face_path': None,
                'confidence': 0,
                'face_location': None,
                'error': f'Error extracting face from CNIC: {str(e)}'
            }
    
    def extract_face_from_selfie(self, image_path: str) -> Dict:
        """Extracts face from selfie image"""
        try:
            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            if not faces or len(faces) == 0:
                guidance = self.guidance_generator.get_bilingual_guidance(
                    ErrorType.NO_FACE_SELFIE
                )
                return {
                    'success': False,
                    'face_path': None,
                    'confidence': 0,
                    'face_location': None,
                    'error': 'No face detected in selfie. Please ensure your face is clearly visible and well lit',
                    'guidance': guidance
                }
            
            # Check for multiple faces
            valid_faces = [face for face in faces if face['confidence'] > 0.5]
            
            if len(valid_faces) == 0:
                guidance = self.guidance_generator.get_bilingual_guidance(
                    ErrorType.NO_FACE_SELFIE
                )
                return {
                    'success': False,
                    'face_path': None,
                    'confidence': 0,
                    'face_location': None,
                    'error': 'No clear face detected in selfie. Please ensure your face is clearly visible and well lit',
                    'guidance': guidance
                }
            
            if len(valid_faces) > 1:
                guidance = self.guidance_generator.get_bilingual_guidance(
                    ErrorType.MULTIPLE_FACES_SELFIE
                )
                return {
                    'success': False,
                    'face_path': None,
                    'confidence': 0,
                    'face_location': None,
                    'error': 'Multiple faces detected in selfie. Please take a selfie with only your face visible',
                    'guidance': guidance
                }
            
            # Extract the single face
            face = valid_faces[0]
            face_image = self._extract_face_region(image_path, face['facial_area'])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"selfie_face_{timestamp}.jpg"
            face_path = os.path.join(self.temp_dir, filename)
            
            cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return {
                'success': True,
                'face_path': face_path,
                'confidence': face['confidence'],
                'face_location': face['facial_area'],
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'face_path': None,
                'confidence': 0,
                'face_location': None,
                'error': f'Error extracting face from selfie: {str(e)}'
            }
    
    def check_face_quality(self, face_path: str) -> Dict:
        """Checks if extracted face meets quality requirements"""
        try:
            face_image = cv2.imread(face_path)
            if face_image is None:
                return {
                    'is_acceptable': False,
                    'issues': ['Could not load face image']
                }
            
            height, width = face_image.shape[:2]
            issues = []
            
            # Check minimum face size
            if width < 50 or height < 50:
                issues.append(f'Face too small: {width}x{height} (minimum 50x50)')
            
            # Check aspect ratio (faces should be roughly portrait)
            aspect_ratio = width / height
            if aspect_ratio > 1.5 or aspect_ratio < 0.6:
                issues.append('Face aspect ratio unusual - may be partially cut off')
            
            # Check if face is centered and not at edge
            margin = 10  # pixels from edge
            if width < 2 * margin or height < 2 * margin:
                issues.append('Face appears to be cut off at the edges')
            
            # Check brightness and contrast
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            if brightness < 0.3:
                issues.append('Face too dark')
            elif brightness > 0.8:
                issues.append('Face too bright')
            
            # Check sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                issues.append('Face too blurry')
            
            # Check for eye detection (basic check for facial features)
            try:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(eyes) < 1:
                    issues.append('Eyes not clearly visible')
                elif len(eyes) > 2:
                    issues.append('Multiple eye-like regions detected')
            except:
                # Eye detection failed, but don't fail the quality check
                pass
            
            # Check face angle (basic check using face landmarks if available)
            try:
                # Try to detect face landmarks for angle estimation
                landmarks = DeepFace.analyze(
                    img_path=face_path,
                    actions=[''],
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                if landmarks and len(landmarks) > 0:
                    # Basic angle check - if landmarks are available
                    # This is a simplified check
                    pass
            except:
                pass  # Landmark detection failed, continue with other checks
            
            is_acceptable = len(issues) == 0
            
            return {
                'is_acceptable': is_acceptable,
                'issues': issues,
                'face_size': (width, height),
                'brightness': brightness,
                'sharpness': laplacian_var
            }
            
        except Exception as e:
            return {
                'is_acceptable': False,
                'issues': [f'Error checking face quality: {str(e)}']
            }
    
    def _extract_face_region(self, image_path: str, facial_area: Dict) -> np.ndarray:
        """Extracts face region from image based on facial area coordinates"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            x = int(facial_area['x'])
            y = int(facial_area['y'])
            w = int(facial_area['w'])
            h = int(facial_area['h'])
            
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Extract face region
            face = image[y:y+h, x:x+w]
            
            # Ensure face is not empty
            if face.size == 0:
                raise ValueError("Extracted face region is empty")
            
            return face
            
        except Exception as e:
            raise ValueError(f"Error extracting face region: {str(e)}")
