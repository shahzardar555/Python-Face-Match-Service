import os
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import hashlib
from datetime import datetime
from typing import Dict, Tuple, Optional
import re


class ImageHandler:
    def __init__(self):
        self.max_size_mb = int(os.getenv('MAX_IMAGE_SIZE_MB', 10))
        self.allowed_formats = os.getenv('ALLOWED_FORMATS', 'jpg,jpeg,png,webp').split(',')
        self.temp_dir = tempfile.gettempdir()
        
    def download_image_from_url(self, url: str) -> str:
        """Downloads image from Cloudinary URL and saves temporarily"""
        try:
            # Generate unique filename
            file_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"download_{file_hash}_{timestamp}.jpg"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not any(img_type in content_type.lower() for img_type in ['image/jpeg', 'image/png', 'image/webp']):
                raise ValueError(f"Invalid content type: {content_type}")
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_size_mb * 1024 * 1024:
                raise ValueError(f"Image too large: {int(content_length) / (1024*1024):.2f}MB")
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Validate saved image
            self._validate_image_file(file_path)
            
            return file_path
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def decode_base64_image(self, base64_str: str) -> str:
        """Decodes base64 image string and saves temporarily"""
        try:
            # Remove data URL prefix if present
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_str)
            
            # Check size
            if len(image_data) > self.max_size_mb * 1024 * 1024:
                raise ValueError(f"Image too large: {len(image_data) / (1024*1024):.2f}MB")
            
            # Generate unique filename
            file_hash = hashlib.md5(image_data).hexdigest()[:10]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"base64_{file_hash}_{timestamp}.jpg"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Validate saved image
            self._validate_image_file(file_path)
            
            return file_path
            
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> str:
        """Preprocesses image for optimal DeepFace performance"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize maintaining aspect ratio (max 800x800)
            height, width = image.shape[:2]
            max_size = 800
            
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                else:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Enhance image quality
            # Normalize brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Improve contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Reduce noise
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Save processed image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_{timestamp}.jpg"
            processed_path = os.path.join(self.temp_dir, filename)
            
            # Convert back to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(processed_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return processed_path
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def cleanup_temp_files(self, *file_paths: str):
        """Deletes temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass  # Silently handle cleanup errors
    
    def validate_image_quality(self, image_path: str) -> Dict:
        """Validates image quality and returns assessment"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'is_valid': False,
                    'brightness_score': 0,
                    'blur_score': 0,
                    'resolution': (0, 0),
                    'issues': ['Could not load image']
                }
            
            height, width = image.shape[:2]
            resolution = (width, height)
            issues = []
            
            # Check minimum resolution
            if width < 100 or height < 100:
                issues.append(f"Resolution too low: {width}x{height} (minimum 100x100)")
            
            # Calculate brightness score
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness_score = np.mean(gray) / 255.0
            
            if brightness_score < 0.2:
                issues.append("Image too dark")
            elif brightness_score > 0.8:
                issues.append("Image too bright")
            
            # Calculate blur score (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score_normalized = min(blur_score / 1000, 1.0)  # Normalize to 0-1
            
            if blur_score < 100:
                issues.append("Image too blurry")
            
            is_valid = len(issues) == 0
            
            return {
                'is_valid': is_valid,
                'brightness_score': brightness_score,
                'blur_score': blur_score_normalized,
                'resolution': resolution,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'brightness_score': 0,
                'blur_score': 0,
                'resolution': (0, 0),
                'issues': [f"Error analyzing image: {str(e)}"]
            }
    
    def _validate_image_file(self, file_path: str):
        """Validates that file is a valid image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_ext not in self.allowed_formats:
                os.remove(file_path)
                raise ValueError(f"Unsupported format: {file_ext}")
                
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def apply_targeted_enhancement(self, image_path: str, quality_issues: list) -> str:
        """Apply targeted enhancement based on specific quality issues"""
        try:
            from utils.enhanced_preprocessor import EnhancedPreprocessor
            preprocessor = EnhancedPreprocessor()
            return preprocessor.enhance_for_specific_issues(image_path, quality_issues)
        except Exception as e:
            raise ValueError(f"Error in targeted enhancement: {str(e)}")
    
    def get_enhanced_preprocessor(self):
        """Get enhanced preprocessor instance"""
        from utils.enhanced_preprocessor import EnhancedPreprocessor
        return EnhancedPreprocessor()
