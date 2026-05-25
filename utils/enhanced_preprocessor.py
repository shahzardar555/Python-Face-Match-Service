import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from datetime import datetime
from typing import Dict, Tuple


class EnhancedPreprocessor:
    """Advanced image enhancement for Tier 2 fallback processing"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def apply_enhanced_preprocessing(self, image_path: str) -> str:
        """Apply strong image enhancement techniques for better face detection"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply multiple enhancement techniques
            enhanced_image = self._enhance_for_face_detection(image)
            
            # Save enhanced image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_{timestamp}.jpg"
            enhanced_path = os.path.join(self.temp_dir, filename)
            
            # Convert back to BGR for OpenCV saving
            enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(enhanced_path, enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return enhanced_path
            
        except Exception as e:
            raise ValueError(f"Error in enhanced preprocessing: {str(e)}")
    
    def _enhance_for_face_detection(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive enhancement for better face detection"""
        enhanced = image.copy()
        
        # 1. Super-resolution style enhancement
        enhanced = self._apply_super_resolution_enhancement(enhanced)
        
        # 2. Advanced denoising
        enhanced = self._apply_advanced_denoising(enhanced)
        
        # 3. Adaptive brightness and contrast
        enhanced = self._apply_adaptive_brightness_contrast(enhanced)
        
        # 4. Local contrast enhancement
        enhanced = self._apply_local_contrast_enhancement(enhanced)
        
        # 5. Edge-preserving smoothing
        enhanced = self._apply_edge_preserving_smoothing(enhanced)
        
        # 6. Final sharpening
        enhanced = self._apply_final_sharpening(enhanced)
        
        return enhanced
    
    def _apply_super_resolution_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply super-resolution style enhancement"""
        # Upscale using advanced interpolation
        height, width = image.shape[:2]
        
        # Upscale by 1.5x using Lanczos interpolation
        upscaled = cv2.resize(image, (int(width * 1.5), int(height * 1.5)), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # Downscale back to original size with sharpening
        enhanced = cv2.resize(upscaled, (width, height), 
                            interpolation=cv2.INTER_AREA)
        
        return enhanced
    
    def _apply_advanced_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced denoising techniques"""
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def _apply_adaptive_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive brightness and contrast adjustment"""
        # Convert to LAB color space for better lightness manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Adaptive brightness correction
        # Calculate current brightness
        current_brightness = np.mean(l)
        target_brightness = 128  # Middle gray
        
        if current_brightness < target_brightness - 20:
            # Image is too dark, apply gamma correction
            gamma = 0.8
            l = self._apply_gamma_correction(l, gamma)
        elif current_brightness > target_brightness + 20:
            # Image is too bright, apply gamma correction
            gamma = 1.2
            l = self._apply_gamma_correction(l, gamma)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _apply_gamma_correction(self, channel: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to a single channel"""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(channel, table)
    
    def _apply_local_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply local contrast enhancement"""
        # Convert to grayscale for contrast analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation for contrast mapping
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Calculate local variance
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_var + 1e-6)
        
        # Create contrast enhancement map
        contrast_map = 1.0 + (local_std - np.mean(local_std)) / (np.std(local_std) + 1e-6)
        contrast_map = np.clip(contrast_map, 0.5, 2.0)
        
        # Apply contrast enhancement to each channel
        enhanced = image.copy().astype(np.float32)
        for i in range(3):
            enhanced[:, :, i] *= contrast_map
        
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _apply_edge_preserving_smoothing(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-preserving smoothing"""
        # Use guided filter for edge-preserving smoothing
        try:
            # Convert to grayscale for guidance
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply guided filter (simplified version)
            radius = 8
            eps = 0.01
            
            # This is a simplified guided filter implementation
            # In production, you might want to use a more sophisticated implementation
            smoothed = cv2.bilateralFilter(image, 9, 75, 75)
            
            return smoothed
        except:
            # Fallback to bilateral filter if guided filter fails
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _apply_final_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply final sharpening to enhance facial features"""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original image to control sharpening strength
        alpha = 0.3  # Sharpening strength
        enhanced = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
        
        return enhanced
    
    def enhance_for_specific_issues(self, image_path: str, quality_issues: list) -> str:
        """Apply targeted enhancement based on specific quality issues"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            enhanced = image.copy()
            
            # Apply specific enhancements based on issues
            for issue in quality_issues:
                issue_lower = issue.lower()
                
                if "dark" in issue_lower or "bright" in issue_lower:
                    enhanced = self._fix_brightness_issues(enhanced)
                
                elif "blurry" in issue_lower or "sharpness" in issue_lower:
                    enhanced = self._fix_blur_issues(enhanced)
                
                elif "small" in issue_lower or "resolution" in issue_lower:
                    enhanced = self._fix_resolution_issues(enhanced)
            
            # Save enhanced image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"targeted_enhanced_{timestamp}.jpg"
            enhanced_path = os.path.join(self.temp_dir, filename)
            
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(enhanced_path, enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return enhanced_path
            
        except Exception as e:
            raise ValueError(f"Error in targeted enhancement: {str(e)}")
    
    def _fix_brightness_issues(self, image: np.ndarray) -> np.ndarray:
        """Fix brightness related issues"""
        # Apply stronger brightness correction
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Additional brightness normalization
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _fix_blur_issues(self, image: np.ndarray) -> np.ndarray:
        """Fix blur related issues"""
        # Apply stronger sharpening
        kernel = np.array([[-2, -2, -2],
                          [-2, 13, -2],
                          [-2, -2, -2]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Apply with higher strength
        alpha = 0.5
        enhanced = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
        
        return enhanced
    
    def _fix_resolution_issues(self, image: np.ndarray) -> np.ndarray:
        """Fix resolution related issues"""
        height, width = image.shape[:2]
        
        # Apply more aggressive upscaling
        scale_factor = 2.0
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use multiple interpolation methods
        upscaled1 = cv2.resize(image, (new_width, new_height), 
                              interpolation=cv2.INTER_CUBIC)
        upscaled2 = cv2.resize(image, (new_width, new_height), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        # Blend results
        blended = cv2.addWeighted(upscaled1, 0.5, upscaled2, 0.5, 0)
        
        # Apply sharpening to upscaled image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(blended, -1, kernel)
        alpha = 0.3
        final = cv2.addWeighted(blended, 1 - alpha, sharpened, alpha, 0)
        
        # Downscale back to original size
        final = cv2.resize(final, (width, height), interpolation=cv2.INTER_AREA)
        
        return final
