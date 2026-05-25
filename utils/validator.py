import re
import base64
import requests
from typing import Dict, Any
from urllib.parse import urlparse


class RequestValidator:
    def __init__(self):
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def validate_match_request(self, data: Dict) -> Dict:
        """Validates incoming face match request"""
        errors = []
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            errors.append("Request body must be a JSON object")
            return {
                'is_valid': False,
                'errors': errors
            }
        
        # Check provider_id
        provider_id = data.get('provider_id')
        if not provider_id:
            errors.append("provider_id is required")
        elif not isinstance(provider_id, str) or len(provider_id.strip()) == 0:
            errors.append("provider_id must be a non-empty string")
        
        # Validate optional threshold override
        threshold = data.get('threshold_override')
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                errors.append("threshold_override must be a number")
            elif not (0.1 <= threshold <= 1.0):
                errors.append("threshold_override must be between 0.1 and 1.0")
        
        # Validate optional language preference
        language = data.get('language')
        if language is not None:
            if not isinstance(language, str) or language not in ['en', 'ur']:
                errors.append("language must be 'en' or 'ur'")
        
        # Validate optional attempt number (for tracking)
        attempt_number = data.get('attempt_number')
        if attempt_number is not None:
            if not isinstance(attempt_number, int) or not (1 <= attempt_number <= 3):
                errors.append("attempt_number must be an integer between 1 and 3")
        
        # Check image sources - either URLs or base64
        has_urls = bool(data.get('cnic_url') and data.get('selfie_url'))
        has_base64 = bool(data.get('cnic_base64') and data.get('selfie_base64'))
        
        if not (has_urls or has_base64):
            errors.append("Either (cnic_url and selfie_url) or (cnic_base64 and selfie_base64) must be provided")
        elif has_urls and has_base64:
            errors.append("Provide either URLs or base64 images, not both")
        
        # Validate URLs if provided
        if has_urls:
            cnic_url = data.get('cnic_url')
            selfie_url = data.get('selfie_url')
            
            if not self.validate_url(cnic_url):
                errors.append("cnic_url is not a valid URL")
            
            if not self.validate_url(selfie_url):
                errors.append("selfie_url is not a valid URL")
        
        # Validate base64 if provided
        if has_base64:
            cnic_base64 = data.get('cnic_base64')
            selfie_base64 = data.get('selfie_base64')
            
            if not self.validate_base64(cnic_base64):
                errors.append("cnic_base64 is not valid base64 image data")
            
            if not self.validate_base64(selfie_base64):
                errors.append("selfie_base64 is not valid base64 image data")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def validate_url(self, url: str) -> bool:
        """Validates URL format and accessibility"""
        if not url or not isinstance(url, str):
            return False
        
        # Check URL format
        if not self.url_pattern.match(url):
            return False
        
        # Check if URL is accessible (optional - can be slow)
        # For performance, we'll skip the accessibility check in production
        # but keep it for development/testing
        try:
            # Just check the URL structure, not actually make a request
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def validate_base64(self, base64_str: str) -> bool:
        """Validates if string is valid base64 image data"""
        if not base64_str or not isinstance(base64_str, str):
            return False
        
        try:
            # Check for data URL prefix
            if 'base64,' in base64_str:
                parts = base64_str.split('base64,')
                if len(parts) != 2:
                    return False
                
                # Check if it has a valid image MIME type
                data_url = parts[0]
                if not data_url.startswith('data:image/'):
                    return False
                
                # Extract the actual base64 part
                base64_part = parts[1]
            else:
                base64_part = base64_str
            
            # Try to decode base64
            decoded = base64.b64decode(base64_part)
            
            # Check if decoded data looks like an image
            # Basic check for common image file signatures
            if len(decoded) < 10:
                return False
            
            # Check for common image headers
            image_signatures = [
                b'\xFF\xD8\xFF',  # JPEG
                b'\x89PNG\r\n\x1a\n',  # PNG
                b'RIFF',  # WEBP (starts with RIFF)
                b'GIF87a',  # GIF
                b'GIF89a',  # GIF
            ]
            
            is_image = any(decoded.startswith(sig) for sig in image_signatures)
            
            return is_image
            
        except Exception:
            return False
    
    def validate_analyze_request(self, data: Dict) -> Dict:
        """Validates face analyze request"""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Request body must be a JSON object")
            return {
                'is_valid': False,
                'errors': errors
            }
        
        # Check image source
        image_url = data.get('image_url')
        image_base64 = data.get('image_base64')
        
        if not image_url and not image_base64:
            errors.append("Either image_url or image_base64 must be provided")
        elif image_url and image_base64:
            errors.append("Provide either image_url or image_base64, not both")
        
        # Validate URL if provided
        if image_url and not self.validate_url(image_url):
            errors.append("image_url is not a valid URL")
        
        # Validate base64 if provided
        if image_base64 and not self.validate_base64(image_base64):
            errors.append("image_base64 is not valid base64 image data")
        
        # Validate image_type
        image_type = data.get('image_type')
        if not image_type:
            errors.append("image_type is required")
        elif image_type not in ['cnic', 'selfie']:
            errors.append("image_type must be either 'cnic' or 'selfie'")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def validate_batch_request(self, data: Dict) -> Dict:
        """Validates batch match request"""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Request body must be a JSON object")
            return {
                'is_valid': False,
                'errors': errors
            }
        
        verifications = data.get('verifications')
        
        if not verifications:
            errors.append("verifications array is required")
        elif not isinstance(verifications, list):
            errors.append("verifications must be an array")
        elif len(verifications) == 0:
            errors.append("verifications array cannot be empty")
        elif len(verifications) > 10:  # Limit batch size
            errors.append("Maximum batch size is 10 verifications")
        else:
            # Validate each verification item
            for i, verification in enumerate(verifications):
                item_errors = []
                
                if not isinstance(verification, dict):
                    item_errors.append(f"Item {i+1} must be an object")
                else:
                    # Check provider_id
                    if not verification.get('provider_id'):
                        item_errors.append(f"Item {i+1}: provider_id is required")
                    
                    # Check image sources
                    has_urls = bool(verification.get('cnic_url') and verification.get('selfie_url'))
                    has_base64 = bool(verification.get('cnic_base64') and verification.get('selfie_base64'))
                    
                    if not (has_urls or has_base64):
                        item_errors.append(f"Item {i+1}: Either URLs or base64 images must be provided")
                    elif has_urls and has_base64:
                        item_errors.append(f"Item {i+1}: Provide either URLs or base64, not both")
                    
                    # Validate URLs if provided
                    if has_urls:
                        if not self.validate_url(verification.get('cnic_url')):
                            item_errors.append(f"Item {i+1}: cnic_url is not valid")
                        if not self.validate_url(verification.get('selfie_url')):
                            item_errors.append(f"Item {i+1}: selfie_url is not valid")
                    
                    # Validate base64 if provided
                    if has_base64:
                        if not self.validate_base64(verification.get('cnic_base64')):
                            item_errors.append(f"Item {i+1}: cnic_base64 is not valid")
                        if not self.validate_base64(verification.get('selfie_base64')):
                            item_errors.append(f"Item {i+1}: selfie_base64 is not valid")
                
                errors.extend(item_errors)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
