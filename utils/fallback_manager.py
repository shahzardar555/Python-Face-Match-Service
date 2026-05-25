import time
from typing import Dict, List, Optional, Tuple
from deepface import DeepFace
from utils.enhanced_preprocessor import EnhancedPreprocessor
from utils.guidance_generator import GuidanceGenerator, ErrorType


class FallbackAttempt:
    """Data class for tracking fallback attempts"""
    
    def __init__(self, tier: int, threshold: float, detector_backend: str, 
                 processing_time_ms: float, success: bool, confidence: float = 0.0,
                 distance: float = 0.0, error: str = None):
        self.tier = tier
        self.threshold = threshold
        self.detector_backend = detector_backend
        self.processing_time_ms = processing_time_ms
        self.success = success
        self.confidence = confidence
        self.distance = distance
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "tier": self.tier,
            "threshold_used": self.threshold,
            "detector_backend": self.detector_backend,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "confidence_score": self.confidence,
            "distance": self.distance,
            "error": self.error,
            "timestamp": self.timestamp
        }


class FallbackManager:
    """Manages 3-tier fallback process for face matching"""
    
    def __init__(self):
        self.enhanced_preprocessor = EnhancedPreprocessor()
        self.guidance_generator = GuidanceGenerator()
        
        # Available detector backends for fallback
        self.detector_backends = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib']
        
        # Threshold tiers
        self.thresholds = {
            "initial": 0.6,
            "tier1": 0.4,
            "tier2": 0.3
        }
    
    def process_with_fallback(self, cnic_face_path: str, selfie_face_path: str,
                            initial_threshold: float = 0.6, provider_id: str = None) -> Dict:
        """Process face matching with 3-tier fallback system"""
        
        fallback_attempts = []
        final_result = None
        guidance_messages = {}
        
        # Tier 0: Initial attempt
        start_time = time.time()
        try:
            result = DeepFace.verify(
                img1_path=cnic_face_path,
                img2_path=selfie_face_path,
                model_name='Facenet',
                detector_backend='opencv',
                distance_metric='cosine',
                enforce_detection=False,
                align=True,
            )
            
            processing_time = (time.time() - start_time) * 1000
            confidence = (1 - result['distance']) * 100
            confidence = max(0, min(100, confidence))
            
            # Create attempt record
            attempt = FallbackAttempt(
                tier=0,
                threshold=initial_threshold,
                detector_backend='opencv',
                processing_time_ms=processing_time,
                success=result['verified'],
                confidence=confidence,
                distance=result['distance']
            )
            fallback_attempts.append(attempt)
            
            # Check if we need fallback
            if result['verified'] and confidence >= initial_threshold * 100:
                final_result = self._create_final_result(
                    result, confidence, initial_threshold, "MATCH", 
                    fallback_attempts, guidance_messages
                )
                return final_result
            elif confidence >= 40 and confidence < initial_threshold * 100:
                # Proceed to Tier 1
                pass
            else:
                # Too low confidence, go directly to manual review
                guidance_messages = self.guidance_generator.get_bilingual_guidance(ErrorType.LOW_CONFIDENCE)
                final_result = self._create_final_result(
                    result, confidence, initial_threshold, "POSSIBLE_MATCH",
                    fallback_attempts, guidance_messages, manual_review=True
                )
                return final_result
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            attempt = FallbackAttempt(
                tier=0,
                threshold=initial_threshold,
                detector_backend='opencv',
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
            fallback_attempts.append(attempt)
        
        # Tier 1: Lower threshold + different detector
        tier1_result = self._process_tier1(cnic_face_path, selfie_face_path, fallback_attempts)
        if tier1_result["success"]:
            return tier1_result["result"]
        
        # Tier 2: Enhanced preprocessing
        tier2_result = self._process_tier2(cnic_face_path, selfie_face_path, fallback_attempts)
        if tier2_result["success"]:
            return tier2_result["result"]
        
        # Tier 3: Manual review (all attempts failed)
        guidance_messages = self.guidance_generator.get_bilingual_guidance(ErrorType.LOW_CONFIDENCE)
        final_result = self._create_final_result(
            None, 0, initial_threshold, "POSSIBLE_MATCH",
            fallback_attempts, guidance_messages, manual_review=True
        )
        
        return final_result
    
    def _process_tier1(self, cnic_face_path: str, selfie_face_path: str,
                      fallback_attempts: List[FallbackAttempt]) -> Dict:
        """Tier 1: Lower threshold + different detector backends"""
        
        threshold = self.thresholds["tier1"]
        
        # Try different detector backends
        for detector in self.detector_backends[1:4]:  # Try retinaface, mtcnn, ssd
            start_time = time.time()
            try:
                result = DeepFace.verify(
                    img1_path=cnic_face_path,
                    img2_path=selfie_face_path,
                    model_name='Facenet',
                    detector_backend=detector,
                    distance_metric='cosine',
                    enforce_detection=False,
                    align=True,
                )
                
                processing_time = (time.time() - start_time) * 1000
                confidence = (1 - result['distance']) * 100
                confidence = max(0, min(100, confidence))
                
                # Record attempt
                attempt = FallbackAttempt(
                    tier=1,
                    threshold=threshold,
                    detector_backend=detector,
                    processing_time_ms=processing_time,
                    success=result['verified'],
                    confidence=confidence,
                    distance=result['distance']
                )
                fallback_attempts.append(attempt)
                
                # Check if successful
                if result['verified'] and confidence >= threshold * 100:
                    guidance_messages = self.guidance_generator.generate_success_guidance(confidence, "MATCH")
                    final_result = self._create_final_result(
                        result, confidence, threshold, "MATCH",
                        fallback_attempts, guidance_messages
                    )
                    return {"success": True, "result": final_result}
                elif confidence >= 40:
                    # Good enough for Tier 2
                    pass
                else:
                    continue
                    
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                attempt = FallbackAttempt(
                    tier=1,
                    threshold=threshold,
                    detector_backend=detector,
                    processing_time_ms=processing_time,
                    success=False,
                    error=str(e)
                )
                fallback_attempts.append(attempt)
                continue
        
        return {"success": False, "result": None}
    
    def _process_tier2(self, cnic_face_path: str, selfie_face_path: str,
                      fallback_attempts: List[FallbackAttempt]) -> Dict:
        """Tier 2: Enhanced preprocessing"""
        
        threshold = self.thresholds["tier2"]
        
        try:
            # Apply enhanced preprocessing to both images
            start_time = time.time()
            enhanced_cnic = self.enhanced_preprocessor.apply_enhanced_preprocessing(cnic_face_path)
            enhanced_selfie = self.enhanced_preprocessor.apply_enhanced_preprocessing(selfie_face_path)
            
            # Try matching with enhanced images
            result = DeepFace.verify(
                img1_path=enhanced_cnic,
                img2_path=enhanced_selfie,
                model_name='Facenet',
                detector_backend='retinaface',  # Use best detector
                distance_metric='cosine',
                enforce_detection=False,
                align=True,
            )
            
            processing_time = (time.time() - start_time) * 1000
            confidence = (1 - result['distance']) * 100
            confidence = max(0, min(100, confidence))
            
            # Record attempt
            attempt = FallbackAttempt(
                tier=2,
                threshold=threshold,
                detector_backend='retinaface_enhanced',
                processing_time_ms=processing_time,
                success=result['verified'],
                confidence=confidence,
                distance=result['distance']
            )
            fallback_attempts.append(attempt)
            
            # Cleanup enhanced images
            import os
            try:
                os.remove(enhanced_cnic)
                os.remove(enhanced_selfie)
            except:
                pass
            
            if result['verified'] and confidence >= threshold * 100:
                guidance_messages = self.guidance_generator.generate_success_guidance(confidence, "MATCH")
                final_result = self._create_final_result(
                    result, confidence, threshold, "MATCH",
                    fallback_attempts, guidance_messages
                )
                return {"success": True, "result": final_result}
            else:
                # Send to manual review
                guidance_messages = self.guidance_generator.get_bilingual_guidance(ErrorType.LOW_CONFIDENCE)
                final_result = self._create_final_result(
                    result, confidence, threshold, "POSSIBLE_MATCH",
                    fallback_attempts, guidance_messages, manual_review=True
                )
                return {"success": True, "result": final_result}
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            attempt = FallbackAttempt(
                tier=2,
                threshold=threshold,
                detector_backend='retinaface_enhanced',
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
            fallback_attempts.append(attempt)
            
            # Cleanup enhanced images
            import os
            try:
                if 'enhanced_cnic' in locals():
                    os.remove(enhanced_cnic)
                if 'enhanced_selfie' in locals():
                    os.remove(enhanced_selfie)
            except:
                pass
        
        return {"success": False, "result": None}
    
    def _create_final_result(self, deepface_result: Optional[Dict], confidence: float,
                           threshold: float, decision: str, fallback_attempts: List[FallbackAttempt],
                           guidance_messages: Dict, manual_review: bool = False) -> Dict:
        """Create final result with all fallback data"""
        
        # Calculate total processing time
        total_time = sum(attempt.processing_time_ms for attempt in fallback_attempts)
        
        # Determine final status
        if manual_review:
            final_status = "manual_review"
            recommendation = "MANUAL_REVIEW"
        elif decision == "MATCH":
            final_status = "approved"
            recommendation = "APPROVE"
        elif decision == "POSSIBLE_MATCH":
            final_status = "manual_review"
            recommendation = "MANUAL_REVIEW"
        else:
            final_status = "rejected"
            recommendation = "REJECT"
        
        # Prepare audit data for Node.js
        audit_data = {
            "fallback_attempts": [attempt.to_dict() for attempt in fallback_attempts],
            "total_processing_time_ms": total_time,
            "final_confidence": confidence,
            "final_threshold": threshold,
            "final_decision": decision,
            "final_status": final_status,
            "manual_review_required": manual_review,
            "guidance_provided": guidance_messages,
            "tiers_attempted": len(set(attempt.tier for attempt in fallback_attempts)),
            "success_tier": next((attempt.tier for attempt in fallback_attempts if attempt.success), None)
        }
        
        # Build response
        response = {
            "success": True,
            "result": {
                "decision": decision,
                "recommendation": recommendation,
                "confidence_score": round(confidence, 1),
                "confidence_percentage": f"{round(confidence, 1)}%",
                "distance": round(deepface_result['distance'], 3) if deepface_result else 0,
                "is_match": deepface_result['verified'] if deepface_result else False,
                "threshold_used": threshold,
                "model_used": "Facenet",
                "fallback_used": len(fallback_attempts) > 1,
                "tiers_attempted": audit_data["tiers_attempted"]
            },
            "fallback_attempts": [attempt.to_dict() for attempt in fallback_attempts],
            "guidance_messages": guidance_messages,
            "audit_data": audit_data,
            "processing_time_ms": round(total_time, 1),
            "timestamp": time.time()
        }
        
        return response
    
    def should_attempt_fallback(self, confidence: float, error_type: str = None) -> bool:
        """Determine if fallback should be attempted"""
        # Don't attempt fallback for certain error types
        no_retry_errors = [
            "Face could not be detected",
            "No face detected",
            "Multiple faces detected"
        ]
        
        if error_type and any(error in error_type for error in no_retry_errors):
            return False
        
        # Attempt fallback for confidence between 35-60%
        return 35 <= confidence <= 60
    
    def get_failure_guidance(self, error_message: str, image_type: str = None) -> Dict:
        """Get appropriate guidance based on failure type"""
        error_lower = error_message.lower()
        
        if "no face detected" in error_lower:
            if image_type == "cnic":
                return self.guidance_generator.get_bilingual_guidance(ErrorType.NO_FACE_CNIC)
            else:
                return self.guidance_generator.get_bilingual_guidance(ErrorType.NO_FACE_SELFIE)
        
        elif "multiple faces" in error_lower:
            return self.guidance_generator.get_bilingual_guidance(ErrorType.MULTIPLE_FACES_SELFIE)
        
        elif "processing error" in error_lower:
            return self.guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
        
        else:
            return self.guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
