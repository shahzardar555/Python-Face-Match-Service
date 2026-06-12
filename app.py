from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from utils.image_handler import ImageHandler
from utils.face_extractor import FaceExtractor
from utils.validator import RequestValidator
from utils.fallback_manager import FallbackManager
from utils.guidance_generator import GuidanceGenerator, ErrorType
from utils.manual_review_handler import ManualReviewHandler
import traceback
import time

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

# Initialize utility classes
image_handler = ImageHandler()
face_extractor = FaceExtractor()
validator = RequestValidator()
fallback_manager = FallbackManager()
guidance_generator = GuidanceGenerator()
manual_review_handler = ManualReviewHandler()

MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', 0.6))


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Bad Request",
        "message": str(error),
        "code": 400
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Not Found",
        "message": "Route not found",
        "code": 404
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal Server Error",
        "message": "Face matching service error",
        "code": 500
    }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test if DeepFace is available
        test_result = DeepFace.analyze(
            img_path=np.zeros((100, 100, 3), dtype=np.uint8),
            actions=[],
            enforce_detection=False
        )
        deepface_available = True
    except:
        deepface_available = False
    
    return jsonify({
        "status": "healthy",
        "service": "Muawin Face Matching",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "deepface_available": deepface_available,
        "model": "Facenet"
    })


@app.route('/api/match-faces', methods=['POST'])
def match_faces():
    """Enhanced face matching endpoint with 3-tier fallback system"""
    start_time = time.time()
    temp_files = []
    
    try:
        # Step 1: Validate request
        data = request.get_json()
        if not data:
            guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "JSON body required",
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "error_type": "validation_error",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        validation_result = validator.validate_match_request(data)
        if not validation_result['is_valid']:
            guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "Invalid request data",
                "errors": validation_result['errors'],
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "error_type": "validation_error",
                    "validation_errors": validation_result['errors'],
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        # Extract request parameters
        provider_id = data.get('provider_id')
        threshold_override = data.get('threshold_override', MATCH_THRESHOLD)
        language = data.get('language', 'en')
        attempt_number = data.get('attempt_number', 1)
        
        # Step 2: Get images
        try:
            if data.get('cnic_url') and data.get('selfie_url'):
                cnic_path = image_handler.download_image_from_url(data['cnic_url'])
                selfie_path = image_handler.download_image_from_url(data['selfie_url'])
            else:
                cnic_path = image_handler.decode_base64_image(data['cnic_base64'])
                selfie_path = image_handler.decode_base64_image(data['selfie_base64'])
            
            temp_files.extend([cnic_path, selfie_path])
            
        except ValueError as e:
            # Determine error type for appropriate guidance
            error_msg = str(e).lower()
            if "too large" in error_msg:
                guidance = guidance_generator.get_bilingual_guidance(ErrorType.IMAGE_TOO_LARGE)
            elif "format" in error_msg:
                guidance = guidance_generator.get_bilingual_guidance(ErrorType.INVALID_FORMAT)
            else:
                guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
            
            return jsonify({
                "success": False,
                "error": "Image Processing Error",
                "message": str(e),
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "error_type": "image_processing_error",
                    "error_message": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        # Step 3: Validate image quality
        cnic_quality = image_handler.validate_image_quality(cnic_path)
        selfie_quality = image_handler.validate_image_quality(selfie_path)
        
        quality_warnings = []
        if not cnic_quality['is_valid']:
            quality_warnings.extend([f"CNIC: {issue}" for issue in cnic_quality['issues']])
        if not selfie_quality['is_valid']:
            quality_warnings.extend([f"Selfie: {issue}" for issue in selfie_quality['issues']])
        
        # Step 4: Preprocess images
        try:
            cnic_processed = image_handler.preprocess_image(cnic_path)
            selfie_processed = image_handler.preprocess_image(selfie_path)
            temp_files.extend([cnic_processed, selfie_processed])
        except ValueError as e:
            guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
            return jsonify({
                "success": False,
                "error": "Image Preprocessing Error",
                "message": str(e),
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "error_type": "preprocessing_error",
                    "error_message": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        # Step 5: Extract faces with enhanced error handling
        cnic_face_result = face_extractor.extract_face_from_cnic(cnic_processed)
        if not cnic_face_result['success']:
            # Return with specific guidance for CNIC face detection failure
            guidance = cnic_face_result.get('guidance', {})
            return jsonify({
                "success": False,
                "error": "Face Extraction Error",
                "message": cnic_face_result['error'],
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "error_type": "cnic_face_detection_failed",
                    "error_message": cnic_face_result['error'],
                    "image_quality": {
                        "cnic_quality": "good" if cnic_quality['is_valid'] else "poor",
                        "selfie_quality": "good" if selfie_quality['is_valid'] else "poor",
                        "quality_warnings": quality_warnings
                    },
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        selfie_face_result = face_extractor.extract_face_from_selfie(selfie_processed)
        if not selfie_face_result['success']:
            # Return with specific guidance for selfie face detection failure
            guidance = selfie_face_result.get('guidance', {})
            return jsonify({
                "success": False,
                "error": "Face Extraction Error",
                "message": selfie_face_result['error'],
                "code": 400,
                "guidance_messages": guidance,
                "audit_data": {
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "error_type": "selfie_face_detection_failed",
                    "error_message": selfie_face_result['error'],
                    "face_extraction": {
                        "cnic_face_detected": cnic_face_result['success'],
                        "cnic_face_confidence": round(cnic_face_result['confidence'], 2)
                    },
                    "image_quality": {
                        "cnic_quality": "good" if cnic_quality['is_valid'] else "poor",
                        "selfie_quality": "good" if selfie_quality['is_valid'] else "poor",
                        "quality_warnings": quality_warnings
                    },
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 400
        
        temp_files.extend([cnic_face_result['face_path'], selfie_face_result['face_path']])
        
        # Step 6: Check face quality
        cnic_face_quality = face_extractor.check_face_quality(cnic_face_result['face_path'])
        selfie_face_quality = face_extractor.check_face_quality(selfie_face_result['face_path'])
        
        # Collect face quality issues for guidance
        face_quality_issues = []
        if not cnic_face_quality['is_acceptable']:
            face_quality_issues.extend([f"CNIC Face: {issue}" for issue in cnic_face_quality['issues']])
        if not selfie_face_quality['is_acceptable']:
            face_quality_issues.extend([f"Selfie Face: {issue}" for issue in selfie_face_quality['issues']])
        
        # Step 7: Process with 3-tier fallback system
        try:
            fallback_result = fallback_manager.process_with_fallback(
                cnic_face_result['face_path'],
                selfie_face_result['face_path'],
                threshold_override,
                provider_id
            )
            
            # Add additional audit data
            fallback_result['audit_data'].update({
                'provider_id': provider_id,
                'attempt_number': attempt_number,
                'language': language,
                'threshold_override': threshold_override,
                'face_extraction': {
                    'cnic_face_detected': cnic_face_result['success'],
                    'selfie_face_detected': selfie_face_result['success'],
                    'cnic_face_confidence': round(cnic_face_result['confidence'], 2),
                    'selfie_face_confidence': round(selfie_face_result['confidence'], 2),
                    'cnic_face_location': cnic_face_result['face_location'],
                    'selfie_face_location': selfie_face_result['face_location']
                },
                'image_quality': {
                    'cnic_quality': "good" if cnic_quality['is_valid'] else "poor",
                    'selfie_quality': "good" if selfie_quality['is_valid'] else "poor",
                    'cnic_quality_score': cnic_quality,
                    'selfie_quality_score': selfie_quality,
                    'quality_warnings': quality_warnings
                },
                'face_quality': {
                    'cnic_face_acceptable': cnic_face_quality['is_acceptable'],
                    'selfie_face_acceptable': selfie_face_quality['is_acceptable'],
                    'cnic_face_quality_score': cnic_face_quality,
                    'selfie_face_quality_score': selfie_face_quality,
                    'face_quality_issues': face_quality_issues
                }
            })
            
            # Add quality-based guidance if needed
            if face_quality_issues and fallback_result['result']['decision'] != 'MATCH':
                quality_error_types = guidance_generator.analyze_quality_issues(face_quality_issues)
                if quality_error_types:
                    quality_guidance = guidance_generator.get_bilingual_guidance(quality_error_types[0])
                    fallback_result['guidance_messages'].update(quality_guidance)
            
            # Step 8: Check if manual review case should be created
            decision = fallback_result['result']['decision']
            fallback_used = fallback_result['result'].get('fallback_used', False)
            
            if decision == 'POSSIBLE_MATCH' or fallback_used:
                # Create manual review case for edge cases
                try:
                    review_case_data = {
                        **fallback_result['audit_data'],
                        "cnic_url": data.get('cnic_url'),
                        "selfie_url": data.get('selfie_url'),
                        "processing_time_ms": fallback_result['processing_time_ms']
                    }
                    
                    manual_review_case = manual_review_handler.create_manual_review_case(review_case_data)
                    
                    if manual_review_case:
                        fallback_result['manual_review_case'] = {
                            "case_id": manual_review_case['case_id'],
                            "created": True,
                            "priority": manual_review_case['priority'],
                            "edge_case_type": manual_review_case['edge_case_type'],
                            "note": "Edge case detected - manual review case created"
                        }
                        
                        # Add admin summary for dashboard
                        fallback_result['admin_summary'] = manual_review_handler.generate_admin_summary(manual_review_case)
                    else:
                        fallback_result['manual_review_case'] = {
                            "created": False,
                            "note": "No edge case detected"
                        }
                        
                except Exception as e:
                    # Log error but don't fail the main process
                    fallback_result['manual_review_case'] = {
                        "created": False,
                        "error": f"Failed to create manual review case: {str(e)}",
                        "note": "Manual review case creation failed, but verification completed"
                    }
            else:
                fallback_result['manual_review_case'] = {
                    "created": False,
                    "note": "No manual review needed - verification clear"
                }
            
            # Step 9: Cleanup temp files
            image_handler.cleanup_temp_files(*temp_files)
            
            return jsonify(fallback_result)
            
        except Exception as e:
            # Cleanup on error
            image_handler.cleanup_temp_files(*temp_files)
            
            guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
            return jsonify({
                "success": False,
                "error": "Matching Error",
                "message": f"Face matching failed: {str(e)}",
                "code": 500,
                "guidance_messages": guidance,
                "audit_data": {
                    "provider_id": provider_id,
                    "attempt_number": attempt_number,
                    "error_type": "matching_error",
                    "error_message": str(e),
                    "face_extraction": {
                        "cnic_face_detected": cnic_face_result['success'],
                        "selfie_face_detected": selfie_face_result['success'],
                        "cnic_face_confidence": round(cnic_face_result['confidence'], 2),
                        "selfie_face_confidence": round(selfie_face_result['confidence'], 2)
                    },
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }), 500
        
    except Exception as e:
        # Cleanup on unexpected error
        image_handler.cleanup_temp_files(*temp_files)
        
        guidance = guidance_generator.get_bilingual_guidance(ErrorType.PROCESSING_ERROR)
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": f"Unexpected error: {str(e)}",
            "code": 500,
            "guidance_messages": guidance,
            "audit_data": {
                "error_type": "unexpected_error",
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        }), 500


@app.route('/api/manual-review/create-case', methods=['POST'])
def create_manual_review_case():
    """Create manual review case for edge cases"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "JSON body required",
                "code": 400
            }), 400
        
        # Create manual review case from verification data
        review_case = manual_review_handler.create_manual_review_case(data)
        
        if not review_case:
            return jsonify({
                "success": False,
                "error": "No Edge Case",
                "message": "This verification does not require manual review",
                "code": 400
            }), 400
        
        return jsonify({
            "success": True,
            "message": "Manual review case created successfully",
            "case_id": review_case["case_id"],
            "review_case": review_case,
            "admin_summary": manual_review_handler.generate_admin_summary(review_case),
            "note": "This is an edge case requiring manual admin attention"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": f"Error creating manual review case: {str(e)}",
            "code": 500
        }), 500


@app.route('/api/manual-review/update-case', methods=['POST'])
def update_manual_review_case():
    """Update manual review case with admin decision"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "JSON body required",
                "code": 400
            }), 400
        
        case_id = data.get("case_id")
        admin_data = {
            "admin_id": data.get("admin_id"),
            "decision": data.get("decision"),  # approved/rejected/reupload
            "notes": data.get("notes"),
            "confidence_override": data.get("confidence_override"),
            "reason_for_override": data.get("reason_for_override")
        }
        
        # Validate required fields
        if not case_id or not admin_data["admin_id"] or not admin_data["decision"]:
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "case_id, admin_id, and decision are required",
                "code": 400
            }), 400
        
        # Update the case
        review_update = manual_review_handler.update_admin_review(case_id, admin_data)
        
        return jsonify({
            "success": True,
            "message": "Manual review case updated successfully",
            "case_id": case_id,
            "review_update": review_update,
            "note": "Admin decision has been recorded and will override AI recommendation"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": f"Error updating manual review case: {str(e)}",
            "code": 500
        }), 500


@app.route('/api/manual-review/edge-cases', methods=['GET'])
def get_edge_cases():
    """Get list of edge cases requiring manual review"""
    try:
        # This would typically query a database
        # For now, return sample structure
        sample_edge_cases = [
            {
                "case_id": "CASE_provider123_20240322120000",
                "provider_id": "provider123",
                "priority": "high",
                "edge_case_type": "low_confidence",
                "created_at": "2024-03-22T12:00:00",
                "status": "pending",
                "ai_confidence": 45.5,
                "risk_level": "medium"
            },
            {
                "case_id": "CASE_provider456_20240322115000",
                "provider_id": "provider456",
                "priority": "medium",
                "edge_case_type": "quality_issues",
                "created_at": "2024-03-22T11:50:00",
                "status": "pending",
                "ai_confidence": 52.3,
                "risk_level": "low"
            }
        ]
        
        return jsonify({
            "success": True,
            "edge_cases": sample_edge_cases,
            "total_cases": len(sample_edge_cases),
            "pending_cases": len([case for case in sample_edge_cases if case["status"] == "pending"]),
            "note": "These are edge cases requiring manual admin attention"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": f"Error retrieving edge cases: {str(e)}",
            "code": 500
        }), 500


@app.route('/api/manual-review/case-details/<case_id>', methods=['GET'])
def get_case_details(case_id):
    """Get detailed information for a specific manual review case"""
    try:
        # This would typically query a database
        # For now, return sample detailed case structure
        sample_case = {
            "case_id": case_id,
            "provider_id": "provider123",
            "created_at": "2024-03-22T12:00:00",
            "status": "pending",
            "priority": "high",
            "edge_case_type": "low_confidence",
            "edge_case_reason": "Confidence score 45.5% is in uncertain range (40-60%)",
            
            "ai_analysis": {
                "final_decision": "POSSIBLE_MATCH",
                "confidence_score": 45.5,
                "distance": 0.545,
                "model_used": "Facenet",
                "fallback_used": True,
                "tiers_attempted": 2
            },
            
            "face_extraction": {
                "cnic_face_detected": True,
                "selfie_face_detected": True,
                "cnic_face_confidence": 0.92,
                "selfie_face_confidence": 0.88
            },
            
            "image_quality": {
                "cnic_quality": "good",
                "selfie_quality": "poor",
                "quality_warnings": ["Selfie: Image too dark"]
            },
            
            "fallback_attempts": [
                {
                    "tier": 0,
                    "threshold_used": 0.6,
                    "detector_backend": "opencv",
                    "processing_time_ms": 1200,
                    "success": False,
                    "confidence_score": 35.2
                },
                {
                    "tier": 1,
                    "threshold_used": 0.4,
                    "detector_backend": "retinaface",
                    "processing_time_ms": 1500,
                    "success": True,
                    "confidence_score": 45.5
                }
            ],
            
            "risk_assessment": {
                "risk_level": "medium",
                "risk_score": 35,
                "risk_factors": ["Uncertain confidence", "Image quality concerns"]
            },
            
            "user_guidance": {
                "en": "Your verification requires manual review. Our team will review it shortly.",
                "ur": "آپ کی تصدیق کے لیے دستی جائزہ ضروری ہے۔ ہماری ٹیم جلد ہی اس کا جائزہ کرے گی۔"
            },
            
            "images_available": {
                "cnic_url": "https://cloudinary.com/.../cnic.jpg",
                "selfie_url": "https://cloudinary.com/.../selfie.jpg"
            },
            
            "admin_review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "decision": None,
                "notes": None
            },
            
            "note": "This is an edge case requiring manual admin review before final decision"
        }
        
        return jsonify({
            "success": True,
            "case_details": sample_case,
            "admin_summary": manual_review_handler.generate_admin_summary(sample_case)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": f"Error retrieving case details: {str(e)}",
            "code": 500
        }), 500


@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    """Analyze single face for testing"""
    temp_files = []
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "JSON body required",
                "code": 400
            }), 400
        
        validation_result = validator.validate_analyze_request(data)
        if not validation_result['is_valid']:
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "Invalid request data",
                "errors": validation_result['errors'],
                "code": 400
            }), 400
        
        image_type = data.get('image_type')
        
        # Get image
        try:
            if data.get('image_url'):
                image_path = image_handler.download_image_from_url(data['image_url'])
            else:
                image_path = image_handler.decode_base64_image(data['image_base64'])
            
            temp_files.append(image_path)
            
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": "Image Processing Error",
                "message": str(e),
                "code": 400
            }), 400
        
        # Preprocess image
        try:
            processed_path = image_handler.preprocess_image(image_path)
            temp_files.append(processed_path)
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": "Image Preprocessing Error",
                "message": str(e),
                "code": 400
            }), 400
        
        # Extract face based on type
        if image_type == 'cnic':
            face_result = face_extractor.extract_face_from_cnic(processed_path)
        else:  # selfie
            face_result = face_extractor.extract_face_from_selfie(processed_path)
        
        if not face_result['success']:
            return jsonify({
                "success": False,
                "error": "Face Detection Error",
                "message": face_result['error'],
                "code": 400
            }), 400
        
        temp_files.append(face_result['face_path'])
        
        # Check face quality
        quality_result = face_extractor.check_face_quality(face_result['face_path'])
        
        # Cleanup
        image_handler.cleanup_temp_files(*temp_files)
        
        return jsonify({
            "success": True,
            "face_detected": True,
            "face_count": 1,
            "face_confidence": round(face_result['confidence'], 2),
            "face_location": face_result['face_location'],
            "quality_check": {
                "is_acceptable": quality_result['is_acceptable'],
                "brightness": "good" if quality_result.get('brightness', 0) > 0.3 and quality_result.get('brightness', 0) < 0.8 else "poor",
                "sharpness": "good" if quality_result.get('sharpness', 0) > 50 else "poor",
                "issues": quality_result['issues']
            }
        })
        
    except Exception as e:
        image_handler.cleanup_temp_files(*temp_files)
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": str(e),
            "code": 500
        }), 500


@app.route('/api/batch-match', methods=['POST'])
def batch_match():
    """Batch face matching for admin"""
    start_time = time.time()
    all_temp_files = []
    results = []
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "JSON body required",
                "code": 400
            }), 400
        
        validation_result = validator.validate_batch_request(data)
        if not validation_result['is_valid']:
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "Invalid request data",
                "errors": validation_result['errors'],
                "code": 400
            }), 400
        
        verifications = data.get('verifications', [])
        
        for verification in verifications:
            temp_files = []
            provider_id = verification.get('provider_id')
            
            try:
                # Get images
                if verification.get('cnic_url'):
                    cnic_path = image_handler.download_image_from_url(verification['cnic_url'])
                    selfie_path = image_handler.download_image_from_url(verification['selfie_url'])
                else:
                    cnic_path = image_handler.decode_base64_image(verification['cnic_base64'])
                    selfie_path = image_handler.decode_base64_image(verification['selfie_base64'])
                
                temp_files.extend([cnic_path, selfie_path])
                
                # Preprocess
                cnic_processed = image_handler.preprocess_image(cnic_path)
                selfie_processed = image_handler.preprocess_image(selfie_path)
                temp_files.extend([cnic_processed, selfie_processed])
                
                # Extract faces
                cnic_face = face_extractor.extract_face_from_cnic(cnic_processed)
                selfie_face = face_extractor.extract_face_from_selfie(selfie_processed)
                
                if not cnic_face['success'] or not selfie_face['success']:
                    results.append({
                        "provider_id": provider_id,
                        "decision": "ERROR",
                        "confidence_score": 0,
                        "error": cnic_face.get('error') or selfie_face.get('error')
                    })
                    image_handler.cleanup_temp_files(*temp_files)
                    continue
                
                temp_files.extend([cnic_face['face_path'], selfie_face['face_path']])
                
                # Match faces
                match_result = DeepFace.verify(
                    img1_path=cnic_face['face_path'],
                    img2_path=selfie_face['face_path'],
                    model_name='Facenet',
                    detector_backend='opencv',
                    distance_metric='cosine',
                    enforce_detection=False,
                    align=True,
                )
                
                confidence = (1 - match_result['distance']) * 100
                confidence = max(0, min(100, confidence))
                
                # Make decision
                if match_result['verified'] and confidence >= MATCH_THRESHOLD * 100:
                    decision = 'MATCH'
                elif confidence >= 50:
                    decision = 'POSSIBLE_MATCH'
                else:
                    decision = 'NO_MATCH'
                
                results.append({
                    "provider_id": provider_id,
                    "decision": decision,
                    "confidence_score": round(confidence, 1)
                })
                
                image_handler.cleanup_temp_files(*temp_files)
                
            except Exception as e:
                results.append({
                    "provider_id": provider_id,
                    "decision": "ERROR",
                    "confidence_score": 0,
                    "error": str(e)
                })
                image_handler.cleanup_temp_files(*temp_files)
        
        # Calculate summary
        matches = sum(1 for r in results if r['decision'] == 'MATCH')
        no_matches = sum(1 for r in results if r['decision'] == 'NO_MATCH')
        manual_reviews = sum(1 for r in results if r['decision'] == 'POSSIBLE_MATCH')
        errors = sum(1 for r in results if r['decision'] == 'ERROR')
        
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "total": len(verifications),
            "processed": len(results),
            "results": results,
            "summary": {
                "matches": matches,
                "no_matches": no_matches,
                "manual_review": manual_reviews,
                "errors": errors
            },
            "processing_time_ms": round(processing_time, 1),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        image_handler.cleanup_temp_files(*all_temp_files)
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": str(e),
            "code": 500
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Muawin Face Matching Service on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Match threshold: {MATCH_THRESHOLD}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
