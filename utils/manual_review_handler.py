import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REUPLOAD = "requires_reupload"


class EdgeCaseType(Enum):
    LOW_CONFIDENCE = "low_confidence"
    QUALITY_ISSUES = "quality_issues"
    MULTIPLE_DETECTION_ATTEMPTS = "multiple_detection_attempts"
    AGE_DIFFERENCE = "age_difference"
    PROCESSING_ERROR = "processing_error"
    MANUAL_OVERRIDE_REQUIRED = "manual_override_required"


class ManualReviewHandler:
    """Handles manual review data for edge cases"""
    
    def __init__(self):
        self.edge_case_thresholds = {
            "low_confidence_min": 40,
            "low_confidence_max": 60,
            "max_fallback_attempts": 3,
            "quality_score_min": 0.5,
            "processing_time_max_ms": 5000
        }
    
    def create_manual_review_case(self, verification_data: Dict) -> Dict:
        """Create a comprehensive manual review case for edge detection"""
        
        # Determine if this is an edge case
        edge_case_info = self._analyze_edge_case(verification_data)
        
        if not edge_case_info["is_edge_case"]:
            return None
        
        # Create manual review case
        review_case = {
            "case_id": self._generate_case_id(verification_data.get("provider_id")),
            "provider_id": verification_data.get("provider_id"),
            "created_at": datetime.now().isoformat(),
            "status": ReviewStatus.PENDING.value,
            "priority": self._determine_priority(edge_case_info),
            "edge_case_type": edge_case_info["type"],
            "edge_case_reason": edge_case_info["reason"],
            
            # Original verification data
            "original_request": {
                "cnic_url": verification_data.get("cnic_url"),
                "selfie_url": verification_data.get("selfie_url"),
                "threshold_used": verification_data.get("threshold_used", 0.6),
                "language": verification_data.get("language", "en"),
                "attempt_number": verification_data.get("attempt_number", 1)
            },
            
            # AI Analysis Results
            "ai_analysis": {
                "final_decision": verification_data.get("result", {}).get("decision"),
                "confidence_score": verification_data.get("result", {}).get("confidence_score"),
                "distance": verification_data.get("result", {}).get("distance"),
                "model_used": verification_data.get("result", {}).get("model_used"),
                "fallback_used": verification_data.get("result", {}).get("fallback_used", False),
                "tiers_attempted": verification_data.get("result", {}).get("tiers_attempted", 1)
            },
            
            # Face Extraction Details
            "face_extraction": {
                "cnic_face_detected": verification_data.get("face_extraction", {}).get("cnic_face_detected", False),
                "selfie_face_detected": verification_data.get("face_extraction", {}).get("selfie_face_detected", False),
                "cnic_face_confidence": verification_data.get("face_extraction", {}).get("cnic_face_confidence", 0),
                "selfie_face_confidence": verification_data.get("face_extraction", {}).get("selfie_face_confidence", 0),
                "cnic_face_location": verification_data.get("face_extraction", {}).get("cnic_face_location"),
                "selfie_face_location": verification_data.get("face_extraction", {}).get("selfie_face_location")
            },
            
            # Image Quality Assessment
            "image_quality": {
                "cnic_quality": verification_data.get("image_quality", {}).get("cnic_quality", "unknown"),
                "selfie_quality": verification_data.get("image_quality", {}).get("selfie_quality", "unknown"),
                "cnic_quality_score": verification_data.get("image_quality", {}).get("cnic_quality_score", {}),
                "selfie_quality_score": verification_data.get("image_quality", {}).get("selfie_quality_score", {}),
                "quality_warnings": verification_data.get("image_quality", {}).get("quality_warnings", [])
            },
            
            # Face Quality Details
            "face_quality": {
                "cnic_face_acceptable": verification_data.get("face_quality", {}).get("cnic_face_acceptable", False),
                "selfie_face_acceptable": verification_data.get("face_quality", {}).get("selfie_face_acceptable", False),
                "face_quality_issues": verification_data.get("face_quality", {}).get("face_quality_issues", [])
            },
            
            # Fallback Attempts History
            "fallback_attempts": verification_data.get("fallback_attempts", []),
            
            # Processing Metrics
            "processing_metrics": {
                "total_processing_time_ms": verification_data.get("processing_time_ms", 0),
                "total_attempts": len(verification_data.get("fallback_attempts", [])),
                "success_tier": verification_data.get("audit_data", {}).get("success_tier"),
                "error_count": sum(1 for attempt in verification_data.get("fallback_attempts", []) if not attempt.get("success", False))
            },
            
            # User Guidance Provided
            "user_guidance": {
                "messages": verification_data.get("guidance_messages", {}),
                "recommended_action": self._get_recommended_action(edge_case_info)
            },
            
            # Risk Assessment
            "risk_assessment": self._assess_risk(verification_data, edge_case_info),
            
            # Admin Notes (to be filled by admin)
            "admin_review": {
                "reviewed_by": None,
                "reviewed_at": None,
                "decision": None,
                "notes": None,
                "confidence_override": None,
                "reason_for_override": None
            },
            
            # Metadata
            "metadata": {
                "service_version": "1.0.0",
                "api_endpoint": "/api/match-faces",
                "requires_manual_attention": True,
                "auto_approve_recommended": False,
                "auto_reject_recommended": False
            }
        }
        
        return review_case
    
    def _analyze_edge_case(self, verification_data: Dict) -> Dict:
        """Analyze verification data to determine if it's an edge case"""
        
        edge_cases = []
        
        # Check for low confidence
        confidence = verification_data.get("result", {}).get("confidence_score", 0)
        if self.edge_case_thresholds["low_confidence_min"] <= confidence <= self.edge_case_thresholds["low_confidence_max"]:
            edge_cases.append({
                "type": EdgeCaseType.LOW_CONFIDENCE.value,
                "reason": f"Confidence score {confidence}% is in uncertain range (40-60%)",
                "severity": "medium"
            })
        
        # Check for quality issues
        quality_warnings = verification_data.get("image_quality", {}).get("quality_warnings", [])
        face_quality_issues = verification_data.get("face_quality", {}).get("face_quality_issues", [])
        
        if quality_warnings or face_quality_issues:
            edge_cases.append({
                "type": EdgeCaseType.QUALITY_ISSUES.value,
                "reason": f"Image quality issues detected: {len(quality_warnings + face_quality_issues)} issues",
                "severity": "high" if len(quality_warnings + face_quality_issues) > 3 else "medium"
            })
        
        # Check for multiple fallback attempts
        fallback_attempts = verification_data.get("fallback_attempts", [])
        if len(fallback_attempts) > 2:
            edge_cases.append({
                "type": EdgeCaseType.MULTIPLE_DETECTION_ATTEMPTS.value,
                "reason": f"Required {len(fallback_attempts)} fallback attempts",
                "severity": "high"
            })
        
        # Check for processing errors
        error_count = sum(1 for attempt in fallback_attempts if not attempt.get("success", False))
        if error_count > 1:
            edge_cases.append({
                "type": EdgeCaseType.PROCESSING_ERROR.value,
                "reason": f"Processing errors in {error_count} attempts",
                "severity": "high"
            })
        
        # Check if manual review was already recommended
        decision = verification_data.get("result", {}).get("decision")
        if decision == "POSSIBLE_MATCH":
            edge_cases.append({
                "type": EdgeCaseType.MANUAL_OVERRIDE_REQUIRED.value,
                "reason": "AI system recommends manual review",
                "severity": "medium"
            })
        
        # Return the most severe edge case
        if edge_cases:
            # Sort by severity (high > medium > low)
            severity_order = {"high": 3, "medium": 2, "low": 1}
            edge_cases.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
            
            return {
                "is_edge_case": True,
                "type": edge_cases[0]["type"],
                "reason": edge_cases[0]["reason"],
                "severity": edge_cases[0]["severity"],
                "all_issues": edge_cases
            }
        
        return {
            "is_edge_case": False,
            "type": None,
            "reason": None,
            "severity": None,
            "all_issues": []
        }
    
    def _determine_priority(self, edge_case_info: Dict) -> str:
        """Determine priority level for manual review"""
        severity = edge_case_info.get("severity", "low")
        
        if severity == "high":
            return "high"
        elif severity == "medium":
            return "medium"
        else:
            return "low"
    
    def _get_recommended_action(self, edge_case_info: Dict) -> str:
        """Get recommended action based on edge case type"""
        edge_type = edge_case_info.get("type")
        
        recommendations = {
            EdgeCaseType.LOW_CONFIDENCE.value: "manual_review",
            EdgeCaseType.QUALITY_ISSUES.value: "request_reupload",
            EdgeCaseType.MULTIPLE_DETECTION_ATTEMPTS.value: "manual_review",
            EdgeCaseType.PROCESSING_ERROR.value: "request_reupload",
            EdgeCaseType.MANUAL_OVERRIDE_REQUIRED.value: "manual_review"
        }
        
        return recommendations.get(edge_type, "manual_review")
    
    def _assess_risk(self, verification_data: Dict, edge_case_info: Dict) -> Dict:
        """Assess risk level for this verification"""
        
        risk_factors = []
        risk_score = 0
        
        # Low confidence adds risk
        confidence = verification_data.get("result", {}).get("confidence_score", 100)
        if confidence < 50:
            risk_factors.append("Low confidence matching")
            risk_score += 30
        elif confidence < 60:
            risk_factors.append("Uncertain confidence")
            risk_score += 15
        
        # Quality issues add risk
        quality_warnings = verification_data.get("image_quality", {}).get("quality_warnings", [])
        if len(quality_warnings) > 2:
            risk_factors.append("Multiple quality issues")
            risk_score += 20
        elif quality_warnings:
            risk_factors.append("Image quality concerns")
            risk_score += 10
        
        # Multiple failed attempts add risk
        failed_attempts = sum(1 for attempt in verification_data.get("fallback_attempts", []) if not attempt.get("success", False))
        if failed_attempts > 2:
            risk_factors.append("Multiple processing failures")
            risk_score += 25
        elif failed_attempts > 0:
            risk_factors.append("Processing inconsistencies")
            risk_score += 10
        
        # Face detection issues add risk
        cnic_detected = verification_data.get("face_extraction", {}).get("cnic_face_detected", False)
        selfie_detected = verification_data.get("face_extraction", {}).get("selfie_face_detected", False)
        
        if not cnic_detected or not selfie_detected:
            risk_factors.append("Face detection problems")
            risk_score += 20
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 100),
            "risk_factors": risk_factors,
            "assessment_notes": f"Risk assessment based on {len(risk_factors)} factors"
        }
    
    def _generate_case_id(self, provider_id: str) -> str:
        """Generate unique case ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CASE_{provider_id}_{timestamp}"
    
    def update_admin_review(self, case_id: str, admin_data: Dict) -> Dict:
        """Update case with admin review decision"""
        
        review_update = {
            "admin_review": {
                "reviewed_by": admin_data.get("admin_id"),
                "reviewed_at": datetime.now().isoformat(),
                "decision": admin_data.get("decision"),  # approved/rejected/reupload
                "notes": admin_data.get("notes"),
                "confidence_override": admin_data.get("confidence_override"),
                "reason_for_override": admin_data.get("reason_for_override")
            },
            "status": admin_data.get("decision"),
            "metadata": {
                "manual_override_applied": True,
                "override_timestamp": datetime.now().isoformat()
            }
        }
        
        return review_update
    
    def generate_admin_summary(self, review_case: Dict) -> Dict:
        """Generate summary for admin review"""
        
        return {
            "case_summary": {
                "case_id": review_case["case_id"],
                "provider_id": review_case["provider_id"],
                "priority": review_case["priority"],
                "edge_case_type": review_case["edge_case_type"],
                "created_at": review_case["created_at"],
                "status": review_case["status"]
            },
            "key_findings": {
                "ai_confidence": review_case["ai_analysis"]["confidence_score"],
                "risk_level": review_case["risk_assessment"]["risk_level"],
                "quality_issues": len(review_case["image_quality"]["quality_warnings"]),
                "processing_attempts": review_case["processing_metrics"]["total_attempts"]
            },
            "recommended_action": review_case["user_guidance"]["recommended_action"],
            "urgent_attention": review_case["priority"] == "high",
            "images_available": {
                "cnic_url": review_case["original_request"]["cnic_url"],
                "selfie_url": review_case["original_request"]["selfie_url"]
            }
        }
