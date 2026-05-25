from typing import Dict, List, Tuple
from enum import Enum


class ErrorType(Enum):
    NO_FACE_CNIC = "no_face_cnic"
    NO_FACE_SELFIE = "no_face_selfie"
    MULTIPLE_FACES_SELFIE = "multiple_faces_selfie"
    LOW_CONFIDENCE = "low_confidence"
    POOR_QUALITY_DARK = "poor_quality_dark"
    POOR_QUALITY_BLURRY = "poor_quality_blurry"
    POOR_QUALITY_SMALL = "poor_quality_small"
    POOR_QUALITY_COVERED = "poor_quality_covered"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_FORMAT = "invalid_format"
    PROCESSING_ERROR = "processing_error"


class GuidanceGenerator:
    def __init__(self):
        self.guidance_messages = self._initialize_guidance_messages()
    
    def _initialize_guidance_messages(self) -> Dict[ErrorType, Dict[str, Dict[str, str]]]:
        """Initialize bilingual guidance messages"""
        return {
            ErrorType.NO_FACE_CNIC: {
                "user": {
                    "en": "Face not detected in your CNIC photo. Please ensure the front of your CNIC is clearly visible and not covered or blurry. Take a new photo in good lighting with the entire CNIC visible.",
                    "ur": "آپ کی شناختی کارڈ میں چہرہ کا پتہ نہیں ملا۔ براہ کرم یقینی بنائیں کہ آپ کی شناختی کارڈ کا سامنا واضح ہے اور ڈھانچے یا دھندلا نہیں ہے۔ اچھی روشنی میں نئی فوٹو لیں۔"
                },
                "action": {
                    "en": "Re-upload CNIC photo",
                    "ur": "شناختی کارڈ کی تصویر دوبارہ اپ لوڈ کریں"
                },
                "technical": {
                    "en": "No face detected in CNIC image during face extraction",
                    "ur": "شناختی کارڈ کی تصویر سے چہرہ کا پتہ نہیں ملا"
                }
            },
            
            ErrorType.NO_FACE_SELFIE: {
                "user": {
                    "en": "No face detected in your selfie. Please take a clear front-facing photo in good lighting. Ensure your entire face is visible and nothing is covering it.",
                    "ur": "آپ کی سیلفی میں چہرہ کا پتہ نہیں ملا۔ براہ کرم اچھی روشنی میں واضح سامنی والی فوٹو لیں۔ یقینی بنائیں کہ آپ کا پورا چہرہ نظر آ رہا ہے۔"
                },
                "action": {
                    "en": "Take new selfie",
                    "ur": "نئی سیلفی لیں"
                },
                "technical": {
                    "en": "No face detected in selfie image during face extraction",
                    "ur": "سیلفی میں سے چہرہ کا پتہ نہیں ملا"
                }
            },
            
            ErrorType.MULTIPLE_FACES_SELFIE: {
                "user": {
                    "en": "Multiple faces detected in your selfie. Please take a selfie with only your face visible. Make sure no one else is in the photo.",
                    "ur": "آپ کی سیلفی میں متعدد چہرے ملا۔ براہ کرم صرف اپنے چہرے کے ساتھ سیلفی لیں۔ یقینی بنائیں کہ تصویر میں کوئی اور نہیں ہے۔"
                },
                "action": {
                    "en": "Take selfie alone",
                    "ur": "تنہا سیلفی لیں"
                },
                "technical": {
                    "en": "Multiple faces detected in selfie image",
                    "ur": "سیلفی میں متعدد چہرے پائے گئے"
                }
            },
            
            ErrorType.LOW_CONFIDENCE: {
                "user": {
                    "en": "Your verification is being reviewed by our team. The face matching needs manual review due to low confidence. You will be notified once the review is complete.",
                    "ur": "آپ کی تصدیق ہماری ٹیم کی جانب سے جائزے کے تحت ہے۔ کم اعتماد کی وجہ سے چہرے کی میچنگ کا جائزہ ضروری ہے۔ جائزہ مکمل ہونے پر آپ کو مطلع کیا جائے گا۔"
                },
                "action": {
                    "en": "Wait for manual review",
                    "ur": "دستی جائزے کا انتظار کریں"
                },
                "technical": {
                    "en": "Low confidence face match, sent for manual review",
                    "ur": "کم اعتماد چہرہ میچ، دستی جائزے کے لیے بھیجا گیا"
                }
            },
            
            ErrorType.POOR_QUALITY_DARK: {
                "user": {
                    "en": "Your photo is too dark. Please take a new photo in better lighting. Face a window or go outdoors for best results.",
                    "ur": "آپ کی تصویر بہت تاریک ہے۔ براہ کرم بہتر روشنی میں نئی تصویر لیں۔ بہترین نتائج کے لیے دریچے کا سامنا کریں یا باہر جائیں۔"
                },
                "action": {
                    "en": "Take photo in brighter light",
                    "ur": "زیادہ روشنی میں تصویر لیں"
                },
                "technical": {
                    "en": "Image brightness too low for reliable face detection",
                    "ur": "可信 چہرہ کی شناخت کے لیے تصویر کی روشنی بہت کم ہے"
                }
            },
            
            ErrorType.POOR_QUALITY_BLURRY: {
                "user": {
                    "en": "Your photo is blurry. Please hold your camera steady and take a new photo. Make sure to focus clearly on your face or CNIC.",
                    "ur": "آپ کی تصویر دھندلی ہے۔ براہ کرم اپنا کیمرہ مستحکم رکھیں اور نئی تصویر لیں۔ یقینی بنائیں کہ آپ کا چہرہ یا شناختی کارڈ واضح فوکس پر ہے۔"
                },
                "action": {
                    "en": "Hold camera steady and retake",
                    "ur": "کیمرہ مستحکم رکھیں اور دوبارہ لیں"
                },
                "technical": {
                    "en": "Image sharpness too low for reliable face detection",
                    "ur": "可信 چہرہ کی شناخت کے لیے تصویر کی وضاحت بہت کم ہے"
                }
            },
            
            ErrorType.POOR_QUALITY_SMALL: {
                "user": {
                    "en": "Your face is too small in the photo. Please move closer to the camera or zoom in so your face takes up more of the frame.",
                    "ur": "آپ کا چہرہ تصویر میں بہت چھوٹا ہے۔ براہ کیمرے کے قریب آئیں یا زوم کریں تاکہ آپ کا چہرہ فریم میں زیادہ جگہ لے۔"
                },
                "action": {
                    "en": "Move closer to camera",
                    "ur": "کیمرے کے قریب آئیں"
                },
                "technical": {
                    "en": "Face size too small for reliable face detection",
                    "ur": "可信 چہرہ کی شناخت کے لیے چہرے کا سائز بہت چھوٹا ہے"
                }
            },
            
            ErrorType.POOR_QUALITY_COVERED: {
                "user": {
                    "en": "Your face appears to be covered. Please remove anything covering your face (sunglasses, mask, hat, etc.) and take a new photo.",
                    "ur": "لگتا ہے کہ آپ کا چہرہ ڈھانچا ہوا ہے۔ براہ کرم اپنے چہرے کو ڈھانکنے والی ہر چیز کو ہٹا دیں (عینک، ماسک، ٹوپی، وغیرہ) اور نئی تصویر لیں۔"
                },
                "action": {
                    "en": "Remove face coverings",
                    "ur": "چہرے کے ڈھانچے ہٹائیں"
                },
                "technical": {
                    "en": "Face appears to be partially covered",
                    "ur": "چہرہ جزوی طور پر ڈھانچا ہوا لگتا ہے"
                }
            },
            
            ErrorType.IMAGE_TOO_LARGE: {
                "user": {
                    "en": "Your photo file is too large. Please compress the image or take a new photo with lower resolution. Maximum file size is 10MB.",
                    "ur": "آپ کی تصویر کا فائل سائز بہت بڑا ہے۔ براہ کرم تصویر کو کمپریس کریں یا کم ریزولوشن کی نئی تصویر لیں۔ زیادہ سے زیادہ فائل سائز 10MB ہے۔"
                },
                "action": {
                    "en": "Compress image or retake",
                    "ur": "تصویر کمپریس کریں یا دوبارہ لیں"
                },
                "technical": {
                    "en": "Image file size exceeds maximum limit",
                    "ur": "تصویر کا فائل سائز زیادہ سے زیادہ حد سے زیادہ ہے"
                }
            },
            
            ErrorType.INVALID_FORMAT: {
                "user": {
                    "en": "Invalid image format. Please use JPG, JPEG, PNG, or WebP format for your photos.",
                    "ur": "غلط تصویر فارمیٹ۔ براہ کرم اپنی تصاویر کے لیے JPG، JPEG، PNG، یا WebP فارمیٹ استعمال کریں۔"
                },
                "action": {
                    "en": "Use correct image format",
                    "ur": "درست تصویر فارمیٹ استعمال کریں"
                },
                "technical": {
                    "en": "Unsupported image format provided",
                    "ur": "غیر معاون تصویر فارمیٹ فراہم کیا گیا"
                }
            },
            
            ErrorType.PROCESSING_ERROR: {
                "user": {
                    "en": "There was an error processing your photos. Please try again with new photos. If the problem persists, contact support.",
                    "ur": "آپ کی تصاویر کو پروسیس کرنے میں ایک خرابی آئی۔ براہ کرم نئی تصاویر کے ساتھ دوبارہ کوشش کریں۔ اگر مسئلہ برقرار رہے، تو سپورٹ سے رابطہ کریں۔"
                },
                "action": {
                    "en": "Try again or contact support",
                    "ur": "دوبارہ کوشش کریں یا سپورٹ سے رابطہ کریں"
                },
                "technical": {
                    "en": "General processing error occurred",
                    "ur": "عام پروسیسنگ خرابی پیش آگئی"
                }
            }
        }
    
    def get_guidance(self, error_type: ErrorType, language: str = "en") -> Dict[str, str]:
        """Get guidance messages for specific error type"""
        if error_type not in self.guidance_messages:
            return self._get_fallback_guidance(language)
        
        messages = self.guidance_messages[error_type]
        return {
            "user_message": messages["user"].get(language, messages["user"]["en"]),
            "recommended_action": messages["action"].get(language, messages["action"]["en"]),
            "technical_message": messages["technical"].get(language, messages["technical"]["en"])
        }
    
    def get_bilingual_guidance(self, error_type: ErrorType) -> Dict[str, str]:
        """Get guidance in both English and Urdu"""
        if error_type not in self.guidance_messages:
            return self._get_fallback_bilingual_guidance()
        
        messages = self.guidance_messages[error_type]
        return {
            "en": messages["user"]["en"],
            "ur": messages["user"]["ur"],
            "action_en": messages["action"]["en"],
            "action_ur": messages["action"]["ur"],
            "technical_en": messages["technical"]["en"],
            "technical_ur": messages["technical"]["ur"]
        }
    
    def analyze_quality_issues(self, quality_issues: List[str]) -> List[ErrorType]:
        """Convert quality issues to error types"""
        error_types = []
        
        for issue in quality_issues:
            issue_lower = issue.lower()
            
            if "dark" in issue_lower or "bright" in issue_lower:
                error_types.append(ErrorType.POOR_QUALITY_DARK)
            elif "blurry" in issue_lower or "sharpness" in issue_lower:
                error_types.append(ErrorType.POOR_QUALITY_BLURRY)
            elif "small" in issue_lower or "resolution" in issue_lower:
                error_types.append(ErrorType.POOR_QUALITY_SMALL)
            elif "covered" in issue_lower or "cut" in issue_lower:
                error_types.append(ErrorType.POOR_QUALITY_COVERED)
        
        return error_types
    
    def _get_fallback_guidance(self, language: str) -> Dict[str, str]:
        """Fallback guidance for unknown error types"""
        fallback_messages = {
            "en": {
                "user_message": "There was an issue with your photo. Please try uploading a clear, well-lit photo.",
                "recommended_action": "Try with better photo",
                "technical_message": "Unknown error occurred during processing"
            },
            "ur": {
                "user_message": "آپ کی تصویر میں کوئی مسئلہ ہے۔ براہ کرم واضح، اچھی روشنی والی تصویر اپ لوڈ کرنے کی کوشش کریں۔",
                "recommended_action": "بہتر تصویر سے کوشش کریں",
                "technical_message": "پروسیسنگ کے دوران نامعلوم خرابی پیش آگئی"
            }
        }
        
        return fallback_messages.get(language, fallback_messages["en"])
    
    def _get_fallback_bilingual_guidance(self) -> Dict[str, str]:
        """Fallback bilingual guidance for unknown error types"""
        return {
            "en": "There was an issue with your photo. Please try uploading a clear, well-lit photo.",
            "ur": "آپ کی تصویر میں کوئی مسئلہ ہے۔ براہ کرم واضح، اچھی روشنی والی تصویر اپ لوڈ کرنے کی کوشش کریں۔",
            "action_en": "Try with better photo",
            "action_ur": "بہتر تصویر سے کوشش کریں",
            "technical_en": "Unknown error occurred during processing",
            "technical_ur": "پروسیسنگ کے دوران نامعلوم خرابی پیش آگئی"
        }
    
    def generate_success_guidance(self, confidence_score: float, decision: str) -> Dict[str, str]:
        """Generate guidance for successful matches"""
        if decision == "MATCH":
            return {
                "en": "Verification successful! Your identity has been confirmed.",
                "ur": "تصدیق کامیاب! آپ کی شناخت کی تصدیق ہو گئی ہے۔"
            }
        elif decision == "POSSIBLE_MATCH":
            return {
                "en": "Your verification requires manual review. Our team will review it shortly.",
                "ur": "آپ کی تصدیق کے لیے دستی جائزہ ضروری ہے۔ ہماری ٹیم جلد ہی اس کا جائزہ کرے گی۔"
            }
        else:
            return {
                "en": "Verification failed. Please try again with better photos.",
                "ur": "تصدیق ناکام ہوئی۔ براہ کرم بہتر تصاویر کے ساتھ دوبارہ کوشش کریں۔"
            }
