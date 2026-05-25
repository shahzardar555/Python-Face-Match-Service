# Manual Review Integration Guide

## Overview

This document explains how to integrate the manual review system with your existing admin dashboard. The face matching microservice provides API endpoints that deliver detailed edge case information for manual admin review.

## 🎯 What This Provides

**Not a dashboard** - Instead, this provides:
- Detailed edge case data structures
- API endpoints for manual review operations
- Risk assessment and recommendations
- Complete audit trail for admin decisions

## 📋 API Endpoints

### 1. Create Manual Review Case
```http
POST /api/manual-review/create-case
```

**Purpose:** Create a manual review case when AI system detects an edge case

**Request Body:** Complete verification data from `/api/match-faces`

**Response:** Detailed manual review case with all analysis data

### 2. Get Edge Cases List
```http
GET /api/manual-review/edge-cases
```

**Purpose:** Get list of all pending edge cases for admin dashboard

**Response:** Array of edge case summaries with priority levels

### 3. Get Case Details
```http
GET /api/manual-review/case-details/{case_id}
```

**Purpose:** Get complete details for a specific case

**Response:** Full case data with AI analysis, images, and recommendations

### 4. Update Case with Admin Decision
```http
POST /api/manual-review/update-case
```

**Purpose:** Record admin's final decision

**Request Body:**
```json
{
  "case_id": "CASE_provider123_20240322120000",
  "admin_id": "admin456",
  "decision": "approved",  // approved/rejected/reupload
  "notes": "Faces match despite low confidence due to lighting",
  "confidence_override": 75.0,
  "reason_for_override": "Manual verification confirms identity"
}
```

## 🔗 Integration with Existing Dashboard

### Step 1: Detect Edge Cases

In your Node.js backend, after calling `/api/match-faces`:

```javascript
const faceMatchResult = await axios.post('http://localhost:5000/api/match-faces', verificationData);

// Check if manual review is needed
if (faceMatchResult.data.result.decision === 'POSSIBLE_MATCH' || 
    faceMatchResult.data.result.fallback_used) {
  
  // Create manual review case
  const reviewCase = await axios.post('http://localhost:5000/api/manual-review/create-case', {
    ...faceMatchResult.data,
    cnic_url: verificationData.cnic_url,
    selfie_url: verificationData.selfie_url
  });
  
  // Store in MongoDB for dashboard
  await ManualReviewCase.create(reviewCase.data.review_case);
}
```

### Step 2: Dashboard Data Display

Your existing dashboard can consume this data structure:

```javascript
// Get edge cases for dashboard
const edgeCases = await axios.get('http://localhost:5000/api/manual-review/edge-cases');

// Display in dashboard with priority indicators
edgeCases.data.edge_cases.forEach(case => {
  const priorityColor = case.priority === 'high' ? 'red' : 
                       case.priority === 'medium' ? 'yellow' : 'green';
  
  // Show in your dashboard UI
  displayCase({
    id: case.case_id,
    provider: case.provider_id,
    priority: case.priority,
    type: case.edge_case_type,
    confidence: case.ai_confidence,
    risk: case.risk_level,
    status: case.status
  });
});
```

### Step 3: Case Review Flow

When admin clicks on a case:

```javascript
// Get full case details
const caseDetails = await axios.get(`http://localhost:5000/api/manual-review/case-details/${caseId}`);

// Display comprehensive information in your dashboard:
showCaseDetails({
  // AI Analysis
  aiConfidence: caseDetails.ai_analysis.confidence_score,
  aiDecision: caseDetails.ai_analysis.final_decision,
  fallbackAttempts: caseDetails.fallback_attempts,
  
  // Face Detection
  cnicFaceDetected: caseDetails.face_extraction.cnic_face_detected,
  selfieFaceDetected: caseDetails.face_extraction.selfie_face_detected,
  
  // Image Quality
  qualityIssues: caseDetails.image_quality.quality_warnings,
  
  // Risk Assessment
  riskLevel: caseDetails.risk_assessment.risk_level,
  riskFactors: caseDetails.risk_assessment.risk_factors,
  
  // Images for visual comparison
  cnicImage: caseDetails.images_available.cnic_url,
  selfieImage: caseDetails.images_available.selfie_url,
  
  // User Guidance (bilingual)
  userMessage: caseDetails.user_guidance
});
```

### Step 4: Admin Decision Recording

When admin makes a decision:

```javascript
// Record admin decision
const decisionResult = await axios.post('http://localhost:5000/api/manual-review/update-case', {
  case_id: caseId,
  admin_id: adminId,
  decision: adminDecision, // approved/rejected/reupload
  notes: adminNotes,
  confidence_override: adminConfidenceOverride,
  reason_for_override: adminReason
});

// Update MongoDB
await ManualReviewCase.updateOne(
  { case_id: caseId },
  { $set: decisionResult.data.review_update }
);

// Update provider verification status
await ProviderVerification.updateOne(
  { provider_id: providerId },
  { 
    status: adminDecision,
    reviewed_by: adminId,
    reviewed_at: new Date(),
    manual_override: true
  }
);
```

## 🎨 Edge Case Types & Handling

### 1. Low Confidence (40-60%)
**Dashboard Display:** Yellow priority, show confidence meter
**Recommendation:** Manual visual comparison
**Admin Actions:** Approve/Reject/Request reupload

### 2. Quality Issues
**Dashboard Display:** Orange priority, show quality warnings
**Recommendation:** Check image quality, request better photos
**Admin Actions:** Approve if clear, else request reupload

### 3. Multiple Detection Attempts
**Dashboard Display:** Red priority, show attempt history
**Recommendation:** Review all attempts manually
**Admin Actions:** Detailed review required

### 4. Processing Errors
**Dashboard Display:** Red priority, show error details
**Recommendation:** Technical review needed
**Admin Actions:** Technical team escalation

## 📊 Risk Assessment Integration

The system provides risk assessment for each case:

```javascript
// Use risk level in dashboard priority
const riskPriority = {
  'high': 1,    // Show first
  'medium': 2,  // Show second  
  'low': 3      // Show last
};

// Display risk factors
caseDetails.risk_assessment.risk_factors.forEach(factor => {
  showRiskWarning(factor);
});
```

## 🌍 Bilingual Support

User guidance is provided in both English and Urdu:

```javascript
// Display based on user preference
const userLang = getUserLanguage(); // 'en' or 'ur'
const message = caseDetails.user_guidance[userLang];

showUserMessage(message);
```

## 📝 MongoDB Schema for Dashboard

Use this schema in your Node.js backend:

```javascript
const manualReviewSchema = new mongoose.Schema({
  case_id: String,
  provider_id: String,
  created_at: Date,
  status: String, // pending/approved/rejected/reupload
  priority: String, // high/medium/low
  edge_case_type: String,
  edge_case_reason: String,
  
  // AI Analysis
  ai_analysis: {
    confidence_score: Number,
    decision: String,
    fallback_used: Boolean,
    tiers_attempted: Number
  },
  
  // Risk Assessment
  risk_assessment: {
    risk_level: String,
    risk_score: Number,
    risk_factors: [String]
  },
  
  // Admin Review
  admin_review: {
    reviewed_by: String,
    reviewed_at: Date,
    decision: String,
    notes: String,
    confidence_override: Number
  }
});
```

## 🔄 Complete Workflow

1. **Face Matching** → AI processes verification
2. **Edge Detection** → System identifies edge cases
3. **Case Creation** → Manual review case created
4. **Dashboard Display** → Case appears in admin dashboard
5. **Admin Review** → Admin reviews detailed information
6. **Decision Recording** → Admin decision recorded
7. **Status Update** → Provider verification updated
8. **User Notification** → User notified of outcome

## 🎯 Key Benefits

- **No Dashboard Development** - Uses your existing dashboard
- **Rich Data** - Complete AI analysis and risk assessment
- **Flexible Decisions** - Admin can override AI recommendations
- **Audit Trail** - Complete record of all decisions
- **Bilingual** - Support for English and Urdu users
- **Risk-Based** - Priority based on risk assessment

## 📞 Support

This manual review system provides all the data your existing dashboard needs to handle edge cases effectively without requiring additional dashboard development.
