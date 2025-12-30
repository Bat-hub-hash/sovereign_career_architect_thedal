"""API routes for Sovereign Career Architect."""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from sovereign_career_architect.api.models import (
    AgentRequest, AgentResponse, ApprovalRequestModel, ApprovalResponse, 
    ApprovalResult, AuditLogEntry, HealthCheck, VoiceWebhookRequest, 
    VoiceWebhookResponse, BrowserActionRequest, BrowserActionResponse,
    JobSearchRequest, JobSearchResponse, InterviewRequest, InterviewResponse,
    ErrorResponse, StatusResponse
)
from sovereign_career_architect.core.agent import SovereignCareerAgent
from sovereign_career_architect.core.safety import SafetyLayer
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)

# Global instances (will be properly initialized in main.py)
agent: Optional[SovereignCareerAgent] = None
safety_layer: Optional[SafetyLayer] = None
browser_agent: Optional[BrowserAgent] = None

# Create routers
agent_router = APIRouter(prefix="/agent", tags=["agent"])
approval_router = APIRouter(prefix="/approvals", tags=["approvals"])
browser_router = APIRouter(prefix="/browser", tags=["browser"])
voice_router = APIRouter(prefix="/voice", tags=["voice"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])
interview_router = APIRouter(prefix="/interview", tags=["interview"])
health_router = APIRouter(prefix="/health", tags=["health"])


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user information from authorization header."""
    if not credentials:
        return None
    
    # In a real implementation, validate the token and extract user info
    # For now, return a placeholder
    return "user_123"


@agent_router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Chat with the Sovereign Career Architect agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(
            "Agent chat request received",
            user_id=user_id or request.user_id,
            session_id=request.session_id,
            message_length=len(request.message)
        )
        
        # Process the request with the agent
        # This is a simplified implementation - in reality, you'd integrate with LangGraph
        response_data = {
            "success": True,
            "message": f"Processed: {request.message[:100]}...",
            "data": {"processed_at": datetime.now(timezone.utc).isoformat()},
            "requires_approval": False,
            "session_id": request.session_id
        }
        
        # Add background task for logging
        background_tasks.add_task(
            log_agent_interaction,
            user_id or request.user_id,
            request.session_id,
            request.message,
            response_data
        )
        
        return AgentResponse(**response_data)
        
    except Exception as e:
        logger.error("Agent chat error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


@agent_router.get("/status", response_model=StatusResponse)
async def get_agent_status():
    """Get current agent status."""
    if not agent:
        return StatusResponse(
            status="unavailable",
            message="Agent not initialized"
        )
    
    return StatusResponse(
        status="active",
        message="Agent is running and ready",
        data={
            "initialized": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@approval_router.get("/pending", response_model=List[ApprovalRequestModel])
async def get_pending_approvals(user_id: Optional[str] = Depends(get_current_user)):
    """Get all pending approval requests."""
    if not safety_layer:
        raise HTTPException(status_code=503, detail="Safety layer not initialized")
    
    try:
        pending_approvals = safety_layer.get_pending_approvals()
        
        # Convert to API models
        api_approvals = []
        for approval in pending_approvals:
            api_approval = ApprovalRequestModel(
                id=approval.id,
                timestamp=approval.timestamp,
                action_type=approval.action_type,
                category=approval.classification.category,
                risk_level=approval.classification.risk_level,
                title=approval.summary.title,
                description=approval.summary.description,
                consequences=approval.summary.consequences,
                risks=approval.summary.risks,
                benefits=approval.summary.benefits,
                affected_systems=approval.summary.affected_systems,
                estimated_duration=approval.summary.estimated_duration,
                reversible=approval.summary.reversible,
                context=approval.context,
                timeout_seconds=approval.classification.timeout_seconds
            )
            api_approvals.append(api_approval)
        
        logger.info("Pending approvals retrieved", count=len(api_approvals), user_id=user_id)
        return api_approvals
        
    except Exception as e:
        logger.error("Error retrieving pending approvals", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve approvals: {str(e)}")


@approval_router.post("/{approval_id}/respond", response_model=ApprovalResult)
async def respond_to_approval(
    approval_id: str,
    response: ApprovalResponse,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Respond to an approval request."""
    if not safety_layer:
        raise HTTPException(status_code=503, detail="Safety layer not initialized")
    
    try:
        success = safety_layer.provide_approval(
            approval_id=approval_id,
            approved=response.approved,
            notes=response.notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Approval request not found")
        
        logger.info(
            "Approval response processed",
            approval_id=approval_id,
            approved=response.approved,
            user_id=user_id
        )
        
        return ApprovalResult(
            success=True,
            message=f"Approval {'granted' if response.approved else 'denied'} successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing approval response", error=str(e), approval_id=approval_id)
        raise HTTPException(status_code=500, detail=f"Failed to process approval: {str(e)}")


@approval_router.get("/audit", response_model=List[AuditLogEntry])
async def get_audit_log(
    limit: Optional[int] = 100,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Get audit log entries."""
    if not safety_layer:
        raise HTTPException(status_code=503, detail="Safety layer not initialized")
    
    try:
        audit_entries = safety_layer.get_audit_log(limit=limit)
        
        # Convert to API models
        api_entries = []
        for entry in audit_entries:
            api_entry = AuditLogEntry(
                id=entry.id,
                timestamp=entry.timestamp,
                action_type=entry.action_type,
                category=entry.classification.category,
                risk_level=entry.classification.risk_level,
                approved=entry.approved,
                executed=entry.executed,
                success=entry.result.success if entry.result else None,
                duration_seconds=entry.duration_seconds,
                error=entry.error
            )
            api_entries.append(api_entry)
        
        logger.info("Audit log retrieved", count=len(api_entries), user_id=user_id)
        return api_entries
        
    except Exception as e:
        logger.error("Error retrieving audit log", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log: {str(e)}")


@browser_router.post("/action", response_model=BrowserActionResponse)
async def execute_browser_action(
    request: BrowserActionRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Execute a browser automation action."""
    if not browser_agent:
        raise HTTPException(status_code=503, detail="Browser agent not initialized")
    
    try:
        logger.info(
            "Browser action requested",
            action_type=request.action_type,
            target_url=request.target_url,
            user_id=user_id
        )
        
        # Execute the browser action based on type
        if request.action_type == "navigate":
            if not request.target_url:
                raise HTTPException(status_code=400, detail="target_url required for navigation")
            
            success = await browser_agent.navigate_to(request.target_url)
            message = f"Navigation to {request.target_url} {'successful' if success else 'failed'}"
            
        elif request.action_type == "click":
            if not request.selector:
                raise HTTPException(status_code=400, detail="selector required for click action")
            
            success = await browser_agent.click_element(request.selector)
            message = f"Click on {request.selector} {'successful' if success else 'failed'}"
            
        elif request.action_type == "fill":
            if not request.selector or not request.value:
                raise HTTPException(status_code=400, detail="selector and value required for fill action")
            
            success = await browser_agent.fill_form_field(request.selector, request.value)
            message = f"Fill {request.selector} {'successful' if success else 'failed'}"
            
        elif request.action_type == "screenshot":
            screenshot_bytes = await browser_agent.take_screenshot()
            success = screenshot_bytes is not None
            
            import base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else None
            
            return BrowserActionResponse(
                success=success,
                message="Screenshot captured" if success else "Screenshot failed",
                screenshot=screenshot_b64
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action type: {request.action_type}")
        
        return BrowserActionResponse(
            success=success,
            message=message,
            data={"action_type": request.action_type, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Browser action error", error=str(e), action_type=request.action_type)
        raise HTTPException(status_code=500, detail=f"Browser action failed: {str(e)}")


@voice_router.post("/webhook", response_model=VoiceWebhookResponse)
async def handle_voice_webhook(request: VoiceWebhookRequest):
    """Handle webhook from Vapi.ai voice interface."""
    try:
        logger.info(
            "Voice webhook received",
            event_type=request.event_type,
            call_id=request.call_id,
            user_id=request.user_id
        )
        
        # Process different webhook events
        if request.event_type == "function_call":
            # Handle function call from voice interface
            if request.function_call:
                function_name = request.function_call.get("name")
                function_args = request.function_call.get("arguments", {})
                
                # Route to appropriate handler
                result = await handle_voice_function_call(function_name, function_args)
                
                return VoiceWebhookResponse(
                    success=True,
                    function_result=result,
                    message="Function executed successfully"
                )
        
        elif request.event_type == "transcript":
            # Handle voice transcript
            if request.transcript:
                # Process the transcript with the agent
                response = await process_voice_transcript(request.transcript, request.user_id)
                
                return VoiceWebhookResponse(
                    success=True,
                    message=response.get("message", "Transcript processed"),
                    next_action=response.get("next_action")
                )
        
        # Default response for unhandled events
        return VoiceWebhookResponse(
            success=True,
            message="Webhook received and processed"
        )
        
    except Exception as e:
        logger.error("Voice webhook error", error=str(e), event_type=request.event_type)
        return VoiceWebhookResponse(
            success=False,
            message=f"Webhook processing failed: {str(e)}"
        )


@jobs_router.post("/search", response_model=JobSearchResponse)
async def search_jobs(
    request: JobSearchRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Search for job opportunities."""
    try:
        logger.info(
            "Job search requested",
            query=request.query,
            location=request.location,
            user_id=user_id
        )
        
        # Mock job search implementation
        # In reality, this would integrate with job search APIs
        mock_jobs = [
            {
                "id": f"job_{i}",
                "title": f"Software Engineer - {request.query}",
                "company": f"Company {i}",
                "location": request.location or "Remote",
                "salary_range": f"${request.salary_min or 80000} - ${request.salary_max or 120000}",
                "description": f"Exciting opportunity for {request.query} role",
                "posted_date": datetime.now(timezone.utc).isoformat(),
                "remote": request.remote or False
            }
            for i in range(min(request.limit, 5))
        ]
        
        return JobSearchResponse(
            success=True,
            jobs=mock_jobs,
            total_count=len(mock_jobs),
            search_metadata={
                "query": request.query,
                "location": request.location,
                "search_time": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error("Job search error", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@interview_router.post("/start", response_model=InterviewResponse)
async def start_interview(
    request: InterviewRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Start an interview simulation session."""
    try:
        logger.info(
            "Interview simulation requested",
            job_role=request.job_role,
            experience_level=request.experience_level,
            language=request.language,
            user_id=user_id
        )
        
        # Generate mock interview questions
        mock_questions = [
            {
                "id": f"q_{i}",
                "question": f"Tell me about your experience with {request.job_role} responsibilities",
                "type": "behavioral",
                "difficulty": request.experience_level,
                "expected_duration": "3-5 minutes"
            }
            for i in range(request.question_count)
        ]
        
        session_id = f"interview_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        return InterviewResponse(
            success=True,
            session_id=session_id,
            questions=mock_questions,
            instructions=f"Welcome to your {request.job_role} interview simulation. Please answer each question thoughtfully."
        )
        
    except Exception as e:
        logger.error("Interview start error", error=str(e), job_role=request.job_role)
        raise HTTPException(status_code=500, detail=f"Interview start failed: {str(e)}")


@health_router.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    components = {
        "agent": "healthy" if agent else "unavailable",
        "safety_layer": "healthy" if safety_layer else "unavailable",
        "browser_agent": "healthy" if browser_agent else "unavailable",
        "database": "healthy",  # Would check actual database connection
        "memory": "healthy"     # Would check memory system
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        components=components
    )


# Helper functions

async def log_agent_interaction(user_id: str, session_id: str, message: str, response: Dict[str, Any]):
    """Log agent interaction for analytics."""
    logger.info(
        "Agent interaction logged",
        user_id=user_id,
        session_id=session_id,
        message_length=len(message),
        response_success=response.get("success", False)
    )


async def handle_voice_function_call(function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle function calls from voice interface."""
    logger.info("Voice function call", function_name=function_name, args=function_args)
    
    # Route to appropriate function handler
    if function_name == "search_jobs":
        # Handle job search function
        return {"result": "Job search initiated", "status": "success"}
    elif function_name == "apply_to_job":
        # Handle job application function
        return {"result": "Job application started", "status": "success"}
    else:
        return {"result": f"Unknown function: {function_name}", "status": "error"}


async def process_voice_transcript(transcript: str, user_id: Optional[str]) -> Dict[str, Any]:
    """Process voice transcript with the agent."""
    logger.info("Processing voice transcript", transcript_length=len(transcript), user_id=user_id)
    
    # In reality, this would integrate with the main agent
    return {
        "message": f"I heard: {transcript}. How can I help you with your job search?",
        "next_action": "wait_for_response"
    }


# Export all routers
all_routers = [
    agent_router,
    approval_router,
    browser_router,
    voice_router,
    jobs_router,
    interview_router,
    health_router
]