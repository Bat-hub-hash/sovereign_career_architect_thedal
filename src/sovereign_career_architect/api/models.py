"""API models for request/response schemas."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from sovereign_career_architect.core.safety import ActionRisk, ActionCategory


class AgentRequest(BaseModel):
    """Request to the agent for task execution."""
    message: str = Field(..., description="User message or task description")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class AgentResponse(BaseModel):
    """Response from the agent."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    requires_approval: bool = Field(False, description="Whether action requires human approval")
    approval_id: Optional[str] = Field(None, description="Approval request ID if applicable")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ApprovalRequestModel(BaseModel):
    """Model for approval requests."""
    id: str = Field(..., description="Unique approval request ID")
    timestamp: datetime = Field(..., description="Request timestamp")
    action_type: str = Field(..., description="Type of action requiring approval")
    category: ActionCategory = Field(..., description="Action category")
    risk_level: ActionRisk = Field(..., description="Risk level of the action")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description")
    consequences: List[str] = Field(..., description="Potential consequences")
    risks: List[str] = Field(..., description="Identified risks")
    benefits: List[str] = Field(..., description="Potential benefits")
    affected_systems: List[str] = Field(..., description="Systems that will be affected")
    estimated_duration: str = Field(..., description="Estimated duration")
    reversible: bool = Field(..., description="Whether the action is reversible")
    context: Dict[str, Any] = Field(..., description="Action context")
    timeout_seconds: int = Field(..., description="Approval timeout in seconds")


class ApprovalResponse(BaseModel):
    """Response for approval decision."""
    approved: bool = Field(..., description="Whether the action is approved")
    notes: Optional[str] = Field(None, description="Optional notes from approver")


class ApprovalResult(BaseModel):
    """Result of approval processing."""
    success: bool = Field(..., description="Whether approval was processed successfully")
    message: str = Field(..., description="Result message")


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    id: str = Field(..., description="Unique entry ID")
    timestamp: datetime = Field(..., description="Entry timestamp")
    action_type: str = Field(..., description="Type of action")
    category: ActionCategory = Field(..., description="Action category")
    risk_level: ActionRisk = Field(..., description="Risk level")
    approved: bool = Field(..., description="Whether action was approved")
    executed: bool = Field(..., description="Whether action was executed")
    success: Optional[bool] = Field(None, description="Whether execution was successful")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    error: Optional[str] = Field(None, description="Error message if any")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    components: Dict[str, str] = Field(..., description="Component status")


class VoiceWebhookRequest(BaseModel):
    """Webhook request from Vapi.ai voice interface."""
    event_type: str = Field(..., description="Type of webhook event")
    call_id: Optional[str] = Field(None, description="Call identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    transcript: Optional[str] = Field(None, description="Voice transcript")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VoiceWebhookResponse(BaseModel):
    """Response to voice webhook."""
    success: bool = Field(..., description="Whether webhook was processed successfully")
    message: Optional[str] = Field(None, description="Response message for TTS")
    function_result: Optional[Dict[str, Any]] = Field(None, description="Function execution result")
    next_action: Optional[str] = Field(None, description="Next action to take")


class BrowserActionRequest(BaseModel):
    """Request for browser automation action."""
    action_type: str = Field(..., description="Type of browser action")
    target_url: Optional[str] = Field(None, description="Target URL")
    selector: Optional[str] = Field(None, description="CSS selector")
    value: Optional[str] = Field(None, description="Value to input")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")


class BrowserActionResponse(BaseModel):
    """Response from browser automation."""
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Action result data")
    screenshot: Optional[str] = Field(None, description="Base64 encoded screenshot")


class JobSearchRequest(BaseModel):
    """Request for job search functionality."""
    query: str = Field(..., description="Job search query")
    location: Optional[str] = Field(None, description="Job location")
    job_type: Optional[str] = Field(None, description="Type of job (full-time, part-time, etc.)")
    experience_level: Optional[str] = Field(None, description="Required experience level")
    salary_min: Optional[int] = Field(None, description="Minimum salary")
    salary_max: Optional[int] = Field(None, description="Maximum salary")
    remote: Optional[bool] = Field(None, description="Remote work option")
    limit: int = Field(10, description="Maximum number of results")


class JobSearchResponse(BaseModel):
    """Response from job search."""
    success: bool = Field(..., description="Whether search was successful")
    jobs: List[Dict[str, Any]] = Field(..., description="List of job opportunities")
    total_count: int = Field(..., description="Total number of jobs found")
    search_metadata: Dict[str, Any] = Field(..., description="Search metadata")


class InterviewRequest(BaseModel):
    """Request for interview simulation."""
    job_role: str = Field(..., description="Job role for interview")
    experience_level: str = Field(..., description="Candidate experience level")
    language: str = Field("en", description="Interview language")
    question_count: int = Field(5, description="Number of questions")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class InterviewResponse(BaseModel):
    """Response from interview simulation."""
    success: bool = Field(..., description="Whether interview was started successfully")
    session_id: str = Field(..., description="Interview session ID")
    questions: List[Dict[str, Any]] = Field(..., description="Interview questions")
    instructions: str = Field(..., description="Interview instructions")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")