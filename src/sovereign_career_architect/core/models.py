"""Core data models for the Sovereign Career Architect."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class MemoryScope(str, Enum):
    """Memory scope enumeration."""
    USER = "user"
    SESSION = "session"
    AGENT = "agent"


class ActionType(str, Enum):
    """Types of actions the agent can perform."""
    JOB_SEARCH = "job_search"
    APPLICATION_SUBMIT = "application_submit"
    INTERVIEW_PRACTICE = "interview_practice"
    PROFILE_UPDATE = "profile_update"
    GENERAL_CHAT = "general_chat"


class JobPreferences(BaseModel):
    """User job preferences and criteria."""
    roles: List[str] = Field(default_factory=list, description="Preferred job roles")
    locations: List[str] = Field(default_factory=list, description="Preferred locations")
    salary_min: Optional[int] = Field(None, description="Minimum salary expectation")
    salary_max: Optional[int] = Field(None, description="Maximum salary expectation")
    company_types: List[str] = Field(default_factory=list, description="Preferred company types")
    work_arrangements: List[str] = Field(default_factory=list, description="Remote, hybrid, onsite")
    industries: List[str] = Field(default_factory=list, description="Preferred industries")
    experience_level: Optional[str] = Field(None, description="Entry, mid, senior, executive")


class Skill(BaseModel):
    """Represents a user skill."""
    name: str = Field(..., description="Skill name")
    level: str = Field(..., description="Proficiency level")
    years_experience: Optional[float] = Field(None, description="Years of experience")
    verified: bool = Field(False, description="Whether skill is verified")


class Experience(BaseModel):
    """Represents work experience."""
    company: str = Field(..., description="Company name")
    role: str = Field(..., description="Job role/title")
    start_date: datetime = Field(..., description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date (None if current)")
    description: Optional[str] = Field(None, description="Role description")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")


class Education(BaseModel):
    """Represents educational background."""
    institution: str = Field(..., description="Educational institution")
    degree: str = Field(..., description="Degree type")
    field_of_study: str = Field(..., description="Field of study")
    graduation_date: Optional[datetime] = Field(None, description="Graduation date")
    gpa: Optional[float] = Field(None, description="Grade point average")


class PersonalInfo(BaseModel):
    """Personal information."""
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    portfolio_url: Optional[str] = Field(None, description="Portfolio website URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    preferred_language: str = Field("en", description="Preferred language code")


class DocumentStore(BaseModel):
    """Document storage references."""
    resume_path: Optional[str] = Field(None, description="Path to resume file")
    cover_letter_template: Optional[str] = Field(None, description="Cover letter template")
    portfolio_files: List[str] = Field(default_factory=list, description="Portfolio file paths")


class UserProfile(BaseModel):
    """Complete user profile."""
    user_id: UUID = Field(default_factory=uuid4, description="Unique user identifier")
    personal_info: PersonalInfo = Field(..., description="Personal information")
    skills: List[Skill] = Field(default_factory=list, description="User skills")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    preferences: JobPreferences = Field(default_factory=JobPreferences, description="Job preferences")
    documents: DocumentStore = Field(default_factory=DocumentStore, description="Document references")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class Memory(BaseModel):
    """Memory entry for persistent storage."""
    id: UUID = Field(default_factory=uuid4, description="Memory identifier")
    content: str = Field(..., description="Memory content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    scope: MemoryScope = Field(..., description="Memory scope")
    user_id: UUID = Field(..., description="Associated user ID")
    session_id: Optional[UUID] = Field(None, description="Associated session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    relationships: List[UUID] = Field(default_factory=list, description="Related memory IDs")
    importance_score: float = Field(0.5, description="Importance score (0-1)")


class ActionResult(BaseModel):
    """Result of an agent action."""
    action_type: ActionType = Field(..., description="Type of action performed")
    success: bool = Field(..., description="Whether action was successful")
    data: Dict[str, Any] = Field(default_factory=dict, description="Action result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class PlanStep(BaseModel):
    """Individual step in an execution plan."""
    id: UUID = Field(default_factory=uuid4, description="Step identifier")
    description: str = Field(..., description="Step description")
    action_type: ActionType = Field(..., description="Type of action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    dependencies: List[UUID] = Field(default_factory=list, description="Dependent step IDs")
    status: str = Field("pending", description="Step status")
    result: Optional[ActionResult] = Field(None, description="Step execution result")


class ExecutionPlan(BaseModel):
    """Complete execution plan."""
    id: UUID = Field(default_factory=uuid4, description="Plan identifier")
    steps: List[PlanStep] = Field(..., description="Plan steps")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Plan creation time")
    status: str = Field("created", description="Plan status")
    user_id: UUID = Field(..., description="Associated user ID")


class InterruptState(BaseModel):
    """State for human-in-the-loop interrupts."""
    interrupt_id: UUID = Field(default_factory=uuid4, description="Interrupt identifier")
    reason: str = Field(..., description="Reason for interrupt")
    action_summary: str = Field(..., description="Summary of pending action")
    requires_approval: bool = Field(True, description="Whether approval is required")
    approved: Optional[bool] = Field(None, description="User approval status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interrupt timestamp")


class JobPosting(BaseModel):
    """Represents a job posting."""
    id: UUID = Field(default_factory=uuid4, description="Job posting identifier")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    description: str = Field(..., description="Job description")
    requirements: List[str] = Field(default_factory=list, description="Job requirements")
    salary_range: Optional[str] = Field(None, description="Salary range")
    url: str = Field(..., description="Job posting URL")
    posted_date: Optional[datetime] = Field(None, description="Job posting date")
    match_score: Optional[float] = Field(None, description="Match score with user profile")


class InterviewSession(BaseModel):
    """Represents an interview practice session."""
    id: UUID = Field(default_factory=uuid4, description="Session identifier")
    user_id: UUID = Field(..., description="User identifier")
    role: str = Field(..., description="Target role for interview")
    language: str = Field("en", description="Interview language")
    questions: List[str] = Field(default_factory=list, description="Interview questions")
    responses: List[str] = Field(default_factory=list, description="User responses")
    feedback: Optional[str] = Field(None, description="Interview feedback")
    score: Optional[float] = Field(None, description="Interview score")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
    completed_at: Optional[datetime] = Field(None, description="Session completion time")