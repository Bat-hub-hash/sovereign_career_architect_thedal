"""Automated job application workflow system."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import structlog

from sovereign_career_architect.jobs.matcher import JobPosting, MatchResult
from sovereign_career_architect.core.models import UserProfile
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.core.safety import SafetyLayer
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class ApplicationStatus(Enum):
    """Application status tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REVIEW = "under_review"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    WITHDRAWN = "withdrawn"
    ERROR = "error"


class ApplicationPriority(Enum):
    """Application priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ApplicationDocument:
    """Document for job application."""
    type: str  # "resume", "cover_letter", "portfolio"
    content: str
    file_path: Optional[str] = None
    customized: bool = False
    customization_notes: Optional[str] = None


@dataclass
class ApplicationAction:
    """Individual action in application workflow."""
    action_type: str  # "navigate", "fill_form", "upload_file", "submit"
    description: str
    target_element: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    completed: bool = False
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class JobApplication:
    """Complete job application record."""
    application_id: str
    job_posting: JobPosting
    user_profile: UserProfile
    match_result: MatchResult
    status: ApplicationStatus
    priority: ApplicationPriority
    documents: List[ApplicationDocument]
    actions: List[ApplicationAction]
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    notes: Optional[str] = None
    tracking_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tracking_info is None:
            self.tracking_info = {}


class ApplicationWorkflow:
    """Manages automated job application workflows."""
    
    def __init__(self, browser_agent: BrowserAgent, safety_layer: SafetyLayer):
        self.logger = logger.bind(component="application_workflow")
        self.browser_agent = browser_agent
        self.safety_layer = safety_layer
        
        # Application tracking
        self.applications: Dict[str, JobApplication] = {}
        
        # Workflow templates by job portal
        self.portal_workflows = self._load_portal_workflows()
        
        # Document templates
        self.document_templates = self._load_document_templates()
    
    def _load_portal_workflows(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load workflow templates for different job portals."""
        return {
            "linkedin": [
                {
                    "action_type": "navigate",
                    "description": "Navigate to job posting",
                    "target_element": "job_url"
                },
                {
                    "action_type": "click",
                    "description": "Click Easy Apply button",
                    "target_element": "button[aria-label*='Easy Apply']"
                },
                {
                    "action_type": "upload_file",
                    "description": "Upload resume",
                    "target_element": "input[type='file']",
                    "data": {"document_type": "resume"}
                },
                {
                    "action_type": "fill_form",
                    "description": "Fill application form",
                    "target_element": "form",
                    "data": {"form_fields": ["phone", "location", "experience"]}
                },
                {
                    "action_type": "review",
                    "description": "Review application before submission",
                    "target_element": "review_section"
                },
                {
                    "action_type": "submit",
                    "description": "Submit application",
                    "target_element": "button[type='submit']"
                }
            ],
            "indeed": [
                {
                    "action_type": "navigate",
                    "description": "Navigate to job posting",
                    "target_element": "job_url"
                },
                {
                    "action_type": "click",
                    "description": "Click Apply Now button",
                    "target_element": "button[data-jk='apply-button']"
                },
                {
                    "action_type": "fill_form",
                    "description": "Fill contact information",
                    "target_element": "form#contact-form",
                    "data": {"form_fields": ["name", "email", "phone"]}
                },
                {
                    "action_type": "upload_file",
                    "description": "Upload resume",
                    "target_element": "input[name='resume']",
                    "data": {"document_type": "resume"}
                },
                {
                    "action_type": "fill_text",
                    "description": "Add cover letter",
                    "target_element": "textarea[name='coverletter']",
                    "data": {"document_type": "cover_letter"}
                },
                {
                    "action_type": "submit",
                    "description": "Submit application",
                    "target_element": "button[type='submit']"
                }
            ],
            "company_website": [
                {
                    "action_type": "navigate",
                    "description": "Navigate to careers page",
                    "target_element": "job_url"
                },
                {
                    "action_type": "find_apply_button",
                    "description": "Find and click apply button",
                    "target_element": "button, a[href*='apply']"
                },
                {
                    "action_type": "fill_form",
                    "description": "Fill application form",
                    "target_element": "form",
                    "data": {"form_fields": ["personal_info", "experience", "education"]}
                },
                {
                    "action_type": "upload_files",
                    "description": "Upload required documents",
                    "target_element": "input[type='file']",
                    "data": {"document_types": ["resume", "cover_letter"]}
                },
                {
                    "action_type": "answer_questions",
                    "description": "Answer screening questions",
                    "target_element": "questionnaire"
                },
                {
                    "action_type": "submit",
                    "description": "Submit application",
                    "target_element": "button[type='submit']"
                }
            ]
        }
    
    def _load_document_templates(self) -> Dict[str, str]:
        """Load document templates for customization."""
        return {
            "cover_letter": """
Dear Hiring Manager,

I am writing to express my strong interest in the {job_title} position at {company}. With my background in {relevant_skills} and {years_experience} years of experience, I am confident I would be a valuable addition to your team.

{customized_paragraph}

I am particularly drawn to {company} because of {company_attraction}. My experience with {key_technologies} aligns well with your requirements, and I am excited about the opportunity to contribute to {specific_project_or_goal}.

{closing_paragraph}

Thank you for considering my application. I look forward to discussing how my skills and enthusiasm can contribute to your team's success.

Best regards,
{candidate_name}
            """,
            "follow_up_email": """
Subject: Following up on {job_title} Application

Dear {hiring_manager_name},

I hope this email finds you well. I wanted to follow up on my application for the {job_title} position that I submitted on {application_date}.

I remain very interested in this opportunity and would welcome the chance to discuss how my {key_qualifications} can contribute to {company}'s goals.

Please let me know if you need any additional information from me.

Best regards,
{candidate_name}
            """
        }
    
    async def create_application(
        self,
        job_posting: JobPosting,
        user_profile: UserProfile,
        match_result: MatchResult,
        priority: ApplicationPriority = ApplicationPriority.MEDIUM,
        deadline: Optional[datetime] = None
    ) -> JobApplication:
        """
        Create a new job application.
        
        Args:
            job_posting: Job posting to apply for
            user_profile: User's profile
            match_result: Job matching result
            priority: Application priority
            deadline: Application deadline
            
        Returns:
            Created job application
        """
        import uuid
        
        application_id = str(uuid.uuid4())
        
        self.logger.info(
            "Creating job application",
            application_id=application_id,
            job_title=job_posting.title,
            company=job_posting.company,
            priority=priority.value
        )
        
        # Generate customized documents
        documents = await self._generate_application_documents(
            job_posting, user_profile, match_result
        )
        
        # Create workflow actions based on job portal
        actions = self._create_workflow_actions(job_posting)
        
        application = JobApplication(
            application_id=application_id,
            job_posting=job_posting,
            user_profile=user_profile,
            match_result=match_result,
            status=ApplicationStatus.PENDING,
            priority=priority,
            documents=documents,
            actions=actions,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            deadline=deadline
        )
        
        self.applications[application_id] = application
        
        self.logger.info(
            "Job application created",
            application_id=application_id,
            documents_count=len(documents),
            actions_count=len(actions)
        )
        
        return application
    
    async def _generate_application_documents(
        self,
        job_posting: JobPosting,
        user_profile: UserProfile,
        match_result: MatchResult
    ) -> List[ApplicationDocument]:
        """Generate customized application documents."""
        documents = []
        
        # Generate customized cover letter
        cover_letter = await self._generate_cover_letter(
            job_posting, user_profile, match_result
        )
        
        documents.append(ApplicationDocument(
            type="cover_letter",
            content=cover_letter,
            customized=True,
            customization_notes="Tailored based on job requirements and user strengths"
        ))
        
        # Add resume reference
        if user_profile.documents.resume_path:
            documents.append(ApplicationDocument(
                type="resume",
                content="",
                file_path=user_profile.documents.resume_path,
                customized=False
            ))
        
        # Add portfolio if available
        if user_profile.documents.portfolio_files:
            for portfolio_file in user_profile.documents.portfolio_files:
                documents.append(ApplicationDocument(
                    type="portfolio",
                    content="",
                    file_path=portfolio_file,
                    customized=False
                ))
        
        return documents
    
    async def _generate_cover_letter(
        self,
        job_posting: JobPosting,
        user_profile: UserProfile,
        match_result: MatchResult
    ) -> str:
        """Generate a customized cover letter."""
        
        # Extract key information
        job_title = job_posting.title
        company = job_posting.company
        candidate_name = user_profile.personal_info.name
        
        # Get relevant skills from match result
        strong_matches = [
            m.requirement.text for m in match_result.skill_matches
            if m.match_level.value in ["exact", "strong"]
        ]
        relevant_skills = ", ".join(strong_matches[:3]) if strong_matches else "relevant technical skills"
        
        # Calculate years of experience
        years_experience = len(user_profile.experience) * 2  # Simplified calculation
        
        # Generate customized paragraph based on strengths
        customized_paragraph = self._generate_customized_paragraph(match_result)
        
        # Company attraction (generic for now)
        company_attraction = f"its innovative approach and strong reputation in the industry"
        
        # Key technologies from requirements
        tech_requirements = [
            req.text for req in job_posting.extracted_requirements.requirements
            if req.type.value == "technical_skill"
        ]
        key_technologies = ", ".join(tech_requirements[:3]) if tech_requirements else "the required technologies"
        
        # Fill template
        cover_letter = self.document_templates["cover_letter"].format(
            job_title=job_title,
            company=company,
            relevant_skills=relevant_skills,
            years_experience=years_experience,
            customized_paragraph=customized_paragraph,
            company_attraction=company_attraction,
            key_technologies=key_technologies,
            specific_project_or_goal="your upcoming projects",
            closing_paragraph="I am excited about the opportunity to bring my expertise to your team and help drive your continued success.",
            candidate_name=candidate_name
        )
        
        return cover_letter
    
    def _generate_customized_paragraph(self, match_result: MatchResult) -> str:
        """Generate a customized paragraph based on match strengths."""
        if match_result.strengths:
            strength_text = match_result.strengths[0]
            return f"My {strength_text.lower()} makes me particularly well-suited for this role. I have consistently delivered results in similar positions and am eager to apply my expertise to new challenges."
        else:
            return "I am passionate about technology and committed to continuous learning and professional growth."
    
    def _create_workflow_actions(self, job_posting: JobPosting) -> List[ApplicationAction]:
        """Create workflow actions based on job posting source."""
        
        # Determine portal type from URL
        portal_type = self._detect_portal_type(job_posting.url)
        
        # Get workflow template
        workflow_template = self.portal_workflows.get(portal_type, self.portal_workflows["company_website"])
        
        actions = []
        for i, template in enumerate(workflow_template):
            action = ApplicationAction(
                action_type=template["action_type"],
                description=template["description"],
                target_element=template.get("target_element"),
                data=template.get("data")
            )
            actions.append(action)
        
        return actions
    
    def _detect_portal_type(self, url: Optional[str]) -> str:
        """Detect job portal type from URL."""
        if not url:
            return "company_website"
        
        url_lower = url.lower()
        if "linkedin.com" in url_lower:
            return "linkedin"
        elif "indeed.com" in url_lower:
            return "indeed"
        else:
            return "company_website"
    
    async def execute_application(self, application_id: str) -> bool:
        """
        Execute the application workflow.
        
        Args:
            application_id: Application ID to execute
            
        Returns:
            True if successful, False otherwise
        """
        if application_id not in self.applications:
            self.logger.error("Application not found", application_id=application_id)
            return False
        
        application = self.applications[application_id]
        
        self.logger.info(
            "Executing application workflow",
            application_id=application_id,
            job_title=application.job_posting.title,
            actions_count=len(application.actions)
        )
        
        application.status = ApplicationStatus.IN_PROGRESS
        application.updated_at = datetime.now()
        
        try:
            # Execute each action in sequence
            for action in application.actions:
                success = await self._execute_action(application, action)
                
                if not success:
                    application.status = ApplicationStatus.ERROR
                    self.logger.error(
                        "Application action failed",
                        application_id=application_id,
                        action_type=action.action_type,
                        error=action.error_message
                    )
                    return False
                
                # Add delay between actions
                await asyncio.sleep(2)
            
            # Mark as submitted
            application.status = ApplicationStatus.SUBMITTED
            application.submitted_at = datetime.now()
            application.updated_at = datetime.now()
            
            self.logger.info(
                "Application submitted successfully",
                application_id=application_id,
                job_title=application.job_posting.title
            )
            
            return True
            
        except Exception as e:
            application.status = ApplicationStatus.ERROR
            self.logger.error(
                "Application execution failed",
                application_id=application_id,
                error=str(e)
            )
            return False
    
    async def _execute_action(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute a single application action."""
        
        self.logger.info(
            "Executing action",
            application_id=application.application_id,
            action_type=action.action_type,
            description=action.description
        )
        
        try:
            if action.action_type == "navigate":
                success = await self._execute_navigate(application, action)
            elif action.action_type == "click":
                success = await self._execute_click(application, action)
            elif action.action_type == "fill_form":
                success = await self._execute_fill_form(application, action)
            elif action.action_type == "upload_file":
                success = await self._execute_upload_file(application, action)
            elif action.action_type == "submit":
                success = await self._execute_submit(application, action)
            elif action.action_type == "review":
                success = await self._execute_review(application, action)
            else:
                self.logger.warning(
                    "Unknown action type",
                    action_type=action.action_type
                )
                success = True  # Skip unknown actions
            
            action.completed = success
            action.timestamp = datetime.now()
            
            return success
            
        except Exception as e:
            action.error_message = str(e)
            action.timestamp = datetime.now()
            return False
    
    async def _execute_navigate(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute navigation action."""
        url = application.job_posting.url
        if not url:
            action.error_message = "No URL available for job posting"
            return False
        
        # Use safety layer to check if navigation is safe
        safety_check = await self.safety_layer.assess_action({
            "action_type": "navigate",
            "url": url,
            "description": f"Navigate to job posting: {application.job_posting.title}"
        })
        
        if not safety_check.approved:
            action.error_message = f"Navigation blocked by safety layer: {safety_check.reason}"
            return False
        
        # Navigate using browser agent
        success = await self.browser_agent.navigate_to_url(url)
        return success
    
    async def _execute_click(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute click action."""
        if not action.target_element:
            action.error_message = "No target element specified for click action"
            return False
        
        # Find and click element
        success = await self.browser_agent.click_element(action.target_element)
        return success
    
    async def _execute_fill_form(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute form filling action."""
        if not action.data or "form_fields" not in action.data:
            action.error_message = "No form fields specified"
            return False
        
        form_fields = action.data["form_fields"]
        form_data = self._prepare_form_data(application, form_fields)
        
        # Fill form using browser agent
        success = await self.browser_agent.fill_form(form_data)
        return success
    
    async def _execute_upload_file(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute file upload action."""
        if not action.data or "document_type" not in action.data:
            action.error_message = "No document type specified"
            return False
        
        document_type = action.data["document_type"]
        
        # Find document to upload
        document = next(
            (doc for doc in application.documents if doc.type == document_type),
            None
        )
        
        if not document:
            action.error_message = f"No {document_type} document available"
            return False
        
        if document.file_path:
            # Upload file
            success = await self.browser_agent.upload_file(
                action.target_element, document.file_path
            )
        else:
            # Handle text content (like cover letter)
            success = await self.browser_agent.fill_text_area(
                action.target_element, document.content
            )
        
        return success
    
    async def _execute_submit(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute submit action with safety check."""
        
        # Safety check for submission
        safety_check = await self.safety_layer.assess_action({
            "action_type": "submit_application",
            "job_title": application.job_posting.title,
            "company": application.job_posting.company,
            "description": f"Submit application for {application.job_posting.title}"
        })
        
        if not safety_check.approved:
            action.error_message = f"Submission blocked by safety layer: {safety_check.reason}"
            return False
        
        # Submit form
        success = await self.browser_agent.click_element(action.target_element)
        return success
    
    async def _execute_review(self, application: JobApplication, action: ApplicationAction) -> bool:
        """Execute review action - capture application details."""
        
        # Capture current page content for review
        page_content = await self.browser_agent.get_page_content()
        
        # Store review information
        application.tracking_info["review_content"] = page_content[:1000]  # Truncate
        application.tracking_info["review_timestamp"] = datetime.now().isoformat()
        
        return True
    
    def _prepare_form_data(self, application: JobApplication, form_fields: List[str]) -> Dict[str, str]:
        """Prepare form data from user profile."""
        user_profile = application.user_profile
        form_data = {}
        
        for field in form_fields:
            if field == "name":
                form_data["name"] = user_profile.personal_info.name
            elif field == "email":
                form_data["email"] = user_profile.personal_info.email
            elif field == "phone":
                form_data["phone"] = user_profile.personal_info.phone or ""
            elif field == "location":
                form_data["location"] = user_profile.personal_info.location or ""
            elif field == "experience":
                # Calculate total experience
                total_exp = len(user_profile.experience) * 2  # Simplified
                form_data["experience"] = str(total_exp)
            elif field == "linkedin":
                form_data["linkedin"] = user_profile.personal_info.linkedin_url or ""
            elif field == "portfolio":
                form_data["portfolio"] = user_profile.personal_info.portfolio_url or ""
        
        return form_data
    
    async def track_application_status(self, application_id: str) -> Optional[ApplicationStatus]:
        """Track application status by checking job portal."""
        if application_id not in self.applications:
            return None
        
        application = self.applications[application_id]
        
        # This would typically involve:
        # 1. Navigating to the application status page
        # 2. Parsing the current status
        # 3. Updating the application record
        
        # For now, simulate status progression
        current_status = application.status
        
        if current_status == ApplicationStatus.SUBMITTED:
            # Simulate progression after some time
            time_since_submission = datetime.now() - (application.submitted_at or datetime.now())
            
            if time_since_submission > timedelta(days=1):
                application.status = ApplicationStatus.UNDER_REVIEW
                application.updated_at = datetime.now()
        
        return application.status
    
    def get_application_summary(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get application summary."""
        if application_id not in self.applications:
            return None
        
        application = self.applications[application_id]
        
        return {
            "application_id": application_id,
            "job_title": application.job_posting.title,
            "company": application.job_posting.company,
            "status": application.status.value,
            "priority": application.priority.value,
            "created_at": application.created_at.isoformat(),
            "submitted_at": application.submitted_at.isoformat() if application.submitted_at else None,
            "match_score": application.match_result.overall_score,
            "documents_count": len(application.documents),
            "actions_completed": sum(1 for action in application.actions if action.completed),
            "total_actions": len(application.actions)
        }
    
    def get_all_applications(self) -> List[Dict[str, Any]]:
        """Get summary of all applications."""
        return [
            self.get_application_summary(app_id)
            for app_id in self.applications.keys()
        ]
    
    async def batch_apply(
        self,
        job_postings: List[JobPosting],
        user_profile: UserProfile,
        match_results: List[MatchResult],
        max_applications: int = 5
    ) -> List[str]:
        """
        Apply to multiple jobs in batch.
        
        Args:
            job_postings: List of job postings
            user_profile: User profile
            match_results: Matching results
            max_applications: Maximum number of applications to submit
            
        Returns:
            List of application IDs created
        """
        self.logger.info(
            "Starting batch application process",
            job_count=len(job_postings),
            max_applications=max_applications
        )
        
        # Sort by match score and priority
        sorted_matches = sorted(
            zip(job_postings, match_results),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        application_ids = []
        
        for i, (job_posting, match_result) in enumerate(sorted_matches[:max_applications]):
            try:
                # Determine priority based on match score
                if match_result.overall_score >= 0.8:
                    priority = ApplicationPriority.HIGH
                elif match_result.overall_score >= 0.6:
                    priority = ApplicationPriority.MEDIUM
                else:
                    priority = ApplicationPriority.LOW
                
                # Create application
                application = await self.create_application(
                    job_posting, user_profile, match_result, priority
                )
                
                application_ids.append(application.application_id)
                
                # Add delay between applications
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(
                    "Failed to create application",
                    job_title=job_posting.title,
                    error=str(e)
                )
                continue
        
        self.logger.info(
            "Batch application creation completed",
            applications_created=len(application_ids)
        )
        
        return application_ids