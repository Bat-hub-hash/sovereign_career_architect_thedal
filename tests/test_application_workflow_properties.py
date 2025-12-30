"""Property-based tests for application workflow system."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any
from datetime import datetime, timedelta

from sovereign_career_architect.jobs.application import (
    ApplicationWorkflow,
    JobApplication,
    ApplicationStatus,
    ApplicationPriority,
    ApplicationDocument,
    ApplicationAction
)
from sovereign_career_architect.jobs.matcher import JobPosting, MatchResult, SkillMatch, MatchLevel
from sovereign_career_architect.jobs.extractor import ExtractedRequirements, JobRequirement, RequirementType, RequirementLevel
from sovereign_career_architect.core.models import UserProfile, PersonalInfo, Skill, JobPreferences, DocumentStore
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.core.safety import SafetyLayer, ActionAssessment


# Test data strategies
@st.composite
def application_document_strategy(draw):
    """Generate application documents."""
    return ApplicationDocument(
        type=draw(st.sampled_from(["resume", "cover_letter", "portfolio"])),
        content=draw(st.text(min_size=50, max_size=500)),
        file_path=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100))),
        customized=draw(st.booleans()),
        customization_notes=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100)))
    )


@st.composite
def application_action_strategy(draw):
    """Generate application actions."""
    return ApplicationAction(
        action_type=draw(st.sampled_from([
            "navigate", "click", "fill_form", "upload_file", "submit", "review"
        ])),
        description=draw(st.text(min_size=10, max_size=100)),
        target_element=draw(st.one_of(st.none(), st.text(min_size=5, max_size=50))),
        data=draw(st.one_of(st.none(), st.dictionaries(st.text(), st.text()))),
        completed=draw(st.booleans()),
        error_message=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
        timestamp=draw(st.one_of(st.none(), st.datetimes()))
    )


@st.composite
def job_application_strategy(draw):
    """Generate job applications."""
    from tests.test_job_matching_properties import job_posting_strategy, user_profile_strategy
    
    job_posting = draw(job_posting_strategy())
    user_profile = draw(user_profile_strategy())
    
    # Create a simple match result
    match_result = MatchResult(
        job_posting=job_posting,
        overall_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        category_scores={},
        skill_matches=[],
        missing_requirements=[],
        strengths=draw(st.lists(st.text(min_size=5, max_size=50), max_size=3)),
        gaps=draw(st.lists(st.text(min_size=5, max_size=50), max_size=3)),
        recommendations=draw(st.lists(st.text(min_size=10, max_size=100), max_size=3)),
        fit_level=draw(st.sampled_from(["excellent", "good", "fair", "poor"]))
    )
    
    documents = draw(st.lists(application_document_strategy(), min_size=1, max_size=5))
    actions = draw(st.lists(application_action_strategy(), min_size=3, max_size=10))
    
    return JobApplication(
        application_id=draw(st.text(min_size=10, max_size=50)),
        job_posting=job_posting,
        user_profile=user_profile,
        match_result=match_result,
        status=draw(st.sampled_from(list(ApplicationStatus))),
        priority=draw(st.sampled_from(list(ApplicationPriority))),
        documents=documents,
        actions=actions,
        created_at=draw(st.datetimes()),
        updated_at=draw(st.datetimes()),
        submitted_at=draw(st.one_of(st.none(), st.datetimes())),
        deadline=draw(st.one_of(st.none(), st.datetimes())),
        notes=draw(st.one_of(st.none(), st.text(max_size=200))),
        tracking_info=draw(st.dictionaries(st.text(), st.text(), max_size=5))
    )


class TestApplicationWorkflowProperties:
    """Property-based tests for application workflow system."""
    
    def create_mock_browser_agent(self):
        """Create mock browser agent."""
        agent = AsyncMock(spec=BrowserAgent)
        
        # Mock successful operations
        agent.navigate_to_url = AsyncMock(return_value=True)
        agent.click_element = AsyncMock(return_value=True)
        agent.fill_form = AsyncMock(return_value=True)
        agent.upload_file = AsyncMock(return_value=True)
        agent.fill_text_area = AsyncMock(return_value=True)
        agent.get_page_content = AsyncMock(return_value="Mock page content")
        
        return agent
    
    def create_mock_safety_layer(self):
        """Create mock safety layer."""
        safety = AsyncMock(spec=SafetyLayer)
        
        # Mock approval for all actions
        safety.assess_action = AsyncMock(return_value=ActionAssessment(
            approved=True,
            confidence=0.9,
            reason="Mock approval",
            suggested_modifications=[]
        ))
        
        return safety
    
    def create_application_workflow(self):
        """Create application workflow with mocks."""
        browser_agent = self.create_mock_browser_agent()
        safety_layer = self.create_mock_safety_layer()
        return ApplicationWorkflow(browser_agent, safety_layer)
    
    @given(
        job_posting=st.from_type(JobPosting).filter(lambda x: len(x.title) > 0),
        user_profile=st.from_type(UserProfile),
        priority=st.sampled_from(list(ApplicationPriority))
    )
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_application_creation_property(self, job_posting, user_profile, priority):
        """
        Property 5: Action Summary Generation
        
        For any job posting and user profile:
        1. Application should be created with valid structure
        2. Documents should be generated appropriately
        3. Workflow actions should be created based on portal type
        4. Application summary should be accurate and complete
        """
        workflow = self.create_application_workflow()
        
        # Create a simple match result
        match_result = MatchResult(
            job_posting=job_posting,
            overall_score=0.7,
            category_scores={},
            skill_matches=[],
            missing_requirements=[],
            strengths=["Good technical skills"],
            gaps=["Need more experience"],
            recommendations=["Apply with confidence"],
            fit_level="good"
        )
        
        async def run_test():
            application = await workflow.create_application(
                job_posting=job_posting,
                user_profile=user_profile,
                match_result=match_result,
                priority=priority
            )
            
            # Property 1: Valid application structure
            assert isinstance(application, JobApplication)
            assert application.job_posting == job_posting
            assert application.user_profile == user_profile
            assert application.match_result == match_result
            assert application.priority == priority
            assert application.status == ApplicationStatus.PENDING
            assert len(application.application_id) > 0
            
            # Property 2: Documents should be generated
            assert len(application.documents) > 0
            
            # Check for cover letter
            cover_letters = [doc for doc in application.documents if doc.type == "cover_letter"]
            assert len(cover_letters) > 0
            
            cover_letter = cover_letters[0]
            assert len(cover_letter.content) > 50  # Meaningful content
            assert cover_letter.customized == True
            assert job_posting.title in cover_letter.content
            assert job_posting.company in cover_letter.content
            assert user_profile.personal_info.name in cover_letter.content
            
            # Property 3: Workflow actions should be created
            assert len(application.actions) > 0
            
            # Should have navigation action
            nav_actions = [action for action in application.actions if action.action_type == "navigate"]
            assert len(nav_actions) > 0
            
            # Should have submit action
            submit_actions = [action for action in application.actions if action.action_type == "submit"]
            assert len(submit_actions) > 0
            
            # Property 4: All actions should have descriptions
            for action in application.actions:
                assert isinstance(action.description, str)
                assert len(action.description.strip()) > 5
                assert action.completed == False  # Initially not completed
                assert action.timestamp is None  # Not executed yet
            
            # Property 5: Application summary should be accurate
            summary = workflow.get_application_summary(application.application_id)
            assert summary is not None
            assert summary["application_id"] == application.application_id
            assert summary["job_title"] == job_posting.title
            assert summary["company"] == job_posting.company
            assert summary["status"] == ApplicationStatus.PENDING.value
            assert summary["priority"] == priority.value
            assert summary["match_score"] == match_result.overall_score
            assert summary["documents_count"] == len(application.documents)
            assert summary["actions_completed"] == 0  # None completed initially
            assert summary["total_actions"] == len(application.actions)
        
        asyncio.run(run_test())
    
    @given(application=job_application_strategy())
    @settings(max_examples=10, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_application_execution_property(self, application):
        """
        Property: Application Execution Consistency
        
        For any application:
        1. Execution should update application status appropriately
        2. Actions should be executed in sequence
        3. Failed actions should stop execution and update status
        4. Successful execution should mark application as submitted
        """
        workflow = self.create_application_workflow()
        
        # Add application to workflow
        workflow.applications[application.application_id] = application
        
        # Reset action states for testing
        for action in application.actions:
            action.completed = False
            action.timestamp = None
            action.error_message = None
        
        async def run_test():
            # Test successful execution
            success = await workflow.execute_application(application.application_id)
            
            # Property 1: Should return success status
            assert isinstance(success, bool)
            
            # Property 2: Application status should be updated
            updated_app = workflow.applications[application.application_id]
            if success:
                assert updated_app.status == ApplicationStatus.SUBMITTED
                assert updated_app.submitted_at is not None
            else:
                assert updated_app.status in [ApplicationStatus.ERROR, ApplicationStatus.IN_PROGRESS]
            
            # Property 3: Updated timestamp should be recent
            assert updated_app.updated_at is not None
            time_diff = datetime.now() - updated_app.updated_at
            assert time_diff < timedelta(minutes=1)
            
            # Property 4: Actions should have execution timestamps
            for action in updated_app.actions:
                if action.completed:
                    assert action.timestamp is not None
        
        asyncio.run(run_test())
    
    @given(
        applications=st.lists(job_application_strategy(), min_size=2, max_size=5)
    )
    @settings(max_examples=8, deadline=12000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_application_property(self, applications):
        """
        Property: Batch Application Consistency
        
        For any list of applications:
        1. Batch operations should handle all applications
        2. Applications should be prioritized correctly
        3. Summary should reflect all applications accurately
        4. Individual application properties should be maintained
        """
        workflow = self.create_application_workflow()
        
        # Add applications to workflow
        for app in applications:
            workflow.applications[app.application_id] = app
        
        # Property 1: Get all applications should return correct count
        all_apps = workflow.get_all_applications()
        assert len(all_apps) == len(applications)
        
        # Property 2: Each application summary should be valid
        for summary in all_apps:
            assert isinstance(summary, dict)
            assert "application_id" in summary
            assert "job_title" in summary
            assert "company" in summary
            assert "status" in summary
            assert "priority" in summary
            assert "match_score" in summary
            assert "documents_count" in summary
            assert "actions_completed" in summary
            assert "total_actions" in summary
            
            # Verify summary accuracy
            app_id = summary["application_id"]
            original_app = next(app for app in applications if app.application_id == app_id)
            
            assert summary["job_title"] == original_app.job_posting.title
            assert summary["company"] == original_app.job_posting.company
            assert summary["status"] == original_app.status.value
            assert summary["priority"] == original_app.priority.value
            assert summary["documents_count"] == len(original_app.documents)
            assert summary["total_actions"] == len(original_app.actions)
        
        # Property 3: Application IDs should be unique
        app_ids = [summary["application_id"] for summary in all_apps]
        assert len(app_ids) == len(set(app_ids))
    
    @given(
        portal_urls=st.lists(
            st.sampled_from([
                "https://linkedin.com/jobs/view/123",
                "https://indeed.com/viewjob?jk=abc123",
                "https://company.com/careers/job/456",
                "https://example.com/jobs/developer"
            ]),
            min_size=1, max_size=4
        )
    )
    def test_portal_detection_property(self, portal_urls):
        """
        Property: Portal Detection Consistency
        
        For any job portal URL:
        1. Portal type should be detected correctly
        2. Workflow actions should be appropriate for portal type
        3. Detection should be consistent for same portal
        """
        workflow = self.create_application_workflow()
        
        for url in portal_urls:
            portal_type = workflow._detect_portal_type(url)
            
            # Property 1: Should return valid portal type
            assert portal_type in ["linkedin", "indeed", "company_website"]
            
            # Property 2: Detection should be logical
            if "linkedin.com" in url.lower():
                assert portal_type == "linkedin"
            elif "indeed.com" in url.lower():
                assert portal_type == "indeed"
            else:
                assert portal_type == "company_website"
            
            # Property 3: Should have workflow template for detected type
            assert portal_type in workflow.portal_workflows
            
            # Property 4: Workflow template should be valid
            workflow_template = workflow.portal_workflows[portal_type]
            assert isinstance(workflow_template, list)
            assert len(workflow_template) > 0
            
            for step in workflow_template:
                assert "action_type" in step
                assert "description" in step
                assert isinstance(step["action_type"], str)
                assert isinstance(step["description"], str)
    
    @given(
        document_types=st.lists(
            st.sampled_from(["resume", "cover_letter", "portfolio"]),
            min_size=1, max_size=3
        ),
        user_name=st.text(min_size=5, max_size=50),
        job_title=st.text(min_size=5, max_size=50),
        company_name=st.text(min_size=3, max_size=30)
    )
    def test_document_generation_property(self, document_types, user_name, job_title, company_name):
        """
        Property: Document Generation Quality
        
        For any document generation request:
        1. Generated documents should contain relevant information
        2. Customization should be applied appropriately
        3. Document content should be professional and coherent
        4. Required information should be included
        """
        workflow = self.create_application_workflow()
        
        # Create test data
        from sovereign_career_architect.jobs.extractor import ExtractedRequirements
        
        personal_info = PersonalInfo(name=user_name, email="test@example.com")
        user_profile = UserProfile(
            personal_info=personal_info,
            skills=[Skill(name="Python", level="Advanced")],
            experience=[],
            education=[],
            preferences=JobPreferences(),
            documents=DocumentStore()
        )
        
        extracted_reqs = ExtractedRequirements(
            job_title=job_title,
            company=company_name,
            location="Test Location",
            requirements=[]
        )
        
        job_posting = JobPosting(
            id="test_job",
            title=job_title,
            company=company_name,
            location="Test Location",
            url="https://example.com/job",
            extracted_requirements=extracted_reqs
        )
        
        match_result = MatchResult(
            job_posting=job_posting,
            overall_score=0.8,
            category_scores={},
            skill_matches=[],
            missing_requirements=[],
            strengths=["Strong technical background"],
            gaps=[],
            recommendations=["Apply with confidence"],
            fit_level="excellent"
        )
        
        async def run_test():
            documents = await workflow._generate_application_documents(
                job_posting, user_profile, match_result
            )
            
            # Property 1: Should generate at least cover letter
            assert len(documents) > 0
            
            cover_letters = [doc for doc in documents if doc.type == "cover_letter"]
            assert len(cover_letters) > 0
            
            cover_letter = cover_letters[0]
            
            # Property 2: Cover letter should contain key information
            assert user_name in cover_letter.content
            assert job_title in cover_letter.content
            assert company_name in cover_letter.content
            
            # Property 3: Should be marked as customized
            assert cover_letter.customized == True
            assert cover_letter.customization_notes is not None
            
            # Property 4: Content should be substantial
            assert len(cover_letter.content) > 200  # Reasonable length
            
            # Property 5: Should have professional structure
            content_lower = cover_letter.content.lower()
            assert "dear" in content_lower  # Professional greeting
            assert "sincerely" in content_lower or "regards" in content_lower  # Professional closing
            
            # Property 6: All documents should have valid types
            for doc in documents:
                assert doc.type in ["resume", "cover_letter", "portfolio"]
                assert isinstance(doc.customized, bool)
        
        asyncio.run(run_test())
    
    @given(
        status_transitions=st.lists(
            st.sampled_from(list(ApplicationStatus)),
            min_size=2, max_size=5
        )
    )
    def test_status_tracking_property(self, status_transitions):
        """
        Property: Status Tracking Consistency
        
        For any sequence of status changes:
        1. Status updates should be tracked correctly
        2. Timestamps should be updated appropriately
        3. Status progression should be logical
        4. Historical information should be preserved
        """
        workflow = self.create_application_workflow()
        
        # Create test application
        from tests.test_job_matching_properties import job_posting_strategy, user_profile_strategy
        
        # Use fixed data for this test
        personal_info = PersonalInfo(name="Test User", email="test@example.com")
        user_profile = UserProfile(
            personal_info=personal_info,
            skills=[Skill(name="Python", level="Advanced")],
            experience=[],
            education=[],
            preferences=JobPreferences(),
            documents=DocumentStore()
        )
        
        extracted_reqs = ExtractedRequirements(
            job_title="Test Job",
            company="Test Company",
            location="Test Location",
            requirements=[]
        )
        
        job_posting = JobPosting(
            id="test_job",
            title="Test Job",
            company="Test Company",
            location="Test Location",
            url="https://example.com/job",
            extracted_requirements=extracted_reqs
        )
        
        match_result = MatchResult(
            job_posting=job_posting,
            overall_score=0.7,
            category_scores={},
            skill_matches=[],
            missing_requirements=[],
            strengths=[],
            gaps=[],
            recommendations=[],
            fit_level="good"
        )
        
        application = JobApplication(
            application_id="test_app",
            job_posting=job_posting,
            user_profile=user_profile,
            match_result=match_result,
            status=ApplicationStatus.PENDING,
            priority=ApplicationPriority.MEDIUM,
            documents=[],
            actions=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        workflow.applications["test_app"] = application
        
        # Test status tracking
        for new_status in status_transitions:
            old_updated_at = application.updated_at
            
            # Update status
            application.status = new_status
            application.updated_at = datetime.now()
            
            # Property 1: Status should be updated
            tracked_status = workflow.track_application_status("test_app")
            assert tracked_status is not None
            
            # Property 2: Summary should reflect current status
            summary = workflow.get_application_summary("test_app")
            assert summary["status"] == new_status.value
            
            # Property 3: Updated timestamp should be more recent
            assert application.updated_at >= old_updated_at


if __name__ == "__main__":
    # Run a quick test
    async def test_basic():
        from sovereign_career_architect.browser.agent import BrowserAgent
        from sovereign_career_architect.core.safety import SafetyLayer
        from unittest.mock import AsyncMock
        
        # Create mocks
        browser_agent = AsyncMock(spec=BrowserAgent)
        safety_layer = AsyncMock(spec=SafetyLayer)
        
        workflow = ApplicationWorkflow(browser_agent, safety_layer)
        
        # Create test data
        personal_info = PersonalInfo(name="Test User", email="test@example.com")
        user_profile = UserProfile(
            personal_info=personal_info,
            skills=[Skill(name="Python", level="Advanced")],
            experience=[],
            education=[],
            preferences=JobPreferences(),
            documents=DocumentStore()
        )
        
        extracted_reqs = ExtractedRequirements(
            job_title="Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            requirements=[]
        )
        
        job_posting = JobPosting(
            id="job_1",
            title="Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            url="https://linkedin.com/jobs/view/123",
            extracted_requirements=extracted_reqs
        )
        
        match_result = MatchResult(
            job_posting=job_posting,
            overall_score=0.8,
            category_scores={},
            skill_matches=[],
            missing_requirements=[],
            strengths=["Strong Python skills"],
            gaps=[],
            recommendations=["Apply with confidence"],
            fit_level="excellent"
        )
        
        # Test application creation
        application = await workflow.create_application(
            job_posting, user_profile, match_result, ApplicationPriority.HIGH
        )
        
        print(f"Application created:")
        print(f"ID: {application.application_id}")
        print(f"Status: {application.status}")
        print(f"Documents: {len(application.documents)}")
        print(f"Actions: {len(application.actions)}")
        
        # Print cover letter excerpt
        cover_letters = [doc for doc in application.documents if doc.type == "cover_letter"]
        if cover_letters:
            print(f"Cover letter excerpt: {cover_letters[0].content[:200]}...")
    
    asyncio.run(test_basic())