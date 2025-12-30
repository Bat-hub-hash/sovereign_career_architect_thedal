"""Tests for the Reviewer Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from datetime import datetime

from sovereign_career_architect.core.nodes.reviewer import ReviewerNode, ReviewDecision, create_reviewer_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionResult,
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
    JobPreferences,
    Skill,
)


class MockReviewResponse:
    """Mock review model response for testing."""
    def __init__(self, content: str):
        self.content = content


class TestReviewerNode:
    """Test cases for ReviewerNode functionality."""
    
    @pytest.fixture
    def mock_review_model(self):
        """Create a mock review model for testing."""
        model = AsyncMock()
        return model
    
    @pytest.fixture
    def reviewer_node(self, mock_review_model):
        """Create a ReviewerNode with mocked dependencies."""
        return ReviewerNode(review_model=mock_review_model)
    
    @pytest.fixture
    def sample_state_with_output(self):
        """Create a sample agent state with tool output for testing."""
        state = AgentState()
        
        # Add a successful job search result
        job_search_result = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={
                "jobs_found": 5,
                "jobs": [
                    {"title": "Python Developer", "company": "Tech Corp", "location": "San Francisco"},
                    {"title": "Backend Engineer", "company": "StartupXYZ", "location": "Remote"}
                ]
            }
        )
        state.add_tool_output(job_search_result)
        
        # Add execution plan
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for Python developer roles",
                    action_type=ActionType.JOB_SEARCH
                ),
                PlanStep(
                    description="Apply to top matches",
                    action_type=ActionType.APPLICATION_SUBMIT
                )
            ],
            user_id=state.user_id
        )
        state.current_plan = plan
        state.current_step_index = 0
        
        return state
    
    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile for testing."""
        return UserProfile(
            personal_info=PersonalInfo(
                name="Jane Developer",
                email="jane@example.com",
                location="San Francisco, CA"
            ),
            skills=[
                Skill(name="Python", level="Expert", years_experience=5),
                Skill(name="Django", level="Advanced", years_experience=3)
            ],
            preferences=JobPreferences(
                roles=["Python Developer", "Backend Engineer"],
                locations=["San Francisco", "Remote"],
                work_arrangements=["Remote", "Hybrid"]
            )
        )
    
    @pytest.mark.asyncio
    async def test_reviewer_node_approve_output(self, reviewer_node, sample_state_with_output, mock_review_model):
        """Test reviewer node approving a good output."""
        # Setup mock to approve
        mock_review_model.ainvoke.return_value = MockReviewResponse("""
DECISION: APPROVE
CONFIDENCE: 0.9
FEEDBACK: Job search results are relevant and match user preferences for Python roles.
        """)
        
        # Execute reviewer node
        result = await reviewer_node(sample_state_with_output)
        
        # Verify approval
        assert isinstance(result, dict)
        assert "critique" in result
        assert "next_node" in result
        assert result["next_node"] == "archivist"
        assert "approved" in result["critique"].lower()
        
        # Verify model was called
        mock_review_model.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reviewer_node_reject_output(self, reviewer_node, sample_state_with_output, mock_review_model):
        """Test reviewer node rejecting a poor output."""
        # Setup mock to reject
        mock_review_model.ainvoke.return_value = MockReviewResponse("""
DECISION: REJECT
CONFIDENCE: 0.8
FEEDBACK: Job search results do not match user's location preferences and skill level.
        """)
        
        # Execute reviewer node
        result = await reviewer_node(sample_state_with_output)
        
        # Verify rejection
        assert result["next_node"] == "planner"  # Should go back to planning
        assert "rejected" in result["critique"].lower()
        
        # Verify retry count was incremented
        assert sample_state_with_output.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_reviewer_node_needs_improvement(self, reviewer_node, sample_state_with_output, mock_review_model):
        """Test reviewer node with needs improvement decision."""
        # Setup mock for needs improvement
        mock_review_model.ainvoke.return_value = MockReviewResponse("""
DECISION: NEEDS_IMPROVEMENT
CONFIDENCE: 0.6
FEEDBACK: Results are acceptable but could include more senior-level positions.
        """)
        
        # Execute reviewer node
        result = await reviewer_node(sample_state_with_output)
        
        # Verify acceptance with notes
        assert result["next_node"] == "archivist"  # Should proceed
        assert "accepted with notes" in result["critique"].lower()
        
        # Retry count should be reset (not incremented for minor issues)
        assert sample_state_with_output.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_reviewer_node_no_outputs(self, reviewer_node, mock_review_model):
        """Test reviewer node with no tool outputs to review."""
        # Create state without outputs
        state = AgentState()
        
        # Execute reviewer node
        result = await reviewer_node(state)
        
        # Should skip to archivist
        assert result["next_node"] == "archivist"
        
        # Model should not be called
        mock_review_model.ainvoke.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_reviewer_node_max_retries_exceeded(self, reviewer_node, sample_state_with_output, mock_review_model):
        """Test reviewer node behavior when max retries are exceeded."""
        # Set state to max retries
        sample_state_with_output.retry_count = sample_state_with_output.max_retries
        
        # Setup mock to reject (but should be overridden)
        mock_review_model.ainvoke.return_value = MockReviewResponse("""
DECISION: REJECT
CONFIDENCE: 0.9
FEEDBACK: Still not good enough.
        """)
        
        # Execute reviewer node
        result = await reviewer_node(sample_state_with_output)
        
        # Should force approval due to max retries
        assert result["next_node"] == "archivist"
        assert "max retries exceeded" in result["critique"].lower()
    
    @pytest.mark.asyncio
    async def test_reviewer_node_error_handling(self, reviewer_node, sample_state_with_output, mock_review_model):
        """Test error handling in reviewer node."""
        # Setup mock to raise exception
        mock_review_model.ainvoke.side_effect = Exception("Review service unavailable")
        
        # Execute reviewer node
        result = await reviewer_node(sample_state_with_output)
        
        # Should default to approval on error
        assert result["next_node"] == "archivist"
        assert "review failed" in result["critique"].lower()
    
    def test_parse_review_response_approve(self, reviewer_node):
        """Test parsing of approval response."""
        response_text = """
DECISION: APPROVE
CONFIDENCE: 0.85
FEEDBACK: Excellent job search results that match all user criteria.
        """
        
        result = reviewer_node._parse_review_response(response_text)
        
        assert result["decision"] == ReviewDecision.APPROVE
        assert result["confidence"] == 0.85
        assert "Excellent job search" in result["feedback"]
    
    def test_parse_review_response_reject(self, reviewer_node):
        """Test parsing of rejection response."""
        response_text = """
DECISION: REJECT
CONFIDENCE: 0.9
FEEDBACK: Results are completely irrelevant to user needs.
        """
        
        result = reviewer_node._parse_review_response(response_text)
        
        assert result["decision"] == ReviewDecision.REJECT
        assert result["confidence"] == 0.9
        assert "completely irrelevant" in result["feedback"]
    
    def test_parse_review_response_malformed(self, reviewer_node):
        """Test parsing of malformed response."""
        response_text = "This is not a properly formatted review response."
        
        result = reviewer_node._parse_review_response(response_text)
        
        # Should default to approve with original text as feedback
        assert result["decision"] == ReviewDecision.APPROVE
        assert result["feedback"] == response_text
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_determine_next_action_approve(self, reviewer_node):
        """Test next action determination for approval."""
        review_result = {
            "decision": ReviewDecision.APPROVE,
            "feedback": "Great results!"
        }
        state = AgentState()
        
        next_node, critique = reviewer_node._determine_next_action(review_result, state)
        
        assert next_node == "archivist"
        assert "approved" in critique.lower()
    
    def test_determine_next_action_reject(self, reviewer_node):
        """Test next action determination for rejection."""
        review_result = {
            "decision": ReviewDecision.REJECT,
            "feedback": "Poor quality results"
        }
        state = AgentState()
        
        next_node, critique = reviewer_node._determine_next_action(review_result, state)
        
        assert next_node == "planner"
        assert "rejected" in critique.lower()
        assert "regenerate" in critique.lower()
    
    def test_build_review_system_prompt(self, reviewer_node):
        """Test system prompt building for reviews."""
        prompt = reviewer_node._build_review_system_prompt()
        
        assert "Reviewer component" in prompt
        assert "EVALUATION CRITERIA" in prompt
        assert "APPROVE" in prompt
        assert "REJECT" in prompt
        assert "NEEDS_IMPROVEMENT" in prompt
        assert "OUTPUT FORMAT" in prompt
    
    def test_build_review_user_prompt(self, reviewer_node, sample_user_profile):
        """Test user prompt building for review."""
        output = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 3, "top_match": "Python Developer at TechCorp"}
        )
        
        plan = ExecutionPlan(
            steps=[PlanStep(description="Search for Python roles", action_type=ActionType.JOB_SEARCH)],
            user_id=uuid4()
        )
        
        prompt = reviewer_node._build_review_user_prompt(
            output, sample_user_profile, plan, 0
        )
        
        assert "OUTPUT TO REVIEW:" in prompt
        assert "job_search" in prompt  # Now lowercase
        assert "Success: True" in prompt
        assert "jobs_found: 3" in prompt
        assert "CURRENT STEP:" in prompt
        assert "Search for Python roles" in prompt
        assert "USER CONTEXT:" in prompt
        assert "Python Developer" in prompt
    
    def test_build_review_user_prompt_with_error(self, reviewer_node):
        """Test user prompt building with error output."""
        output = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=False,
            error_message="API rate limit exceeded"
        )
        
        prompt = reviewer_node._build_review_user_prompt(output, None, None, 0)
        
        assert "Success: False" in prompt
        assert "Error: API rate limit exceeded" in prompt
    
    def test_summarize_user_profile_for_review(self, reviewer_node, sample_user_profile):
        """Test user profile summarization for review context."""
        summary = reviewer_node._summarize_user_profile_for_review(sample_user_profile)
        
        assert "Python Developer" in summary
        assert "Backend Engineer" in summary
        assert "San Francisco" in summary
        assert "Remote" in summary
        assert "Python" in summary
        assert "Expert" in summary
    
    def test_summarize_user_profile_minimal(self, reviewer_node):
        """Test profile summarization with minimal data."""
        minimal_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com")
        )
        
        summary = reviewer_node._summarize_user_profile_for_review(minimal_profile)
        
        assert "No specific preferences available" in summary
    
    def test_should_skip_review(self, reviewer_node):
        """Test review skipping logic."""
        # General chat should be skipped
        chat_output = ActionResult(
            action_type=ActionType.GENERAL_CHAT,
            success=True,
            data={"response": "Hello!"}
        )
        assert reviewer_node._should_skip_review(chat_output) is True
        
        # Profile update should be skipped
        profile_output = ActionResult(
            action_type=ActionType.PROFILE_UPDATE,
            success=True,
            data={"updated_fields": ["skills"]}
        )
        assert reviewer_node._should_skip_review(profile_output) is True
        
        # Job search should not be skipped
        job_output = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        )
        assert reviewer_node._should_skip_review(job_output) is False
    
    def test_create_reviewer_node_factory(self):
        """Test the factory function for creating reviewer nodes."""
        # Should create a reviewer node even without API keys (for testing)
        reviewer = create_reviewer_node()
        assert isinstance(reviewer, ReviewerNode)
        # Model should be None when no API keys are available
        assert reviewer.review_model is None
    
    def test_create_reviewer_node_with_custom_model(self, mock_review_model):
        """Test factory function with custom review model."""
        node = create_reviewer_node(review_model=mock_review_model)
        
        assert isinstance(node, ReviewerNode)
        assert node.review_model == mock_review_model


class TestReviewerNodeIntegration:
    """Integration tests for ReviewerNode."""
    
    @pytest.mark.asyncio
    async def test_complete_review_workflow(self):
        """Test complete review workflow with realistic data."""
        # Create mock model
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = MockReviewResponse("""
DECISION: APPROVE
CONFIDENCE: 0.85
FEEDBACK: Job search results are highly relevant to user's Python expertise and location preferences.
        """)
        
        # Create reviewer with mock model
        reviewer = ReviewerNode(review_model=mock_model)
        
        # Create realistic state
        state = AgentState()
        
        # Add user profile
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Alice", email="alice@example.com"),
            skills=[Skill(name="Python", level="Expert", years_experience=5)],
            preferences=JobPreferences(
                roles=["Python Developer"],
                locations=["Remote"],
                work_arrangements=["Remote"]
            )
        )
        
        # Add execution plan
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for remote Python roles", action_type=ActionType.JOB_SEARCH)
            ],
            user_id=state.user_id
        )
        
        # Add tool output to review
        job_result = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={
                "jobs_found": 8,
                "matches": [
                    {"title": "Senior Python Developer", "company": "RemoteTech", "location": "Remote"},
                    {"title": "Python Backend Engineer", "company": "CloudCorp", "location": "Remote"}
                ]
            }
        )
        state.add_tool_output(job_result)
        
        # Execute reviewer
        result = await reviewer(state)
        
        # Verify comprehensive results
        assert result["next_node"] == "archivist"
        assert "approved" in result["critique"].lower()
        
        # Verify model was called with proper context
        mock_model.ainvoke.assert_called_once()
        call_args = mock_model.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System + User message
        
        user_prompt = call_args[1].content
        assert "job_search" in user_prompt  # Now lowercase
        assert "jobs_found: 8" in user_prompt
        assert "Python Developer" in user_prompt
        assert "Remote" in user_prompt