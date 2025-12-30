"""Property-based tests for Reviewer Node output validation."""

from typing import Dict, Any
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import AsyncMock

from sovereign_career_architect.core.nodes.reviewer import ReviewerNode, ReviewDecision
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionResult,
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
    JobPreferences,
)


class MockReviewResponse:
    """Mock review model response for property testing."""
    def __init__(self, content: str):
        self.content = content


# Hypothesis strategies for generating test data
@st.composite
def action_result_strategy(draw):
    """Generate realistic action results for testing."""
    action_type = draw(st.sampled_from(ActionType))
    success = draw(st.booleans())
    
    # Generate realistic data based on action type
    if action_type == ActionType.JOB_SEARCH:
        if success:
            data = {
                "jobs_found": draw(st.integers(min_value=0, max_value=50)),
                "jobs": draw(st.lists(
                    st.fixed_dictionaries({
                        "title": st.text(min_size=5, max_size=30),
                        "company": st.text(min_size=3, max_size=20),
                        "location": st.sampled_from(["Remote", "San Francisco", "New York", "Seattle"])
                    }),
                    min_size=0, max_size=5
                ))
            }
        else:
            data = {"error_details": "Search service unavailable"}
    
    elif action_type == ActionType.APPLICATION_SUBMIT:
        if success:
            data = {
                "applications_submitted": draw(st.integers(min_value=1, max_value=10)),
                "confirmation_ids": draw(st.lists(st.text(min_size=5, max_size=15), min_size=1, max_size=5))
            }
        else:
            data = {"failed_applications": draw(st.integers(min_value=1, max_value=5))}
    
    else:
        # Generic data for other action types
        data = draw(st.dictionaries(
            st.text(min_size=1, max_size=10), 
            st.one_of(st.text(max_size=20), st.integers(), st.booleans()),
            min_size=0, max_size=3
        ))
    
    error_message = None if success else draw(st.one_of(st.none(), st.text(min_size=5, max_size=50)))
    
    return ActionResult(
        action_type=action_type,
        success=success,
        data=data,
        error_message=error_message
    )


@st.composite
def review_response_strategy(draw):
    """Generate realistic review responses."""
    decisions = ["APPROVE", "REJECT", "NEEDS_IMPROVEMENT"]
    decision = draw(st.sampled_from(decisions))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    feedback_templates = {
        "APPROVE": [
            "Results are excellent and match all criteria",
            "Output quality is high and relevant to user needs",
            "Perfect match for user preferences and requirements"
        ],
        "REJECT": [
            "Results do not match user requirements",
            "Output quality is poor and needs improvement",
            "Completely irrelevant to user's needs and preferences"
        ],
        "NEEDS_IMPROVEMENT": [
            "Results are acceptable but could be enhanced",
            "Good output but missing some important details",
            "Mostly relevant but needs minor improvements"
        ]
    }
    
    feedback = draw(st.sampled_from(feedback_templates[decision]))
    
    return f"""
DECISION: {decision}
CONFIDENCE: {confidence:.2f}
FEEDBACK: {feedback}
    """.strip()


@st.composite
def execution_plan_strategy(draw):
    """Generate realistic execution plans."""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = []
    
    for i in range(num_steps):
        action_type = draw(st.sampled_from(ActionType))
        description = draw(st.text(min_size=10, max_size=50))
        steps.append(PlanStep(description=description, action_type=action_type))
    
    return ExecutionPlan(steps=steps, user_id=uuid4())


class TestReviewerNodeProperties:
    """Property-based tests for ReviewerNode validation consistency."""
    
    @pytest.fixture
    def mock_review_model(self):
        """Create a mock review model that returns consistent responses."""
        return AsyncMock()
    
    @pytest.fixture
    def reviewer_node(self, mock_review_model):
        """Create a ReviewerNode with mocked model."""
        return ReviewerNode(review_model=mock_review_model)
    
    # Feature: sovereign-career-architect, Property 7: Output Validation Reliability
    @given(
        action_result=action_result_strategy(),
        review_response=review_response_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_output_validation_reliability(
        self,
        reviewer_node,
        mock_review_model,
        action_result: ActionResult,
        review_response: str
    ):
        """
        For any action result and review response, the reviewer should
        validate the output and return a consistent decision structure.
        
        **Validates: Requirements 2.3**
        """
        # Setup mock response
        mock_review_model.ainvoke.return_value = MockReviewResponse(review_response)
        
        # Create state with the action result
        state = AgentState()
        state.add_tool_output(action_result)
        
        # Add a simple plan for context
        plan = ExecutionPlan(
            steps=[PlanStep(description="Test step", action_type=action_result.action_type)],
            user_id=state.user_id
        )
        state.current_plan = plan
        state.current_step_index = 0
        
        # Execute reviewer
        result = await reviewer_node(state)
        
        # Verify consistent result structure
        assert isinstance(result, dict)
        assert "critique" in result
        assert "next_node" in result
        
        # Next node should be valid
        valid_next_nodes = ["archivist", "planner"]
        assert result["next_node"] in valid_next_nodes
        
        # Critique should be a non-empty string
        assert isinstance(result["critique"], str)
        assert len(result["critique"].strip()) > 0
    
    @given(
        action_results=st.lists(action_result_strategy(), min_size=1, max_size=3),
        review_response=review_response_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_validation_consistency_with_multiple_outputs(
        self,
        reviewer_node,
        mock_review_model,
        action_results: list,
        review_response: str
    ):
        """
        For any sequence of action results, the reviewer should
        consistently validate the latest output.
        """
        # Reset mock for this test example
        mock_review_model.reset_mock()
        
        # Setup mock response
        mock_review_model.ainvoke.return_value = MockReviewResponse(review_response)
        
        # Create state with multiple outputs
        state = AgentState()
        for result in action_results:
            state.add_tool_output(result)
        
        # Execute reviewer
        result = await reviewer_node(state)
        
        # Should have reviewed the latest output
        assert isinstance(result, dict)
        assert "critique" in result
        assert "next_node" in result
        
        # Model should have been called exactly once (for latest output)
        mock_review_model.ainvoke.assert_called_once()
    
    @given(review_response_text=st.text(min_size=0, max_size=100))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=15
    )
    def test_review_response_parsing_robustness(
        self,
        reviewer_node,
        review_response_text: str
    ):
        """
        For any review response text, parsing should be robust
        and always return a valid decision structure.
        """
        result = reviewer_node._parse_review_response(review_response_text)
        
        # Should always return a valid structure
        assert isinstance(result, dict)
        assert "decision" in result
        assert "feedback" in result
        assert "confidence" in result
        
        # Decision should be a valid ReviewDecision
        assert isinstance(result["decision"], ReviewDecision)
        
        # Confidence should be a valid float between 0 and 1
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Feedback should be a string
        assert isinstance(result["feedback"], str)
    
    @given(
        action_result=action_result_strategy(),
        execution_plan=st.one_of(st.none(), execution_plan_strategy())
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    def test_review_prompt_building_consistency(
        self,
        reviewer_node,
        action_result: ActionResult,
        execution_plan: ExecutionPlan
    ):
        """
        For any action result and execution plan, prompt building
        should be consistent and include all necessary context.
        """
        prompt = reviewer_node._build_review_user_prompt(
            action_result, None, execution_plan, 0
        )
        
        # Should always return a string
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Should contain key sections
        assert "OUTPUT TO REVIEW:" in prompt
        assert action_result.action_type.value in prompt
        assert f"Success: {action_result.success}" in prompt
        
        # If plan exists, should include current step info
        if execution_plan and execution_plan.steps:
            assert "CURRENT STEP:" in prompt
    
    @given(
        retry_count=st.integers(min_value=0, max_value=10),
        max_retries=st.integers(min_value=1, max_value=5)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    def test_retry_logic_consistency(
        self,
        reviewer_node,
        retry_count: int,
        max_retries: int
    ):
        """
        For any retry count and max retries configuration,
        the retry logic should behave consistently.
        """
        # Create mock review result that would normally cause rejection
        review_result = {
            "decision": ReviewDecision.REJECT,
            "feedback": "Test rejection"
        }
        
        # Create state with specified retry configuration
        state = AgentState()
        state.retry_count = retry_count
        state.max_retries = max_retries
        
        next_node, critique = reviewer_node._determine_next_action(review_result, state)
        
        # Logic should be consistent
        if retry_count >= max_retries:
            # Should force approval when max retries exceeded
            assert next_node == "archivist"
        else:
            # Should reject and go back to planner
            assert next_node == "planner"
            assert "rejected" in critique.lower()
        
        # Critique should always be a non-empty string
        assert isinstance(critique, str)
        assert len(critique.strip()) > 0
    
    @given(
        decision=st.sampled_from([ReviewDecision.APPROVE, ReviewDecision.REJECT, ReviewDecision.NEEDS_IMPROVEMENT]),
        feedback=st.text(min_size=1, max_size=100)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=15
    )
    def test_next_action_determination_consistency(
        self,
        reviewer_node,
        decision: ReviewDecision,
        feedback: str
    ):
        """
        For any review decision and feedback, next action determination
        should be consistent and follow the expected routing logic.
        """
        review_result = {
            "decision": decision,
            "feedback": feedback
        }
        state = AgentState()
        
        next_node, critique = reviewer_node._determine_next_action(review_result, state)
        
        # Routing should be consistent with decision
        if decision == ReviewDecision.APPROVE:
            assert next_node == "archivist"
            assert "approved" in critique.lower()
        elif decision == ReviewDecision.REJECT:
            assert next_node == "planner"
            assert "rejected" in critique.lower()
        elif decision == ReviewDecision.NEEDS_IMPROVEMENT:
            assert next_node == "archivist"
            assert "accepted with notes" in critique.lower()
        
        # Critique should contain the original feedback
        assert feedback in critique
    
    @given(action_result=action_result_strategy())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    def test_skip_review_logic_consistency(
        self,
        reviewer_node,
        action_result: ActionResult
    ):
        """
        For any action result, skip review logic should be
        consistent and deterministic.
        """
        should_skip = reviewer_node._should_skip_review(action_result)
        
        # Should return a boolean
        assert isinstance(should_skip, bool)
        
        # Logic should be consistent with action type and success
        if action_result.action_type == ActionType.GENERAL_CHAT and action_result.success:
            assert should_skip is True
        elif action_result.action_type == ActionType.PROFILE_UPDATE and action_result.success:
            assert should_skip is True
        else:
            # Most other cases should not skip review
            assert should_skip is False


class TestReviewerNodeEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def reviewer_node_no_model(self):
        """Create a ReviewerNode without a model to test error handling."""
        return ReviewerNode(review_model=None)
    
    @pytest.mark.asyncio
    async def test_reviewer_with_no_model(self, reviewer_node_no_model):
        """Test reviewer behavior when no model is available."""
        state = AgentState()
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs": []}
        ))
        
        # Should handle gracefully
        result = await reviewer_node_no_model(state)
        
        assert isinstance(result, dict)
        assert result["next_node"] == "archivist"  # Should default to approval
        assert "no review model" in result["critique"].lower()
    
    def test_parse_empty_review_response(self):
        """Test parsing of empty or minimal review responses."""
        reviewer = ReviewerNode(review_model=AsyncMock())
        
        # Test empty string
        result = reviewer._parse_review_response("")
        assert result["decision"] == ReviewDecision.APPROVE  # Default
        
        # Test whitespace only
        result = reviewer._parse_review_response("   \n\t  ")
        assert result["decision"] == ReviewDecision.APPROVE
        
        # Test partial response
        result = reviewer._parse_review_response("DECISION: REJECT")
        assert result["decision"] == ReviewDecision.REJECT
    
    def test_confidence_value_clamping(self):
        """Test that confidence values are properly clamped to [0, 1] range."""
        reviewer = ReviewerNode(review_model=AsyncMock())
        
        # Test out-of-range values
        test_cases = [
            "CONFIDENCE: -0.5",  # Below 0
            "CONFIDENCE: 1.5",   # Above 1
            "CONFIDENCE: 999",   # Way above 1
            "CONFIDENCE: invalid", # Invalid format
        ]
        
        for test_case in test_cases:
            result = reviewer._parse_review_response(test_case)
            assert 0.0 <= result["confidence"] <= 1.0