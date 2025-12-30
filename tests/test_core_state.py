"""Property-based tests for core state management."""

from datetime import datetime
from uuid import UUID, uuid4
from typing import Any, Dict, List

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from langchain_core.messages import HumanMessage, AIMessage

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


# Hypothesis strategies for generating test data
@st.composite
def user_profile_strategy(draw):
    """Generate a valid UserProfile for testing."""
    return UserProfile(
        personal_info=PersonalInfo(
            name=draw(st.text(min_size=1, max_size=20)),
            email=draw(st.emails()),
            preferred_language=draw(st.sampled_from(["en", "hi", "ta"]))
        ),
        preferences=JobPreferences(
            roles=draw(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=3))
        )
    )


@st.composite
def action_result_strategy(draw):
    """Generate a valid ActionResult for testing."""
    return ActionResult(
        action_type=draw(st.sampled_from(ActionType)),
        success=draw(st.booleans()),
        data=draw(st.dictionaries(st.text(max_size=10), st.text(max_size=10), min_size=0, max_size=2)),
        error_message=draw(st.one_of(st.none(), st.text(min_size=1, max_size=20)))
    )


@st.composite
def plan_step_strategy(draw):
    """Generate a valid PlanStep for testing."""
    return PlanStep(
        description=draw(st.text(min_size=1, max_size=20)),
        action_type=draw(st.sampled_from(ActionType)),
        parameters=draw(st.dictionaries(st.text(max_size=5), st.text(max_size=5), min_size=0, max_size=2))
    )


@st.composite
def execution_plan_strategy(draw):
    """Generate a valid ExecutionPlan for testing."""
    steps = draw(st.lists(plan_step_strategy(), min_size=1, max_size=3))
    return ExecutionPlan(
        steps=steps,
        user_id=uuid4()
    )


class TestAgentStateProperties:
    """Property-based tests for AgentState management."""
    
    # Feature: sovereign-career-architect, Property 6: Plan Structure Consistency
    @given(execution_plan_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    def test_plan_structure_consistency(self, plan: ExecutionPlan):
        """
        For any execution plan, when assigned to agent state, 
        the plan should maintain its structure and step ordering.
        
        **Validates: Requirements 2.2**
        """
        state = AgentState()
        state.current_plan = plan
        
        # Plan should be preserved exactly
        assert state.current_plan is not None
        assert state.current_plan.id == plan.id
        assert len(state.current_plan.steps) == len(plan.steps)
        
        # Step ordering should be preserved
        for i, step in enumerate(plan.steps):
            assert state.current_plan.steps[i].id == step.id
            assert state.current_plan.steps[i].description == step.description
            assert state.current_plan.steps[i].action_type == step.action_type
    
    @given(st.lists(action_result_strategy(), min_size=0, max_size=5))
    @settings(max_examples=10)
    def test_tool_output_accumulation(self, results: List[ActionResult]):
        """
        For any sequence of action results, when added to agent state,
        all results should be preserved in order.
        """
        state = AgentState()
        
        for result in results:
            state.add_tool_output(result)
        
        assert len(state.tool_outputs) == len(results)
        
        # Results should be in the same order
        for i, original_result in enumerate(results):
            stored_result = state.tool_outputs[i]
            assert stored_result.action_type == original_result.action_type
            assert stored_result.success == original_result.success
            assert stored_result.data == original_result.data
    
    @given(st.integers(min_value=0, max_value=10))
    @settings(max_examples=10)
    def test_retry_counter_behavior(self, retry_attempts: int):
        """
        For any number of retry attempts, the retry counter should
        accurately track attempts and correctly identify when max is exceeded.
        """
        state = AgentState(max_retries=3)
        
        exceeded_count = 0
        for _ in range(retry_attempts):
            exceeded = state.increment_retry()
            if exceeded:
                exceeded_count += 1
        
        # Should only exceed once we hit max_retries
        if retry_attempts >= 3:
            assert exceeded_count > 0
            assert state.retry_count == retry_attempts
        else:
            assert exceeded_count == 0
            assert state.retry_count == retry_attempts
    
    @given(execution_plan_strategy())
    @settings(max_examples=10)
    def test_step_advancement_consistency(self, plan: ExecutionPlan):
        """
        For any execution plan, step advancement should correctly
        track progress and identify plan completion.
        """
        state = AgentState()
        state.current_plan = plan
        
        steps_advanced = 0
        while state.advance_step():
            steps_advanced += 1
            current_step = state.get_current_step()
            
            # Should be able to get current step while in bounds
            if state.current_step_index < len(plan.steps):
                assert current_step is not None
                assert current_step.id == plan.steps[state.current_step_index].id
        
        # Should advance through all steps exactly once
        assert steps_advanced == len(plan.steps) - 1  # -1 because we start at index 0
        assert state.current_step_index == len(plan.steps)
        assert state.get_current_step() is None
    
    @given(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=20))
    @settings(max_examples=10)
    def test_interrupt_state_management(self, reason: str, action_summary: str):
        """
        For any interrupt reason and action summary, interrupt state
        should be correctly managed through its lifecycle.
        """
        state = AgentState()
        
        # Initially not interrupted
        assert not state.is_interrupted()
        assert state.interrupt_state is None
        
        # Set interrupt
        state.set_interrupt(reason, action_summary)
        
        # Should be interrupted with correct data
        assert state.is_interrupted()
        assert state.interrupt_state is not None
        assert state.interrupt_state.reason == reason
        assert state.interrupt_state.action_summary == action_summary
        assert state.interrupt_state.requires_approval is True
        assert state.interrupt_state.approved is None
        
        # Clear interrupt
        state.clear_interrupt()
        
        # Should no longer be interrupted
        assert not state.is_interrupted()
        assert state.interrupt_state is None
    
    @given(user_profile_strategy())
    @settings(max_examples=10)
    def test_user_profile_preservation(self, profile: UserProfile):
        """
        For any user profile, when assigned to agent state,
        all profile data should be preserved accurately.
        """
        state = AgentState()
        state.user_profile = profile
        
        assert state.user_profile is not None
        assert state.user_profile.user_id == profile.user_id
        assert state.user_profile.personal_info.name == profile.personal_info.name
        assert state.user_profile.personal_info.email == profile.personal_info.email
        assert state.user_profile.preferences.roles == profile.preferences.roles
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    @settings(max_examples=10)
    def test_message_accumulation(self, message_contents: List[str]):
        """
        For any sequence of messages, the state should accumulate
        them in the correct order.
        """
        state = AgentState()
        
        for i, content in enumerate(message_contents):
            if i % 2 == 0:
                message = HumanMessage(content=content)
            else:
                message = AIMessage(content=content)
            
            # Simulate message addition (normally done by LangGraph)
            state.messages.append(message)
        
        assert len(state.messages) == len(message_contents)
        
        for i, original_content in enumerate(message_contents):
            assert state.messages[i].content == original_content
    
    def test_context_summary_completeness(self):
        """
        For any agent state, the context summary should include
        all essential state information.
        """
        state = AgentState()
        summary = state.get_context_summary()
        
        # Required fields should always be present
        required_fields = [
            "user_id", "session_id", "messages_count", "has_user_profile",
            "has_current_plan", "current_step_index", "tool_outputs_count",
            "retry_count", "is_interrupted", "voice_active", "language",
            "browser_active"
        ]
        
        for field in required_fields:
            assert field in summary
        
        # Values should match state
        assert summary["messages_count"] == len(state.messages)
        assert summary["has_user_profile"] == (state.user_profile is not None)
        assert summary["has_current_plan"] == (state.current_plan is not None)
        assert summary["retry_count"] == state.retry_count
        assert summary["is_interrupted"] == state.is_interrupted()


# Integration test for state transitions
class TestStateTransitions:
    """Test state transitions and workflows."""
    
    def test_complete_workflow_state_transitions(self):
        """Test a complete workflow through state transitions."""
        state = AgentState()
        
        # Start with empty state
        assert len(state.messages) == 0
        assert state.user_profile is None
        assert state.current_plan is None
        
        # Add user profile
        profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.user_profile = profile
        
        # Create and assign plan
        plan = ExecutionPlan(
            steps=[
                PlanStep(description="Step 1", action_type=ActionType.JOB_SEARCH),
                PlanStep(description="Step 2", action_type=ActionType.APPLICATION_SUBMIT),
            ],
            user_id=profile.user_id
        )
        state.current_plan = plan
        
        # Execute steps
        assert state.get_current_step() is not None
        assert state.advance_step()
        assert state.get_current_step() is not None
        assert not state.advance_step()  # Should be complete
        
        # Add tool outputs
        result = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        )
        state.add_tool_output(result)
        
        # Verify final state
        summary = state.get_context_summary()
        assert summary["has_user_profile"] is True
        assert summary["has_current_plan"] is True
        assert summary["tool_outputs_count"] == 1