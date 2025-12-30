"""Property-based tests for Supervisor Node self-correction loops."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import AsyncMock

from sovereign_career_architect.core.nodes.supervisor import create_supervisor_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionResult,
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
)
from langchain_core.messages import HumanMessage, AIMessage


# Feature: sovereign-career-architect, Property 8: Self-Correction Loop Activation
@pytest.mark.property
class TestSupervisorSelfCorrectionProperties:
    """Property-based tests for supervisor self-correction behavior."""
    
    @given(
        retry_count=st.integers(min_value=0, max_value=5),
        critique_satisfactory=st.booleans(),
        has_tool_outputs=st.booleans(),
        step_index=st.integers(min_value=0, max_value=3),
        total_steps=st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_self_correction_loop_activation(
        self,
        retry_count: int,
        critique_satisfactory: bool,
        has_tool_outputs: bool,
        step_index: int,
        total_steps: int
    ):
        """
        Property 8: Self-Correction Loop Activation
        
        For any execution state with unsatisfactory critique and retry count < 3,
        the supervisor should route back to planner for self-correction.
        
        Validates: Requirements 2.4
        """
        assume(step_index < total_steps)  # Ensure valid step index
        
        supervisor = create_supervisor_node()
        
        # Create state with critique
        state = AgentState()
        state.messages = [HumanMessage(content="Test request")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description=f"Step {i}", action_type=ActionType.JOB_SEARCH)
                for i in range(total_steps)
            ],
            user_id=state.user_id
        )
        state.current_step_index = step_index
        state.retry_count = retry_count
        
        # Add tool outputs if specified
        if has_tool_outputs:
            state.add_tool_output(ActionResult(
                action_type=ActionType.JOB_SEARCH,
                success=True,
                data={"test": "data"}
            ))
        
        # Set critique based on satisfactory flag
        if critique_satisfactory:
            state.critique = "Results are satisfactory and meet requirements"
        else:
            state.critique = "Results are unsatisfactory and need improvement"
        
        result = await supervisor(state)
        
        # Property: Self-correction loop should activate for unsatisfactory results
        if not critique_satisfactory and retry_count < 3:
            # Should route back to planner for self-correction
            assert result["next_node"] == "planner", (
                f"Expected planner for self-correction with retry_count={retry_count}, "
                f"but got {result['next_node']}"
            )
        elif not critique_satisfactory and retry_count >= 3:
            # Should end after max retries
            assert result["next_node"] == "end", (
                f"Expected end after max retries with retry_count={retry_count}, "
                f"but got {result['next_node']}"
            )
    
    @given(
        has_user_profile=st.booleans(),
        has_plan=st.booleans(),
        has_tool_outputs=st.booleans(),
        has_critique=st.booleans()
    )
    @settings(max_examples=30, deadline=3000)
    @pytest.mark.asyncio
    async def test_routing_consistency(
        self,
        has_user_profile: bool,
        has_plan: bool,
        has_tool_outputs: bool,
        has_critique: bool
    ):
        """
        Property: Routing Consistency
        
        For any valid agent state, the supervisor should always return a valid
        next_node and routing_reason.
        """
        supervisor = create_supervisor_node()
        
        # Create state based on parameters
        state = AgentState()
        state.messages = [HumanMessage(content="Test message")]
        
        if has_user_profile:
            state.user_profile = UserProfile(
                personal_info=PersonalInfo(name="Test User", email="test@example.com")
            )
        
        if has_plan:
            state.current_plan = ExecutionPlan(
                steps=[
                    PlanStep(description="Test step", action_type=ActionType.JOB_SEARCH)
                ],
                user_id=state.user_id
            )
            state.current_step_index = 0
        
        if has_tool_outputs:
            state.add_tool_output(ActionResult(
                action_type=ActionType.JOB_SEARCH,
                success=True,
                data={"test": "data"}
            ))
        
        if has_critique:
            state.critique = "Test critique"
        
        result = await supervisor(state)
        
        # Property: Always returns valid routing decision
        assert "next_node" in result
        assert "routing_reason" in result
        assert isinstance(result["next_node"], str)
        assert isinstance(result["routing_reason"], str)
        assert len(result["routing_reason"]) > 0
        
        # Property: Next node is always valid
        valid_nodes = {"profiler", "planner", "executor", "reviewer", "archivist", "end"}
        assert result["next_node"] in valid_nodes, (
            f"Invalid next_node: {result['next_node']}"
        )
    
    @given(
        step_count=st.integers(min_value=1, max_value=5),
        current_step=st.integers(min_value=0, max_value=4),
        critique_type=st.sampled_from(["satisfactory", "unsatisfactory", None])
    )
    @settings(max_examples=40, deadline=4000)
    @pytest.mark.asyncio
    async def test_plan_execution_flow(
        self,
        step_count: int,
        current_step: int,
        critique_type: str
    ):
        """
        Property: Plan Execution Flow
        
        For any execution plan, the supervisor should route appropriately
        based on current step and critique status.
        """
        assume(current_step < step_count)  # Ensure valid step
        
        supervisor = create_supervisor_node()
        
        # Create state with execution plan
        state = AgentState()
        state.messages = [HumanMessage(content="Execute plan")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description=f"Step {i}", action_type=ActionType.JOB_SEARCH)
                for i in range(step_count)
            ],
            user_id=state.user_id
        )
        state.current_step_index = current_step
        
        # Add tool outputs for current step
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"step": current_step}
        ))
        
        # Set critique if specified
        if critique_type == "satisfactory":
            state.critique = "Results are satisfactory"
        elif critique_type == "unsatisfactory":
            state.critique = "Results are unsatisfactory"
            state.retry_count = 1
        
        result = await supervisor(state)
        
        # Property: Routing follows execution flow logic
        if critique_type == "satisfactory":
            if current_step < step_count - 1:
                # Should continue to next step
                assert result["next_node"] == "executor", (
                    f"Expected executor for step continuation, got {result['next_node']}"
                )
            else:
                # At last step with satisfactory critique, should continue execution
                # The supervisor will route to executor to complete the step
                # Only after step_index >= len(steps) will it go to archivist
                assert result["next_node"] == "executor", (
                    f"Expected executor for last step execution, got {result['next_node']}"
                )
        elif critique_type == "unsatisfactory":
            # Should retry planning
            assert result["next_node"] == "planner", (
                f"Expected planner for unsatisfactory critique, got {result['next_node']}"
            )
        elif critique_type is None:
            # Should review outputs
            assert result["next_node"] == "reviewer", (
                f"Expected reviewer for unreviewed outputs, got {result['next_node']}"
            )
    
    @given(
        message_count=st.integers(min_value=0, max_value=5),
        has_ai_messages=st.booleans()
    )
    @settings(max_examples=20, deadline=2000)
    @pytest.mark.asyncio
    async def test_conversation_state_handling(
        self,
        message_count: int,
        has_ai_messages: bool
    ):
        """
        Property: Conversation State Handling
        
        For any conversation state, the supervisor should handle
        message history appropriately.
        """
        supervisor = create_supervisor_node()
        
        # Create state with message history
        state = AgentState()
        
        # Add messages
        for i in range(message_count):
            if has_ai_messages and i % 2 == 1:
                state.messages.append(AIMessage(content=f"AI response {i}"))
            else:
                state.messages.append(HumanMessage(content=f"User message {i}"))
        
        result = await supervisor(state)
        
        # Property: Always handles conversation state gracefully
        assert "next_node" in result
        assert "routing_reason" in result
        
        # Property: Empty conversation starts with profiler
        if message_count == 0:
            assert result["next_node"] == "profiler", (
                "Empty conversation should start with profiler"
            )
    
    @pytest.mark.asyncio
    async def test_error_state_recovery(self):
        """
        Property: Error State Recovery
        
        The supervisor should handle error states gracefully and
        always provide a valid routing decision.
        """
        supervisor = create_supervisor_node()
        
        # Test with None state
        result = await supervisor(None)
        assert result["next_node"] == "end"
        assert "error" in result["routing_reason"].lower()
        
        # Test with corrupted state
        corrupted_state = AgentState()
        corrupted_state.current_step_index = -1  # Invalid step index
        corrupted_state.retry_count = -1  # Invalid retry count
        
        result = await supervisor(corrupted_state)
        assert "next_node" in result
        assert "routing_reason" in result
        
        # Should handle gracefully without crashing
        valid_nodes = {"profiler", "planner", "executor", "reviewer", "archivist", "end"}
        assert result["next_node"] in valid_nodes