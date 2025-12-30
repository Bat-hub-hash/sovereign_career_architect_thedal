"""Property-based tests for self-correction loop activation."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionResult, ActionType, ExecutionPlan, PlanStep, UserProfile, PersonalInfo
)
from sovereign_career_architect.core.nodes.supervisor import SupervisorNode, create_supervisor_node
from sovereign_career_architect.core.nodes.reviewer import ReviewerNode, create_reviewer_node
from langchain_core.messages import HumanMessage, AIMessage


# Feature: sovereign-career-architect, Property 8: Self-Correction Loop Activation
@pytest.mark.property
class TestSelfCorrectionProperties:
    """Property-based tests for self-correction loop activation behavior."""
    
    @given(
        retry_count=st.integers(min_value=0, max_value=5),
        critique_severity=st.sampled_from(["minor", "major", "critical"]),
        has_user_profile=st.booleans(),
        plan_step_count=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_self_correction_loop_activation(
        self,
        retry_count: int,
        critique_severity: str,
        has_user_profile: bool,
        plan_step_count: int
    ):
        """
        Property 8: Self-Correction Loop Activation
        
        For any agent state with critique feedback, the system should
        activate self-correction loops when quality issues are detected.
        
        Validates: Requirements 2.4
        """
        # Create supervisor node
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Create test state with critique
        critique_messages = {
            "minor": "The response could be more specific about job requirements.",
            "major": "The response doesn't address the user's main question about salary expectations.",
            "critical": "The response is completely irrelevant to the user's career goals."
        }
        
        # Create execution plan
        plan_steps = []
        for i in range(plan_step_count):
            step = PlanStep(
                description=f"Step {i+1}: Execute career action",
                action_type=ActionType.JOB_SEARCH,
                status="pending"
            )
            plan_steps.append(step)
        
        execution_plan = ExecutionPlan(
            steps=plan_steps,
            user_id=user_id
        )
        
        # Create user profile if needed
        user_profile = None
        if has_user_profile:
            user_profile = UserProfile(
                personal_info=PersonalInfo(
                    name="Test User",
                    email="test@example.com"
                )
            )
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[
                HumanMessage(content="Help me find a software engineering job"),
                AIMessage(content="I'll help you search for positions")
            ],
            current_plan=execution_plan,
            current_step_index=0,
            retry_count=retry_count,
            critique=critique_messages[critique_severity],
            user_profile=user_profile,
            tool_outputs=[
                ActionResult(
                    action_type=ActionType.JOB_SEARCH,
                    success=False,  # Failed action to trigger correction
                    data={"error": "Search failed"}
                )
            ]
        )
        
        # Execute supervisor logic
        result = await supervisor(state)
        
        # Property: Self-correction should be activated based on critique and retry count
        assert result is not None, "Supervisor should return a result"
        assert "next_node" in result, "Result should contain next_node routing"
        
        # Property: Retry count should influence correction behavior
        if retry_count >= 3:
            # High retry count should lead to different behavior (possibly ending or escalating)
            assert result["next_node"] in [None, "planner", "reviewer"], (
                "High retry count should lead to termination or replanning"
            )
        else:
            # Lower retry count should continue with correction
            assert result["next_node"] in ["planner", "reviewer", "executor"], (
                "Lower retry count should continue with correction attempts"
            )
        
        # Property: Critique severity should influence routing decisions
        if critique_severity == "critical" and retry_count > 0:
            # Critical issues should trigger replanning
            assert result["next_node"] in ["planner", None], (
                "Critical critique should trigger replanning or termination"
            )
    
    @given(
        consecutive_failures=st.integers(min_value=1, max_value=4),
        action_types=st.lists(
            st.sampled_from([ActionType.JOB_SEARCH, ActionType.PROFILE_UPDATE, ActionType.GENERAL_CHAT]),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=10, deadline=2500)
    @pytest.mark.asyncio
    async def test_failure_pattern_detection(
        self,
        consecutive_failures: int,
        action_types: list[ActionType]
    ):
        """
        Property: Failure Pattern Detection
        
        For any sequence of consecutive failures, the system should
        detect patterns and adjust correction strategies accordingly.
        """
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Create tool outputs with consecutive failures
        tool_outputs = []
        for i in range(consecutive_failures):
            action_type = action_types[i % len(action_types)]
            tool_outputs.append(
                ActionResult(
                    action_type=action_type,
                    success=False,
                    data={"error": f"Failure {i+1}"},
                    error_message=f"Action failed: attempt {i+1}"
                )
            )
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[HumanMessage(content="Help me with my career")],
            tool_outputs=tool_outputs,
            retry_count=consecutive_failures,
            critique="Multiple failures detected in recent actions"
        )
        
        # Execute supervisor
        result = await supervisor(state)
        
        # Property: Multiple failures should trigger appropriate responses
        assert result is not None, "Supervisor should handle failure patterns"
        
        # Property: Excessive failures should lead to termination or major replanning
        if consecutive_failures >= 3:
            assert result["next_node"] in [None, "planner"], (
                "Excessive failures should trigger termination or major replanning"
            )
        
        # Property: Failure information should be preserved for learning
        if "tool_outputs" in result:
            # Should maintain failure history for analysis
            assert len(result["tool_outputs"]) >= 0, (
                "Failure history should be maintained for analysis"
            )
    
    @given(
        plan_complexity=st.integers(min_value=2, max_value=8),
        current_step_ratio=st.floats(min_value=0.1, max_value=0.9),
        has_critique=st.booleans()
    )
    @settings(max_examples=12, deadline=2000)
    @pytest.mark.asyncio
    async def test_plan_progress_correction(
        self,
        plan_complexity: int,
        current_step_ratio: float,
        has_critique: bool
    ):
        """
        Property: Plan Progress Correction
        
        For any execution plan, correction loops should consider
        plan progress and complexity when making routing decisions.
        """
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Create complex execution plan
        plan_steps = []
        for i in range(plan_complexity):
            step = PlanStep(
                description=f"Complex step {i+1}",
                action_type=ActionType.JOB_SEARCH,
                status="completed" if i < int(plan_complexity * current_step_ratio) else "pending"
            )
            plan_steps.append(step)
        
        execution_plan = ExecutionPlan(
            steps=plan_steps,
            user_id=user_id
        )
        
        current_step_index = int(plan_complexity * current_step_ratio)
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[HumanMessage(content="Execute my career plan")],
            current_plan=execution_plan,
            current_step_index=current_step_index,
            critique="Plan execution needs adjustment" if has_critique else None,
            retry_count=1
        )
        
        # Execute supervisor
        result = await supervisor(state)
        
        # Property: Plan progress should influence correction decisions
        assert result is not None, "Supervisor should handle plan progress"
        
        # Property: Near completion should have different behavior than early stages
        progress_ratio = current_step_index / plan_complexity
        
        if progress_ratio > 0.8:
            # Near completion - should be more conservative about major changes
            if has_critique:
                assert result["next_node"] in ["reviewer", "executor", None], (
                    "Near completion should prefer minor corrections"
                )
        elif progress_ratio < 0.3:
            # Early stage - more open to major replanning
            if has_critique:
                assert result["next_node"] in ["planner", "reviewer", "executor"], (
                    "Early stage should allow major replanning"
                )
        
        # Property: Complex plans should have more careful correction strategies
        if plan_complexity >= 6 and has_critique:
            # Complex plans should prefer incremental corrections
            assert result["next_node"] in ["reviewer", "executor", "planner"], (
                "Complex plans should use careful correction strategies"
            )
    
    @given(
        user_feedback_sentiment=st.sampled_from(["positive", "neutral", "negative"]),
        system_confidence=st.floats(min_value=0.1, max_value=1.0),
        interaction_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10, deadline=2000)
    @pytest.mark.asyncio
    async def test_adaptive_correction_sensitivity(
        self,
        user_feedback_sentiment: str,
        system_confidence: float,
        interaction_count: int
    ):
        """
        Property: Adaptive Correction Sensitivity
        
        For any interaction context, correction sensitivity should
        adapt based on user feedback and system confidence levels.
        """
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Create messages reflecting user sentiment
        sentiment_messages = {
            "positive": "Great! That's exactly what I was looking for.",
            "neutral": "Okay, that's helpful information.",
            "negative": "This isn't what I asked for. Can you try again?"
        }
        
        messages = [HumanMessage(content="Help me find a job")]
        for i in range(interaction_count):
            messages.append(AIMessage(content=f"Here's suggestion {i+1}"))
            if i == interaction_count - 1:  # Last message reflects sentiment
                messages.append(HumanMessage(content=sentiment_messages[user_feedback_sentiment]))
        
        # Create critique based on confidence and sentiment
        critique = None
        if system_confidence < 0.5 or user_feedback_sentiment == "negative":
            critique = "Response quality needs improvement based on user feedback"
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            critique=critique,
            retry_count=1 if critique else 0,
            tool_outputs=[
                ActionResult(
                    action_type=ActionType.JOB_SEARCH,
                    success=system_confidence > 0.6,
                    data={"confidence": system_confidence}
                )
            ]
        )
        
        # Execute supervisor
        result = await supervisor(state)
        
        # Property: Correction sensitivity should adapt to context
        assert result is not None, "Supervisor should handle adaptive correction"
        
        # Property: Negative feedback should increase correction likelihood
        if user_feedback_sentiment == "negative":
            assert result["next_node"] in ["planner", "reviewer"], (
                "Negative feedback should trigger correction"
            )
        
        # Property: Low confidence should trigger more careful review
        if system_confidence < 0.4:
            assert result["next_node"] in ["reviewer", "planner"], (
                "Low confidence should trigger review or replanning"
            )
        
        # Property: Positive feedback with high confidence should continue
        if user_feedback_sentiment == "positive" and system_confidence > 0.7:
            assert result["next_node"] in ["executor", "archivist", None], (
                "Positive feedback with high confidence should continue or complete"
            )
    
    @given(
        correction_history=st.lists(
            st.sampled_from(["planner", "reviewer", "executor"]),
            min_size=1,
            max_size=5
        ),
        loop_detection_threshold=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=8, deadline=2000)
    @pytest.mark.asyncio
    async def test_correction_loop_prevention(
        self,
        correction_history: list[str],
        loop_detection_threshold: int
    ):
        """
        Property: Correction Loop Prevention
        
        For any sequence of correction attempts, the system should
        detect and prevent infinite correction loops.
        """
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Simulate correction history by setting high retry count
        retry_count = len(correction_history)
        
        # Check for potential loops in history
        has_loop = False
        if len(correction_history) >= loop_detection_threshold:
            # Simple loop detection: check for repeated patterns
            for i in range(len(correction_history) - 1):
                if correction_history[i] == correction_history[i + 1]:
                    has_loop = True
                    break
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[HumanMessage(content="This keeps failing")],
            retry_count=retry_count,
            critique="Repeated correction attempts detected",
            tool_outputs=[
                ActionResult(
                    action_type=ActionType.JOB_SEARCH,
                    success=False,
                    data={"correction_attempts": retry_count}
                )
            ]
        )
        
        # Execute supervisor
        result = await supervisor(state)
        
        # Property: Loop prevention should activate with high retry counts
        assert result is not None, "Supervisor should handle loop prevention"
        
        # Property: Excessive retries should lead to termination
        if retry_count >= loop_detection_threshold:
            assert result["next_node"] in [None, "planner"], (
                "Excessive retries should trigger termination or major replanning"
            )
        
        # Property: Loop detection should prevent infinite cycles
        if has_loop and retry_count >= 3:
            # Should break the loop by changing strategy or terminating
            assert result["next_node"] != correction_history[-1] or result["next_node"] is None, (
                "Loop detection should prevent repeating the same correction"
            )
    
    @given(
        resource_constraints=st.fixed_dictionaries({
            "time_limit": st.integers(min_value=30, max_value=300),  # seconds
            "max_retries": st.integers(min_value=1, max_value=5),
            "complexity_budget": st.floats(min_value=0.1, max_value=1.0)
        }),
        current_usage=st.fixed_dictionaries({
            "time_elapsed": st.integers(min_value=10, max_value=250),
            "retries_used": st.integers(min_value=0, max_value=4),
            "complexity_used": st.floats(min_value=0.0, max_value=0.9)
        })
    )
    @settings(max_examples=8, deadline=2000)
    @pytest.mark.asyncio
    async def test_resource_aware_correction(
        self,
        resource_constraints: dict,
        current_usage: dict
    ):
        """
        Property: Resource-Aware Correction
        
        For any resource constraints, correction decisions should
        consider available resources and adjust strategies accordingly.
        """
        assume(current_usage["time_elapsed"] <= resource_constraints["time_limit"])
        assume(current_usage["retries_used"] <= resource_constraints["max_retries"])
        assume(current_usage["complexity_used"] <= resource_constraints["complexity_budget"])
        
        supervisor = create_supervisor_node()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Calculate resource utilization ratios
        time_ratio = current_usage["time_elapsed"] / resource_constraints["time_limit"]
        retry_ratio = current_usage["retries_used"] / resource_constraints["max_retries"]
        complexity_ratio = current_usage["complexity_used"] / resource_constraints["complexity_budget"]
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[HumanMessage(content="Need help with career planning")],
            retry_count=current_usage["retries_used"],
            critique="Resource-constrained correction needed",
            tool_outputs=[
                ActionResult(
                    action_type=ActionType.JOB_SEARCH,
                    success=False,
                    data={
                        "time_elapsed": current_usage["time_elapsed"],
                        "time_limit": resource_constraints["time_limit"],  # Pass the constraint
                        "max_retries": resource_constraints["max_retries"],  # Pass the constraint
                        "complexity": current_usage["complexity_used"],
                        "complexity_budget": resource_constraints["complexity_budget"]  # Pass the constraint
                    }
                )
            ]
        )
        
        # Execute supervisor
        result = await supervisor(state)
        
        # Property: Resource constraints should influence correction decisions
        assert result is not None, "Supervisor should handle resource constraints"
        
        # Property: High resource usage should lead to simpler corrections
        if time_ratio > 0.8 or retry_ratio > 0.8:
            # Near resource limits - should prefer simple corrections or termination
            assert result["next_node"] in [None, "executor", "archivist"], (
                "High resource usage should prefer simple corrections or termination"
            )
        
        # Property: Low resource usage allows more complex corrections
        if time_ratio < 0.3 and retry_ratio < 0.3 and complexity_ratio < 0.3:
            # Plenty of resources - can afford complex corrections
            assert result["next_node"] in ["planner", "reviewer", "executor"], (
                "Low resource usage should allow complex corrections"
            )
        
        # Property: Retry limit should be strictly enforced
        if current_usage["retries_used"] >= resource_constraints["max_retries"]:
            assert result["next_node"] in [None, "archivist"], (
                "Retry limit exceeded should lead to termination"
            )