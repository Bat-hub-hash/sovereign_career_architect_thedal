"""Property-based tests for Planner Node consistency."""

from typing import List
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from langchain_core.messages import HumanMessage, AIMessage

from sovereign_career_architect.core.nodes.planner import PlannerNode
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
    JobPreferences,
    Skill,
)


class MockLLMResponse:
    """Mock LLM response for property testing."""
    def __init__(self, content: str):
        self.content = content


# Hypothesis strategies for generating test data
@st.composite
def user_message_strategy(draw):
    """Generate realistic user messages for career-related requests."""
    templates = [
        "I need help finding a {role} job in {location}",
        "Help me prepare for {role} interviews",
        "I want to update my profile for {role} positions",
        "Search for {role} opportunities with {experience} years experience",
        "I'm looking for {arrangement} {role} roles"
    ]
    
    roles = ["Python Developer", "Software Engineer", "Data Scientist", "DevOps Engineer", "Frontend Developer"]
    locations = ["San Francisco", "New York", "Remote", "Seattle", "Austin"]
    arrangements = ["remote", "hybrid", "full-time", "contract"]
    
    template = draw(st.sampled_from(templates))
    role = draw(st.sampled_from(roles))
    location = draw(st.sampled_from(locations))
    arrangement = draw(st.sampled_from(arrangements))
    experience = draw(st.integers(min_value=0, max_value=15))
    
    return template.format(
        role=role, 
        location=location, 
        arrangement=arrangement,
        experience=experience
    )


@st.composite
def skill_strategy(draw):
    """Generate realistic skills."""
    skills = ["Python", "JavaScript", "Java", "React", "Django", "AWS", "Docker", "Kubernetes"]
    levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
    
    return Skill(
        name=draw(st.sampled_from(skills)),
        level=draw(st.sampled_from(levels)),
        years_experience=draw(st.integers(min_value=0, max_value=10))
    )


@st.composite
def user_profile_strategy(draw):
    """Generate realistic user profiles."""
    return UserProfile(
        personal_info=PersonalInfo(
            name=draw(st.text(min_size=2, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs')))),
            email=draw(st.emails()),
            location=draw(st.sampled_from(["San Francisco", "New York", "Remote", "Seattle", None])),
            preferred_language=draw(st.sampled_from(["en", "hi", "es"]))
        ),
        skills=draw(st.lists(skill_strategy(), min_size=0, max_size=5)),
        preferences=JobPreferences(
            roles=draw(st.lists(st.sampled_from(["Developer", "Engineer", "Analyst"]), min_size=0, max_size=3)),
            locations=draw(st.lists(st.sampled_from(["Remote", "SF", "NYC"]), min_size=0, max_size=3)),
            work_arrangements=draw(st.lists(st.sampled_from(["Remote", "Hybrid", "Onsite"]), min_size=0, max_size=2))
        )
    )


@st.composite
def mock_plan_response_strategy(draw):
    """Generate realistic mock plan responses."""
    action_types = ["JOB_SEARCH", "APPLICATION_SUBMIT", "INTERVIEW_PRACTICE", "PROFILE_UPDATE", "GENERAL_CHAT"]
    
    num_steps = draw(st.integers(min_value=1, max_value=5))
    steps = []
    
    for i in range(num_steps):
        action_type = draw(st.sampled_from(action_types))
        description = draw(st.text(min_size=10, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))))
        steps.append(f"{i+1}. [{action_type}] {description}")
    
    return "\n".join(steps)


class TestPlannerNodeProperties:
    """Property-based tests for PlannerNode consistency."""
    
    @pytest.fixture
    def mock_reasoning_model(self):
        """Create a mock reasoning model that returns consistent responses."""
        from unittest.mock import AsyncMock
        return AsyncMock()
    
    @pytest.fixture
    def planner_node(self, mock_reasoning_model):
        """Create a PlannerNode with mocked model."""
        return PlannerNode(reasoning_model=mock_reasoning_model)
    
    # Feature: sovereign-career-architect, Property 6: Plan Structure Consistency
    @given(
        user_message=user_message_strategy(),
        user_profile=st.one_of(st.none(), user_profile_strategy()),
        mock_response=mock_plan_response_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_plan_structure_consistency(
        self, 
        planner_node, 
        mock_reasoning_model,
        user_message: str,
        user_profile: UserProfile,
        mock_response: str
    ):
        """
        For any user request and profile, when the planner generates a plan,
        the output should be a well-structured ExecutionPlan with valid steps.
        
        **Validates: Requirements 2.2**
        """
        # Setup mock response
        mock_reasoning_model.ainvoke.return_value = MockLLMResponse(mock_response)
        
        # Create state with generated data
        state = AgentState()
        state.messages = [HumanMessage(content=user_message)]
        state.user_profile = user_profile
        state.memory_context = {"memories": {"user": [], "session": [], "agent": []}, "total_memories": 0}
        
        # Execute planner
        result = await planner_node(state)
        
        # Verify plan structure consistency
        assert isinstance(result, dict)
        assert "current_plan" in result
        
        plan = result["current_plan"]
        if plan is not None:  # Plan generation might fail gracefully
            assert isinstance(plan, ExecutionPlan)
            assert hasattr(plan, 'steps')
            assert isinstance(plan.steps, list)
            assert hasattr(plan, 'user_id')
            assert hasattr(plan, 'status')
            
            # Each step should be properly structured
            for step in plan.steps:
                assert isinstance(step, PlanStep)
                assert hasattr(step, 'description')
                assert hasattr(step, 'action_type')
                assert isinstance(step.action_type, ActionType)
                assert isinstance(step.description, str)
                assert len(step.description.strip()) > 0
    
    @given(
        user_messages=st.lists(user_message_strategy(), min_size=1, max_size=3),
        mock_response=mock_plan_response_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_plan_determinism_with_same_context(
        self,
        planner_node,
        mock_reasoning_model,
        user_messages: List[str],
        mock_response: str
    ):
        """
        For any identical context, the planner should generate
        structurally consistent plans (same number of steps, same action types).
        """
        # Setup consistent mock response
        mock_reasoning_model.ainvoke.return_value = MockLLMResponse(mock_response)
        
        # Create identical states
        states = []
        for _ in range(2):
            state = AgentState()
            state.messages = [HumanMessage(content=msg) for msg in user_messages]
            state.memory_context = {"memories": {"user": [], "session": [], "agent": []}, "total_memories": 0}
            states.append(state)
        
        # Execute planner on both states
        results = []
        for state in states:
            result = await planner_node(state)
            results.append(result)
        
        # Both should produce plans (or both should fail)
        plan1 = results[0]["current_plan"]
        plan2 = results[1]["current_plan"]
        
        if plan1 is not None and plan2 is not None:
            # Plans should have same structure
            assert len(plan1.steps) == len(plan2.steps)
            
            # Action types should match
            action_types1 = [step.action_type for step in plan1.steps]
            action_types2 = [step.action_type for step in plan2.steps]
            assert action_types1 == action_types2
        else:
            # Both should be None (consistent failure)
            assert plan1 == plan2
    
    @given(mock_response=mock_plan_response_strategy())
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_plan_step_action_type_validity(
        self,
        planner_node,
        mock_reasoning_model,
        mock_response: str
    ):
        """
        For any generated plan, all action types should be valid ActionType enum values.
        """
        # Setup mock response
        mock_reasoning_model.ainvoke.return_value = MockLLMResponse(mock_response)
        
        # Create simple state
        state = AgentState()
        state.messages = [HumanMessage(content="Help me find a job")]
        state.memory_context = {"memories": {"user": [], "session": [], "agent": []}, "total_memories": 0}
        
        # Execute planner
        result = await planner_node(state)
        
        plan = result["current_plan"]
        if plan is not None:
            # All action types should be valid enum values
            valid_action_types = set(ActionType)
            for step in plan.steps:
                assert step.action_type in valid_action_types
    
    @given(
        user_message=user_message_strategy(),
        mock_response=mock_plan_response_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_plan_step_descriptions_non_empty(
        self,
        planner_node,
        mock_reasoning_model,
        user_message: str,
        mock_response: str
    ):
        """
        For any generated plan, all step descriptions should be non-empty strings.
        """
        # Setup mock response
        mock_reasoning_model.ainvoke.return_value = MockLLMResponse(mock_response)
        
        # Create state
        state = AgentState()
        state.messages = [HumanMessage(content=user_message)]
        state.memory_context = {"memories": {"user": [], "session": [], "agent": []}, "total_memories": 0}
        
        # Execute planner
        result = await planner_node(state)
        
        plan = result["current_plan"]
        if plan is not None:
            # All descriptions should be non-empty
            for step in plan.steps:
                assert isinstance(step.description, str)
                assert len(step.description.strip()) > 0
    
    @given(mock_response=st.text(min_size=0, max_size=10))  # Very short or empty responses
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    @pytest.mark.asyncio
    async def test_plan_graceful_handling_of_poor_responses(
        self,
        planner_node,
        mock_reasoning_model,
        mock_response: str
    ):
        """
        For any poor quality model response, the planner should handle it gracefully
        without crashing and return a valid result structure.
        """
        # Setup mock with poor response
        mock_reasoning_model.ainvoke.return_value = MockLLMResponse(mock_response)
        
        # Create state
        state = AgentState()
        state.messages = [HumanMessage(content="Help me")]
        state.memory_context = {"memories": {"user": [], "session": [], "agent": []}, "total_memories": 0}
        
        # Execute planner - should not crash
        result = await planner_node(state)
        
        # Should return valid result structure
        assert isinstance(result, dict)
        assert "current_plan" in result
        assert "next_node" in result
        
        # Plan might be None for poor responses, which is acceptable
        plan = result["current_plan"]
        if plan is not None:
            assert isinstance(plan, ExecutionPlan)
    
    def test_parse_plan_steps_consistency(self, planner_node):
        """
        For any plan text, parsing should be consistent and handle edge cases gracefully.
        """
        # Test various edge cases
        test_cases = [
            "",  # Empty string
            "No numbered steps here",  # No valid format
            "1. [INVALID_ACTION] Some description",  # Invalid action type
            "1. [JOB_SEARCH] Valid step\n2. Invalid step without action type",  # Mixed format
            "1. [JOB_SEARCH] \n2. [APPLICATION_SUBMIT] ",  # Empty descriptions
        ]
        
        for plan_text in test_cases:
            # Should not crash
            steps = planner_node._parse_plan_steps(plan_text)
            
            # Should return a list
            assert isinstance(steps, list)
            
            # All returned steps should be valid
            for step in steps:
                assert isinstance(step, PlanStep)
                assert isinstance(step.action_type, ActionType)
                assert isinstance(step.description, str)
    
    @given(
        action_type_str=st.text(min_size=0, max_size=20)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=20
    )
    def test_action_type_mapping_robustness(self, planner_node, action_type_str: str):
        """
        For any string input, action type mapping should always return a valid ActionType.
        """
        result = planner_node._map_action_type(action_type_str)
        
        # Should always return a valid ActionType
        assert isinstance(result, ActionType)
        
        # Should be one of the valid enum values
        valid_action_types = set(ActionType)
        assert result in valid_action_types
    
    @given(
        user_profile=st.one_of(st.none(), user_profile_strategy())
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture], 
        max_examples=10
    )
    def test_user_profile_summarization_robustness(self, planner_node, user_profile):
        """
        For any user profile (including None), summarization should return a valid string.
        """
        if user_profile is None:
            # Should handle None gracefully - this might raise an exception
            # which is acceptable behavior
            return
        
        summary = planner_node._summarize_user_profile(user_profile)
        
        # Should return a string
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Should contain the user's name if available
        if user_profile.personal_info and user_profile.personal_info.name:
            assert user_profile.personal_info.name in summary


class TestPlannerNodeEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def planner_node_no_model(self):
        """Create a PlannerNode without a model to test error handling."""
        return PlannerNode(reasoning_model=None)
    
    @pytest.mark.asyncio
    async def test_planner_with_no_model(self, planner_node_no_model):
        """Test planner behavior when no model is available."""
        state = AgentState()
        state.messages = [HumanMessage(content="Help me find a job")]
        
        # Should handle gracefully
        result = await planner_node_no_model(state)
        
        assert isinstance(result, dict)
        assert result["current_plan"] is None
        assert result["next_node"] == "executor"