"""Tests for the Planner Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage

from sovereign_career_architect.core.nodes.planner import PlannerNode, create_planner_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
    JobPreferences,
    Skill,
    Experience,
)


class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, content: str):
        self.content = content


class TestPlannerNode:
    """Test cases for PlannerNode functionality."""
    
    @pytest.fixture
    def mock_reasoning_model(self):
        """Create a mock reasoning model for testing."""
        model = AsyncMock()
        return model
    
    @pytest.fixture
    def planner_node(self, mock_reasoning_model):
        """Create a PlannerNode with mocked dependencies."""
        return PlannerNode(reasoning_model=mock_reasoning_model)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state for testing."""
        state = AgentState()
        state.messages = [
            HumanMessage(content="I need help finding a Python developer job in San Francisco"),
            AIMessage(content="I'd be happy to help you with your job search!")
        ]
        state.memory_context = {
            "memories": {
                "user": [{"content": "User has 5 years Python experience"}],
                "session": [{"content": "Looking for remote opportunities"}],
                "agent": []
            },
            "total_memories": 2
        }
        return state
    
    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile for testing."""
        return UserProfile(
            personal_info=PersonalInfo(
                name="John Doe",
                email="john@example.com",
                location="San Francisco, CA",
                preferred_language="en"
            ),
            skills=[
                Skill(name="Python", level="Expert", years_experience=5),
                Skill(name="Django", level="Advanced", years_experience=3),
                Skill(name="React", level="Intermediate", years_experience=2)
            ],
            experience=[
                Experience(
                    company="Tech Corp",
                    role="Senior Python Developer",
                    start_date="2020-01-01",
                    description="Led backend development team"
                )
            ],
            preferences=JobPreferences(
                roles=["Python Developer", "Backend Engineer"],
                locations=["San Francisco", "Remote"],
                work_arrangements=["Remote", "Hybrid"]
            )
        )
    
    @pytest.mark.asyncio
    async def test_planner_node_basic_execution(self, planner_node, sample_state, mock_reasoning_model):
        """Test basic planner node execution."""
        # Setup mock response
        mock_response = MockLLMResponse("""
1. [JOB_SEARCH] Search for Python developer roles in San Francisco
2. [JOB_SEARCH] Filter by experience level and salary requirements
3. [APPLICATION_SUBMIT] Apply to top 3 matching positions
        """)
        mock_reasoning_model.ainvoke.return_value = mock_response
        
        # Execute planner node
        result = await planner_node(sample_state)
        
        # Verify results
        assert isinstance(result, dict)
        assert "current_plan" in result
        assert "current_step_index" in result
        assert "next_node" in result
        assert result["next_node"] == "executor"
        assert result["current_step_index"] == 0
        
        # Verify plan structure
        plan = result["current_plan"]
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 3
        assert plan.steps[0].action_type == ActionType.JOB_SEARCH
        assert plan.steps[2].action_type == ActionType.APPLICATION_SUBMIT
    
    @pytest.mark.asyncio
    async def test_planner_node_with_user_profile(self, planner_node, sample_state, sample_user_profile, mock_reasoning_model):
        """Test planner node with user profile context."""
        sample_state.user_profile = sample_user_profile
        
        # Setup mock response
        mock_response = MockLLMResponse("""
1. [JOB_SEARCH] Search for Senior Python Developer roles matching 5+ years experience
2. [PROFILE_UPDATE] Update portfolio with recent Django projects
3. [INTERVIEW_PRACTICE] Practice system design interviews
        """)
        mock_reasoning_model.ainvoke.return_value = mock_response
        
        # Execute planner node
        result = await planner_node(sample_state)
        
        # Verify plan incorporates profile context
        plan = result["current_plan"]
        assert len(plan.steps) == 3
        assert any("Senior Python" in step.description for step in plan.steps)
        assert any(step.action_type == ActionType.PROFILE_UPDATE for step in plan.steps)
        assert any(step.action_type == ActionType.INTERVIEW_PRACTICE for step in plan.steps)
        
        # Verify reasoning model was called with profile context
        mock_reasoning_model.ainvoke.assert_called_once()
        call_args = mock_reasoning_model.ainvoke.call_args[0][0]
        user_message = call_args[1].content  # Second message should be user prompt
        assert "John Doe" in user_message
        assert "Python" in user_message
        assert "Expert" in user_message
    
    @pytest.mark.asyncio
    async def test_planner_node_no_user_message(self, planner_node, mock_reasoning_model):
        """Test planner node with no user messages."""
        # Create state with only AI messages
        state = AgentState()
        state.messages = [AIMessage(content="Hello! How can I help you?")]
        
        # Execute planner node
        result = await planner_node(state)
        
        # Should skip planning and route to executor
        assert result["next_node"] == "executor"
        assert result.get("current_plan") is None
        
        # Reasoning model should not be called
        mock_reasoning_model.ainvoke.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_planner_node_error_handling(self, planner_node, sample_state, mock_reasoning_model):
        """Test error handling in planner node."""
        # Setup mock to raise exception
        mock_reasoning_model.ainvoke.side_effect = Exception("Model service unavailable")
        
        # Execute planner node
        result = await planner_node(sample_state)
        
        # Verify graceful error handling
        assert isinstance(result, dict)
        assert result["current_plan"] is None
        assert result["next_node"] == "executor"  # Should still route to next node
    
    def test_parse_plan_steps_valid_format(self, planner_node):
        """Test parsing of well-formatted plan steps."""
        plan_text = """
1. [JOB_SEARCH] Search for Python developer roles in San Francisco
2. [APPLICATION_SUBMIT] Apply to top 3 matching positions with tailored resumes
3. [INTERVIEW_PRACTICE] Prepare for technical interviews focusing on algorithms
4. [PROFILE_UPDATE] Update LinkedIn profile with recent achievements
        """
        
        steps = planner_node._parse_plan_steps(plan_text)
        
        assert len(steps) == 4
        assert steps[0].action_type == ActionType.JOB_SEARCH
        assert "Python developer" in steps[0].description
        assert steps[1].action_type == ActionType.APPLICATION_SUBMIT
        assert steps[2].action_type == ActionType.INTERVIEW_PRACTICE
        assert steps[3].action_type == ActionType.PROFILE_UPDATE
    
    def test_parse_plan_steps_mixed_format(self, planner_node):
        """Test parsing of mixed format plan steps."""
        plan_text = """
1. [JOB_SEARCH] Search for roles
2. Review and shortlist candidates
3. [APPLICATION_SUBMIT] Submit applications
Some random text that should be ignored
4. [GENERAL_CHAT] Provide career advice
        """
        
        steps = planner_node._parse_plan_steps(plan_text)
        
        assert len(steps) == 4
        assert steps[0].action_type == ActionType.JOB_SEARCH
        assert steps[1].action_type == ActionType.GENERAL_CHAT  # Default for no action type
        assert "Review and shortlist" in steps[1].description
        assert steps[2].action_type == ActionType.APPLICATION_SUBMIT
        assert steps[3].action_type == ActionType.GENERAL_CHAT
    
    def test_parse_plan_steps_invalid_format(self, planner_node):
        """Test parsing of invalid format plan steps."""
        plan_text = """
This is just random text without proper formatting.
No numbered steps here.
[JOB_SEARCH] This has action type but no number.
        """
        
        steps = planner_node._parse_plan_steps(plan_text)
        
        # Should handle gracefully and extract what it can
        assert isinstance(steps, list)
        # Might be empty or have minimal steps depending on parsing logic
    
    def test_map_action_type(self, planner_node):
        """Test action type mapping."""
        assert planner_node._map_action_type("JOB_SEARCH") == ActionType.JOB_SEARCH
        assert planner_node._map_action_type("job_search") == ActionType.JOB_SEARCH
        assert planner_node._map_action_type("APPLICATION_SUBMIT") == ActionType.APPLICATION_SUBMIT
        assert planner_node._map_action_type("INTERVIEW_PRACTICE") == ActionType.INTERVIEW_PRACTICE
        assert planner_node._map_action_type("PROFILE_UPDATE") == ActionType.PROFILE_UPDATE
        assert planner_node._map_action_type("GENERAL_CHAT") == ActionType.GENERAL_CHAT
        
        # Test unknown action type defaults to GENERAL_CHAT
        assert planner_node._map_action_type("UNKNOWN_ACTION") == ActionType.GENERAL_CHAT
    
    def test_summarize_user_profile(self, planner_node, sample_user_profile):
        """Test user profile summarization."""
        summary = planner_node._summarize_user_profile(sample_user_profile)
        
        assert "John Doe" in summary
        assert "San Francisco" in summary
        assert "Python" in summary
        assert "Expert" in summary
        assert "Tech Corp" in summary
        assert "Senior Python Developer" in summary
        assert "Remote" in summary
    
    def test_summarize_user_profile_minimal(self, planner_node):
        """Test user profile summarization with minimal data."""
        minimal_profile = UserProfile(
            personal_info=PersonalInfo(name="Jane", email="jane@example.com")
        )
        
        summary = planner_node._summarize_user_profile(minimal_profile)
        
        assert "Jane" in summary
        assert "en" in summary  # Default language
    
    def test_summarize_memory_context(self, planner_node):
        """Test memory context summarization."""
        memory_context = {
            "memories": {
                "user": [
                    {"content": "User has 5 years Python experience"},
                    {"content": "Prefers remote work"}
                ],
                "session": [
                    {"content": "Currently looking for senior roles"}
                ],
                "agent": [
                    {"content": "User responds well to detailed explanations"}
                ]
            },
            "total_memories": 4
        }
        
        summary = planner_node._summarize_memory_context(memory_context)
        
        assert "Long-term context:" in summary
        assert "Python experience" in summary
        assert "Current session context:" in summary
        assert "senior roles" in summary
        assert "Interaction patterns:" in summary
    
    def test_summarize_memory_context_empty(self, planner_node):
        """Test memory context summarization with empty context."""
        empty_context = {"memories": {"user": [], "session": [], "agent": []}}
        
        summary = planner_node._summarize_memory_context(empty_context)
        
        assert summary == "No relevant context found"
    
    def test_build_system_prompt(self, planner_node):
        """Test system prompt building."""
        prompt = planner_node._build_system_prompt()
        
        assert "Sovereign Career Architect" in prompt
        assert "ACTION TYPES" in prompt
        assert "JOB_SEARCH" in prompt
        assert "OUTPUT FORMAT" in prompt
        assert "step-by-step" in prompt
    
    def test_build_user_prompt_full_context(self, planner_node, sample_user_profile):
        """Test user prompt building with full context."""
        memory_context = {
            "memories": {"user": [{"content": "Has Python experience"}]},
            "total_memories": 1
        }
        
        prompt = planner_node._build_user_prompt(
            "Find me a job",
            sample_user_profile,
            memory_context
        )
        
        assert "USER REQUEST: Find me a job" in prompt
        assert "USER PROFILE:" in prompt
        assert "John Doe" in prompt
        assert "RELEVANT CONTEXT:" in prompt
        assert "Python experience" in prompt
    
    def test_build_user_prompt_minimal_context(self, planner_node):
        """Test user prompt building with minimal context."""
        prompt = planner_node._build_user_prompt(
            "Help me with my career",
            None,
            {}
        )
        
        assert "USER REQUEST: Help me with my career" in prompt
        assert "USER PROFILE:" not in prompt
        assert "RELEVANT CONTEXT:" not in prompt
    
    def test_get_latest_user_message(self, planner_node):
        """Test extraction of latest user message."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Second message"),
            AIMessage(content="Another AI response")
        ]
        
        latest = planner_node._get_latest_user_message(messages)
        
        assert latest is not None
        assert latest.content == "Second message"
    
    def test_get_latest_user_message_no_user_messages(self, planner_node):
        """Test extraction when no user messages exist."""
        messages = [
            AIMessage(content="AI response 1"),
            AIMessage(content="AI response 2")
        ]
        
        latest = planner_node._get_latest_user_message(messages)
        
        assert latest is None
    
    def test_create_planner_node_factory(self):
        """Test the factory function for creating planner nodes."""
        # Should create a planner node even without API keys (for testing)
        planner = create_planner_node()
        assert isinstance(planner, PlannerNode)
        # Model should be None when no API keys are available
        assert planner.reasoning_model is None
    
    def test_create_planner_node_with_custom_model(self, mock_reasoning_model):
        """Test factory function with custom reasoning model."""
        node = create_planner_node(reasoning_model=mock_reasoning_model)
        
        assert isinstance(node, PlannerNode)
        assert node.reasoning_model == mock_reasoning_model


class TestPlannerNodeIntegration:
    """Integration tests for PlannerNode."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_planning_workflow(self):
        """Test complete planning workflow with mock model."""
        # Create mock model
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = MockLLMResponse("""
1. [JOB_SEARCH] Search for Python developer positions in target location
2. [JOB_SEARCH] Filter results by experience level and company size preferences
3. [APPLICATION_SUBMIT] Prepare and submit applications to top 5 matching roles
4. [INTERVIEW_PRACTICE] Schedule mock interviews for common Python interview questions
5. [PROFILE_UPDATE] Update resume and LinkedIn profile with latest projects
        """)
        
        # Create planner with mock model
        planner = PlannerNode(reasoning_model=mock_model)
        
        # Create realistic state
        state = AgentState()
        state.messages = [
            HumanMessage(content="I want to find a new Python developer job. I have 3 years of experience and prefer remote work.")
        ]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com"),
            skills=[Skill(name="Python", level="Advanced", years_experience=3)]
        )
        state.memory_context = {
            "memories": {"user": [{"content": "Prefers remote work opportunities"}]},
            "total_memories": 1
        }
        
        # Execute planner
        result = await planner(state)
        
        # Verify comprehensive results
        assert result["next_node"] == "executor"
        assert result["current_step_index"] == 0
        
        plan = result["current_plan"]
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 5
        
        # Verify step types and content
        step_types = [step.action_type for step in plan.steps]
        assert ActionType.JOB_SEARCH in step_types
        assert ActionType.APPLICATION_SUBMIT in step_types
        assert ActionType.INTERVIEW_PRACTICE in step_types
        assert ActionType.PROFILE_UPDATE in step_types
        
        # Verify step descriptions contain relevant keywords
        descriptions = [step.description for step in plan.steps]
        combined_description = " ".join(descriptions)
        assert "Python" in combined_description
        assert any("search" in desc.lower() for desc in descriptions)
        
        # Verify model was called with proper context
        mock_model.ainvoke.assert_called_once()
        call_args = mock_model.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System + User message
        assert "Sovereign Career Architect" in call_args[0].content  # System prompt
        assert "Python developer job" in call_args[1].content  # User prompt with context