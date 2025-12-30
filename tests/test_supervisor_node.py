"""Tests for the Supervisor Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage

from sovereign_career_architect.core.nodes.supervisor import SupervisorNode, create_supervisor_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    ActionResult,
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    PersonalInfo,
)


class MockRoutingResponse:
    """Mock routing model response for testing."""
    def __init__(self, content: str):
        self.content = content


class TestSupervisorNode:
    """Test cases for SupervisorNode functionality."""
    
    @pytest.fixture
    def mock_routing_model(self):
        """Create a mock routing model for testing."""
        model = AsyncMock()
        return model
    
    @pytest.fixture
    def supervisor_node(self, mock_routing_model):
        """Create a SupervisorNode with mocked dependencies."""
        return SupervisorNode(routing_model=mock_routing_model)
    
    @pytest.fixture
    def supervisor_node_no_llm(self):
        """Create a SupervisorNode without LLM for fallback testing."""
        return SupervisorNode(routing_model=None)
    
    @pytest.fixture
    def empty_state(self):
        """Create an empty agent state."""
        return AgentState()
    
    @pytest.fixture
    def state_with_user_message(self):
        """Create state with a user message but no profile."""
        state = AgentState()
        state.messages = [HumanMessage(content="I want to find a job")]
        return state
    
    @pytest.fixture
    def state_with_profile_no_plan(self):
        """Create state with user profile but no plan."""
        state = AgentState()
        state.messages = [HumanMessage(content="Help me find a Python job")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        return state
    
    @pytest.fixture
    def state_with_plan_ready(self):
        """Create state with plan ready for execution."""
        state = AgentState()
        state.messages = [HumanMessage(content="Find me a job")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH)
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        return state
    
    @pytest.fixture
    def state_with_tool_outputs(self):
        """Create state with tool outputs needing review."""
        state = AgentState()
        state.messages = [HumanMessage(content="Find me a job")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH)
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        ))
        return state
    
    @pytest.fixture
    def state_with_critique(self):
        """Create state with critique for plan revision."""
        state = AgentState()
        state.messages = [HumanMessage(content="Find me a job")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH)
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        ))
        state.critique = "Results are unsatisfactory, need more specific search"
        state.retry_count = 1
        return state
    
    @pytest.fixture
    def state_ready_for_archival(self):
        """Create state ready for archival."""
        state = AgentState()
        state.messages = [
            HumanMessage(content="Find me a job"),
            AIMessage(content="I found 5 relevant positions for you")
        ]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH)
            ],
            user_id=state.user_id
        )
        state.current_step_index = 1  # Completed all steps
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        ))
        state.critique = "Results are satisfactory"
        return state
    
    @pytest.mark.asyncio
    async def test_supervisor_empty_state_routes_to_profiler(
        self, 
        supervisor_node_no_llm, 
        empty_state
    ):
        """Test that empty state routes to profiler."""
        result = await supervisor_node_no_llm(empty_state)
        
        assert result["next_node"] == "profiler"
        assert "routing_reason" in result
    
    @pytest.mark.asyncio
    async def test_supervisor_user_message_no_profile_routes_to_profiler(
        self, 
        supervisor_node_no_llm, 
        state_with_user_message
    ):
        """Test that user message without profile routes to profiler."""
        result = await supervisor_node_no_llm(state_with_user_message)
        
        assert result["next_node"] == "profiler"
    
    @pytest.mark.asyncio
    async def test_supervisor_profile_no_plan_routes_to_planner(
        self, 
        supervisor_node_no_llm, 
        state_with_profile_no_plan
    ):
        """Test that profile without plan routes to planner."""
        result = await supervisor_node_no_llm(state_with_profile_no_plan)
        
        assert result["next_node"] == "planner"
    
    @pytest.mark.asyncio
    async def test_supervisor_plan_ready_routes_to_executor(
        self, 
        supervisor_node_no_llm, 
        state_with_plan_ready
    ):
        """Test that ready plan routes to executor."""
        result = await supervisor_node_no_llm(state_with_plan_ready)
        
        assert result["next_node"] == "executor"
    
    @pytest.mark.asyncio
    async def test_supervisor_tool_outputs_route_to_reviewer(
        self, 
        supervisor_node_no_llm, 
        state_with_tool_outputs
    ):
        """Test that tool outputs route to reviewer."""
        result = await supervisor_node_no_llm(state_with_tool_outputs)
        
        assert result["next_node"] == "reviewer"
    
    @pytest.mark.asyncio
    async def test_supervisor_unsatisfactory_critique_routes_to_planner(
        self, 
        supervisor_node_no_llm, 
        state_with_critique
    ):
        """Test that unsatisfactory critique routes back to planner."""
        result = await supervisor_node_no_llm(state_with_critique)
        
        assert result["next_node"] == "planner"
    
    @pytest.mark.asyncio
    async def test_supervisor_completed_work_routes_to_archivist(
        self, 
        supervisor_node_no_llm, 
        state_ready_for_archival
    ):
        """Test that completed work routes to archivist."""
        result = await supervisor_node_no_llm(state_ready_for_archival)
        
        assert result["next_node"] == "archivist"
    
    @pytest.mark.asyncio
    async def test_supervisor_llm_routing_fallback_on_error(
        self, 
        supervisor_node, 
        mock_routing_model
    ):
        """Test fallback to LLM routing when heuristic routing is unclear."""
        # Setup mock response
        mock_routing_model.ainvoke.return_value = MockRoutingResponse("planner")
        
        # Create a supervisor that forces LLM usage by overriding deterministic logic
        class TestSupervisor(SupervisorNode):
            async def _determine_next_node(self, state):
                # Force LLM routing for this test
                if self.routing_model:
                    return await self._llm_based_routing(state)
                return "end"
        
        test_supervisor = TestSupervisor(routing_model=mock_routing_model)
        
        state = AgentState()
        state.messages = [HumanMessage(content="Complex scenario")]
        
        result = await test_supervisor(state)
        
        # Should use LLM routing
        mock_routing_model.ainvoke.assert_called_once()
        assert result["next_node"] == "planner"
    
    @pytest.mark.asyncio
    async def test_supervisor_llm_routing_fallback_on_error(
        self, 
        supervisor_node, 
        mock_routing_model,
        state_with_user_message
    ):
        """Test fallback to heuristic routing when LLM fails."""
        # Setup mock to raise exception
        mock_routing_model.ainvoke.side_effect = Exception("LLM service unavailable")
        
        result = await supervisor_node(state_with_user_message)
        
        # Should fallback to heuristic routing
        assert result["next_node"] in ["profiler", "planner", "end"]
    
    @pytest.mark.asyncio
    async def test_supervisor_error_handling(self, supervisor_node_no_llm):
        """Test error handling in supervisor node."""
        # Create invalid state that might cause errors
        invalid_state = None
        
        result = await supervisor_node_no_llm(invalid_state)
        
        # Should handle gracefully and route to end
        assert result["next_node"] == "end"
        assert "error" in result["routing_reason"].lower()
    
    def test_fallback_routing_job_search_intent(self, supervisor_node_no_llm):
        """Test fallback routing for job search intent."""
        state = AgentState()
        state.messages = [HumanMessage(content="I need to find a software engineering job")]
        
        result = supervisor_node_no_llm._fallback_routing(state)
        
        assert result == "planner"  # Should plan for job search
    
    def test_fallback_routing_interview_intent(self, supervisor_node_no_llm):
        """Test fallback routing for interview intent."""
        state = AgentState()
        state.messages = [HumanMessage(content="Help me practice for interviews")]
        
        result = supervisor_node_no_llm._fallback_routing(state)
        
        assert result == "planner"  # Should plan for interview practice
    
    def test_fallback_routing_unknown_intent(self, supervisor_node_no_llm):
        """Test fallback routing for unknown intent."""
        state = AgentState()
        state.messages = [HumanMessage(content="Hello, how are you?")]
        
        result = supervisor_node_no_llm._fallback_routing(state)
        
        assert result == "end"  # Should end for general chat
    
    def test_parse_routing_response_valid_nodes(self, supervisor_node_no_llm):
        """Test parsing of valid routing responses."""
        test_cases = [
            ("profiler", "profiler"),
            ("The next node should be planner", "planner"),
            ("EXECUTOR", "executor"),
            ("reviewer node", "reviewer"),
            ("archivist", "archivist"),
            ("end the conversation", "end"),
            ("unknown response", "end")  # Default fallback
        ]
        
        for response, expected in test_cases:
            result = supervisor_node_no_llm._parse_routing_response(response)
            assert result == expected
    
    def test_build_routing_system_prompt(self, supervisor_node_no_llm):
        """Test system prompt building for routing."""
        prompt = supervisor_node_no_llm._build_routing_system_prompt()
        
        assert "Supervisor component" in prompt
        assert "profiler" in prompt
        assert "planner" in prompt
        assert "executor" in prompt
        assert "reviewer" in prompt
        assert "archivist" in prompt
        assert "end" in prompt
    
    def test_build_routing_user_prompt_full_context(self, supervisor_node_no_llm):
        """Test user prompt building with full context."""
        state = AgentState()
        state.messages = [
            HumanMessage(content="Find me a job"),
            AIMessage(content="I'll help you search")
        ]
        state.current_plan = ExecutionPlan(
            steps=[PlanStep(description="Search", action_type=ActionType.JOB_SEARCH)],
            user_id=state.user_id
        )
        state.current_step_index = 0
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs": 5}
        ))
        state.critique = "Good results"
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com")
        )
        
        prompt = supervisor_node_no_llm._build_routing_user_prompt(state)
        
        assert "CONVERSATION HISTORY:" in prompt
        assert "CURRENT PLAN:" in prompt
        assert "TOOL OUTPUTS:" in prompt
        assert "LAST CRITIQUE:" in prompt
        assert "USER PROFILE:" in prompt
        assert "RETRY COUNT:" in prompt
    
    def test_get_routing_reason(self, supervisor_node_no_llm):
        """Test routing reason generation."""
        state = AgentState()
        
        reasons = {
            "profiler": supervisor_node_no_llm._get_routing_reason(state, "profiler"),
            "planner": supervisor_node_no_llm._get_routing_reason(state, "planner"),
            "executor": supervisor_node_no_llm._get_routing_reason(state, "executor"),
            "reviewer": supervisor_node_no_llm._get_routing_reason(state, "reviewer"),
            "archivist": supervisor_node_no_llm._get_routing_reason(state, "archivist"),
            "end": supervisor_node_no_llm._get_routing_reason(state, "end")
        }
        
        for node, reason in reasons.items():
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    def test_create_supervisor_node_factory(self):
        """Test the factory function for creating supervisor nodes."""
        # Test without dependencies
        supervisor = create_supervisor_node()
        assert isinstance(supervisor, SupervisorNode)
        assert supervisor.routing_model is None
        
        # Test with custom model
        mock_model = AsyncMock()
        supervisor_with_model = create_supervisor_node(routing_model=mock_model)
        assert isinstance(supervisor_with_model, SupervisorNode)
        assert supervisor_with_model.routing_model == mock_model


class TestSupervisorNodeIntegration:
    """Integration tests for SupervisorNode."""
    
    @pytest.mark.asyncio
    async def test_complete_routing_workflow(self):
        """Test complete routing workflow through different states."""
        supervisor = create_supervisor_node()
        
        # Test progression through typical workflow
        states = []
        
        # 1. Empty state -> profiler
        state1 = AgentState()
        result1 = await supervisor(state1)
        states.append((state1, result1))
        assert result1["next_node"] == "profiler"
        
        # 2. User message, no profile -> profiler
        state2 = AgentState()
        state2.messages = [HumanMessage(content="I want a job")]
        result2 = await supervisor(state2)
        states.append((state2, result2))
        assert result2["next_node"] == "profiler"
        
        # 3. Has profile, no plan -> planner
        state3 = AgentState()
        state3.messages = [HumanMessage(content="Find me a Python job")]
        state3.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com")
        )
        result3 = await supervisor(state3)
        states.append((state3, result3))
        assert result3["next_node"] == "planner"
        
        # 4. Has plan -> executor
        state4 = AgentState()
        state4.messages = [HumanMessage(content="Find me a Python job")]
        state4.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com")
        )
        state4.current_plan = ExecutionPlan(
            steps=[PlanStep(description="Search", action_type=ActionType.JOB_SEARCH)],
            user_id=state4.user_id
        )
        result4 = await supervisor(state4)
        states.append((state4, result4))
        assert result4["next_node"] == "executor"
        
        # Verify all states have routing reasons
        for state, result in states:
            assert "routing_reason" in result
            assert isinstance(result["routing_reason"], str)
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_max_retries(self):
        """Test retry logic respects maximum retry count."""
        supervisor = create_supervisor_node()
        
        # Create state with max retries reached
        state = AgentState()
        state.messages = [HumanMessage(content="Find me a job")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[PlanStep(description="Search", action_type=ActionType.JOB_SEARCH)],
            user_id=state.user_id
        )
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs": 0}
        ))
        state.critique = "Results are unsatisfactory"
        state.retry_count = 3  # Max retries reached
        
        result = await supervisor(state)
        
        # Should not retry, should end or move to archival
        assert result["next_node"] in ["end", "archivist"]
    
    @pytest.mark.asyncio
    async def test_multi_step_plan_execution_routing(self):
        """Test routing for multi-step plan execution."""
        supervisor = create_supervisor_node()
        
        # Create state with multi-step plan
        state = AgentState()
        state.messages = [HumanMessage(content="Find and apply to jobs")]
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH),
                PlanStep(description="Apply to jobs", action_type=ActionType.APPLICATION_SUBMIT)
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        ))
        state.critique = "Results are satisfactory"
        
        result = await supervisor(state)
        
        # Should continue to executor for next step
        assert result["next_node"] == "executor"