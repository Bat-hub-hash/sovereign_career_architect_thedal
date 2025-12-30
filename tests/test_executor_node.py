"""Tests for the Executor Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sovereign_career_architect.core.nodes.executor import ExecutorNode, create_executor_node
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


class TestExecutorNode:
    """Test cases for ExecutorNode functionality."""
    
    @pytest.fixture
    def mock_browser_agent(self):
        """Create a mock browser agent for testing."""
        agent = AsyncMock()
        return agent
    
    @pytest.fixture
    def mock_api_clients(self):
        """Create mock API clients for testing."""
        return {
            "job_api": AsyncMock(),
            "profile_api": AsyncMock()
        }
    
    @pytest.fixture
    def executor_node(self, mock_browser_agent, mock_api_clients):
        """Create an ExecutorNode with mocked dependencies."""
        return ExecutorNode(
            browser_agent=mock_browser_agent,
            api_clients=mock_api_clients,
            max_retries=3
        )
    
    @pytest.fixture
    def executor_node_minimal(self):
        """Create an ExecutorNode with minimal dependencies."""
        return ExecutorNode()
    
    @pytest.fixture
    def state_no_plan(self):
        """Create state without execution plan."""
        return AgentState()
    
    @pytest.fixture
    def state_with_job_search_plan(self):
        """Create state with job search plan."""
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for Python developer jobs",
                    action_type=ActionType.JOB_SEARCH,
                    parameters={"query": "Python developer", "location": "San Francisco"}
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        return state
    
    @pytest.fixture
    def state_with_application_plan(self):
        """Create state with application submission plan."""
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Apply to TechCorp position",
                    action_type=ActionType.APPLICATION_SUBMIT,
                    parameters={"job_url": "https://example.com/job/123"}
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com")
        )
        return state
    
    @pytest.fixture
    def state_with_interview_plan(self):
        """Create state with interview practice plan."""
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Practice Python developer interview",
                    action_type=ActionType.INTERVIEW_PRACTICE,
                    parameters={"role": "Python Developer", "difficulty": "medium"}
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        return state
    
    @pytest.fixture
    def state_with_profile_update_plan(self):
        """Create state with profile update plan."""
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Update user skills",
                    action_type=ActionType.PROFILE_UPDATE,
                    parameters={"updates": {"skills": ["Python", "Django", "React"]}}
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        return state
    
    @pytest.fixture
    def state_completed_plan(self):
        """Create state with completed plan."""
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for jobs",
                    action_type=ActionType.JOB_SEARCH
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 1  # Beyond last step
        return state
    
    @pytest.mark.asyncio
    async def test_executor_no_plan_returns_error(
        self, 
        executor_node_minimal, 
        state_no_plan
    ):
        """Test executor behavior when no plan is available."""
        result = await executor_node_minimal(state_no_plan)
        
        assert "tool_outputs" in result
        assert len(result["tool_outputs"]) == 1
        assert not result["tool_outputs"][0].success
        assert "No execution plan" in result["tool_outputs"][0].error_message
        assert result["next_node"] == "planner"
    
    @pytest.mark.asyncio
    async def test_executor_completed_plan_routes_to_reviewer(
        self, 
        executor_node_minimal, 
        state_completed_plan
    ):
        """Test executor behavior when plan is completed."""
        result = await executor_node_minimal(state_completed_plan)
        
        assert result["next_node"] == "reviewer"
    
    @pytest.mark.asyncio
    async def test_executor_job_search_execution(
        self, 
        executor_node_minimal, 
        state_with_job_search_plan
    ):
        """Test job search action execution."""
        result = await executor_node_minimal(state_with_job_search_plan)
        
        assert "tool_outputs" in result
        assert len(result["tool_outputs"]) == 1
        
        action_result = result["tool_outputs"][0]
        assert action_result.action_type == ActionType.JOB_SEARCH
        assert action_result.success
        assert "jobs_found" in action_result.data
        assert "matches" in action_result.data
        assert result["current_step_index"] == 1  # Incremented
        assert result["next_node"] == "reviewer"
    
    @pytest.mark.asyncio
    async def test_executor_job_search_with_user_profile(
        self, 
        executor_node_minimal, 
        state_with_job_search_plan
    ):
        """Test job search uses user profile for context."""
        # Add user profile with preferences
        state_with_job_search_plan.user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test User", email="test@example.com"),
            preferences=JobPreferences(
                roles=["Senior Python Developer"],
                locations=["New York"]
            )
        )
        
        # Clear search parameters to test profile usage
        state_with_job_search_plan.current_plan.steps[0].parameters = {}
        
        result = await executor_node_minimal(state_with_job_search_plan)
        
        action_result = result["tool_outputs"][0]
        assert action_result.success
        assert "Senior Python Developer" in action_result.data["query"]
        assert action_result.data["location"] == "New York"
    
    @pytest.mark.asyncio
    async def test_executor_application_submission(
        self, 
        executor_node_minimal, 
        state_with_application_plan
    ):
        """Test job application submission execution."""
        result = await executor_node_minimal(state_with_application_plan)
        
        assert "tool_outputs" in result
        action_result = result["tool_outputs"][0]
        assert action_result.action_type == ActionType.APPLICATION_SUBMIT
        assert action_result.success
        assert "job_url" in action_result.data
        assert "submitted_at" in action_result.data
        assert "application_id" in action_result.data
    
    @pytest.mark.asyncio
    async def test_executor_application_no_url_fails(
        self, 
        executor_node_minimal, 
        state_with_application_plan
    ):
        """Test application submission fails without job URL."""
        # Remove job URL from parameters
        state_with_application_plan.current_plan.steps[0].parameters = {}
        
        result = await executor_node_minimal(state_with_application_plan)
        
        action_result = result["tool_outputs"][0]
        assert not action_result.success
        assert "No job URL" in action_result.error_message
    
    @pytest.mark.asyncio
    async def test_executor_interview_practice(
        self, 
        executor_node_minimal, 
        state_with_interview_plan
    ):
        """Test interview practice execution."""
        result = await executor_node_minimal(state_with_interview_plan)
        
        action_result = result["tool_outputs"][0]
        assert action_result.action_type == ActionType.INTERVIEW_PRACTICE
        assert action_result.success
        assert "role" in action_result.data
        assert "difficulty" in action_result.data
        assert "questions_generated" in action_result.data
        assert "questions" in action_result.data
        assert action_result.data["role"] == "Python Developer"
    
    @pytest.mark.asyncio
    async def test_executor_profile_update(
        self, 
        executor_node_minimal, 
        state_with_profile_update_plan
    ):
        """Test profile update execution."""
        result = await executor_node_minimal(state_with_profile_update_plan)
        
        action_result = result["tool_outputs"][0]
        assert action_result.action_type == ActionType.PROFILE_UPDATE
        assert action_result.success
        assert "updated_fields" in action_result.data
        assert "update_count" in action_result.data
        assert "updated_at" in action_result.data
        assert "skills" in action_result.data["updated_fields"]
    
    @pytest.mark.asyncio
    async def test_executor_unknown_action_type(self, executor_node_minimal):
        """Test executor behavior with unknown action type."""
        # Create a custom executor that simulates unknown action handling
        class TestExecutor(ExecutorNode):
            async def _execute_action(self, step, state):
                # Simulate unknown action type handling
                return ActionResult(
                    action_type=step.action_type,
                    success=False,
                    error_message=f"Unknown action type: {step.action_type}"
                )
        
        executor = TestExecutor()
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Unknown action",
                    action_type=ActionType.GENERAL_CHAT  # Use valid enum but simulate unknown handling
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        
        result = await executor(state)
        
        action_result = result["tool_outputs"][0]
        assert not action_result.success
        assert "Unknown action type" in action_result.error_message
    
    @pytest.mark.asyncio
    async def test_executor_retry_logic_success_after_failure(self, executor_node_minimal):
        """Test retry logic succeeds after initial failure."""
        # Create a custom executor that fails first attempt
        class FailOnceExecutor(ExecutorNode):
            def __init__(self):
                super().__init__(max_retries=3)
                self.attempt_count = 0
            
            async def _execute_action(self, step, state):
                self.attempt_count += 1
                if self.attempt_count == 1:
                    raise Exception("Simulated failure")
                return ActionResult(
                    action_type=step.action_type,
                    success=True,
                    data={"attempt": self.attempt_count}
                )
        
        executor = FailOnceExecutor()
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Test step",
                    action_type=ActionType.JOB_SEARCH
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        
        result = await executor(state)
        
        action_result = result["tool_outputs"][0]
        assert action_result.success
        assert action_result.data["attempt"] == 2  # Succeeded on second attempt
    
    @pytest.mark.asyncio
    async def test_executor_retry_logic_max_retries_exceeded(self, executor_node_minimal):
        """Test retry logic fails after max retries."""
        # Create a custom executor that always fails
        class AlwaysFailExecutor(ExecutorNode):
            def __init__(self):
                super().__init__(max_retries=2)  # Lower retry count for faster test
            
            async def _execute_action(self, step, state):
                raise Exception("Always fails")
        
        executor = AlwaysFailExecutor()
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Test step",
                    action_type=ActionType.JOB_SEARCH
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        
        result = await executor(state)
        
        action_result = result["tool_outputs"][0]
        assert not action_result.success
        assert "Failed after 2 attempts" in action_result.error_message
        assert result["current_step_index"] == 0  # Should not increment on failure
    
    @pytest.mark.asyncio
    async def test_executor_execution_time_tracking(
        self, 
        executor_node_minimal, 
        state_with_job_search_plan
    ):
        """Test that execution time is tracked."""
        result = await executor_node_minimal(state_with_job_search_plan)
        
        action_result = result["tool_outputs"][0]
        assert action_result.execution_time is not None
        assert action_result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_executor_error_handling(self, executor_node_minimal):
        """Test executor error handling for unexpected exceptions."""
        # Create state that might cause errors
        state = AgentState()
        state.current_plan = None  # This should be handled gracefully
        
        result = await executor_node_minimal(state)
        
        assert "tool_outputs" in result
        assert len(result["tool_outputs"]) == 1
        assert not result["tool_outputs"][0].success
    
    def test_create_executor_node_factory(self):
        """Test the factory function for creating executor nodes."""
        # Test minimal creation
        executor = create_executor_node()
        assert isinstance(executor, ExecutorNode)
        assert executor.browser_agent is None
        assert executor.api_clients == {}
        assert executor.max_retries == 3
        
        # Test with custom parameters
        mock_browser = AsyncMock()
        mock_clients = {"test": "client"}
        
        executor_custom = create_executor_node(
            browser_agent=mock_browser,
            api_clients=mock_clients,
            max_retries=5
        )
        
        assert executor_custom.browser_agent == mock_browser
        assert executor_custom.api_clients == mock_clients
        assert executor_custom.max_retries == 5


class TestExecutorNodeIntegration:
    """Integration tests for ExecutorNode."""
    
    @pytest.mark.asyncio
    async def test_multi_step_execution_workflow(self):
        """Test execution of multi-step plan."""
        executor = create_executor_node()
        
        # Create multi-step plan
        state = AgentState()
        state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for jobs",
                    action_type=ActionType.JOB_SEARCH,
                    parameters={"query": "Python", "location": "Remote"}
                ),
                PlanStep(
                    description="Practice interview",
                    action_type=ActionType.INTERVIEW_PRACTICE,
                    parameters={"role": "Python Developer"}
                )
            ],
            user_id=state.user_id
        )
        state.current_step_index = 0
        
        # Execute first step
        result1 = await executor(state)
        assert result1["current_step_index"] == 1
        assert len(result1["tool_outputs"]) == 1
        assert result1["tool_outputs"][0].action_type == ActionType.JOB_SEARCH
        
        # Update state for second step
        state.current_step_index = result1["current_step_index"]
        state.tool_outputs = result1["tool_outputs"]
        
        # Execute second step
        result2 = await executor(state)
        assert result2["current_step_index"] == 2
        assert len(result2["tool_outputs"]) == 2
        assert result2["tool_outputs"][1].action_type == ActionType.INTERVIEW_PRACTICE
    
    @pytest.mark.asyncio
    async def test_execution_with_state_persistence(self):
        """Test that execution properly updates state."""
        executor = create_executor_node()
        
        # Create initial state
        initial_state = AgentState()
        initial_state.current_plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for Python jobs",
                    action_type=ActionType.JOB_SEARCH,
                    parameters={"query": "Python developer"}
                )
            ],
            user_id=initial_state.user_id
        )
        initial_state.current_step_index = 0
        initial_state.tool_outputs = []
        
        # Execute step
        result = await executor(initial_state)
        
        # Verify state updates
        assert result["current_step_index"] == 1
        assert len(result["tool_outputs"]) == 1
        assert result["tool_outputs"][0].success
        assert result["next_node"] == "reviewer"
        
        # Verify original state wasn't mutated
        assert initial_state.current_step_index == 0
        assert len(initial_state.tool_outputs) == 0
    
    @pytest.mark.asyncio
    async def test_mock_implementations_realistic_data(self):
        """Test that mock implementations return realistic data."""
        executor = create_executor_node()
        
        # Test job search mock
        jobs = await executor._mock_job_search("Python developer", "San Francisco")
        assert len(jobs) > 0
        for job in jobs:
            assert "title" in job
            assert "company" in job
            assert "location" in job
            assert "url" in job
            assert "match_score" in job
            assert "Python" in job["title"]
        
        # Test application submission mock
        success = await executor._mock_application_submit("https://example.com/job", None)
        assert success is True
        
        # Test interview questions mock
        questions = await executor._mock_generate_interview_questions("Software Engineer", "medium")
        assert len(questions) > 0
        for question in questions:
            assert isinstance(question, str)
            assert len(question) > 10  # Reasonable question length