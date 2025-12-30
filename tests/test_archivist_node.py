"""Tests for the Archivist Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from sovereign_career_architect.core.nodes.archivist import ArchivistNode, create_archivist_node
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
    MemoryScope,
)
from sovereign_career_architect.memory.client import MemoryClient


class MockExtractionResponse:
    """Mock extraction model response for testing."""
    def __init__(self, content: str):
        self.content = content


class TestArchivistNode:
    """Test cases for ArchivistNode functionality."""
    
    @pytest.fixture
    def mock_memory_client(self):
        """Create a mock memory client for testing."""
        client = AsyncMock(spec=MemoryClient)
        return client
    
    @pytest.fixture
    def mock_extraction_model(self):
        """Create a mock extraction model for testing."""
        model = AsyncMock()
        return model
    
    @pytest.fixture
    def archivist_node(self, mock_memory_client, mock_extraction_model):
        """Create an ArchivistNode with mocked dependencies."""
        return ArchivistNode(
            memory_client=mock_memory_client,
            extraction_model=mock_extraction_model
        )
    
    @pytest.fixture
    def sample_state_with_interaction(self):
        """Create a sample agent state with interaction history."""
        state = AgentState()
        
        # Add conversation history
        state.messages = [
            HumanMessage(content="I want to find a Python developer job in San Francisco"),
            AIMessage(content="I'll help you search for Python developer positions in San Francisco"),
            HumanMessage(content="I have 5 years of experience with Django and React"),
            AIMessage(content="Great! That's valuable experience. Let me search for senior roles.")
        ]
        
        # Add tool outputs
        job_search_result = ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={
                "jobs_found": 8,
                "matches": [
                    {"title": "Senior Python Developer", "company": "TechCorp", "location": "San Francisco"},
                    {"title": "Full Stack Developer", "company": "StartupXYZ", "location": "San Francisco"}
                ]
            }
        )
        state.add_tool_output(job_search_result)
        
        # Add execution plan
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    description="Search for Python developer roles in San Francisco",
                    action_type=ActionType.JOB_SEARCH
                ),
                PlanStep(
                    description="Apply to top matching positions",
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
                name="Alice Developer",
                email="alice@example.com",
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
    async def test_archivist_node_basic_execution(
        self, 
        archivist_node, 
        sample_state_with_interaction, 
        mock_extraction_model,
        mock_memory_client
    ):
        """Test basic archivist node execution with fact extraction."""
        # Setup mock extraction response
        mock_response = MockExtractionResponse("""
SCOPE: USER
CATEGORY: skill
FACT: User has 5 years of Python development experience
CONFIDENCE: 0.9

SCOPE: USER
CATEGORY: preference
FACT: User prefers Python developer roles in San Francisco
CONFIDENCE: 0.8

SCOPE: SESSION
CATEGORY: context
FACT: User is currently searching for senior Python developer positions
CONFIDENCE: 0.7
        """)
        mock_extraction_model.ainvoke.return_value = mock_response
        
        # Execute archivist node
        result = await archivist_node(sample_state_with_interaction)
        
        # Verify results
        assert isinstance(result, dict)
        assert result["tool_outputs"] == []  # Should clear outputs
        assert result["critique"] is None    # Should clear critique
        assert result["next_node"] is None   # Should end workflow
        
        # Verify extraction model was called
        mock_extraction_model.ainvoke.assert_called_once()
        
        # Verify memory updates were called
        assert mock_memory_client.add_memory.call_count == 3  # Three facts total
        
        # Verify the calls were made with correct scopes
        calls = mock_memory_client.add_memory.call_args_list
        scopes = [call[1]["scope"] for call in calls]
        assert MemoryScope.USER in scopes
        assert MemoryScope.SESSION in scopes
    
    @pytest.mark.asyncio
    async def test_archivist_node_with_user_profile(
        self, 
        archivist_node, 
        sample_state_with_interaction,
        sample_user_profile,
        mock_extraction_model,
        mock_memory_client
    ):
        """Test archivist node with existing user profile context."""
        sample_state_with_interaction.user_profile = sample_user_profile
        
        # Setup mock extraction response
        mock_response = MockExtractionResponse("""
SCOPE: USER
CATEGORY: goal
FACT: User wants to transition to senior Python developer roles
CONFIDENCE: 0.8
        """)
        mock_extraction_model.ainvoke.return_value = mock_response
        
        # Execute archivist node
        result = await archivist_node(sample_state_with_interaction)
        
        # Verify extraction model was called with profile context
        mock_extraction_model.ainvoke.assert_called_once()
        call_args = mock_extraction_model.ainvoke.call_args[0][0]
        user_prompt = call_args[1].content
        
        # Should include existing profile information
        assert "Alice Developer" in user_prompt
        assert "Python (Expert)" in user_prompt
        
        # Verify memory was updated
        mock_memory_client.add_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_archivist_node_no_extraction_model(self, mock_memory_client):
        """Test archivist node behavior when no extraction model is available."""
        # Create archivist without extraction model
        archivist = ArchivistNode(memory_client=mock_memory_client, extraction_model=None)
        
        state = AgentState()
        state.messages = [HumanMessage(content="Test message")]
        
        # Execute archivist node
        result = await archivist(state)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["tool_outputs"] == []
        assert result["next_node"] is None
        
        # No memory updates should be called
        mock_memory_client.add_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_archivist_node_extraction_error(
        self, 
        archivist_node, 
        sample_state_with_interaction,
        mock_extraction_model,
        mock_memory_client
    ):
        """Test error handling in fact extraction."""
        # Setup mock to raise exception
        mock_extraction_model.ainvoke.side_effect = Exception("Extraction service unavailable")
        
        # Execute archivist node
        result = await archivist_node(sample_state_with_interaction)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["tool_outputs"] == []
        assert result["next_node"] is None
        
        # No memory updates should be called due to extraction failure
        mock_memory_client.add_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_archivist_node_memory_update_error(
        self, 
        archivist_node, 
        sample_state_with_interaction,
        mock_extraction_model,
        mock_memory_client
    ):
        """Test error handling in memory updates."""
        # Setup successful extraction
        mock_response = MockExtractionResponse("""
SCOPE: USER
CATEGORY: skill
FACT: User knows Python
CONFIDENCE: 0.9
        """)
        mock_extraction_model.ainvoke.return_value = mock_response
        
        # Setup memory client to raise exception
        mock_memory_client.add_memory.side_effect = Exception("Memory service unavailable")
        
        # Execute archivist node
        result = await archivist_node(sample_state_with_interaction)
        
        # Should handle gracefully and still complete
        assert isinstance(result, dict)
        assert result["tool_outputs"] == []
        assert result["next_node"] is None
    
    def test_parse_extracted_facts_valid_format(self, archivist_node):
        """Test parsing of well-formatted extracted facts."""
        facts_text = """
SCOPE: USER
CATEGORY: skill
FACT: User has 5 years of Python experience
CONFIDENCE: 0.9

SCOPE: SESSION
CATEGORY: context
FACT: User is looking for remote work
CONFIDENCE: 0.8

SCOPE: AGENT
CATEGORY: strategy
FACT: User responds well to detailed job descriptions
CONFIDENCE: 0.7
        """
        
        state = AgentState()
        facts = archivist_node._parse_extracted_facts(facts_text, state)
        
        assert len(facts) == 3
        
        # Check first fact
        assert facts[0]["scope"] == "USER"
        assert facts[0]["category"] == "skill"
        assert "Python experience" in facts[0]["fact"]
        assert facts[0]["confidence"] == 0.9
        
        # Check metadata
        for fact in facts:
            assert "user_id" in fact
            assert "session_id" in fact
            assert "timestamp" in fact
            assert "source" in fact
    
    def test_parse_extracted_facts_malformed(self, archivist_node):
        """Test parsing of malformed extracted facts."""
        facts_text = """
This is not properly formatted.
SCOPE: USER
FACT: Missing category
CONFIDENCE: invalid_number

SCOPE: INVALID_SCOPE
CATEGORY: test
FACT: Invalid scope should be ignored
CONFIDENCE: 0.5
        """
        
        state = AgentState()
        facts = archivist_node._parse_extracted_facts(facts_text, state)
        
        # Should handle gracefully and extract what it can
        assert isinstance(facts, list)
        # May have some facts depending on parsing logic
    
    def test_build_extraction_system_prompt(self, archivist_node):
        """Test system prompt building for fact extraction."""
        prompt = archivist_node._build_extraction_system_prompt()
        
        assert "Archivist component" in prompt
        assert "salient facts" in prompt
        assert "USER" in prompt
        assert "SESSION" in prompt
        assert "AGENT" in prompt
        assert "SCOPE:" in prompt
        assert "CATEGORY:" in prompt
        assert "FACT:" in prompt
        assert "CONFIDENCE:" in prompt
    
    def test_build_extraction_user_prompt_full_context(self, archivist_node, sample_user_profile):
        """Test user prompt building with full context."""
        state = AgentState()
        state.messages = [
            HumanMessage(content="I want to find a job"),
            AIMessage(content="I'll help you with that")
        ]
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={"jobs_found": 5}
        ))
        state.user_profile = sample_user_profile
        
        plan = ExecutionPlan(
            steps=[PlanStep(description="Search for jobs", action_type=ActionType.JOB_SEARCH)],
            user_id=state.user_id
        )
        state.current_plan = plan
        state.current_step_index = 0
        
        prompt = archivist_node._build_extraction_user_prompt(state)
        
        assert "CONVERSATION HISTORY:" in prompt
        assert "ACTIONS PERFORMED:" in prompt
        assert "PLAN CONTEXT:" in prompt
        assert "EXISTING USER PROFILE:" in prompt
        assert "Alice Developer" in prompt
        assert "job_search" in prompt
        assert "jobs_found: 5" in prompt
    
    def test_build_extraction_user_prompt_minimal_context(self, archivist_node):
        """Test user prompt building with minimal context."""
        state = AgentState()
        state.messages = [HumanMessage(content="Hello")]
        
        prompt = archivist_node._build_extraction_user_prompt(state)
        
        assert "CONVERSATION HISTORY:" in prompt
        assert "Hello" in prompt
        assert "Extract salient facts" in prompt
    
    def test_create_archivist_node_factory(self):
        """Test the factory function for creating archivist nodes."""
        # Should create an archivist node even without dependencies
        archivist = create_archivist_node()
        assert isinstance(archivist, ArchivistNode)
        assert isinstance(archivist.memory_client, MemoryClient)
    
    def test_create_archivist_node_with_custom_dependencies(self):
        """Test factory function with custom dependencies."""
        mock_memory = AsyncMock(spec=MemoryClient)
        mock_model = AsyncMock()
        
        node = create_archivist_node(
            memory_client=mock_memory,
            extraction_model=mock_model
        )
        
        assert isinstance(node, ArchivistNode)
        assert node.memory_client == mock_memory
        assert node.extraction_model == mock_model


class TestArchivistNodeIntegration:
    """Integration tests for ArchivistNode."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_memory_consolidation(self):
        """Test complete memory consolidation workflow."""
        # Create mock dependencies
        mock_memory = AsyncMock(spec=MemoryClient)
        mock_model = AsyncMock()
        
        # Setup realistic extraction response
        mock_model.ainvoke.return_value = MockExtractionResponse("""
SCOPE: USER
CATEGORY: skill
FACT: User has expertise in Python and Django development
CONFIDENCE: 0.9

SCOPE: USER
CATEGORY: preference
FACT: User prefers remote or hybrid work arrangements
CONFIDENCE: 0.8

SCOPE: SESSION
CATEGORY: context
FACT: User successfully found 8 relevant job matches in current search
CONFIDENCE: 0.7

SCOPE: AGENT
CATEGORY: strategy
FACT: Detailed job descriptions with company info work well for this user
CONFIDENCE: 0.6
        """)
        
        # Create archivist with mocks
        archivist = ArchivistNode(memory_client=mock_memory, extraction_model=mock_model)
        
        # Create realistic state
        state = AgentState()
        state.messages = [
            HumanMessage(content="I'm looking for a Python developer job with remote options"),
            AIMessage(content="I'll search for Python developer positions with remote work options"),
            HumanMessage(content="I have 5 years experience with Django and prefer hybrid/remote work"),
            AIMessage(content="Perfect! Let me find senior Python roles that match your preferences")
        ]
        
        # Add successful job search result
        state.add_tool_output(ActionResult(
            action_type=ActionType.JOB_SEARCH,
            success=True,
            data={
                "jobs_found": 8,
                "matches": [
                    {"title": "Senior Python Developer", "company": "RemoteTech", "remote": True},
                    {"title": "Django Developer", "company": "HybridCorp", "hybrid": True}
                ]
            }
        ))
        
        # Add user profile
        state.user_profile = UserProfile(
            personal_info=PersonalInfo(name="John Developer", email="john@example.com"),
            skills=[Skill(name="Python", level="Expert", years_experience=5)]
        )
        
        # Execute archivist
        result = await archivist(state)
        
        # Verify workflow completion
        assert result["tool_outputs"] == []
        assert result["critique"] is None
        assert result["next_node"] is None
        
        # Verify extraction was called with proper context
        mock_model.ainvoke.assert_called_once()
        call_args = mock_model.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System + User message
        
        user_prompt = call_args[1].content
        assert "Python developer job" in user_prompt
        assert "John Developer" in user_prompt
        assert "jobs_found: 8" in user_prompt
        
        # Verify memory updates
        assert mock_memory.add_memory.call_count == 4  # Four facts total
        
        # Verify the calls were made with correct scopes
        calls = mock_memory.add_memory.call_args_list
        scopes = [call[1]["scope"] for call in calls]
        user_facts = [call for call in calls if call[1]["scope"] == MemoryScope.USER]
        session_facts = [call for call in calls if call[1]["scope"] == MemoryScope.SESSION]
        agent_facts = [call for call in calls if call[1]["scope"] == MemoryScope.AGENT]
        
        assert len(user_facts) == 2    # Two USER facts
        assert len(session_facts) == 1  # One SESSION fact
        assert len(agent_facts) == 1    # One AGENT fact
        
        # Verify memory content
        assert "Python and Django" in user_facts[0][1]["content"]  # First user fact
        assert "remote or hybrid" in user_facts[1][1]["content"]   # Second user fact