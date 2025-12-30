"""Tests for the Profiler Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage

from sovereign_career_architect.core.nodes.profiler import ProfilerNode, create_profiler_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import (
    Memory, 
    MemoryScope, 
    UserProfile, 
    PersonalInfo
)
from sovereign_career_architect.memory.client import MemoryClient


class TestProfilerNode:
    """Test cases for ProfilerNode functionality."""
    
    @pytest.fixture
    def mock_memory_client(self):
        """Create a mock memory client for testing."""
        client = AsyncMock(spec=MemoryClient)
        return client
    
    @pytest.fixture
    def profiler_node(self, mock_memory_client):
        """Create a ProfilerNode with mocked dependencies."""
        return ProfilerNode(memory_client=mock_memory_client)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state for testing."""
        state = AgentState()
        state.messages = [
            HumanMessage(content="Hello, I'm looking for a job in software engineering"),
            AIMessage(content="I'd be happy to help you with your job search!")
        ]
        return state
    
    @pytest.mark.asyncio
    async def test_profiler_node_basic_execution(self, profiler_node, sample_state, mock_memory_client):
        """Test basic profiler node execution."""
        # Setup mocks
        mock_memory_client.search_memories.return_value = []
        
        # Execute profiler node
        result = await profiler_node(sample_state)
        
        # Verify results
        assert isinstance(result, dict)
        assert "memory_context" in result
        assert "next_node" in result
        assert result["next_node"] == "planner"
        
        # Verify memory client was called
        assert mock_memory_client.search_memories.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_profiler_node_with_existing_profile(self, profiler_node, sample_state, mock_memory_client):
        """Test profiler node when user profile already exists in state."""
        # Setup state with existing profile
        sample_state.user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Test User",
                email="test@example.com"
            )
        )
        
        # Setup mocks
        mock_memory_client.search_memories.return_value = []
        
        # Execute profiler node
        result = await profiler_node(sample_state)
        
        # Verify existing profile is preserved
        assert result["user_profile"] == sample_state.user_profile
    
    @pytest.mark.asyncio
    async def test_profiler_node_memory_retrieval(self, profiler_node, sample_state, mock_memory_client):
        """Test memory retrieval functionality."""
        # Setup mock memories
        user_memory = Memory(
            content="User prefers remote work",
            scope=MemoryScope.USER,
            user_id=sample_state.user_id
        )
        session_memory = Memory(
            content="Currently looking for Python developer roles",
            scope=MemoryScope.SESSION,
            user_id=sample_state.user_id,
            session_id=sample_state.session_id
        )
        
        # Configure mock to return different memories for different scopes
        def mock_search_side_effect(*args, **kwargs):
            scope = kwargs.get('scope')
            if scope == MemoryScope.USER:
                return [user_memory]
            elif scope == MemoryScope.SESSION:
                return [session_memory]
            else:
                return []
        
        mock_memory_client.search_memories.side_effect = mock_search_side_effect
        
        # Execute profiler node
        result = await profiler_node(sample_state)
        
        # Verify memory context structure
        memory_context = result["memory_context"]
        assert "memories" in memory_context
        assert "user" in memory_context["memories"]
        assert "session" in memory_context["memories"]
        assert "agent" in memory_context["memories"]
        assert memory_context["total_memories"] > 0
    
    @pytest.mark.asyncio
    async def test_profiler_node_error_handling(self, profiler_node, sample_state, mock_memory_client):
        """Test error handling in profiler node."""
        # Setup mock to raise exception
        mock_memory_client.search_memories.side_effect = Exception("Memory service unavailable")
        
        # Execute profiler node
        result = await profiler_node(sample_state)
        
        # Verify graceful error handling
        assert isinstance(result, dict)
        assert "memory_context" in result
        assert "error" in result["memory_context"]
        assert result["next_node"] == "planner"  # Should still route to next node
    
    @pytest.mark.asyncio
    async def test_profiler_node_no_user_messages(self, profiler_node, mock_memory_client):
        """Test profiler node with no user messages in state."""
        # Create state with only AI messages
        state = AgentState()
        state.messages = [
            AIMessage(content="Hello! How can I help you today?")
        ]
        
        mock_memory_client.search_memories.return_value = []
        
        # Execute profiler node
        result = await profiler_node(state)
        
        # Should still work without user messages
        assert isinstance(result, dict)
        assert "memory_context" in result
        assert "next_node" in result
    
    def test_get_latest_user_message(self, profiler_node):
        """Test extraction of latest user message."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Second message"),
            AIMessage(content="Another AI response")
        ]
        
        latest = profiler_node._get_latest_user_message(messages)
        
        assert latest is not None
        assert latest.content == "Second message"
    
    def test_get_latest_user_message_no_user_messages(self, profiler_node):
        """Test extraction when no user messages exist."""
        messages = [
            AIMessage(content="AI response 1"),
            AIMessage(content="AI response 2")
        ]
        
        latest = profiler_node._get_latest_user_message(messages)
        
        assert latest is None
    
    def test_memory_to_dict_conversion(self, profiler_node):
        """Test memory object to dictionary conversion."""
        memory = Memory(
            content="Test memory content",
            scope=MemoryScope.USER,
            user_id=uuid4(),
            importance_score=0.8,
            metadata={"category": "skills"}
        )
        
        result = profiler_node._memory_to_dict(memory)
        
        assert isinstance(result, dict)
        assert result["content"] == "Test memory content"
        assert result["scope"] == "user"
        assert result["importance_score"] == 0.8
        assert result["metadata"]["category"] == "skills"
        assert "id" in result
        assert "timestamp" in result
    
    def test_create_profiler_node_factory(self):
        """Test the factory function for creating profiler nodes."""
        node = create_profiler_node()
        
        assert isinstance(node, ProfilerNode)
        assert node.memory_client is not None
    
    def test_create_profiler_node_with_custom_client(self, mock_memory_client):
        """Test factory function with custom memory client."""
        node = create_profiler_node(memory_client=mock_memory_client)
        
        assert isinstance(node, ProfilerNode)
        assert node.memory_client == mock_memory_client


class TestProfilerNodeIntegration:
    """Integration tests for ProfilerNode with real MemoryClient."""
    
    @pytest.mark.asyncio
    async def test_profiler_with_real_memory_client(self):
        """Test profiler node with actual MemoryClient (mock storage)."""
        # Create real memory client (uses in-memory mock storage)
        memory_client = MemoryClient()
        profiler_node = ProfilerNode(memory_client=memory_client)
        
        # Create state
        state = AgentState()
        state.messages = [HumanMessage(content="I need help finding a job")]
        
        # Add some test memories
        await memory_client.add_memory(
            user_id=state.user_id,
            content="User has 5 years Python experience",
            scope=MemoryScope.USER,
            importance_score=0.9
        )
        
        await memory_client.add_memory(
            user_id=state.user_id,
            content="Looking for remote opportunities",
            scope=MemoryScope.SESSION,
            session_id=state.session_id,
            importance_score=0.7
        )
        
        # Execute profiler node
        result = await profiler_node(state)
        
        # Verify results
        assert isinstance(result, dict)
        assert "memory_context" in result
        assert result["memory_context"]["total_memories"] >= 0
        assert "next_node" in result