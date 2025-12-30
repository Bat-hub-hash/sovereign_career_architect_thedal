"""Tests for enhanced Mem0 memory client integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sovereign_career_architect.memory.client import (
    MemoryClient,
    create_memory_client,
    create_test_memory_client
)
from sovereign_career_architect.core.models import Memory, MemoryScope


class TestEnhancedMemoryClient:
    """Test enhanced memory client with Mem0 integration."""
    
    @pytest.fixture
    def mock_mem0_client(self):
        """Create a mock Mem0 client."""
        mock_client = MagicMock()
        mock_client.add = MagicMock()
        mock_client.search = MagicMock(return_value=[])
        mock_client.update = MagicMock()
        mock_client.delete = MagicMock()
        return mock_client
    
    @pytest.fixture
    def memory_client_with_mem0(self, mock_mem0_client):
        """Create memory client with mocked Mem0."""
        config = {
            "provider": "qdrant",
            "config": {
                "url": "http://localhost:6333",
                "collection_name": "test_collection",
                "embedding_model": "text-embedding-3-small"
            }
        }
        client = MemoryClient(
            mem0_client=mock_mem0_client,
            vector_store_config=config
        )
        client._initialized = True
        return client
    
    @pytest.fixture
    def test_memory_client(self):
        """Create test memory client."""
        return create_test_memory_client()
    
    def test_memory_client_initialization(self):
        """Test memory client initialization with different configurations."""
        # Test with default configuration
        client = create_memory_client()
        assert client is not None
        assert client.vector_store_config["provider"] in ["qdrant", "chroma", "memory"]
        
        # Test with custom configuration
        custom_config = {
            "provider": "chroma",
            "config": {
                "persist_directory": "./test_data",
                "collection_name": "test_collection"
            }
        }
        client = MemoryClient(vector_store_config=custom_config)
        assert client.vector_store_config == custom_config
    
    def test_vector_store_config_generation(self):
        """Test vector store configuration generation."""
        client = MemoryClient()
        config = client._get_default_vector_config()
        
        assert "provider" in config
        assert "config" in config
        assert config["provider"] in ["qdrant", "chroma", "memory"]
    
    @pytest.mark.asyncio
    async def test_initialization_process(self, memory_client_with_mem0):
        """Test the initialization process."""
        client = memory_client_with_mem0
        client._initialized = False
        
        # Mock the connection test
        with patch.object(client, '_test_mem0_connection', return_value=True):
            result = await client.initialize()
            assert result is True
            assert client._initialized is True
    
    @pytest.mark.asyncio
    async def test_add_memory_with_mem0(self, memory_client_with_mem0):
        """Test adding memory with Mem0 client."""
        user_id = uuid4()
        content = "User has 5 years of Python experience"
        
        memory = await memory_client_with_mem0.add_memory(
            user_id=user_id,
            content=content,
            scope=MemoryScope.USER,
            importance_score=0.8
        )
        
        assert memory.content == content
        assert memory.user_id == user_id
        assert memory.scope == MemoryScope.USER
        assert memory.importance_score == 0.8
        
        # Verify Mem0 client was called
        memory_client_with_mem0.mem0_client.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_memories_with_mem0(self, memory_client_with_mem0):
        """Test searching memories with Mem0 client."""
        user_id = uuid4()
        query = "Python experience"
        
        # Mock Mem0 search results
        mock_results = [
            {
                "memory": "User has 5 years of Python experience",
                "score": 0.9,
                "metadata": {"skill": "python"}
            },
            {
                "memory": "User worked on Django projects",
                "score": 0.8,
                "metadata": {"framework": "django"}
            }
        ]
        memory_client_with_mem0.mem0_client.search.return_value = mock_results
        
        memories = await memory_client_with_mem0.search_memories(
            user_id=user_id,
            query=query,
            scope=MemoryScope.USER,
            limit=5
        )
        
        assert len(memories) == 2
        assert memories[0].content == "User has 5 years of Python experience"
        assert memories[0].importance_score == 0.9
        assert memories[1].content == "User worked on Django projects"
        
        # Verify Mem0 client was called with correct parameters
        memory_client_with_mem0.mem0_client.search.assert_called_once()
        call_args = memory_client_with_mem0.mem0_client.search.call_args[1]
        assert call_args["query"] == query
        assert call_args["limit"] == 5
        assert call_args["user_id"] == str(user_id)
    
    @pytest.mark.asyncio
    async def test_search_memories_fallback(self, memory_client_with_mem0):
        """Test fallback to mock search when Mem0 fails."""
        user_id = uuid4()
        query = "test query"
        
        # Make Mem0 search raise an exception
        memory_client_with_mem0.mem0_client.search.side_effect = Exception("Mem0 error")
        
        # Add a memory to mock storage first
        await memory_client_with_mem0._add_to_mock_storage(Memory(
            content="test memory content",
            scope=MemoryScope.USER,
            user_id=user_id
        ))
        
        memories = await memory_client_with_mem0.search_memories(
            user_id=user_id,
            query="test",
            scope=MemoryScope.USER
        )
        
        # Should fall back to mock search and find the memory
        assert len(memories) == 1
        assert "test" in memories[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_get_user_profile_memories(self, memory_client_with_mem0):
        """Test getting user profile memories."""
        user_id = uuid4()
        
        # Mock search results
        mock_results = [
            {
                "memory": "User is a software engineer",
                "score": 0.9,
                "metadata": {"type": "profile"}
            }
        ]
        memory_client_with_mem0.mem0_client.search.return_value = mock_results
        
        memories = await memory_client_with_mem0.get_user_profile_memories(user_id)
        
        assert len(memories) == 1
        assert "software engineer" in memories[0].content
        
        # Verify search was called with profile-related query
        call_args = memory_client_with_mem0.mem0_client.search.call_args[1]
        assert "profile" in call_args["query"]
        assert call_args["limit"] == 50
    
    @pytest.mark.asyncio
    async def test_get_session_memories(self, memory_client_with_mem0):
        """Test getting session-specific memories."""
        user_id = uuid4()
        session_id = uuid4()
        
        memory_client_with_mem0.mem0_client.search.return_value = []
        
        memories = await memory_client_with_mem0.get_session_memories(
            user_id=user_id,
            session_id=session_id,
            limit=10
        )
        
        assert isinstance(memories, list)
        
        # Verify search was called with session parameters
        call_args = memory_client_with_mem0.mem0_client.search.call_args[1]
        assert call_args["user_id"] == str(user_id)
        assert call_args["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_get_agent_memories(self, memory_client_with_mem0):
        """Test getting agent-level memories."""
        query = "user interaction patterns"
        
        memory_client_with_mem0.mem0_client.search.return_value = []
        
        memories = await memory_client_with_mem0.get_agent_memories(
            query=query,
            limit=5
        )
        
        assert isinstance(memories, list)
        
        # Verify search was called with agent parameters
        call_args = memory_client_with_mem0.mem0_client.search.call_args[1]
        assert call_args["query"] == query
        assert call_args["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_update_memory_importance(self, memory_client_with_mem0):
        """Test updating memory importance score."""
        memory_id = "test-memory-id"
        new_score = 0.9
        
        result = await memory_client_with_mem0.update_memory_importance(
            memory_id=memory_id,
            importance_score=new_score
        )
        
        assert result is True
        memory_client_with_mem0.mem0_client.update.assert_called_once_with(
            memory_id=memory_id,
            data={"importance_score": new_score}
        )
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_client_with_mem0):
        """Test deleting a memory."""
        memory_id = "test-memory-id"
        
        result = await memory_client_with_mem0.delete_memory(memory_id)
        
        assert result is True
        memory_client_with_mem0.mem0_client.delete.assert_called_once_with(
            memory_id=memory_id
        )
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_client_with_mem0):
        """Test getting memory statistics."""
        user_id = uuid4()
        
        # Mock different search results for different scopes
        def mock_search(**kwargs):
            scope_filters = kwargs.get("filters", {})
            if kwargs.get("user_id") == "agent":
                return [{"memory": "agent memory", "score": 0.5}]
            else:
                return [{"memory": "user memory", "score": 0.8}]
        
        memory_client_with_mem0.mem0_client.search.side_effect = mock_search
        
        stats = await memory_client_with_mem0.get_memory_stats(user_id)
        
        assert "user_memories" in stats
        assert "session_memories" in stats
        assert "agent_memories" in stats
        assert "total_memories" in stats
        assert isinstance(stats["total_memories"], int)
    
    @pytest.mark.asyncio
    async def test_mock_storage_functionality(self, test_memory_client):
        """Test mock storage functionality when Mem0 is not available."""
        user_id = uuid4()
        content = "Test memory content"
        
        # Add memory
        memory = await test_memory_client.add_memory(
            user_id=user_id,
            content=content,
            scope=MemoryScope.USER
        )
        
        assert memory.content == content
        
        # Search for memory
        memories = await test_memory_client.search_memories(
            user_id=user_id,
            query="test",
            scope=MemoryScope.USER
        )
        
        assert len(memories) == 1
        assert memories[0].content == content
    
    @pytest.mark.asyncio
    async def test_scope_based_memory_storage(self, memory_client_with_mem0):
        """Test that memories are stored with correct scope parameters."""
        user_id = uuid4()
        session_id = uuid4()
        
        # Test user scope
        await memory_client_with_mem0.add_memory(
            user_id=user_id,
            content="User memory",
            scope=MemoryScope.USER
        )
        
        # Test session scope
        await memory_client_with_mem0.add_memory(
            user_id=user_id,
            content="Session memory",
            scope=MemoryScope.SESSION,
            session_id=session_id
        )
        
        # Test agent scope
        await memory_client_with_mem0.add_memory(
            user_id=user_id,
            content="Agent memory",
            scope=MemoryScope.AGENT
        )
        
        # Verify all three calls were made to Mem0
        assert memory_client_with_mem0.mem0_client.add.call_count == 3
        
        # Check that different user_ids were used for different scopes
        calls = memory_client_with_mem0.mem0_client.add.call_args_list
        
        # User and session should use the actual user_id
        assert calls[0][1]["user_id"] == str(user_id)
        assert calls[1][1]["user_id"] == str(user_id)
        
        # Agent should use "agent" as user_id
        assert calls[2][1]["user_id"] == "agent"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_client_with_mem0):
        """Test error handling in various operations."""
        user_id = uuid4()
        
        # Test search error handling
        memory_client_with_mem0.mem0_client.search.side_effect = Exception("Search failed")
        
        memories = await memory_client_with_mem0.search_memories(
            user_id=user_id,
            query="test",
            scope=MemoryScope.USER
        )
        
        # Should return empty list on error
        assert memories == []
        
        # Test add error handling (should still return memory object)
        memory_client_with_mem0.mem0_client.add.side_effect = Exception("Add failed")
        
        memory = await memory_client_with_mem0.add_memory(
            user_id=user_id,
            content="Test content",
            scope=MemoryScope.USER
        )
        
        # Should still return memory object even if storage failed
        assert memory.content == "Test content"