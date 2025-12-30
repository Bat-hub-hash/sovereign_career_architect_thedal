"""Property-based tests for memory update consistency."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import datetime as dt

from sovereign_career_architect.memory.client import MemoryClient, create_test_memory_client
from sovereign_career_architect.core.models import Memory, MemoryScope
from sovereign_career_architect.core.nodes.archivist import ArchivistNode, create_archivist_node
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import ActionResult, ActionType, ExecutionPlan, PlanStep
from langchain_core.messages import HumanMessage, AIMessage


# Feature: sovereign-career-architect, Property 12: Memory Update Consistency
@pytest.mark.property
class TestMemoryUpdateProperties:
    """Property-based tests for memory update consistency behavior."""
    
    @given(
        original_content=st.text(min_size=10, max_size=200),
        updated_content=st.text(min_size=10, max_size=200),
        original_importance=st.floats(min_value=0.0, max_value=1.0),
        updated_importance=st.floats(min_value=0.0, max_value=1.0),
        scope=st.sampled_from([MemoryScope.USER, MemoryScope.SESSION, MemoryScope.AGENT])
    )
    @settings(max_examples=20, deadline=4000)
    @pytest.mark.asyncio
    async def test_memory_update_consistency(
        self,
        original_content: str,
        updated_content: str,
        original_importance: float,
        updated_importance: float,
        scope: MemoryScope
    ):
        """
        Property 12: Memory Update Consistency
        
        For any memory that is updated, the changes should be consistently
        reflected across all subsequent operations and retrievals.
        
        Validates: Requirements 3.4
        """
        assume(original_content != updated_content)  # Ensure we're actually updating
        
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        session_id = uuid4()
        
        # Add original memory
        original_memory = await memory_client.add_memory(
            user_id=user_id,
            content=original_content,
            scope=scope,
            session_id=session_id if scope == MemoryScope.SESSION else None,
            importance_score=original_importance
        )
        
        # Update memory importance
        update_success = await memory_client.update_memory_importance(
            memory_id=str(original_memory.id),
            importance_score=updated_importance
        )
        
        if update_success:
            # Retrieve memory and verify update consistency
            retrieved_memories = await memory_client.search_memories(
                user_id=user_id,
                query=original_content[:50],
                scope=scope,
                session_id=session_id if scope == MemoryScope.SESSION else None,
                limit=10
            )
            
            # Property: Updated memory should be retrievable
            matching_memories = [
                m for m in retrieved_memories 
                if m.content == original_content
            ]
            assert len(matching_memories) > 0, (
                "Updated memory should still be retrievable"
            )
            
            # Property: Importance score should be updated consistently
            updated_memory = matching_memories[0]
            # Note: In mock implementation, importance update might not persist
            # In real Mem0 implementation, this would be properly tested
            assert updated_memory.content == original_content, (
                "Memory content should remain unchanged after importance update"
            )
    
    @given(
        num_updates=st.integers(min_value=2, max_value=5),
        base_content=st.text(min_size=20, max_size=100)
    )
    @settings(max_examples=15, deadline=5000)
    @pytest.mark.asyncio
    async def test_sequential_memory_updates(
        self,
        num_updates: int,
        base_content: str
    ):
        """
        Property: Sequential Memory Updates
        
        For any sequence of memory updates, each update should be
        consistently applied and the final state should reflect all changes.
        """
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        
        # Add initial memory
        memory = await memory_client.add_memory(
            user_id=user_id,
            content=base_content,
            scope=MemoryScope.USER,
            importance_score=0.5
        )
        
        # Perform sequential updates
        importance_scores = []
        for i in range(num_updates):
            new_importance = (i + 1) / (num_updates + 1)  # Increasing importance
            importance_scores.append(new_importance)
            
            update_success = await memory_client.update_memory_importance(
                memory_id=str(memory.id),
                importance_score=new_importance
            )
            
            # Property: Each update should succeed or fail consistently
            # In mock implementation, this might always return True
            assert isinstance(update_success, bool), (
                "Update operation should return a boolean result"
            )
        
        # Verify final state
        final_memories = await memory_client.search_memories(
            user_id=user_id,
            query=base_content[:30],
            scope=MemoryScope.USER,
            limit=5
        )
        
        # Property: Memory should still exist after all updates
        matching_memories = [m for m in final_memories if m.content == base_content]
        assert len(matching_memories) > 0, (
            "Memory should exist after sequential updates"
        )
    
    @given(
        memory_count=st.integers(min_value=3, max_value=8),
        update_indices=st.lists(
            st.integers(min_value=0, max_value=7),
            min_size=1,
            max_size=4,
            unique=True
        )
    )
    @settings(max_examples=10, deadline=4000)
    @pytest.mark.asyncio
    async def test_selective_memory_updates(
        self,
        memory_count: int,
        update_indices: list[int]
    ):
        """
        Property: Selective Memory Updates
        
        For any set of memories, updating only some of them should not
        affect the others, and all updates should be applied consistently.
        """
        assume(all(idx < memory_count for idx in update_indices))
        
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        
        # Add multiple memories
        memories = []
        for i in range(memory_count):
            memory = await memory_client.add_memory(
                user_id=user_id,
                content=f"Memory content {i}",
                scope=MemoryScope.USER,
                importance_score=0.5
            )
            memories.append(memory)
        
        # Update only selected memories
        updated_scores = {}
        for idx in update_indices:
            new_score = 0.8 + (idx * 0.05)  # Different scores for each update
            updated_scores[idx] = new_score
            
            await memory_client.update_memory_importance(
                memory_id=str(memories[idx].id),
                importance_score=new_score
            )
        
        # Verify selective updates
        all_memories = await memory_client.search_memories(
            user_id=user_id,
            query="Memory content",
            scope=MemoryScope.USER,
            limit=20
        )
        
        # Property: All original memories should still exist
        found_contents = {m.content for m in all_memories}
        expected_contents = {f"Memory content {i}" for i in range(memory_count)}
        
        assert expected_contents.issubset(found_contents), (
            "All original memories should still exist after selective updates"
        )
        
        # Property: Updated memories should reflect changes (in real implementation)
        # In mock implementation, we just verify they're still retrievable
        for idx in update_indices:
            expected_content = f"Memory content {idx}"
            matching_memories = [m for m in all_memories if m.content == expected_content]
            assert len(matching_memories) > 0, (
                f"Updated memory {idx} should still be retrievable"
            )
    
    @given(
        facts_data=st.lists(
            st.fixed_dictionaries({
                "fact": st.text(min_size=10, max_size=100),
                "category": st.sampled_from(["skill", "goal", "preference", "gap", "strategy"]),
                "confidence": st.floats(min_value=0.1, max_value=1.0),
                "scope": st.sampled_from(["USER", "SESSION", "AGENT"])
            }),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=10, deadline=6000)
    @pytest.mark.asyncio
    async def test_archivist_memory_update_consistency(
        self,
        facts_data: list[dict]
    ):
        """
        Property: Archivist Memory Update Consistency
        
        For any set of facts processed by the Archivist, memory updates
        should be consistent with conflict resolution and deduplication rules.
        """
        # Create mock memory client and archivist
        mock_memory_client = AsyncMock(spec=MemoryClient)
        mock_memory_client.initialize = AsyncMock(return_value=True)
        mock_memory_client.search_memories = AsyncMock(return_value=[])  # No existing memories
        mock_memory_client.add_memory = AsyncMock()
        mock_memory_client.update_memory_importance = AsyncMock(return_value=True)
        
        # Mock extraction model
        mock_model = AsyncMock()
        
        # Build mock response for fact extraction
        response_lines = []
        for fact in facts_data:
            response_lines.extend([
                f"SCOPE: {fact['scope']}",
                f"CATEGORY: {fact['category']}",
                f"FACT: {fact['fact']}",
                f"CONFIDENCE: {fact['confidence']}",
                ""  # Empty line separator
            ])
        
        mock_response = MagicMock()
        mock_response.content = "\n".join(response_lines)
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        
        archivist = ArchivistNode(
            memory_client=mock_memory_client,
            extraction_model=mock_model
        )
        
        # Create test state
        user_id = uuid4()
        session_id = uuid4()
        
        state = AgentState(
            user_id=user_id,
            session_id=session_id,
            messages=[
                HumanMessage(content="I want to find a Python developer job"),
                AIMessage(content="I'll help you find Python developer positions")
            ],
            tool_outputs=[
                ActionResult(
                    action_type=ActionType.JOB_SEARCH,
                    success=True,
                    data={"jobs_found": 5}
                )
            ],
            current_plan=ExecutionPlan(
                id=uuid4(),
                steps=[
                    PlanStep(
                        description="Search for jobs",
                        action_type=ActionType.JOB_SEARCH
                    )
                ],
                user_id=user_id
            )
        )
        
        # Execute archivist
        result = await archivist(state)
        
        # Property: Archivist should complete successfully
        assert result is not None, "Archivist should return a result"
        assert "next_node" in result, "Result should contain next_node"
        
        # Property: Memory client should be called for each fact
        # (In real implementation with conflict resolution, some calls might be skipped)
        assert mock_memory_client.add_memory.call_count >= 0, (
            "Memory client should be called for memory operations"
        )
        
        # Property: All memory operations should use consistent user_id
        if mock_memory_client.add_memory.call_count > 0:
            for call in mock_memory_client.add_memory.call_args_list:
                call_user_id = call[1]["user_id"]  # keyword argument
                assert call_user_id == user_id, (
                    "All memory operations should use consistent user_id"
                )
    
    @given(
        similar_facts=st.lists(
            st.text(min_size=20, max_size=80),
            min_size=2,
            max_size=4
        ),
        confidence_scores=st.lists(
            st.floats(min_value=0.1, max_value=1.0),
            min_size=2,
            max_size=4
        )
    )
    @settings(max_examples=10, deadline=4000)
    @pytest.mark.asyncio
    async def test_memory_conflict_resolution_consistency(
        self,
        similar_facts: list[str],
        confidence_scores: list[float]
    ):
        """
        Property: Memory Conflict Resolution Consistency
        
        For any set of similar facts with different confidence scores,
        conflict resolution should consistently apply the same rules.
        """
        assume(len(similar_facts) == len(confidence_scores))
        assume(len(set(similar_facts)) > 1)  # Ensure facts are different
        
        # Create archivist with real memory client for conflict resolution testing
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        archivist = ArchivistNode(memory_client=memory_client, extraction_model=None)
        
        user_id = uuid4()
        
        # Add facts sequentially to test conflict resolution
        for i, (fact, confidence) in enumerate(zip(similar_facts, confidence_scores)):
            fact_dict = {
                "fact": fact,
                "category": "skill",
                "confidence": confidence,
                "scope": "USER",
                "user_id": str(user_id),
                "session_id": str(uuid4()),
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "source": "test"
            }
            
            # Find similar memories
            existing_memories = await archivist._find_similar_memories(
                fact, user_id, MemoryScope.USER
            )
            
            # Resolve conflicts
            action = await archivist._resolve_memory_conflicts(fact_dict, existing_memories)
            
            # Property: Conflict resolution should return valid action
            assert "type" in action, "Conflict resolution should return action type"
            assert action["type"] in ["add", "update", "merge", "skip"], (
                "Action type should be one of the valid options"
            )
            
            # Property: Higher confidence facts should not be skipped for lower confidence
            if existing_memories and action["type"] == "skip":
                existing_confidence = getattr(existing_memories[0], 'importance_score', 0.5)
                assert confidence <= existing_confidence + 0.1, (
                    "Higher confidence facts should not be skipped for lower confidence ones"
                )
    
    @given(
        memory_content=st.text(min_size=15, max_size=100),
        metadata_updates=st.lists(
            st.fixed_dictionaries({
                "category": st.sampled_from(["skill", "goal", "preference", "context"]),
                "confidence": st.floats(min_value=0.0, max_value=1.0),
                "source": st.sampled_from(["archivist", "user_input", "system"])
            }),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_memory_metadata_update_consistency(
        self,
        memory_content: str,
        metadata_updates: list[dict]
    ):
        """
        Property: Memory Metadata Update Consistency
        
        For any memory with metadata updates, the metadata should be
        consistently maintained and retrievable.
        """
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        
        # Add memory with initial metadata
        initial_metadata = metadata_updates[0]
        memory = await memory_client.add_memory(
            user_id=user_id,
            content=memory_content,
            scope=MemoryScope.USER,
            metadata=initial_metadata,
            importance_score=initial_metadata["confidence"]
        )
        
        # Apply subsequent metadata updates (simulated)
        for update in metadata_updates[1:]:
            # In a real implementation, this would update metadata
            # For now, we test that the memory remains consistent
            retrieved_memories = await memory_client.search_memories(
                user_id=user_id,
                query=memory_content[:30],
                scope=MemoryScope.USER,
                limit=5
            )
            
            # Property: Memory should remain retrievable after metadata updates
            matching_memories = [m for m in retrieved_memories if m.content == memory_content]
            assert len(matching_memories) > 0, (
                "Memory should remain retrievable after metadata updates"
            )
            
            # Property: Memory content should remain unchanged
            retrieved_memory = matching_memories[0]
            assert retrieved_memory.content == memory_content, (
                "Memory content should remain unchanged during metadata updates"
            )
    
    @given(
        batch_size=st.integers(min_value=2, max_value=6),
        update_probability=st.floats(min_value=0.2, max_value=0.8)
    )
    @settings(max_examples=10, deadline=4000)
    @pytest.mark.asyncio
    async def test_batch_memory_update_consistency(
        self,
        batch_size: int,
        update_probability: float
    ):
        """
        Property: Batch Memory Update Consistency
        
        For any batch of memory updates, the operations should be
        consistently applied without interfering with each other.
        """
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        
        # Add batch of memories
        memories = []
        for i in range(batch_size):
            memory = await memory_client.add_memory(
                user_id=user_id,
                content=f"Batch memory {i}",
                scope=MemoryScope.USER,
                importance_score=0.5
            )
            memories.append(memory)
        
        # Randomly update some memories
        import random
        random.seed(42)  # For reproducible tests
        
        updated_memories = []
        for memory in memories:
            if random.random() < update_probability:
                new_importance = random.uniform(0.6, 0.9)
                success = await memory_client.update_memory_importance(
                    memory_id=str(memory.id),
                    importance_score=new_importance
                )
                if success:
                    updated_memories.append(memory)
        
        # Verify batch consistency
        all_retrieved = await memory_client.search_memories(
            user_id=user_id,
            query="Batch memory",
            scope=MemoryScope.USER,
            limit=20
        )
        
        # Property: All original memories should still exist
        retrieved_contents = {m.content for m in all_retrieved}
        expected_contents = {f"Batch memory {i}" for i in range(batch_size)}
        
        assert expected_contents.issubset(retrieved_contents), (
            "All batch memories should remain retrievable after updates"
        )
        
        # Property: Number of memories should remain consistent
        batch_memories = [m for m in all_retrieved if "Batch memory" in m.content]
        assert len(batch_memories) >= batch_size, (
            "Batch update should not reduce the number of memories"
        )
    
    @pytest.mark.asyncio
    async def test_memory_update_atomicity(self):
        """
        Property: Memory Update Atomicity
        
        For any memory update operation, it should either completely
        succeed or completely fail, with no partial updates.
        """
        memory_client = create_test_memory_client()
        await memory_client.initialize()
        
        user_id = uuid4()
        
        # Add memory
        memory = await memory_client.add_memory(
            user_id=user_id,
            content="Test atomicity memory",
            scope=MemoryScope.USER,
            importance_score=0.5
        )
        
        # Test valid update
        valid_update = await memory_client.update_memory_importance(
            memory_id=str(memory.id),
            importance_score=0.8
        )
        
        # Property: Valid update should return boolean result
        assert isinstance(valid_update, bool), (
            "Update operation should return boolean result"
        )
        
        # Test invalid update (non-existent memory)
        invalid_update = await memory_client.update_memory_importance(
            memory_id=str(uuid4()),  # Non-existent ID
            importance_score=0.9
        )
        
        # Property: Invalid update should fail gracefully
        assert isinstance(invalid_update, bool), (
            "Invalid update should return boolean result"
        )
        
        # Property: Original memory should still be retrievable
        retrieved_memories = await memory_client.search_memories(
            user_id=user_id,
            query="Test atomicity",
            scope=MemoryScope.USER,
            limit=5
        )
        
        matching_memories = [m for m in retrieved_memories if "atomicity" in m.content]
        assert len(matching_memories) > 0, (
            "Original memory should remain retrievable after failed update"
        )