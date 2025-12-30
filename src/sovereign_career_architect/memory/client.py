"""Memory client for persistent context management using Mem0."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
from datetime import datetime
import datetime as dt

import structlog

from sovereign_career_architect.core.models import Memory, MemoryScope
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryClient:
    """
    Memory client for interacting with Mem0 persistent memory layer.
    
    This client provides an interface to store and retrieve memories across
    different scopes (User, Session, Agent) with semantic search capabilities.
    """
    
    def __init__(self, mem0_client=None, vector_store_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory client.
        
        Args:
            mem0_client: Optional Mem0 client instance for dependency injection
            vector_store_config: Optional vector store configuration
        """
        self.mem0_client = mem0_client
        self.vector_store_config = vector_store_config or self._get_default_vector_config()
        self.logger = logger.bind(component="memory_client")
        
        # Initialize Mem0 client if not provided
        if self.mem0_client is None:
            self.mem0_client = self._create_mem0_client()
        
        # In-memory storage for development/testing when Mem0 is not available
        self._memory_store: Dict[str, List[Memory]] = {}
        self._initialized = False
    
    def _get_default_vector_config(self) -> Dict[str, Any]:
        """Get default vector store configuration from settings."""
        if settings.vector_store_type == "qdrant":
            return {
                "provider": "qdrant",
                "config": {
                    "url": settings.qdrant_url or "http://localhost:6333",
                    "api_key": settings.qdrant_api_key,
                    "collection_name": settings.qdrant_collection_name,
                    "embedding_model": settings.embedding_model,
                    "vector_size": 1536,  # text-embedding-3-small dimension
                    "distance": "cosine"
                }
            }
        elif settings.vector_store_type == "chroma":
            return {
                "provider": "chroma",
                "config": {
                    "persist_directory": settings.chroma_persist_directory,
                    "collection_name": settings.chroma_collection_name,
                    "embedding_model": settings.embedding_model
                }
            }
        else:
            # Default to in-memory for testing
            return {
                "provider": "memory",
                "config": {}
            }
    
    def _create_mem0_client(self):
        """Create Mem0 client with proper configuration."""
        try:
            # Try to import and initialize Mem0
            from mem0 import Memory as Mem0Client
            
            # Configure Mem0 with vector store
            config = {
                "vector_store": self.vector_store_config,
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "api_key": settings.openai_api_key
                    }
                } if settings.openai_api_key else None,
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": settings.embedding_model,
                        "api_key": settings.openai_api_key
                    }
                } if settings.openai_api_key else None
            }
            
            # Remove None values
            config = {k: v for k, v in config.items() if v is not None}
            
            self.logger.info(
                "Initializing Mem0 client",
                vector_store=self.vector_store_config["provider"],
                has_llm=bool(config.get("llm")),
                has_embedder=bool(config.get("embedder"))
            )
            
            return Mem0Client.from_config(config)
            
        except ImportError:
            self.logger.warning("Mem0 not available, using mock implementation")
            return None
        except Exception as e:
            self.logger.error(
                "Failed to initialize Mem0 client",
                error=str(e),
                error_type=type(e).__name__
            )
            return None
    
    async def initialize(self) -> bool:
        """
        Initialize the memory client and vector store.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            if self.mem0_client:
                # Test Mem0 connection
                self.logger.info("Testing Mem0 connection")
                # Add a test memory to verify connection
                test_result = await self._test_mem0_connection()
                if test_result:
                    self.logger.info("Mem0 client initialized successfully")
                    self._initialized = True
                    return True
                else:
                    self.logger.warning("Mem0 connection test failed, falling back to mock")
            
            # Fall back to mock implementation
            self.logger.info("Using mock memory implementation")
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(
                "Memory client initialization failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
    
    async def _test_mem0_connection(self) -> bool:
        """Test Mem0 connection with a simple operation."""
        try:
            # Try to add and retrieve a test memory
            test_user_id = "test-connection"
            test_content = "Connection test memory"
            
            # Add test memory
            if hasattr(self.mem0_client, 'add'):
                self.mem0_client.add(
                    messages=[{"role": "user", "content": test_content}],
                    user_id=test_user_id
                )
            
            # Search for test memory
            if hasattr(self.mem0_client, 'search'):
                results = self.mem0_client.search(
                    query="connection test",
                    user_id=test_user_id,
                    limit=1
                )
                return len(results) > 0
            
            return True
            
        except Exception as e:
            self.logger.debug(
                "Mem0 connection test failed",
                error=str(e)
            )
            return False
    
    async def search_memories(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search for relevant memories using semantic similarity.
        
        Args:
            user_id: User identifier
            query: Search query
            scope: Memory scope to search within
            session_id: Optional session identifier for session-scoped searches
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        self.logger.debug(
            "Searching memories",
            user_id=str(user_id),
            query=query[:100],
            scope=scope.value,
            limit=limit
        )
        
        try:
            # Ensure client is initialized
            if not self._initialized:
                await self.initialize()
            
            # Use Mem0 client if available
            if self.mem0_client and hasattr(self.mem0_client, 'search'):
                memories = await self._search_with_mem0(user_id, query, scope, session_id, limit)
            else:
                # Fall back to mock implementation
                memories = await self._mock_search_memories(user_id, query, scope, session_id, limit)
            
            self.logger.debug(
                "Memory search completed",
                memories_found=len(memories),
                user_id=str(user_id)
            )
            
            return memories
            
        except Exception as e:
            self.logger.error(
                "Memory search failed",
                error=str(e),
                user_id=str(user_id),
                query=query[:100]
            )
            return []
    
    async def _search_with_mem0(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search memories using Mem0 client."""
        try:
            # Prepare search parameters based on scope
            search_params = {
                "query": query,
                "limit": limit
            }
            
            # Add scope-specific parameters
            if scope == MemoryScope.USER:
                search_params["user_id"] = str(user_id)
            elif scope == MemoryScope.SESSION and session_id:
                search_params["user_id"] = str(user_id)
                search_params["filters"] = {"session_id": str(session_id)}
            elif scope == MemoryScope.AGENT:
                search_params["user_id"] = "agent"
                search_params["filters"] = {"scope": "agent"}
            
            # Execute search
            results = self.mem0_client.search(**search_params)
            
            # Convert Mem0 results to our Memory objects
            memories = []
            for result in results:
                memory = Memory(
                    content=result.get("memory", ""),
                    scope=scope,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=result.get("metadata", {}),
                    importance_score=result.get("score", 0.5)
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            self.logger.error(
                "Mem0 search failed",
                error=str(e),
                user_id=str(user_id)
            )
            # Fall back to mock search
            return await self._mock_search_memories(user_id, query, scope, session_id, limit)
    
    async def add_memory(
        self,
        user_id: UUID,
        content: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 0.5
    ) -> Memory:
        """
        Add a new memory to the persistent store.
        
        Args:
            user_id: User identifier
            content: Memory content
            scope: Memory scope
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
            importance_score: Importance score (0-1)
            
        Returns:
            Created memory object
        """
        memory = Memory(
            content=content,
            scope=scope,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            importance_score=importance_score
        )
        
        try:
            # Ensure client is initialized
            if not self._initialized:
                await self.initialize()
            
            # Use Mem0 client if available
            if self.mem0_client and hasattr(self.mem0_client, 'add'):
                await self._add_with_mem0(memory)
            else:
                # Fall back to mock storage
                await self._add_to_mock_storage(memory)
            
            self.logger.info(
                "Memory added",
                memory_id=str(memory.id),
                user_id=str(user_id),
                scope=scope.value,
                content_length=len(content)
            )
            
            return memory
            
        except Exception as e:
            self.logger.error(
                "Failed to add memory",
                error=str(e),
                user_id=str(user_id),
                scope=scope.value
            )
            # Still return the memory object even if storage failed
            return memory
    
    async def _add_with_mem0(self, memory: Memory) -> None:
        """Add memory using Mem0 client."""
        try:
            # Prepare memory data for Mem0
            messages = [{"role": "user", "content": memory.content}]
            
            # Prepare metadata
            mem0_metadata = {
                "scope": memory.scope.value,
                "importance_score": memory.importance_score,
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata
            }
            
            # Add session_id to metadata if present
            if memory.session_id:
                mem0_metadata["session_id"] = str(memory.session_id)
            
            # Add to Mem0 based on scope
            if memory.scope == MemoryScope.USER:
                self.mem0_client.add(
                    messages=messages,
                    user_id=str(memory.user_id),
                    metadata=mem0_metadata
                )
            elif memory.scope == MemoryScope.SESSION:
                self.mem0_client.add(
                    messages=messages,
                    user_id=str(memory.user_id),
                    metadata=mem0_metadata
                )
            elif memory.scope == MemoryScope.AGENT:
                self.mem0_client.add(
                    messages=messages,
                    user_id="agent",
                    metadata=mem0_metadata
                )
                
        except Exception as e:
            self.logger.error(
                "Mem0 add failed",
                error=str(e),
                memory_id=str(memory.id)
            )
            # Fall back to mock storage
            await self._add_to_mock_storage(memory)
    
    async def _add_to_mock_storage(self, memory: Memory) -> None:
        """Add memory to mock storage."""
        key = f"{memory.user_id}:{memory.scope.value}"
        if key not in self._memory_store:
            self._memory_store[key] = []
        self._memory_store[key].append(memory)
    
    async def get_user_profile_memories(self, user_id: UUID) -> List[Memory]:
        """
        Get all user profile related memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of profile-related memories
        """
        return await self.search_memories(
            user_id=user_id,
            query="profile personal information skills experience education",
            scope=MemoryScope.USER,
            limit=50
        )
    
    async def get_session_memories(self, user_id: UUID, session_id: UUID, limit: int = 20) -> List[Memory]:
        """
        Get memories for a specific session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of session memories
        """
        return await self.search_memories(
            user_id=user_id,
            query="",  # Empty query to get all session memories
            scope=MemoryScope.SESSION,
            session_id=session_id,
            limit=limit
        )
    
    async def get_agent_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Get agent-level memories (learned patterns and behaviors).
        
        Args:
            query: Search query for agent memories
            limit: Maximum number of memories to return
            
        Returns:
            List of agent memories
        """
        # Use a dummy user_id for agent memories
        from uuid import uuid4
        dummy_user_id = uuid4()
        
        return await self.search_memories(
            user_id=dummy_user_id,
            query=query,
            scope=MemoryScope.AGENT,
            limit=limit
        )
    
    async def update_memory_importance(self, memory_id: str, importance_score: float) -> bool:
        """
        Update the importance score of a memory.
        
        Args:
            memory_id: Memory identifier
            importance_score: New importance score (0-1)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if self.mem0_client and hasattr(self.mem0_client, 'update'):
                # Update in Mem0
                self.mem0_client.update(
                    memory_id=memory_id,
                    data={"importance_score": importance_score}
                )
                
                self.logger.info(
                    "Memory importance updated",
                    memory_id=memory_id,
                    importance_score=importance_score
                )
                return True
            else:
                # Update in mock storage
                for memories in self._memory_store.values():
                    for memory in memories:
                        if str(memory.id) == memory_id:
                            memory.importance_score = importance_score
                            return True
                return False
                
        except Exception as e:
            self.logger.error(
                "Failed to update memory importance",
                error=str(e),
                memory_id=memory_id
            )
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the store.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if self.mem0_client and hasattr(self.mem0_client, 'delete'):
                # Delete from Mem0
                self.mem0_client.delete(memory_id=memory_id)
                
                self.logger.info("Memory deleted", memory_id=memory_id)
                return True
            else:
                # Delete from mock storage
                for key, memories in self._memory_store.items():
                    for i, memory in enumerate(memories):
                        if str(memory.id) == memory_id:
                            del memories[i]
                            return True
                return False
                
        except Exception as e:
            self.logger.error(
                "Failed to delete memory",
                error=str(e),
                memory_id=memory_id
            )
            return False
    
    async def semantic_search_memories(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10,
        similarity_threshold: float = 0.3,
        include_metadata_search: bool = True
    ) -> List[Memory]:
        """
        Advanced semantic search for memories with relevance ranking and context filtering.
        
        Args:
            user_id: User identifier
            query: Search query
            scope: Memory scope to search within
            session_id: Optional session identifier for session-scoped searches
            limit: Maximum number of memories to return
            similarity_threshold: Minimum similarity score for results
            include_metadata_search: Whether to include metadata in search
            
        Returns:
            List of relevant memories ranked by relevance
        """
        self.logger.debug(
            "Performing semantic search",
            user_id=str(user_id),
            query=query[:100],
            scope=scope.value,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        try:
            # Ensure client is initialized
            if not self._initialized:
                await self.initialize()
            
            # Use enhanced search with Mem0 if available
            if self.mem0_client and hasattr(self.mem0_client, 'search'):
                memories = await self._semantic_search_with_mem0(
                    user_id, query, scope, session_id, limit, similarity_threshold
                )
            else:
                # Enhanced mock semantic search
                memories = await self._enhanced_mock_semantic_search(
                    user_id, query, scope, session_id, limit, similarity_threshold, include_metadata_search
                )
            
            # Apply relevance ranking and context filtering
            ranked_memories = await self._rank_and_filter_memories(
                memories, query, similarity_threshold
            )
            
            self.logger.debug(
                "Semantic search completed",
                memories_found=len(ranked_memories),
                user_id=str(user_id),
                avg_relevance=sum(m.importance_score for m in ranked_memories) / len(ranked_memories) if ranked_memories else 0
            )
            
            return ranked_memories[:limit]
            
        except Exception as e:
            self.logger.error(
                "Semantic search failed",
                error=str(e),
                user_id=str(user_id),
                query=query[:100]
            )
            return []
    
    async def _semantic_search_with_mem0(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Memory]:
        """Enhanced semantic search using Mem0 client with advanced filtering."""
        try:
            # Prepare advanced search parameters
            search_params = {
                "query": query,
                "limit": limit * 2,  # Get more results for better filtering
                "filters": {}
            }
            
            # Add scope-specific parameters and filters
            if scope == MemoryScope.USER:
                search_params["user_id"] = str(user_id)
                search_params["filters"]["scope"] = "user"
            elif scope == MemoryScope.SESSION and session_id:
                search_params["user_id"] = str(user_id)
                search_params["filters"]["session_id"] = str(session_id)
                search_params["filters"]["scope"] = "session"
            elif scope == MemoryScope.AGENT:
                search_params["user_id"] = "agent"
                search_params["filters"]["scope"] = "agent"
            
            # Add similarity threshold filter if supported
            if hasattr(self.mem0_client, 'search') and 'threshold' in self.mem0_client.search.__code__.co_varnames:
                search_params["threshold"] = similarity_threshold
            
            # Execute enhanced search
            results = self.mem0_client.search(**search_params)
            
            # Convert Mem0 results to our Memory objects with enhanced metadata
            memories = []
            for result in results:
                # Extract relevance score from Mem0 result
                relevance_score = result.get("score", 0.5)
                
                # Skip results below similarity threshold
                if relevance_score < similarity_threshold:
                    continue
                
                memory = Memory(
                    content=result.get("memory", ""),
                    scope=scope,
                    user_id=user_id,
                    session_id=session_id,
                    metadata={
                        **result.get("metadata", {}),
                        "search_relevance": relevance_score,
                        "search_query": query,
                        "search_timestamp": datetime.now(dt.timezone.utc).isoformat()
                    },
                    importance_score=relevance_score
                )
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            self.logger.error(
                "Mem0 semantic search failed",
                error=str(e),
                user_id=str(user_id)
            )
            # Fall back to enhanced mock search
            return await self._enhanced_mock_semantic_search(
                user_id, query, scope, session_id, limit, similarity_threshold, True
            )
    
    async def _enhanced_mock_semantic_search(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10,
        similarity_threshold: float = 0.3,
        include_metadata_search: bool = True
    ) -> List[Memory]:
        """
        Enhanced mock implementation of semantic search with better relevance scoring.
        """
        # Simulate async operation
        await asyncio.sleep(0.01)
        
        key = f"{user_id}:{scope.value}"
        memories = self._memory_store.get(key, [])
        
        # Filter by session_id for session-scoped memories
        if scope == MemoryScope.SESSION and session_id is not None:
            memories = [m for m in memories if m.session_id == session_id]
        
        if not query:
            return memories[:limit]
        
        # Enhanced semantic similarity calculation
        query_terms = self._extract_search_terms(query)
        scored_memories = []
        
        for memory in memories:
            # Calculate content similarity
            content_score = self._calculate_semantic_similarity(
                query_terms, memory.content
            )
            
            # Calculate metadata similarity if enabled
            metadata_score = 0.0
            if include_metadata_search and memory.metadata:
                metadata_score = self._calculate_metadata_similarity(
                    query_terms, memory.metadata
                )
            
            # Combined relevance score
            relevance_score = (content_score * 0.8) + (metadata_score * 0.2)
            
            # Apply similarity threshold
            if relevance_score >= similarity_threshold:
                # Update memory with search metadata
                memory.importance_score = relevance_score
                memory.metadata = {
                    **memory.metadata,
                    "search_relevance": relevance_score,
                    "content_similarity": content_score,
                    "metadata_similarity": metadata_score,
                    "search_query": query
                }
                scored_memories.append(memory)
        
        # Sort by relevance score (descending)
        scored_memories.sort(key=lambda m: m.importance_score, reverse=True)
        
        return scored_memories[:limit]
    
    def _extract_search_terms(self, query: str) -> Dict[str, float]:
        """
        Extract and weight search terms from query.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary of terms with their weights
        """
        if not query:
            return {}
        
        # Simple term extraction with basic weighting
        terms = {}
        words = query.lower().split()
        
        # Common stop words to filter out
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "of", "with", "by", "is", "are", "was", "were", "be", 
            "been", "have", "has", "had", "do", "does", "did", "will", "would"
        }
        
        for i, word in enumerate(words):
            if word not in stop_words and len(word) > 2:
                # Weight terms higher if they appear earlier in query
                position_weight = 1.0 - (i * 0.1)
                # Weight longer terms higher
                length_weight = min(len(word) / 10.0, 1.0)
                
                terms[word] = max(position_weight, 0.1) * (1.0 + length_weight)
        
        return terms
    
    def _calculate_semantic_similarity(
        self, 
        query_terms: Dict[str, float], 
        content: str
    ) -> float:
        """
        Calculate semantic similarity between query terms and content.
        
        Args:
            query_terms: Weighted query terms
            content: Memory content
            
        Returns:
            Similarity score between 0 and 1
        """
        if not query_terms or not content:
            return 0.0
        
        content_words = set(content.lower().split())
        
        # Calculate weighted term matches
        total_weight = sum(query_terms.values())
        matched_weight = 0.0
        
        for term, weight in query_terms.items():
            if term in content_words:
                matched_weight += weight
            else:
                # Check for partial matches (substring)
                for content_word in content_words:
                    if term in content_word or content_word in term:
                        matched_weight += weight * 0.5  # Partial match gets half weight
                        break
        
        # Normalize by total weight
        similarity = matched_weight / total_weight if total_weight > 0 else 0.0
        
        # Boost score for exact phrase matches
        if len(query_terms) > 1:
            query_phrase = " ".join(query_terms.keys())
            if query_phrase in content.lower():
                similarity = min(similarity * 1.5, 1.0)
        
        return similarity
    
    def _calculate_metadata_similarity(
        self, 
        query_terms: Dict[str, float], 
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between query terms and metadata.
        
        Args:
            query_terms: Weighted query terms
            metadata: Memory metadata
            
        Returns:
            Similarity score between 0 and 1
        """
        if not query_terms or not metadata:
            return 0.0
        
        # Convert metadata values to searchable text
        metadata_text = []
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata_text.append(value.lower())
            elif isinstance(value, (list, tuple)):
                metadata_text.extend(str(v).lower() for v in value)
            else:
                metadata_text.append(str(value).lower())
        
        metadata_content = " ".join(metadata_text)
        
        # Use same similarity calculation as content
        return self._calculate_semantic_similarity(query_terms, metadata_content)
    
    async def _rank_and_filter_memories(
        self,
        memories: List[Memory],
        query: str,
        similarity_threshold: float
    ) -> List[Memory]:
        """
        Apply advanced ranking and filtering to search results.
        
        Args:
            memories: List of memories to rank
            query: Original search query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Ranked and filtered memories
        """
        if not memories:
            return []
        
        # Apply additional filtering
        filtered_memories = []
        
        for memory in memories:
            # Check similarity threshold
            relevance_score = getattr(memory, 'importance_score', 0.5)
            if relevance_score < similarity_threshold:
                continue
            
            # Apply recency boost for recent memories
            recency_boost = self._calculate_recency_boost(memory.timestamp)
            
            # Apply importance boost based on original importance
            original_importance = memory.metadata.get('confidence', 0.5)
            importance_boost = original_importance * 0.2
            
            # Calculate final ranking score
            final_score = relevance_score + recency_boost + importance_boost
            memory.importance_score = min(final_score, 1.0)
            
            filtered_memories.append(memory)
        
        # Sort by final ranking score
        filtered_memories.sort(key=lambda m: m.importance_score, reverse=True)
        
        return filtered_memories
    
    def _calculate_recency_boost(self, timestamp: datetime) -> float:
        """
        Calculate recency boost for memory ranking.
        
        Args:
            timestamp: Memory timestamp
            
        Returns:
            Recency boost score (0-0.2)
        """
        try:
            now = datetime.now(dt.timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
            
            age_days = (now - timestamp).days
            
            # Boost recent memories (within 7 days)
            if age_days <= 1:
                return 0.2  # Very recent
            elif age_days <= 7:
                return 0.1  # Recent
            elif age_days <= 30:
                return 0.05  # Somewhat recent
            else:
                return 0.0  # Older memories get no boost
                
        except Exception:
            return 0.0
    
    async def search_memories_by_category(
        self,
        user_id: UUID,
        category: str,
        scope: MemoryScope,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search memories by specific category.
        
        Args:
            user_id: User identifier
            category: Memory category to search for
            scope: Memory scope
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in the specified category
        """
        try:
            # Use semantic search with category-specific query
            category_query = f"category:{category} {category}"
            
            memories = await self.semantic_search_memories(
                user_id=user_id,
                query=category_query,
                scope=scope,
                limit=limit,
                similarity_threshold=0.1,  # Lower threshold for category search
                include_metadata_search=True
            )
            
            # Additional filtering by metadata category
            category_memories = [
                m for m in memories 
                if m.metadata.get("category", "").lower() == category.lower()
            ]
            
            # If no exact category matches, return semantic matches
            return category_memories if category_memories else memories
            
        except Exception as e:
            self.logger.error(
                "Category search failed",
                error=str(e),
                category=category,
                user_id=str(user_id)
            )
            return []
    
    async def search_memories_by_timeframe(
        self,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime,
        scope: MemoryScope,
        query: Optional[str] = None,
        limit: int = 20
    ) -> List[Memory]:
        """
        Search memories within a specific timeframe.
        
        Args:
            user_id: User identifier
            start_date: Start of timeframe
            end_date: End of timeframe
            scope: Memory scope
            query: Optional search query
            limit: Maximum number of memories to return
            
        Returns:
            List of memories within the timeframe
        """
        try:
            # Get all memories in scope
            if query:
                memories = await self.semantic_search_memories(
                    user_id=user_id,
                    query=query,
                    scope=scope,
                    limit=limit * 2,  # Get more for filtering
                    similarity_threshold=0.1
                )
            else:
                memories = await self.search_memories(
                    user_id=user_id,
                    query="",
                    scope=scope,
                    limit=limit * 2
                )
            
            # Filter by timeframe
            timeframe_memories = []
            for memory in memories:
                memory_time = memory.timestamp
                if memory_time.tzinfo is None:
                    memory_time = memory_time.replace(tzinfo=dt.timezone.utc)
                
                if start_date <= memory_time <= end_date:
                    timeframe_memories.append(memory)
            
            # Sort by timestamp (most recent first)
            timeframe_memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            return timeframe_memories[:limit]
            
        except Exception as e:
            self.logger.error(
                "Timeframe search failed",
                error=str(e),
                user_id=str(user_id)
            )
            return []
    
    async def get_memory_stats(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            stats = {
                "user_memories": 0,
                "session_memories": 0,
                "agent_memories": 0,
                "total_memories": 0
            }
            
            # Count memories by scope
            for scope in MemoryScope:
                memories = await self.search_memories(
                    user_id=user_id,
                    query="",
                    scope=scope,
                    limit=1000  # High limit to count all
                )
                
                if scope == MemoryScope.USER:
                    stats["user_memories"] = len(memories)
                elif scope == MemoryScope.SESSION:
                    stats["session_memories"] = len(memories)
                elif scope == MemoryScope.AGENT:
                    stats["agent_memories"] = len(memories)
                
                stats["total_memories"] += len(memories)
            
            return stats
            
        except Exception as e:
            self.logger.error(
                "Failed to get memory stats",
                error=str(e),
                user_id=str(user_id)
            )
            return {"error": str(e)}
    
    async def _mock_search_memories(
        self,
        user_id: UUID,
        query: str,
        scope: MemoryScope,
        session_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Mock implementation of memory search for development/testing.
        
        In production, this would be replaced with actual Mem0 semantic search.
        """
        # Simulate async operation
        await asyncio.sleep(0.01)
        
        # Return empty list for now - in production this would use Mem0
        key = f"{user_id}:{scope.value}"
        memories = self._memory_store.get(key, [])
        
        # Filter by session_id for session-scoped memories
        if scope == MemoryScope.SESSION and session_id is not None:
            memories = [m for m in memories if m.session_id == session_id]
        
        # Simple keyword matching for mock implementation
        if query:
            query_lower = query.lower()
            filtered_memories = [
                m for m in memories 
                if query_lower in m.content.lower()
            ]
        else:
            filtered_memories = memories
        
        # Sort by importance score and timestamp
        filtered_memories.sort(
            key=lambda m: (m.importance_score, m.timestamp),
            reverse=True
        )
        
        return filtered_memories[:limit]


def create_memory_client(
    vector_store_type: Optional[str] = None,
    vector_store_config: Optional[Dict[str, Any]] = None
) -> MemoryClient:
    """
    Factory function to create a properly configured memory client.
    
    Args:
        vector_store_type: Type of vector store to use (qdrant/chroma/memory)
        vector_store_config: Custom vector store configuration
        
    Returns:
        Configured MemoryClient instance
    """
    # Use provided config or create from settings
    if vector_store_config is None:
        client = MemoryClient()
    else:
        client = MemoryClient(vector_store_config=vector_store_config)
    
    logger.info(
        "Memory client created",
        vector_store=client.vector_store_config["provider"],
        has_mem0_client=client.mem0_client is not None
    )
    
    return client


def create_test_memory_client() -> MemoryClient:
    """
    Create a memory client for testing with in-memory storage.
    
    Returns:
        MemoryClient configured for testing
    """
    config = {
        "provider": "memory",
        "config": {}
    }
    
    return MemoryClient(vector_store_config=config)