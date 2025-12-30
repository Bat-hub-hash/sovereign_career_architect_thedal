"""
Profiler Node: Context retrieval and user profile enrichment.

This node is the entry point for the cognitive architecture. It ingests the user's
raw input and queries the Mem0 layer to retrieve relevant long-term context.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from langchain_core.messages import BaseMessage

from sovereign_career_architect.core.state import AgentState, StateUpdate
from sovereign_career_architect.core.models import UserProfile, Memory, MemoryScope
from sovereign_career_architect.memory.client import MemoryClient
from sovereign_career_architect.utils.logging import get_logger, log_function_call, log_agent_state

logger = get_logger(__name__)


class ProfilerNode:
    """
    Profiler Node for context retrieval and user profile enrichment.
    
    This node acts as the entry point to the cognitive architecture, responsible for:
    1. Retrieving user profile from persistent memory
    2. Enriching incoming requests with relevant context
    3. Preparing the state for downstream reasoning nodes
    """
    
    def __init__(self, memory_client: Optional[MemoryClient] = None):
        """
        Initialize the Profiler Node.
        
        Args:
            memory_client: Optional memory client for testing. If None, creates default client.
        """
        from sovereign_career_architect.memory.client import create_memory_client
        self.memory_client = memory_client or create_memory_client()
        self.logger = logger.bind(node="profiler")
        self._initialized = False
    
    async def __call__(self, state: AgentState) -> StateUpdate:
        """
        Execute the profiler node logic.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with enriched context
        """
        self.logger.info(
            "Profiler node executing",
            **log_function_call("profiler_node", user_id=str(state.user_id)),
            **log_agent_state(state.model_dump())
        )
        
        try:
            # Initialize memory client if needed
            if not self._initialized:
                await self.memory_client.initialize()
                self._initialized = True
            
            # Get the latest user message for context
            latest_message = self._get_latest_user_message(state.messages)
            
            # Retrieve user profile if not already loaded
            user_profile = await self._retrieve_user_profile(state)
            
            # Retrieve relevant memories based on current context
            memory_context = await self._retrieve_memory_context(
                state.user_id, 
                latest_message.content if latest_message else "",
                state.session_id
            )
            
            # Enrich context with semantic analysis
            enriched_context = await self._enrich_memory_context(
                memory_context, 
                latest_message.content if latest_message else ""
            )
            
            # Prepare state updates
            updates: StateUpdate = {
                "user_profile": user_profile,
                "memory_context": enriched_context,
                "next_node": "planner"  # Route to planner for next step
            }
            
            self.logger.info(
                "Profiler node completed successfully",
                user_profile_loaded=user_profile is not None,
                memory_entries_retrieved=enriched_context.get("total_memories", 0),
                next_node=updates["next_node"]
            )
            
            return updates
            
        except Exception as e:
            self.logger.error(
                "Profiler node execution failed",
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Return minimal updates to allow graceful degradation
            return {
                "memory_context": {"error": str(e)},
                "next_node": "planner"  # Still route to planner
            }
    
    def _get_latest_user_message(self, messages: List[BaseMessage]) -> Optional[BaseMessage]:
        """
        Get the most recent user message from the conversation history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Latest user message or None if no user messages found
        """
        # Iterate backwards to find the most recent user message
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message
        return None
    
    async def _retrieve_user_profile(self, state: AgentState) -> Optional[UserProfile]:
        """
        Retrieve user profile from memory or return existing profile.
        
        Args:
            state: Current agent state
            
        Returns:
            User profile or None if not found
        """
        # If profile already loaded, return it
        if state.user_profile:
            self.logger.debug("User profile already loaded in state")
            return state.user_profile
        
        try:
            # Use the enhanced memory client method for profile retrieval
            profile_memories = await self.memory_client.get_user_profile_memories(state.user_id)
            
            if not profile_memories:
                self.logger.info("No user profile found in memory")
                return None
            
            # Reconstruct user profile from memories
            profile_data = await self._reconstruct_profile_from_memories(profile_memories)
            
            if profile_data:
                user_profile = UserProfile(**profile_data)
                self.logger.info(
                    "User profile reconstructed from memory",
                    profile_id=str(user_profile.user_id),
                    skills_count=len(user_profile.skills),
                    experience_count=len(user_profile.experience)
                )
                return user_profile
            
        except Exception as e:
            self.logger.warning(
                "Failed to retrieve user profile",
                error=str(e),
                user_id=str(state.user_id)
            )
        
        return None
    
    async def _retrieve_memory_context(
        self, 
        user_id: UUID, 
        query: str, 
        session_id: UUID
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memory context for the current interaction.
        
        Args:
            user_id: User identifier
            query: Current user query/message
            session_id: Current session identifier
            
        Returns:
            Dictionary containing relevant memories and context
        """
        try:
            # Use enhanced memory client methods for better retrieval
            user_memories = await self.memory_client.search_memories(
                user_id=user_id,
                query=query,
                scope=MemoryScope.USER,
                limit=5
            )
            
            # Get session-specific memories
            session_memories = await self.memory_client.get_session_memories(
                user_id=user_id,
                session_id=session_id,
                limit=3
            )
            
            # Get relevant agent memories
            agent_memories = await self.memory_client.get_agent_memories(
                query=query,
                limit=2
            )
            
            # Combine and structure the context
            context = {
                "memories": {
                    "user": [self._memory_to_dict(m) for m in user_memories],
                    "session": [self._memory_to_dict(m) for m in session_memories],
                    "agent": [self._memory_to_dict(m) for m in agent_memories]
                },
                "total_memories": len(user_memories) + len(session_memories) + len(agent_memories),
                "query": query,
                "retrieved_at": str(session_id),
                "memory_stats": await self._get_memory_statistics(user_id)
            }
            
            self.logger.debug(
                "Memory context retrieved",
                user_memories=len(user_memories),
                session_memories=len(session_memories),
                agent_memories=len(agent_memories),
                total=context["total_memories"]
            )
            
            return context
            
        except Exception as e:
            self.logger.error(
                "Failed to retrieve memory context",
                error=str(e),
                user_id=str(user_id),
                query=query[:100] if query else ""
            )
            
            return {
                "memories": {"user": [], "session": [], "agent": []},
                "total_memories": 0,
                "error": str(e),
                "retrieved_at": str(session_id)
            }
    
    def _reconstruct_profile_from_memories(self, memories: List[Memory]) -> Optional[Dict[str, Any]]:
        """
        Reconstruct user profile data from memory entries.
        
        This is a simplified implementation. In practice, you'd have more
        sophisticated logic to parse and reconstruct structured profile data.
        
        Args:
            memories: List of memory entries
            
        Returns:
            Profile data dictionary or None
        """
        if not memories:
            return None
        
        # For now, return a minimal profile structure
        # In practice, you'd parse memory content to extract structured data
        return {
            "personal_info": {
                "name": "User",  # Would be extracted from memories
                "email": "user@example.com",  # Would be extracted from memories
                "preferred_language": "en"
            },
            "skills": [],  # Would be extracted from memories
            "experience": [],  # Would be extracted from memories
            "education": [],  # Would be extracted from memories
            "preferences": {
                "roles": [],
                "locations": [],
                "company_types": [],
                "work_arrangements": []
            },
            "documents": {
                "resume_path": None,
                "cover_letter_template": None,
                "portfolio_files": []
            }
        }
    
    async def _enrich_memory_context(
        self, 
        context: Dict[str, Any], 
        current_query: str
    ) -> Dict[str, Any]:
        """
        Enrich memory context with additional semantic analysis.
        
        Args:
            context: Base memory context
            current_query: Current user query
            
        Returns:
            Enriched context with additional insights
        """
        try:
            # Add query analysis
            context["query_analysis"] = {
                "intent": self._analyze_query_intent(current_query),
                "entities": self._extract_entities(current_query),
                "sentiment": "neutral",  # Could be enhanced with sentiment analysis
                "complexity": len(current_query.split()) if current_query else 0
            }
            
            # Add memory relevance scoring
            for scope in ["user", "session", "agent"]:
                memories = context["memories"].get(scope, [])
                for memory in memories:
                    memory["relevance_score"] = self._calculate_relevance_score(
                        memory["content"], 
                        current_query
                    )
            
            # Add context summary
            context["summary"] = self._generate_context_summary(context)
            
            return context
            
        except Exception as e:
            self.logger.warning(
                "Failed to enrich memory context",
                error=str(e)
            )
            return context
    
    async def _get_memory_statistics(self, user_id: UUID) -> Dict[str, Any]:
        """Get memory statistics for the user."""
        try:
            return await self.memory_client.get_memory_stats(user_id)
        except Exception as e:
            self.logger.warning(
                "Failed to get memory statistics",
                error=str(e),
                user_id=str(user_id)
            )
            return {}
    
    def _analyze_query_intent(self, query: str) -> str:
        """
        Analyze the intent of the user query.
        
        Args:
            query: User query string
            
        Returns:
            Detected intent category
        """
        if not query:
            return "unknown"
        
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(word in query_lower for word in ["job", "position", "role", "career", "work"]):
            return "job_search"
        elif any(word in query_lower for word in ["interview", "prepare", "practice", "questions"]):
            return "interview_prep"
        elif any(word in query_lower for word in ["resume", "cv", "profile", "update"]):
            return "profile_management"
        elif any(word in query_lower for word in ["apply", "application", "submit"]):
            return "job_application"
        elif any(word in query_lower for word in ["help", "how", "what", "?"]):
            return "information_seeking"
        else:
            return "general"
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from the user query.
        
        Args:
            query: User query string
            
        Returns:
            List of extracted entities
        """
        if not query:
            return []
        
        # Simple entity extraction based on common patterns
        entities = []
        query_lower = query.lower()
        
        # Technology entities
        tech_keywords = [
            "python", "javascript", "java", "react", "node.js", "django", 
            "flask", "aws", "docker", "kubernetes", "sql", "mongodb"
        ]
        for tech in tech_keywords:
            if tech in query_lower:
                entities.append(f"technology:{tech}")
        
        # Role entities
        role_keywords = [
            "engineer", "developer", "manager", "analyst", "designer", 
            "architect", "consultant", "specialist"
        ]
        for role in role_keywords:
            if role in query_lower:
                entities.append(f"role:{role}")
        
        # Company entities (simplified)
        company_keywords = [
            "google", "microsoft", "amazon", "apple", "facebook", "meta",
            "netflix", "uber", "airbnb", "startup", "enterprise"
        ]
        for company in company_keywords:
            if company in query_lower:
                entities.append(f"company:{company}")
        
        return entities
    
    def _calculate_relevance_score(self, memory_content: str, query: str) -> float:
        """
        Calculate relevance score between memory content and query.
        
        Args:
            memory_content: Content of the memory
            query: Current user query
            
        Returns:
            Relevance score between 0 and 1
        """
        if not memory_content or not query:
            return 0.0
        
        # Simple keyword overlap scoring
        memory_words = set(memory_content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(memory_words.intersection(query_words))
        return min(overlap / len(query_words), 1.0)
    
    def _generate_context_summary(self, context: Dict[str, Any]) -> str:
        """
        Generate a summary of the memory context.
        
        Args:
            context: Memory context dictionary
            
        Returns:
            Context summary string
        """
        total_memories = context.get("total_memories", 0)
        query_intent = context.get("query_analysis", {}).get("intent", "unknown")
        
        if total_memories == 0:
            return "No relevant memories found for this interaction."
        
        user_count = len(context.get("memories", {}).get("user", []))
        session_count = len(context.get("memories", {}).get("session", []))
        agent_count = len(context.get("memories", {}).get("agent", []))
        
        summary_parts = []
        
        if user_count > 0:
            summary_parts.append(f"{user_count} user memories")
        if session_count > 0:
            summary_parts.append(f"{session_count} session memories")
        if agent_count > 0:
            summary_parts.append(f"{agent_count} agent memories")
        
        summary = f"Retrieved {', '.join(summary_parts)} for {query_intent} intent."
        return summary
    
    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
        """
        Convert memory object to dictionary for context.
        
        Args:
            memory: Memory object
            
        Returns:
            Dictionary representation of memory
        """
        return {
            "id": str(memory.id),
            "content": memory.content,
            "scope": memory.scope.value,
            "timestamp": memory.timestamp.isoformat(),
            "importance_score": memory.importance_score,
            "metadata": memory.metadata
        }


# Factory function for creating profiler node
def create_profiler_node(memory_client: Optional[MemoryClient] = None) -> ProfilerNode:
    """
    Factory function to create a profiler node.
    
    Args:
        memory_client: Optional memory client for dependency injection
        
    Returns:
        Configured ProfilerNode instance
    """
    return ProfilerNode(memory_client=memory_client)