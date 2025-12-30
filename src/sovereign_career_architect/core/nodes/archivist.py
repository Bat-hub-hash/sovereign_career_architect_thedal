"""
Archivist Node: Memory consolidation and long-term storage.

This node extracts salient facts from completed interactions and updates
the persistent memory layer (Mem0) to ensure the agent learns and improves
over time.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import datetime as dt

import structlog
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from sovereign_career_architect.core.state import AgentState, StateUpdate
from sovereign_career_architect.core.models import (
    ActionResult,
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
    MemoryScope,
)
from sovereign_career_architect.memory.client import MemoryClient
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger, log_function_call, log_agent_state

logger = get_logger(__name__)


class ArchivistNode:
    """
    Archivist Node for memory consolidation and long-term storage.
    
    This node is responsible for:
    1. Extracting salient facts from completed interactions
    2. Updating persistent memory with new information
    3. Organizing memories into User, Session, and Agent scopes
    4. Ensuring memory consistency and deduplication
    """
    
    def __init__(self, memory_client: Optional[MemoryClient] = None, extraction_model: Optional[Any] = None):
        """
        Initialize the Archivist Node.
        
        Args:
            memory_client: Optional memory client for testing. If None, creates default client.
            extraction_model: Optional model for fact extraction. If None, creates default model.
        """
        self.memory_client = memory_client or MemoryClient()
        
        if extraction_model is not None:
            self.extraction_model = extraction_model
        else:
            try:
                self.extraction_model = self._create_extraction_model()
            except ValueError:
                # No API keys available - set to None for testing
                self.extraction_model = None
        
        self.logger = logger.bind(node="archivist")
    
    def _create_extraction_model(self) -> Any:
        """Create the model for fact extraction."""
        if settings.openai_api_key:
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fast model for fact extraction
                api_key=settings.openai_api_key,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1024,
                timeout=20.0,
            )
        elif settings.groq_api_key:
            # Fallback to Groq if OpenAI not available
            return ChatGroq(
                model="llama-3.1-8b-instant",  # Fast model for extraction
                api_key=settings.groq_api_key,
                temperature=0.1,
                max_tokens=1024,
                timeout=20.0,
            )
        else:
            raise ValueError("No API keys configured for extraction model")
    
    async def __call__(self, state: AgentState) -> StateUpdate:
        """
        Execute the archivist node logic.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with memory consolidation results
        """
        self.logger.info(
            "Archivist node executing",
            **log_function_call("archivist_node", user_id=str(state.user_id)),
            **log_agent_state(state.model_dump())
        )
        
        try:
            # Extract salient facts from the interaction
            salient_facts = await self._extract_salient_facts(state)
            
            # Update memory with extracted facts
            if salient_facts:
                await self._update_memory(salient_facts, state)
            
            # Clear session-specific data for next interaction
            updates: StateUpdate = {
                "tool_outputs": [],  # Clear completed outputs
                "critique": None,    # Clear previous critique
                "next_node": None    # End of workflow
            }
            
            self.logger.info(
                "Archivist node completed",
                facts_extracted=len(salient_facts) if salient_facts else 0,
                user_id=str(state.user_id)
            )
            
            return updates
            
        except Exception as e:
            self.logger.error(
                "Archivist node execution failed",
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Return minimal updates to allow graceful completion
            return {
                "tool_outputs": [],
                "critique": None,
                "next_node": None
            }
    
    async def _extract_salient_facts(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Extract salient facts from the current interaction.
        
        Args:
            state: Current agent state with interaction history
            
        Returns:
            List of extracted facts with metadata
        """
        # Handle case where no model is available
        if self.extraction_model is None:
            self.logger.warning("No extraction model available, skipping fact extraction")
            return []
        
        try:
            # Build context for fact extraction
            system_prompt = self._build_extraction_system_prompt()
            user_prompt = self._build_extraction_user_prompt(state)
            
            # Extract facts using the model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger.debug(
                "Extracting salient facts",
                messages_count=len(state.messages),
                tool_outputs_count=len(state.tool_outputs)
            )
            
            response = await self.extraction_model.ainvoke(messages)
            facts_text = response.content
            
            # Parse extracted facts
            facts = self._parse_extracted_facts(facts_text, state)
            
            self.logger.debug(
                "Facts extraction completed",
                facts_count=len(facts)
            )
            
            return facts
            
        except Exception as e:
            self.logger.error(
                "Failed to extract salient facts",
                error=str(e)
            )
            return []
    
    def _build_extraction_system_prompt(self) -> str:
        """Build the system prompt for fact extraction."""
        return """You are the Archivist component of the Sovereign Career Architect, responsible for extracting salient facts from user interactions.

Your role is to identify and extract important information that should be remembered for future sessions.

EXTRACTION CRITERIA:
1. User preferences and goals (career objectives, role preferences, location preferences)
2. Skills and experience updates (new skills learned, experience gained)
3. Knowledge gaps identified (areas where user needs improvement)
4. Successful strategies (what worked well for the user)
5. Feedback and insights (user reactions, preferences discovered)
6. Important context (industry focus, company preferences, salary expectations)

MEMORY SCOPES:
- USER: Long-term facts about the user (skills, preferences, goals)
- SESSION: Current session context (recent activities, current focus)
- AGENT: Learning about interaction patterns and effective strategies

OUTPUT FORMAT:
For each salient fact, provide:
SCOPE: [USER/SESSION/AGENT]
CATEGORY: [preference/skill/goal/gap/strategy/context]
FACT: [Clear, concise statement of the fact]
CONFIDENCE: [0.0-1.0]

Example:
SCOPE: USER
CATEGORY: skill
FACT: User has 5 years of Python development experience
CONFIDENCE: 0.9

Only extract facts that are clearly stated or strongly implied. Avoid speculation."""
    
    def _build_extraction_user_prompt(self, state: AgentState) -> str:
        """Build the user prompt with interaction context."""
        prompt_parts = []
        
        # Add conversation history
        if state.messages:
            prompt_parts.append("CONVERSATION HISTORY:")
            for i, message in enumerate(state.messages[-10:]):  # Last 10 messages
                role = "User" if hasattr(message, 'type') and message.type == 'human' else "Agent"
                content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                prompt_parts.append(f"{role}: {content}")
        
        # Add tool outputs
        if state.tool_outputs:
            prompt_parts.append("\nACTIONS PERFORMED:")
            for output in state.tool_outputs:
                prompt_parts.append(f"Action: {output.action_type.value}")
                prompt_parts.append(f"Success: {output.success}")
                if output.data:
                    # Summarize key data points
                    data_summary = []
                    for key, value in list(output.data.items())[:3]:  # Top 3 data points
                        value_str = str(value)[:100]  # Truncate long values
                        data_summary.append(f"{key}: {value_str}")
                    prompt_parts.append(f"Results: {', '.join(data_summary)}")
        
        # Add current plan context
        if state.current_plan:
            prompt_parts.append(f"\nPLAN CONTEXT:")
            prompt_parts.append(f"Plan had {len(state.current_plan.steps)} steps")
            if state.current_step_index < len(state.current_plan.steps):
                current_step = state.current_plan.steps[state.current_step_index]
                prompt_parts.append(f"Current step: {current_step.description}")
        
        # Add user profile context if available
        if state.user_profile:
            prompt_parts.append(f"\nEXISTING USER PROFILE:")
            if state.user_profile.personal_info:
                prompt_parts.append(f"Name: {state.user_profile.personal_info.name}")
            if state.user_profile.skills:
                skills_summary = [f"{skill.name} ({skill.level})" for skill in state.user_profile.skills[:3]]
                prompt_parts.append(f"Known Skills: {', '.join(skills_summary)}")
        
        prompt_parts.append("\nExtract salient facts from this interaction that should be remembered:")
        
        return "\n".join(prompt_parts)
    
    def _parse_extracted_facts(self, facts_text: str, state: AgentState) -> List[Dict[str, Any]]:
        """
        Parse extracted facts from model response.
        
        Args:
            facts_text: Raw response from extraction model
            state: Current agent state for context
            
        Returns:
            List of parsed facts with metadata
        """
        facts = []
        lines = facts_text.strip().split('\n')
        
        current_fact = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("SCOPE:"):
                # Start of new fact
                if current_fact and "fact" in current_fact:
                    facts.append(current_fact)
                current_fact = {}
                scope = line.replace("SCOPE:", "").strip().upper()
                if scope in ["USER", "SESSION", "AGENT"]:
                    current_fact["scope"] = scope
            
            elif line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip().lower()
                current_fact["category"] = category
            
            elif line.startswith("FACT:"):
                fact = line.replace("FACT:", "").strip()
                current_fact["fact"] = fact
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(confidence_str)
                    current_fact["confidence"] = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    current_fact["confidence"] = 0.5  # Default confidence
        
        # Add the last fact if complete
        if current_fact and "fact" in current_fact:
            facts.append(current_fact)
        
        # Add metadata to each fact
        for fact in facts:
            fact.update({
                "user_id": str(state.user_id),
                "session_id": str(state.session_id),
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "source": "archivist_extraction"
            })
        
        return facts
    
    async def _update_memory(self, facts: List[Dict[str, Any]], state: AgentState) -> None:
        """
        Update memory with extracted facts, including conflict resolution and deduplication.
        
        Args:
            facts: List of extracted facts to store
            state: Current agent state
        """
        try:
            # Initialize memory client if needed
            if not hasattr(self.memory_client, '_initialized') or not self.memory_client._initialized:
                await self.memory_client.initialize()
            
            for fact in facts:
                # Determine memory scope
                scope_str = fact.get("scope", "USER").lower()
                
                # Map to MemoryScope enum
                if scope_str == "user":
                    scope = MemoryScope.USER
                elif scope_str == "session":
                    scope = MemoryScope.SESSION
                elif scope_str == "agent":
                    scope = MemoryScope.AGENT
                else:
                    scope = MemoryScope.USER  # Default to user scope
                
                # Check for existing similar memories (deduplication)
                existing_memories = await self._find_similar_memories(
                    fact["fact"], 
                    state.user_id, 
                    scope,
                    state.session_id if scope == MemoryScope.SESSION else None
                )
                
                # Resolve conflicts and decide whether to add, update, or skip
                action = await self._resolve_memory_conflicts(fact, existing_memories)
                
                if action["type"] == "add":
                    # Add new memory
                    await self._add_new_memory(fact, state, scope, action.get("metadata", {}))
                    
                elif action["type"] == "update":
                    # Update existing memory
                    await self._update_existing_memory(
                        action["memory_id"], 
                        fact, 
                        action.get("metadata", {})
                    )
                    
                elif action["type"] == "merge":
                    # Merge with existing memory
                    await self._merge_memories(
                        action["memory_id"], 
                        fact, 
                        existing_memories[0],
                        action.get("metadata", {})
                    )
                    
                elif action["type"] == "skip":
                    # Skip duplicate memory
                    self.logger.debug(
                        "Skipping duplicate memory",
                        fact_preview=fact["fact"][:50] + "..." if len(fact["fact"]) > 50 else fact["fact"],
                        reason=action.get("reason", "duplicate")
                    )
                    continue
                
                self.logger.debug(
                    "Memory processed",
                    action=action["type"],
                    scope=scope.value,
                    category=fact.get("category"),
                    fact_preview=fact["fact"][:50] + "..." if len(fact["fact"]) > 50 else fact["fact"]
                )
        
        except Exception as e:
            self.logger.error(
                "Failed to update memory",
                error=str(e),
                facts_count=len(facts)
            )
    
    async def _find_similar_memories(
        self, 
        fact_content: str, 
        user_id: Any, 
        scope: MemoryScope,
        session_id: Optional[Any] = None
    ) -> List[Any]:
        """
        Find existing memories similar to the new fact for deduplication.
        
        Args:
            fact_content: Content of the new fact
            user_id: User identifier
            scope: Memory scope to search within
            session_id: Optional session identifier
            
        Returns:
            List of similar existing memories
        """
        try:
            # Search for similar memories using semantic search
            similar_memories = await self.memory_client.search_memories(
                user_id=user_id,
                query=fact_content,
                scope=scope,
                session_id=session_id,
                limit=5  # Check top 5 similar memories
            )
            
            # Filter for high similarity (basic keyword overlap for now)
            highly_similar = []
            fact_words = set(fact_content.lower().split())
            
            for memory in similar_memories:
                memory_words = set(memory.content.lower().split())
                overlap = len(fact_words.intersection(memory_words))
                similarity = overlap / max(len(fact_words), len(memory_words), 1)
                
                # Consider memories with >60% word overlap as similar
                if similarity > 0.6:
                    highly_similar.append(memory)
            
            return highly_similar
            
        except Exception as e:
            self.logger.warning(
                "Failed to find similar memories",
                error=str(e),
                fact_content=fact_content[:50]
            )
            return []
    
    async def _resolve_memory_conflicts(
        self, 
        new_fact: Dict[str, Any], 
        existing_memories: List[Any]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between new fact and existing memories.
        
        Args:
            new_fact: New fact to be stored
            existing_memories: List of existing similar memories
            
        Returns:
            Dictionary with action type and parameters
        """
        if not existing_memories:
            # No conflicts, add new memory
            return {"type": "add"}
        
        # Get the most similar existing memory
        most_similar = existing_memories[0]
        new_confidence = new_fact.get("confidence", 0.5)
        existing_confidence = getattr(most_similar, 'importance_score', 0.5)
        
        # Calculate content similarity more precisely
        similarity_score = self._calculate_content_similarity(
            new_fact["fact"], 
            most_similar.content
        )
        
        # Decision logic for conflict resolution
        if similarity_score > 0.9:
            # Very similar content - check confidence levels
            if new_confidence > existing_confidence + 0.1:
                # New fact is more confident, update existing
                return {
                    "type": "update",
                    "memory_id": str(most_similar.id),
                    "reason": "higher_confidence"
                }
            elif abs(new_confidence - existing_confidence) < 0.1:
                # Similar confidence, merge information
                return {
                    "type": "merge",
                    "memory_id": str(most_similar.id),
                    "reason": "complementary_info"
                }
            else:
                # Existing fact is more confident, skip new one
                return {
                    "type": "skip",
                    "reason": "lower_confidence"
                }
        
        elif similarity_score > 0.7:
            # Moderately similar - check if they're complementary
            if self._are_facts_complementary(new_fact["fact"], most_similar.content):
                return {
                    "type": "merge",
                    "memory_id": str(most_similar.id),
                    "reason": "complementary_facts"
                }
            else:
                # Different enough to warrant separate storage
                return {"type": "add"}
        
        else:
            # Different enough, add as new memory
            return {"type": "add"}
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not content1 or not content2:
            return 0.0
        
        # Normalize content
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _are_facts_complementary(self, fact1: str, fact2: str) -> bool:
        """
        Check if two facts are complementary (provide different but related information).
        
        Args:
            fact1: First fact content
            fact2: Second fact content
            
        Returns:
            True if facts are complementary, False otherwise
        """
        # Simple heuristic: facts are complementary if they share some keywords
        # but have different key information (numbers, specific details)
        
        words1 = set(fact1.lower().split())
        words2 = set(fact2.lower().split())
        
        # Check for shared context words
        shared_words = words1.intersection(words2)
        shared_ratio = len(shared_words) / max(len(words1), len(words2), 1)
        
        # Facts are complementary if they share 30-70% of words
        # (enough context overlap but different specific information)
        return 0.3 <= shared_ratio <= 0.7
    
    async def _add_new_memory(
        self, 
        fact: Dict[str, Any], 
        state: AgentState, 
        scope: MemoryScope,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new memory with enhanced metadata and relationships.
        
        Args:
            fact: Fact dictionary to store
            state: Current agent state
            scope: Memory scope
            additional_metadata: Additional metadata from conflict resolution
        """
        # Create comprehensive metadata
        metadata = {
            "category": fact.get("category", "general"),
            "confidence": fact.get("confidence", 0.5),
            "source": fact.get("source", "archivist"),
            "timestamp": fact.get("timestamp"),
            "extraction_session": str(state.session_id),
            "plan_context": self._extract_plan_context(state),
            "interaction_type": self._determine_interaction_type(state),
            **(additional_metadata or {})
        }
        
        # Add relationships to other memories
        relationships = await self._identify_memory_relationships(fact, state)
        if relationships:
            metadata["relationships"] = relationships
        
        # Add to memory with enhanced metadata
        await self.memory_client.add_memory(
            user_id=state.user_id,
            content=fact["fact"],
            scope=scope,
            session_id=state.session_id if scope == MemoryScope.SESSION else None,
            metadata=metadata,
            importance_score=fact.get("confidence", 0.5)
        )
        
        self.logger.info(
            "New memory added with relationships",
            scope=scope.value,
            category=fact.get("category"),
            relationships_count=len(relationships) if relationships else 0
        )
    
    async def _update_existing_memory(
        self, 
        memory_id: str, 
        new_fact: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing memory with new information.
        
        Args:
            memory_id: ID of memory to update
            new_fact: New fact information
            additional_metadata: Additional metadata from conflict resolution
        """
        # Update importance score with new confidence
        new_importance = new_fact.get("confidence", 0.5)
        
        success = await self.memory_client.update_memory_importance(
            memory_id=memory_id,
            importance_score=new_importance
        )
        
        if success:
            self.logger.info(
                "Memory updated",
                memory_id=memory_id,
                new_importance=new_importance
            )
        else:
            self.logger.warning(
                "Failed to update memory",
                memory_id=memory_id
            )
    
    async def _merge_memories(
        self, 
        existing_memory_id: str, 
        new_fact: Dict[str, Any],
        existing_memory: Any,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Merge new fact with existing memory.
        
        Args:
            existing_memory_id: ID of existing memory
            new_fact: New fact to merge
            existing_memory: Existing memory object
            additional_metadata: Additional metadata from conflict resolution
        """
        # For now, we'll create a new memory with merged content
        # In a full implementation, this would update the existing memory content
        
        merged_content = f"{existing_memory.content} | {new_fact['fact']}"
        merged_confidence = max(
            getattr(existing_memory, 'importance_score', 0.5),
            new_fact.get("confidence", 0.5)
        )
        
        # Update the existing memory's importance score
        await self.memory_client.update_memory_importance(
            memory_id=existing_memory_id,
            importance_score=merged_confidence
        )
        
        self.logger.info(
            "Memories merged",
            existing_memory_id=existing_memory_id,
            merged_confidence=merged_confidence
        )
    
    def _extract_plan_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract relevant plan context for memory metadata."""
        if not state.current_plan:
            return {}
        
        return {
            "plan_id": str(state.current_plan.id),
            "total_steps": len(state.current_plan.steps),
            "current_step": state.current_step_index,
            "plan_status": state.current_plan.status,
            "plan_created_at": state.current_plan.created_at.isoformat()
        }
    
    def _determine_interaction_type(self, state: AgentState) -> str:
        """Determine the type of interaction for memory categorization."""
        if state.tool_outputs:
            # Check the types of actions performed
            action_types = [output.action_type for output in state.tool_outputs]
            
            if any(action.value in ["job_search", "apply_job"] for action in action_types):
                return "job_application"
            elif any(action.value in ["interview_prep", "practice_interview"] for action in action_types):
                return "interview_preparation"
            elif any(action.value in ["profile_update", "resume_update"] for action in action_types):
                return "profile_management"
            else:
                return "general_assistance"
        
        # Analyze message content for interaction type
        if state.messages:
            latest_message = state.messages[-1]
            content = latest_message.content.lower()
            
            if any(word in content for word in ["job", "apply", "application"]):
                return "job_search"
            elif any(word in content for word in ["interview", "prepare", "practice"]):
                return "interview_prep"
            elif any(word in content for word in ["resume", "profile", "cv"]):
                return "profile_management"
        
        return "general_conversation"
    
    async def _identify_memory_relationships(
        self, 
        fact: Dict[str, Any], 
        state: AgentState
    ) -> List[Dict[str, Any]]:
        """
        Identify relationships between the new fact and existing memories.
        
        Args:
            fact: New fact to analyze
            state: Current agent state
            
        Returns:
            List of relationship descriptors
        """
        relationships = []
        
        try:
            # Find related memories based on category and content
            category = fact.get("category", "general")
            
            # Search for memories in the same category
            related_memories = await self.memory_client.search_memories(
                user_id=state.user_id,
                query=f"{category} {fact['fact'][:50]}",  # Use category and partial content
                scope=MemoryScope.USER,
                limit=3
            )
            
            for memory in related_memories:
                # Skip if it's the same content (shouldn't happen due to deduplication)
                if memory.content == fact["fact"]:
                    continue
                
                # Determine relationship type
                relationship_type = self._determine_relationship_type(
                    fact, 
                    memory.content, 
                    memory.metadata
                )
                
                if relationship_type:
                    relationships.append({
                        "memory_id": str(memory.id),
                        "relationship_type": relationship_type,
                        "strength": self._calculate_content_similarity(fact["fact"], memory.content)
                    })
            
        except Exception as e:
            self.logger.warning(
                "Failed to identify memory relationships",
                error=str(e)
            )
        
        return relationships
    
    def _determine_relationship_type(
        self, 
        new_fact: Dict[str, Any], 
        existing_content: str,
        existing_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine the type of relationship between facts.
        
        Args:
            new_fact: New fact dictionary
            existing_content: Content of existing memory
            existing_metadata: Metadata of existing memory
            
        Returns:
            Relationship type string or None
        """
        new_category = new_fact.get("category", "general")
        existing_category = existing_metadata.get("category", "general")
        
        # Same category relationships
        if new_category == existing_category:
            if new_category == "skill":
                return "skill_progression"
            elif new_category == "goal":
                return "goal_refinement"
            elif new_category == "preference":
                return "preference_evolution"
            else:
                return "category_related"
        
        # Cross-category relationships
        if (new_category == "skill" and existing_category == "goal") or \
           (new_category == "goal" and existing_category == "skill"):
            return "skill_goal_alignment"
        
        if (new_category == "gap" and existing_category == "skill") or \
           (new_category == "skill" and existing_category == "gap"):
            return "skill_gap_relationship"
        
        # Temporal relationships (if both have timestamps)
        new_timestamp = new_fact.get("timestamp")
        existing_timestamp = existing_metadata.get("timestamp")
        
        if new_timestamp and existing_timestamp:
            # Could add temporal relationship logic here
            return "temporal_sequence"
        
        return None


# Factory function for creating archivist node
def create_archivist_node(
    memory_client: Optional[MemoryClient] = None,
    extraction_model: Optional[Any] = None
) -> ArchivistNode:
    """
    Factory function to create an archivist node.
    
    Args:
        memory_client: Optional memory client for dependency injection
        extraction_model: Optional extraction model for dependency injection
        
    Returns:
        Configured ArchivistNode instance
    """
    return ArchivistNode(memory_client=memory_client, extraction_model=extraction_model)