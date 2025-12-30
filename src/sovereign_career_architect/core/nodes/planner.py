"""
Planner Node: High-reasoning model for step-by-step plan generation.

This node acts as the "Prefrontal Cortex" of the cognitive architecture, analyzing
enriched context and generating detailed execution plans using advanced reasoning models.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from sovereign_career_architect.core.state import AgentState, StateUpdate
from sovereign_career_architect.core.models import (
    ActionType,
    ExecutionPlan,
    PlanStep,
    UserProfile,
)
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger, log_function_call, log_agent_state

logger = get_logger(__name__)


class PlannerNode:
    """
    Planner Node for intelligent plan generation using high-reasoning models.
    
    This node is responsible for:
    1. Analyzing user context and memory
    2. Generating step-by-step execution plans
    3. Adapting plans based on user profile and preferences
    4. Ensuring plans are actionable and well-structured
    """
    
    def __init__(self, reasoning_model: Optional[Any] = None):
        """
        Initialize the Planner Node.
        
        Args:
            reasoning_model: Optional model for testing. If None, creates default Groq model.
        """
        if reasoning_model is not None:
            self.reasoning_model = reasoning_model
        else:
            try:
                self.reasoning_model = self._create_reasoning_model()
            except ValueError:
                # No API keys available - set to None for testing
                self.reasoning_model = None
        self.logger = logger.bind(node="planner")
    
    def _create_reasoning_model(self) -> Any:
        """Create the high-reasoning model (Llama-3-70B via Groq)."""
        if settings.groq_api_key:
            return ChatGroq(
                model=settings.reasoning_model,
                api_key=settings.groq_api_key,
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=2048,
                timeout=30.0,
            )
        elif settings.openai_api_key:
            # Fallback to OpenAI if Groq not available
            self.logger.warning("Groq API key not found, falling back to OpenAI GPT-4")
            return ChatOpenAI(
                model="gpt-4",
                api_key=settings.openai_api_key,
                temperature=0.1,
                max_tokens=2048,
                timeout=30.0,
            )
        else:
            raise ValueError("No API keys configured for reasoning model")
    
    async def __call__(self, state: AgentState) -> StateUpdate:
        """
        Execute the planner node logic.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with generated execution plan
        """
        self.logger.info(
            "Planner node executing",
            **log_function_call("planner_node", user_id=str(state.user_id)),
            **log_agent_state(state.model_dump())
        )
        
        try:
            # Get the latest user message for planning context
            latest_message = self._get_latest_user_message(state.messages)
            if not latest_message:
                self.logger.warning("No user message found for planning")
                return {"next_node": "executor"}  # Skip to executor with empty plan
            
            # Generate execution plan
            execution_plan = await self._generate_execution_plan(
                user_message=latest_message.content,
                user_profile=state.user_profile,
                memory_context=state.memory_context,
                user_id=state.user_id
            )
            
            # Prepare state updates
            updates: StateUpdate = {
                "current_plan": execution_plan,
                "current_step_index": 0,  # Reset to start of new plan
                "next_node": "executor"  # Route to executor for plan execution
            }
            
            self.logger.info(
                "Planner node completed successfully",
                plan_id=str(execution_plan.id) if execution_plan else None,
                steps_count=len(execution_plan.steps) if execution_plan else 0,
                next_node=updates["next_node"]
            )
            
            return updates
            
        except Exception as e:
            self.logger.error(
                "Planner node execution failed",
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Return minimal updates to allow graceful degradation
            return {
                "current_plan": None,
                "next_node": "executor"  # Still route to executor
            }
    
    async def _generate_execution_plan(
        self,
        user_message: str,
        user_profile: Optional[UserProfile],
        memory_context: Dict[str, Any],
        user_id: Any
    ) -> Optional[ExecutionPlan]:
        """
        Generate a detailed execution plan using the reasoning model.
        
        Args:
            user_message: Latest user message/request
            user_profile: User profile information
            memory_context: Retrieved memory context
            user_id: User identifier
            
        Returns:
            Generated execution plan or None if generation fails
        """
        # Handle case where no model is available
        if self.reasoning_model is None:
            self.logger.warning("No reasoning model available, cannot generate plan")
            return None
        
        try:
            # Build context for the reasoning model
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                user_message, user_profile, memory_context
            )
            
            # Generate plan using reasoning model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger.debug(
                "Generating execution plan",
                user_message_length=len(user_message),
                has_profile=user_profile is not None,
                memory_entries=memory_context.get("total_memories", 0)
            )
            
            response = await self.reasoning_model.ainvoke(messages)
            plan_text = response.content
            
            # Parse the generated plan into structured steps
            steps = self._parse_plan_steps(plan_text)
            
            if not steps:
                self.logger.warning("No valid steps generated from plan")
                return None
            
            # Create execution plan
            execution_plan = ExecutionPlan(
                steps=steps,
                user_id=user_id,
                status="created"
            )
            
            self.logger.info(
                "Execution plan generated successfully",
                plan_id=str(execution_plan.id),
                steps_count=len(steps),
                plan_text_length=len(plan_text)
            )
            
            return execution_plan
            
        except Exception as e:
            self.logger.error(
                "Failed to generate execution plan",
                error=str(e),
                user_message=user_message[:100] if user_message else ""
            )
            return None
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the reasoning model."""
        return """You are the Planner component of the Sovereign Career Architect, an autonomous AI agent for career navigation.

Your role is to analyze user requests and generate detailed, actionable execution plans. You have access to:
- User profile information (skills, experience, preferences)
- Memory context from previous interactions
- Current user request/message

IMPORTANT GUIDELINES:
1. Generate step-by-step plans that are specific and actionable
2. Consider the user's profile and preferences when planning
3. Break complex requests into manageable steps
4. Each step should have a clear action type and description
5. Steps should build logically upon each other
6. Include validation and review steps where appropriate

ACTION TYPES available:
- JOB_SEARCH: Search for job opportunities
- APPLICATION_SUBMIT: Submit job applications
- INTERVIEW_PRACTICE: Conduct mock interviews
- PROFILE_UPDATE: Update user profile information
- GENERAL_CHAT: General conversation/advice

OUTPUT FORMAT:
Generate a numbered list of steps, each with:
1. [ACTION_TYPE] Step description with specific details
2. [ACTION_TYPE] Next step description
...

Example:
1. [JOB_SEARCH] Search for Python developer roles in San Francisco with 3-5 years experience requirement
2. [JOB_SEARCH] Filter results by remote work options and salary range $80k-$120k
3. [APPLICATION_SUBMIT] Apply to top 3 matching positions with tailored cover letters
4. [INTERVIEW_PRACTICE] Prepare for technical interviews focusing on Python and system design

Be specific, actionable, and consider the user's context."""
    
    def _build_user_prompt(
        self,
        user_message: str,
        user_profile: Optional[UserProfile],
        memory_context: Dict[str, Any]
    ) -> str:
        """Build the user prompt with context."""
        prompt_parts = [f"USER REQUEST: {user_message}"]
        
        # Add user profile context
        if user_profile:
            profile_summary = self._summarize_user_profile(user_profile)
            prompt_parts.append(f"USER PROFILE:\n{profile_summary}")
        
        # Add memory context
        if memory_context and memory_context.get("total_memories", 0) > 0:
            memory_summary = self._summarize_memory_context(memory_context)
            prompt_parts.append(f"RELEVANT CONTEXT:\n{memory_summary}")
        
        prompt_parts.append("Generate a detailed execution plan for this request:")
        
        return "\n\n".join(prompt_parts)
    
    def _summarize_user_profile(self, profile: UserProfile) -> str:
        """Create a concise summary of the user profile."""
        parts = []
        
        # Personal info
        if profile.personal_info:
            parts.append(f"Name: {profile.personal_info.name}")
            if profile.personal_info.location:
                parts.append(f"Location: {profile.personal_info.location}")
            parts.append(f"Preferred Language: {profile.personal_info.preferred_language}")
        
        # Skills
        if profile.skills:
            skills_list = [f"{skill.name} ({skill.level})" for skill in profile.skills[:5]]
            parts.append(f"Key Skills: {', '.join(skills_list)}")
        
        # Experience
        if profile.experience:
            exp_count = len(profile.experience)
            latest_role = profile.experience[0] if profile.experience else None
            if latest_role:
                parts.append(f"Latest Role: {latest_role.role} at {latest_role.company}")
            parts.append(f"Total Experience Entries: {exp_count}")
        
        # Preferences
        if profile.preferences:
            prefs = profile.preferences
            if prefs.roles:
                parts.append(f"Preferred Roles: {', '.join(prefs.roles[:3])}")
            if prefs.locations:
                parts.append(f"Preferred Locations: {', '.join(prefs.locations[:3])}")
            if prefs.work_arrangements:
                parts.append(f"Work Preferences: {', '.join(prefs.work_arrangements)}")
        
        return "\n".join(parts) if parts else "No profile information available"
    
    def _summarize_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """Create a summary of relevant memory context."""
        memories = memory_context.get("memories", {})
        parts = []
        
        # User memories (long-term facts)
        user_memories = memories.get("user", [])
        if user_memories:
            parts.append("Long-term context:")
            for memory in user_memories[:3]:  # Top 3 most relevant
                parts.append(f"- {memory['content']}")
        
        # Session memories (current context)
        session_memories = memories.get("session", [])
        if session_memories:
            parts.append("Current session context:")
            for memory in session_memories[:2]:
                parts.append(f"- {memory['content']}")
        
        # Agent memories (learned patterns)
        agent_memories = memories.get("agent", [])
        if agent_memories:
            parts.append("Interaction patterns:")
            for memory in agent_memories[:2]:
                parts.append(f"- {memory['content']}")
        
        return "\n".join(parts) if parts else "No relevant context found"
    
    def _parse_plan_steps(self, plan_text: str) -> List[PlanStep]:
        """
        Parse the generated plan text into structured PlanStep objects.
        
        Args:
            plan_text: Generated plan text from the reasoning model
            
        Returns:
            List of structured plan steps
        """
        steps = []
        lines = plan_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not any(char.isdigit() for char in line[:5]):
                continue  # Skip empty lines or lines without numbers
            
            try:
                # Extract action type and description
                # Expected format: "1. [ACTION_TYPE] Description"
                if '[' in line and ']' in line:
                    # Extract action type
                    start_bracket = line.find('[')
                    end_bracket = line.find(']')
                    action_type_str = line[start_bracket+1:end_bracket].strip()
                    
                    # Map string to ActionType enum
                    action_type = self._map_action_type(action_type_str)
                    
                    # Extract description (everything after the closing bracket)
                    description = line[end_bracket+1:].strip()
                    
                    # Create plan step
                    step = PlanStep(
                        description=description,
                        action_type=action_type,
                        parameters={"raw_text": line}  # Store original for reference
                    )
                    steps.append(step)
                    
                else:
                    # Fallback: treat as general chat if no action type specified
                    description = line
                    # Remove leading numbers/bullets
                    import re
                    description = re.sub(r'^\d+\.?\s*', '', description)
                    
                    if description:  # Only add if there's actual content
                        step = PlanStep(
                            description=description,
                            action_type=ActionType.GENERAL_CHAT,
                            parameters={"raw_text": line}
                        )
                        steps.append(step)
                        
            except Exception as e:
                self.logger.warning(
                    "Failed to parse plan step",
                    line=line,
                    error=str(e)
                )
                continue
        
        self.logger.debug(
            "Parsed plan steps",
            total_lines=len(lines),
            valid_steps=len(steps)
        )
        
        return steps
    
    def _map_action_type(self, action_type_str: str) -> ActionType:
        """Map string action type to ActionType enum."""
        action_type_str = action_type_str.upper().strip()
        
        mapping = {
            "JOB_SEARCH": ActionType.JOB_SEARCH,
            "APPLICATION_SUBMIT": ActionType.APPLICATION_SUBMIT,
            "INTERVIEW_PRACTICE": ActionType.INTERVIEW_PRACTICE,
            "PROFILE_UPDATE": ActionType.PROFILE_UPDATE,
            "GENERAL_CHAT": ActionType.GENERAL_CHAT,
        }
        
        return mapping.get(action_type_str, ActionType.GENERAL_CHAT)
    
    def _get_latest_user_message(self, messages: List[BaseMessage]) -> Optional[BaseMessage]:
        """Get the most recent user message from the conversation history."""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message
        return None


# Factory function for creating planner node
def create_planner_node(reasoning_model: Optional[Any] = None) -> PlannerNode:
    """
    Factory function to create a planner node.
    
    Args:
        reasoning_model: Optional reasoning model for dependency injection
        
    Returns:
        Configured PlannerNode instance
    """
    return PlannerNode(reasoning_model=reasoning_model)