"""
Reviewer Node: Output validation and quality gate for agent actions.

This node acts as a quality gate, comparing tool outputs against user profile
and current plan to ensure results meet expectations and provide feedback
for self-correction when needed.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

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
)
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger, log_function_call, log_agent_state

logger = get_logger(__name__)


class ReviewDecision(str, Enum):
    """Possible review decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_IMPROVEMENT = "needs_improvement"


class ReviewerNode:
    """
    Reviewer Node for output validation and quality assurance.
    
    This node is responsible for:
    1. Validating tool outputs against user profile and plan
    2. Checking output quality and relevance
    3. Providing specific feedback for improvements
    4. Deciding whether to approve, reject, or request improvements
    """
    
    def __init__(self, review_model: Optional[Any] = None):
        """
        Initialize the Reviewer Node.
        
        Args:
            review_model: Optional model for testing. If None, creates default model.
        """
        if review_model is not None:
            self.review_model = review_model
        else:
            try:
                self.review_model = self._create_review_model()
            except ValueError:
                # No API keys available - set to None for testing
                self.review_model = None
        self.logger = logger.bind(node="reviewer")
    
    def _create_review_model(self) -> Any:
        """Create the review model for output validation."""
        if settings.openai_api_key:
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fast model for review tasks
                api_key=settings.openai_api_key,
                temperature=0.0,  # Deterministic for consistent reviews
                max_tokens=1024,
                timeout=20.0,
            )
        elif settings.groq_api_key:
            # Fallback to Groq if OpenAI not available
            return ChatGroq(
                model="llama-3.1-8b-instant",  # Fast model for reviews
                api_key=settings.groq_api_key,
                temperature=0.0,
                max_tokens=1024,
                timeout=20.0,
            )
        else:
            raise ValueError("No API keys configured for review model")
    
    async def __call__(self, state: AgentState) -> StateUpdate:
        """
        Execute the reviewer node logic.
        
        Args:
            state: Current agent state
            
        Returns:
            State updates with review results and routing decisions
        """
        self.logger.info(
            "Reviewer node executing",
            **log_function_call("reviewer_node", user_id=str(state.user_id)),
            **log_agent_state(state.model_dump())
        )
        
        try:
            # Get the latest tool outputs to review
            if not state.tool_outputs:
                self.logger.info("No tool outputs to review, routing to archivist")
                return {"next_node": "archivist"}
            
            latest_output = state.tool_outputs[-1]
            
            # Perform review
            review_result = await self._review_output(
                output=latest_output,
                user_profile=state.user_profile,
                current_plan=state.current_plan,
                current_step_index=state.current_step_index
            )
            
            # Determine next action based on review
            next_node, critique = self._determine_next_action(review_result, state)
            
            # Prepare state updates
            updates: StateUpdate = {
                "critique": critique,
                "next_node": next_node
            }
            
            # Handle retry count updates
            if review_result["decision"] == ReviewDecision.REJECT:
                if state.retry_count < state.max_retries:
                    # Increment retry count for next attempt
                    state.increment_retry()
                # Note: if max retries exceeded, _determine_next_action already handled routing
            else:
                # Reset retry count on success
                state.reset_retry_count()
            
            self.logger.info(
                "Reviewer node completed",
                decision=review_result["decision"],
                next_node=updates["next_node"],
                retry_count=state.retry_count
            )
            
            return updates
            
        except Exception as e:
            self.logger.error(
                "Reviewer node execution failed",
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Default to approval on error to prevent blocking
            return {
                "critique": f"Review failed due to error: {str(e)}. Proceeding with output.",
                "next_node": "archivist"
            }
    
    async def _review_output(
        self,
        output: ActionResult,
        user_profile: Optional[UserProfile],
        current_plan: Optional[ExecutionPlan],
        current_step_index: int
    ) -> Dict[str, Any]:
        """
        Review a tool output for quality and relevance.
        
        Args:
            output: The action result to review
            user_profile: User profile for context
            current_plan: Current execution plan
            current_step_index: Index of current step being executed
            
        Returns:
            Review result with decision and feedback
        """
        # Handle case where no model is available
        if self.review_model is None:
            self.logger.warning("No review model available, defaulting to approval")
            return {
                "decision": ReviewDecision.APPROVE,
                "feedback": "No review model available, defaulting to approval",
                "confidence": 0.0
            }
        
        try:
            # Build review context
            system_prompt = self._build_review_system_prompt()
            user_prompt = self._build_review_user_prompt(
                output, user_profile, current_plan, current_step_index
            )
            
            # Get review from model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger.debug(
                "Requesting output review",
                action_type=output.action_type.value,
                success=output.success,
                data_keys=list(output.data.keys()) if output.data else []
            )
            
            response = await self.review_model.ainvoke(messages)
            review_text = response.content
            
            # Parse review response
            review_result = self._parse_review_response(review_text)
            
            self.logger.debug(
                "Review completed",
                decision=review_result["decision"],
                feedback_length=len(review_result.get("feedback", ""))
            )
            
            return review_result
            
        except Exception as e:
            self.logger.error(
                "Failed to review output",
                error=str(e),
                action_type=output.action_type.value if output else "unknown"
            )
            
            # Default to approval on review failure
            return {
                "decision": ReviewDecision.APPROVE,
                "feedback": f"Review failed: {str(e)}. Defaulting to approval.",
                "confidence": 0.0
            }
    
    def _build_review_system_prompt(self) -> str:
        """Build the system prompt for output review."""
        return """You are the Reviewer component of the Sovereign Career Architect, responsible for quality assurance.

Your role is to evaluate tool outputs and determine if they meet the user's needs and plan requirements.

EVALUATION CRITERIA:
1. Relevance: Does the output match the intended action and user context?
2. Quality: Is the output complete, accurate, and useful?
3. Alignment: Does it align with the user's profile, preferences, and goals?
4. Completeness: Are all expected data fields present and meaningful?

REVIEW DECISIONS:
- APPROVE: Output meets all criteria and should be accepted
- REJECT: Output has significant issues and should be regenerated
- NEEDS_IMPROVEMENT: Output is acceptable but could be enhanced

OUTPUT FORMAT:
DECISION: [APPROVE/REJECT/NEEDS_IMPROVEMENT]
CONFIDENCE: [0.0-1.0]
FEEDBACK: [Specific feedback explaining the decision]

Be thorough but concise. Focus on actionable feedback that can improve future outputs."""
    
    def _build_review_user_prompt(
        self,
        output: ActionResult,
        user_profile: Optional[UserProfile],
        current_plan: Optional[ExecutionPlan],
        current_step_index: int
    ) -> str:
        """Build the user prompt for review with context."""
        prompt_parts = []
        
        # Output to review
        prompt_parts.append("OUTPUT TO REVIEW:")
        prompt_parts.append(f"Action Type: {output.action_type.value}")
        prompt_parts.append(f"Success: {output.success}")
        
        if output.error_message:
            prompt_parts.append(f"Error: {output.error_message}")
        
        if output.data:
            prompt_parts.append("Output Data:")
            for key, value in output.data.items():
                # Truncate long values for readability
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                prompt_parts.append(f"  {key}: {value_str}")
        
        # Current step context
        if current_plan and current_step_index < len(current_plan.steps):
            current_step = current_plan.steps[current_step_index]
            prompt_parts.append(f"\nCURRENT STEP:")
            prompt_parts.append(f"Description: {current_step.description}")
            prompt_parts.append(f"Expected Action: {current_step.action_type.value}")
        
        # User profile context
        if user_profile:
            profile_summary = self._summarize_user_profile_for_review(user_profile)
            prompt_parts.append(f"\nUSER CONTEXT:\n{profile_summary}")
        
        prompt_parts.append("\nPlease review this output and provide your assessment:")
        
        return "\n".join(prompt_parts)
    
    def _summarize_user_profile_for_review(self, profile: UserProfile) -> str:
        """Create a concise profile summary for review context."""
        parts = []
        
        # Key preferences for relevance checking
        if profile.preferences:
            prefs = profile.preferences
            if prefs.roles:
                parts.append(f"Target Roles: {', '.join(prefs.roles[:3])}")
            if prefs.locations:
                parts.append(f"Preferred Locations: {', '.join(prefs.locations[:3])}")
            if prefs.work_arrangements:
                parts.append(f"Work Preferences: {', '.join(prefs.work_arrangements)}")
        
        # Key skills for matching
        if profile.skills:
            top_skills = [f"{skill.name} ({skill.level})" for skill in profile.skills[:3]]
            parts.append(f"Key Skills: {', '.join(top_skills)}")
        
        # Experience level
        if profile.experience:
            parts.append(f"Experience Entries: {len(profile.experience)}")
        
        return "\n".join(parts) if parts else "No specific preferences available"
    
    def _parse_review_response(self, review_text: str) -> Dict[str, Any]:
        """
        Parse the review response from the model.
        
        Args:
            review_text: Raw response from review model
            
        Returns:
            Parsed review result
        """
        result = {
            "decision": ReviewDecision.APPROVE,  # Default
            "feedback": review_text,
            "confidence": 0.5
        }
        
        lines = review_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("DECISION:"):
                decision_str = line.replace("DECISION:", "").strip().upper()
                if decision_str in ["APPROVE", "ACCEPT"]:
                    result["decision"] = ReviewDecision.APPROVE
                elif decision_str in ["REJECT", "DENY"]:
                    result["decision"] = ReviewDecision.REJECT
                elif decision_str in ["NEEDS_IMPROVEMENT", "IMPROVE"]:
                    result["decision"] = ReviewDecision.NEEDS_IMPROVEMENT
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(confidence_str)
                    result["confidence"] = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    pass  # Keep default confidence
            
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
                if feedback:
                    result["feedback"] = feedback
        
        return result
    
    def _determine_next_action(
        self, 
        review_result: Dict[str, Any], 
        state: AgentState
    ) -> tuple[str, str]:
        """
        Determine the next node and critique based on review result and retry logic.
        
        Args:
            review_result: Result from output review
            state: Current agent state
            
        Returns:
            Tuple of (next_node, critique)
        """
        decision = review_result["decision"]
        feedback = review_result.get("feedback", "No specific feedback provided")
        
        if decision == ReviewDecision.APPROVE:
            return "archivist", f"Output approved: {feedback}"
        
        elif decision == ReviewDecision.REJECT:
            # Check retry logic for rejections
            if state.retry_count >= state.max_retries:
                # Force approval when max retries exceeded
                return "archivist", f"Max retries exceeded, accepting output: {feedback}"
            else:
                # Normal rejection - go back to planner
                return "planner", f"Output rejected: {feedback}. Please regenerate with improvements."
        
        elif decision == ReviewDecision.NEEDS_IMPROVEMENT:
            # For minor improvements, we can proceed but note the issues
            return "archivist", f"Output accepted with notes: {feedback}"
        
        else:
            # Fallback
            return "archivist", f"Unknown review decision, proceeding: {feedback}"
    
    def _should_skip_review(self, output: ActionResult) -> bool:
        """
        Determine if an output should skip detailed review.
        
        Some outputs (like simple confirmations) may not need full review.
        """
        # Skip review for simple general chat responses
        if output.action_type == ActionType.GENERAL_CHAT and output.success:
            return True
        
        # Skip review for profile updates (usually safe)
        if output.action_type == ActionType.PROFILE_UPDATE and output.success:
            return True
        
        return False


# Factory function for creating reviewer node
def create_reviewer_node(review_model: Optional[Any] = None) -> ReviewerNode:
    """
    Factory function to create a reviewer node.
    
    Args:
        review_model: Optional review model for dependency injection
        
    Returns:
        Configured ReviewerNode instance
    """
    return ReviewerNode(review_model=review_model)