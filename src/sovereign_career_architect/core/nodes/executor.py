"""Executor Node for performing actions via browser automation and external tools."""

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

import structlog
from langchain_core.language_models import BaseChatModel

from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import ActionResult, ActionType, PlanStep
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutorNode:
    """
    Executor Node for performing actions via browser automation and external tools.
    
    This node acts as the "Motor Cortex" of the cognitive architecture, interfacing
    with external tools like Browser-use for web interaction, APIs for data retrieval,
    and other services to execute the specific steps defined by the Planner.
    """
    
    def __init__(
        self,
        browser_agent=None,
        api_clients: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ):
        """
        Initialize the Executor Node.
        
        Args:
            browser_agent: Browser automation agent (Browser-use)
            api_clients: Dictionary of API clients for external services
            max_retries: Maximum retry attempts for failed actions
        """
        self.browser_agent = browser_agent
        self.api_clients = api_clients or {}
        self.max_retries = max_retries
        self.logger = logger.bind(component="executor_node")
    
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute the current plan step.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution results
        """
        self.logger.info(
            "Executor node executing",
            session_id=str(state.session_id),
            current_step=state.current_step_index,
            has_plan=state.current_plan is not None
        )
        
        try:
            # Validate we have a plan to execute
            if not state.current_plan or not state.current_plan.steps:
                self.logger.warning("No execution plan available")
                return {
                    "tool_outputs": [ActionResult(
                        action_type=ActionType.GENERAL_CHAT,
                        success=False,
                        error_message="No execution plan available"
                    )],
                    "next_node": "planner"
                }
            
            # Get current step to execute
            if state.current_step_index >= len(state.current_plan.steps):
                self.logger.info("All plan steps completed")
                return {
                    "tool_outputs": state.tool_outputs,
                    "next_node": "reviewer"
                }
            
            current_step = state.current_plan.steps[state.current_step_index]
            
            self.logger.info(
                "Executing plan step",
                step_description=current_step.description,
                action_type=current_step.action_type.value,
                step_index=state.current_step_index
            )
            
            # Execute the step with retry logic
            result = await self._execute_step_with_retry(current_step, state)
            
            # Update state with results
            updated_outputs = state.tool_outputs.copy()
            updated_outputs.append(result)
            
            # Increment step index if successful
            next_step_index = state.current_step_index
            if result.success:
                next_step_index += 1
                self.logger.info(
                    "Step executed successfully",
                    step_index=state.current_step_index,
                    next_step=next_step_index
                )
            else:
                self.logger.warning(
                    "Step execution failed",
                    step_index=state.current_step_index,
                    error=result.error_message
                )
            
            return {
                "tool_outputs": updated_outputs,
                "current_step_index": next_step_index,
                "next_node": "reviewer"
            }
            
        except Exception as e:
            self.logger.error(
                "Executor node failed",
                error=str(e),
                session_id=str(state.session_id)
            )
            
            # Return error result
            error_result = ActionResult(
                action_type=ActionType.GENERAL_CHAT,
                success=False,
                error_message=f"Executor error: {str(e)}"
            )
            
            return {
                "tool_outputs": state.tool_outputs + [error_result],
                "next_node": "reviewer"
            }
    
    async def _execute_step_with_retry(
        self, 
        step: PlanStep, 
        state: AgentState
    ) -> ActionResult:
        """
        Execute a plan step with retry logic.
        
        Args:
            step: Plan step to execute
            state: Current agent state
            
        Returns:
            Action result with execution outcome
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    "Attempting step execution",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    step_id=str(step.id)
                )
                
                # Execute based on action type
                result = await self._execute_action(step, state)
                
                if result.success:
                    self.logger.info(
                        "Step execution successful",
                        attempt=attempt + 1,
                        step_id=str(step.id)
                    )
                    return result
                else:
                    last_error = result.error_message
                    self.logger.warning(
                        "Step execution failed, retrying",
                        attempt=attempt + 1,
                        error=result.error_message
                    )
                    
                    # Add delay between retries
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            except Exception as e:
                last_error = str(e)
                self.logger.error(
                    "Step execution exception",
                    attempt=attempt + 1,
                    error=str(e),
                    step_id=str(step.id)
                )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        return ActionResult(
            action_type=step.action_type,
            success=False,
            error_message=f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )
    
    async def _execute_action(self, step: PlanStep, state: AgentState) -> ActionResult:
        """
        Execute a specific action based on its type.
        
        Args:
            step: Plan step to execute
            state: Current agent state
            
        Returns:
            Action result
        """
        start_time = datetime.utcnow()
        
        try:
            if step.action_type == ActionType.JOB_SEARCH:
                result = await self._execute_job_search(step, state)
            elif step.action_type == ActionType.APPLICATION_SUBMIT:
                result = await self._execute_application_submit(step, state)
            elif step.action_type == ActionType.INTERVIEW_PRACTICE:
                result = await self._execute_interview_practice(step, state)
            elif step.action_type == ActionType.PROFILE_UPDATE:
                result = await self._execute_profile_update(step, state)
            else:
                result = ActionResult(
                    action_type=step.action_type,
                    success=False,
                    error_message=f"Unknown action type: {step.action_type}"
                )
            
            # Add execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return ActionResult(
                action_type=step.action_type,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _execute_job_search(self, step: PlanStep, state: AgentState) -> ActionResult:
        """Execute job search action."""
        self.logger.info("Executing job search", step_id=str(step.id))
        
        try:
            # Extract search parameters from step
            search_params = step.parameters
            query = search_params.get("query", "")
            location = search_params.get("location", "")
            
            # Use user profile for context if available
            if state.user_profile:
                if not query and state.user_profile.preferences.roles:
                    query = " OR ".join(state.user_profile.preferences.roles)
                if not location and state.user_profile.preferences.locations:
                    location = state.user_profile.preferences.locations[0]
            
            # Mock job search implementation
            # In production, this would use browser automation or job APIs
            jobs_found = await self._mock_job_search(query, location)
            
            return ActionResult(
                action_type=ActionType.JOB_SEARCH,
                success=True,
                data={
                    "query": query,
                    "location": location,
                    "jobs_found": len(jobs_found),
                    "matches": jobs_found[:5]  # Top 5 matches
                }
            )
            
        except Exception as e:
            return ActionResult(
                action_type=ActionType.JOB_SEARCH,
                success=False,
                error_message=f"Job search failed: {str(e)}"
            )
    
    async def _execute_application_submit(self, step: PlanStep, state: AgentState) -> ActionResult:
        """Execute job application submission."""
        self.logger.info("Executing application submission", step_id=str(step.id))
        
        try:
            # Extract application parameters
            app_params = step.parameters
            job_url = app_params.get("job_url", "")
            
            if not job_url:
                return ActionResult(
                    action_type=ActionType.APPLICATION_SUBMIT,
                    success=False,
                    error_message="No job URL provided for application"
                )
            
            # Mock application submission
            # In production, this would use browser automation
            success = await self._mock_application_submit(job_url, state.user_profile)
            
            return ActionResult(
                action_type=ActionType.APPLICATION_SUBMIT,
                success=success,
                data={
                    "job_url": job_url,
                    "submitted_at": datetime.utcnow().isoformat(),
                    "application_id": f"app_{datetime.utcnow().timestamp()}"
                }
            )
            
        except Exception as e:
            return ActionResult(
                action_type=ActionType.APPLICATION_SUBMIT,
                success=False,
                error_message=f"Application submission failed: {str(e)}"
            )
    
    async def _execute_interview_practice(self, step: PlanStep, state: AgentState) -> ActionResult:
        """Execute interview practice session."""
        self.logger.info("Executing interview practice", step_id=str(step.id))
        
        try:
            # Extract practice parameters
            practice_params = step.parameters
            role = practice_params.get("role", "Software Engineer")
            difficulty = practice_params.get("difficulty", "medium")
            
            # Mock interview practice
            questions = await self._mock_generate_interview_questions(role, difficulty)
            
            return ActionResult(
                action_type=ActionType.INTERVIEW_PRACTICE,
                success=True,
                data={
                    "role": role,
                    "difficulty": difficulty,
                    "questions_generated": len(questions),
                    "questions": questions[:3]  # First 3 questions
                }
            )
            
        except Exception as e:
            return ActionResult(
                action_type=ActionType.INTERVIEW_PRACTICE,
                success=False,
                error_message=f"Interview practice failed: {str(e)}"
            )
    
    async def _execute_profile_update(self, step: PlanStep, state: AgentState) -> ActionResult:
        """Execute profile update action."""
        self.logger.info("Executing profile update", step_id=str(step.id))
        
        try:
            # Extract update parameters
            update_params = step.parameters
            updates = update_params.get("updates", {})
            
            # Mock profile update
            # In production, this would update the actual user profile
            updated_fields = list(updates.keys())
            
            return ActionResult(
                action_type=ActionType.PROFILE_UPDATE,
                success=True,
                data={
                    "updated_fields": updated_fields,
                    "update_count": len(updated_fields),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            return ActionResult(
                action_type=ActionType.PROFILE_UPDATE,
                success=False,
                error_message=f"Profile update failed: {str(e)}"
            )
    
    # Mock implementations for development/testing
    async def _mock_job_search(self, query: str, location: str) -> List[Dict[str, Any]]:
        """Mock job search implementation."""
        await asyncio.sleep(0.5)  # Simulate API call
        
        return [
            {
                "title": f"Senior {query} Developer",
                "company": "TechCorp Inc",
                "location": location or "Remote",
                "url": "https://example.com/job1",
                "match_score": 0.95
            },
            {
                "title": f"{query} Engineer",
                "company": "StartupXYZ",
                "location": location or "San Francisco",
                "url": "https://example.com/job2",
                "match_score": 0.87
            },
            {
                "title": f"Lead {query} Architect",
                "company": "BigTech Corp",
                "location": location or "New York",
                "url": "https://example.com/job3",
                "match_score": 0.82
            }
        ]
    
    async def _mock_application_submit(self, job_url: str, user_profile) -> bool:
        """Mock application submission."""
        await asyncio.sleep(1.0)  # Simulate form filling
        return True  # Always succeed in mock
    
    async def _mock_generate_interview_questions(self, role: str, difficulty: str) -> List[str]:
        """Mock interview question generation."""
        await asyncio.sleep(0.3)  # Simulate generation
        
        questions = [
            f"Tell me about your experience with {role} responsibilities",
            f"How would you approach a challenging {role} project?",
            f"What technologies are you most excited about in the {role} field?",
            f"Describe a time when you had to solve a complex problem as a {role}",
            f"How do you stay updated with the latest trends in {role}?"
        ]
        
        return questions


def create_executor_node(
    browser_agent=None,
    api_clients: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
) -> ExecutorNode:
    """
    Factory function to create an Executor Node.
    
    Args:
        browser_agent: Browser automation agent
        api_clients: Dictionary of API clients
        max_retries: Maximum retry attempts
        
    Returns:
        Configured ExecutorNode instance
    """
    return ExecutorNode(
        browser_agent=browser_agent,
        api_clients=api_clients,
        max_retries=max_retries
    )