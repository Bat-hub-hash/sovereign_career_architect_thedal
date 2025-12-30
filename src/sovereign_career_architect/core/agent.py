"""Main Sovereign Career Architect agent integrating all components."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import structlog

from sovereign_career_architect.core.graph import (
    create_career_architect_graph, create_career_architect_with_interrupts
)
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.models import UserProfile
from sovereign_career_architect.memory.client import MemoryClient
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.voice.orchestrator import VoiceOrchestrator
from sovereign_career_architect.jobs.matcher import JobMatcher
from sovereign_career_architect.jobs.application import ApplicationWorkflow
from sovereign_career_architect.interview.generator import InterviewQuestionGenerator
from sovereign_career_architect.interview.multilingual import MultilingualInterviewManager
from sovereign_career_architect.interview.feedback import InterviewFeedbackAnalyzer
from sovereign_career_architect.cultural.adapter import CulturalAdapter
from sovereign_career_architect.cultural.code_switching import CodeSwitchingManager
from sovereign_career_architect.core.safety import SafetyLayer
from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class SovereignCareerArchitect:
    """
    Main agent class that orchestrates all career development components.
    
    This class integrates:
    - LangGraph cognitive architecture
    - Memory management with Mem0
    - Browser automation for job applications
    - Voice interface with multilingual support
    - Interview simulation and feedback
    - Cultural adaptation and code-switching
    - Safety layer for human oversight
    """
    
    def __init__(
        self,
        enable_voice: bool = True,
        enable_browser: bool = True,
        enable_interrupts: bool = True,
        memory_config: Optional[Dict[str, Any]] = None,
        browser_config: Optional[Dict[str, Any]] = None,
        voice_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Sovereign Career Architect.
        
        Args:
            enable_voice: Enable voice interface
            enable_browser: Enable browser automation
            enable_interrupts: Enable human-in-the-loop interrupts
            memory_config: Memory system configuration
            browser_config: Browser automation configuration
            voice_config: Voice interface configuration
        """
        self.logger = logger.bind(component="sovereign_agent")
        
        # Configuration
        self.enable_voice = enable_voice
        self.enable_browser = enable_browser
        self.enable_interrupts = enable_interrupts
        
        # Core components
        self.memory_client: Optional[MemoryClient] = None
        self.browser_agent: Optional[BrowserAgent] = None
        self.voice_orchestrator: Optional[VoiceOrchestrator] = None
        self.safety_layer: Optional[SafetyLayer] = None
        
        # Specialized components
        self.job_matcher: Optional[JobMatcher] = None
        self.application_workflow: Optional[ApplicationWorkflow] = None
        self.interview_generator: Optional[InterviewQuestionGenerator] = None
        self.multilingual_interview: Optional[MultilingualInterviewManager] = None
        self.interview_feedback: Optional[InterviewFeedbackAnalyzer] = None
        self.cultural_adapter: Optional[CulturalAdapter] = None
        self.code_switching: Optional[CodeSwitchingManager] = None
        
        # LangGraph workflow
        self.workflow = None
        
        # Configuration storage
        self.memory_config = memory_config or {}
        self.browser_config = browser_config or {}
        self.voice_config = voice_config or {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(
            "Sovereign Career Architect initialized",
            enable_voice=enable_voice,
            enable_browser=enable_browser,
            enable_interrupts=enable_interrupts
        )
    
    async def initialize(self) -> None:
        """Initialize all components and establish connections."""
        self.logger.info("Initializing Sovereign Career Architect components")
        
        try:
            # Initialize safety layer first
            await self._initialize_safety_layer()
            
            # Initialize memory system
            await self._initialize_memory()
            
            # Initialize browser automation if enabled
            if self.enable_browser:
                await self._initialize_browser()
            
            # Initialize voice interface if enabled
            if self.enable_voice:
                await self._initialize_voice()
            
            # Initialize specialized components
            await self._initialize_job_components()
            await self._initialize_interview_components()
            await self._initialize_cultural_components()
            
            # Initialize LangGraph workflow
            await self._initialize_workflow()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error("Component initialization failed", error=str(e))
            raise
    
    async def _initialize_safety_layer(self) -> None:
        """Initialize the safety layer."""
        self.safety_layer = SafetyLayer()
        self.logger.info("Safety layer initialized")
    
    async def _initialize_memory(self) -> None:
        """Initialize the memory system."""
        self.memory_client = MemoryClient(
            config=self.memory_config,
            enable_qdrant=self.memory_config.get("enable_qdrant", False),
            enable_chroma=self.memory_config.get("enable_chroma", False)
        )
        
        # Test memory connection
        await self.memory_client.initialize()
        self.logger.info("Memory system initialized")
    
    async def _initialize_browser(self) -> None:
        """Initialize browser automation."""
        self.browser_agent = BrowserAgent(
            headless=self.browser_config.get("headless", True),
            stealth_mode=self.browser_config.get("stealth_mode", True),
            **self.browser_config
        )
        
        await self.browser_agent.initialize()
        self.logger.info("Browser automation initialized")
    
    async def _initialize_voice(self) -> None:
        """Initialize voice interface."""
        self.voice_orchestrator = VoiceOrchestrator(
            vapi_token=self.voice_config.get("vapi_token", settings.vapi_token),
            **self.voice_config
        )
        
        await self.voice_orchestrator.initialize()
        self.logger.info("Voice interface initialized")
    
    async def _initialize_job_components(self) -> None:
        """Initialize job-related components."""
        # Job matcher
        self.job_matcher = JobMatcher()
        
        # Application workflow
        if self.browser_agent and self.safety_layer:
            self.application_workflow = ApplicationWorkflow(
                browser_agent=self.browser_agent,
                safety_layer=self.safety_layer
            )
        
        self.logger.info("Job components initialized")
    
    async def _initialize_interview_components(self) -> None:
        """Initialize interview-related components."""
        # Interview question generator
        self.interview_generator = InterviewQuestionGenerator()
        
        # Multilingual interview manager
        self.multilingual_interview = MultilingualInterviewManager()
        
        # Interview feedback analyzer
        self.interview_feedback = InterviewFeedbackAnalyzer()
        
        self.logger.info("Interview components initialized")
    
    async def _initialize_cultural_components(self) -> None:
        """Initialize cultural adaptation components."""
        # Cultural adapter
        self.cultural_adapter = CulturalAdapter()
        
        # Code-switching manager
        self.code_switching = CodeSwitchingManager()
        
        self.logger.info("Cultural components initialized")
    
    async def _initialize_workflow(self) -> None:
        """Initialize the LangGraph workflow."""
        # Prepare API clients dictionary
        api_clients = {}
        
        if self.voice_orchestrator:
            api_clients["voice"] = self.voice_orchestrator
        
        if self.job_matcher:
            api_clients["job_matcher"] = self.job_matcher
        
        if self.application_workflow:
            api_clients["application_workflow"] = self.application_workflow
        
        if self.interview_generator:
            api_clients["interview_generator"] = self.interview_generator
        
        if self.multilingual_interview:
            api_clients["multilingual_interview"] = self.multilingual_interview
        
        if self.interview_feedback:
            api_clients["interview_feedback"] = self.interview_feedback
        
        if self.cultural_adapter:
            api_clients["cultural_adapter"] = self.cultural_adapter
        
        if self.code_switching:
            api_clients["code_switching"] = self.code_switching
        
        # Create workflow with or without interrupts
        if self.enable_interrupts:
            self.workflow = create_career_architect_with_interrupts(
                memory_client=self.memory_client,
                browser_agent=self.browser_agent,
                api_clients=api_clients
            )
        else:
            self.workflow = create_career_architect_graph(
                memory_client=self.memory_client,
                browser_agent=self.browser_agent,
                api_clients=api_clients
            )
        
        self.logger.info("LangGraph workflow initialized")
    
    async def process_user_request(
        self,
        user_id: str,
        session_id: str,
        message: str,
        user_profile: Optional[UserProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user request through the complete system.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message: User message/request
            user_profile: User profile (optional, will be retrieved if not provided)
            context: Additional context
            
        Returns:
            Processing result with response and metadata
        """
        self.logger.info(
            "Processing user request",
            user_id=user_id,
            session_id=session_id,
            message_length=len(message)
        )
        
        try:
            # Initialize session if needed
            if session_id not in self.active_sessions:
                await self._initialize_session(user_id, session_id)
            
            # Get or create user profile
            if not user_profile:
                user_profile = await self._get_user_profile(user_id)
            
            # Create initial agent state
            initial_state = AgentState(
                user_id=user_id,
                session_id=session_id,
                user_message=message,
                user_profile=user_profile,
                context=context or {},
                timestamp=datetime.now()
            )
            
            # Process through LangGraph workflow
            result = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Extract response
            response = {
                "success": True,
                "message": result.get("response", "Request processed successfully"),
                "data": result.get("output_data", {}),
                "session_id": session_id,
                "requires_approval": result.get("requires_approval", False),
                "next_steps": result.get("next_steps", []),
                "metadata": {
                    "processing_time": (datetime.now() - initial_state.timestamp).total_seconds(),
                    "components_used": self._get_components_used(result),
                    "memory_updates": result.get("memory_updates", 0)
                }
            }
            
            # Update session
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            self.active_sessions[session_id]["message_count"] += 1
            
            self.logger.info(
                "User request processed successfully",
                user_id=user_id,
                session_id=session_id,
                processing_time=response["metadata"]["processing_time"]
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "User request processing failed",
                user_id=user_id,
                session_id=session_id,
                error=str(e)
            )
            
            return {
                "success": False,
                "message": f"Processing failed: {str(e)}",
                "error": str(e),
                "session_id": session_id
            }
    
    async def _initialize_session(self, user_id: str, session_id: str) -> None:
        """Initialize a new user session."""
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
            "context": {}
        }
        
        self.logger.info(
            "Session initialized",
            user_id=user_id,
            session_id=session_id
        )
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if self.memory_client:
            # Try to retrieve from memory
            memories = await self.memory_client.search_memories(
                query=f"user profile for {user_id}",
                user_id=user_id,
                limit=1
            )
            
            if memories:
                # Extract profile from memory (simplified)
                # In practice, you'd have a more sophisticated profile reconstruction
                pass
        
        # Return default profile for now
        from sovereign_career_architect.core.models import PersonalInfo
        
        return UserProfile(
            personal_info=PersonalInfo(
                name="User",
                email="user@example.com",
                preferred_language="en"
            ),
            skills=[],
            experience=[],
            education=[],
            preferences={},
            documents=None
        )
    
    def _get_components_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract which components were used in processing."""
        components = []
        
        if result.get("memory_accessed"):
            components.append("memory")
        
        if result.get("browser_used"):
            components.append("browser")
        
        if result.get("voice_used"):
            components.append("voice")
        
        if result.get("job_matching_used"):
            components.append("job_matching")
        
        if result.get("interview_used"):
            components.append("interview")
        
        if result.get("cultural_adaptation_used"):
            components.append("cultural_adaptation")
        
        return components
    
    async def start_job_search(
        self,
        user_id: str,
        session_id: str,
        search_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start a job search process."""
        self.logger.info(
            "Starting job search",
            user_id=user_id,
            session_id=session_id,
            criteria=search_criteria
        )
        
        if not self.job_matcher:
            return {"success": False, "error": "Job matching not available"}
        
        try:
            # This would integrate with the job matching system
            # For now, return a placeholder response
            return {
                "success": True,
                "message": "Job search initiated",
                "search_id": f"search_{session_id}_{datetime.now().timestamp()}",
                "criteria": search_criteria
            }
            
        except Exception as e:
            self.logger.error("Job search failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def start_interview_simulation(
        self,
        user_id: str,
        session_id: str,
        interview_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start an interview simulation."""
        self.logger.info(
            "Starting interview simulation",
            user_id=user_id,
            session_id=session_id,
            config=interview_config
        )
        
        if not self.interview_generator:
            return {"success": False, "error": "Interview simulation not available"}
        
        try:
            # Generate interview questions
            questions = await self.interview_generator.generate_interview_questions(
                job_role=interview_config.get("job_role", "Software Engineer"),
                experience_level=interview_config.get("experience_level", "mid"),
                question_count=interview_config.get("question_count", 5)
            )
            
            return {
                "success": True,
                "message": "Interview simulation started",
                "interview_id": f"interview_{session_id}_{datetime.now().timestamp()}",
                "questions": [q.dict() for q in questions],
                "config": interview_config
            }
            
        except Exception as e:
            self.logger.error("Interview simulation failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def apply_cultural_adaptation(
        self,
        message: str,
        user_profile: UserProfile,
        target_culture: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply cultural adaptation to a message."""
        if not self.cultural_adapter:
            return {"success": False, "error": "Cultural adaptation not available"}
        
        try:
            # Create cultural profile
            cultural_profile = await self.cultural_adapter.create_cultural_profile(
                user_profile,
                user_profile.preferences.get("cultural_background", "american"),
                context
            )
            
            # Create cultural context
            from sovereign_career_architect.cultural.adapter import CulturalContext
            cultural_context = CulturalContext(
                situation_type=context.get("situation_type", "professional"),
                formality_level=context.get("formality_level", "formal"),
                audience=context.get("audience", "professional")
            )
            
            # Adapt message
            adapted_message = await self.cultural_adapter.adapt_message(
                message, cultural_profile, cultural_context, target_culture
            )
            
            return {
                "success": True,
                "original_message": message,
                "adapted_message": adapted_message,
                "target_culture": target_culture
            }
            
        except Exception as e:
            self.logger.error("Cultural adaptation failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "overall_status": "healthy",
            "components": {},
            "active_sessions": len(self.active_sessions),
            "timestamp": datetime.now().isoformat()
        }
        
        # Check component status
        components = {
            "memory": self.memory_client,
            "browser": self.browser_agent,
            "voice": self.voice_orchestrator,
            "safety": self.safety_layer,
            "job_matcher": self.job_matcher,
            "application_workflow": self.application_workflow,
            "interview_generator": self.interview_generator,
            "cultural_adapter": self.cultural_adapter,
            "workflow": self.workflow
        }
        
        for name, component in components.items():
            if component is not None:
                status["components"][name] = "healthy"
            else:
                status["components"][name] = "unavailable"
                if name in ["memory", "workflow"]:  # Critical components
                    status["overall_status"] = "degraded"
        
        return status
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down Sovereign Career Architect")
        
        try:
            # Close browser agent
            if self.browser_agent:
                await self.browser_agent.close()
            
            # Close voice orchestrator
            if self.voice_orchestrator:
                await self.voice_orchestrator.close()
            
            # Close memory client
            if self.memory_client:
                await self.memory_client.close()
            
            # Clear active sessions
            self.active_sessions.clear()
            
            self.logger.info("Shutdown completed successfully")
            
        except Exception as e:
            self.logger.error("Shutdown error", error=str(e))
            raise


# Convenience function for creating a fully configured agent
async def create_sovereign_career_architect(
    config: Optional[Dict[str, Any]] = None
) -> SovereignCareerArchitect:
    """
    Create and initialize a fully configured Sovereign Career Architect.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized agent instance
    """
    config = config or {}
    
    agent = SovereignCareerArchitect(
        enable_voice=config.get("enable_voice", True),
        enable_browser=config.get("enable_browser", True),
        enable_interrupts=config.get("enable_interrupts", True),
        memory_config=config.get("memory", {}),
        browser_config=config.get("browser", {}),
        voice_config=config.get("voice", {})
    )
    
    await agent.initialize()
    return agent