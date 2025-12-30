"""Voice interface orchestrator using Vapi.ai."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import structlog
import httpx

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class VoiceEventType(Enum):
    """Types of voice events from Vapi.ai."""
    CALL_STARTED = "call-started"
    CALL_ENDED = "call-ended"
    FUNCTION_CALL = "function-call"
    TRANSCRIPT = "transcript"
    SPEECH_UPDATE = "speech-update"
    HANG = "hang"
    ERROR = "error"


@dataclass
class VoiceConfig:
    """Configuration for voice interface."""
    api_key: str
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    voice_id: str = "jennifer"  # Default Vapi.ai voice
    language: str = "en"
    max_duration: int = 1800  # 30 minutes
    silence_timeout: int = 30  # 30 seconds
    background_sound: str = "office"
    
    # TTS settings
    tts_provider: str = "11labs"
    tts_voice_id: str = "jennifer"
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75
    
    # STT settings
    stt_provider: str = "deepgram"
    stt_model: str = "nova-2"
    stt_language: str = "en-US"


@dataclass
class FunctionDefinition:
    """Definition of a function that can be called from voice interface."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable


class VoiceOrchestrator:
    """Orchestrates voice interactions using Vapi.ai."""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig(
            api_key=settings.vapi_api_key or "",
            webhook_url=settings.vapi_webhook_url,
            webhook_secret=settings.vapi_webhook_secret
        )
        self.logger = logger.bind(component="voice_orchestrator")
        
        # Function registry
        self.functions: Dict[str, FunctionDefinition] = {}
        
        # Active calls
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        
        # HTTP client for Vapi.ai API
        self.client = httpx.AsyncClient(
            base_url="https://api.vapi.ai",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
        # Register default functions
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default voice functions."""
        
        # Job search function
        self.register_function(
            name="search_jobs",
            description="Search for job opportunities based on criteria",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Job search query (e.g., 'software engineer', 'data scientist')"
                    },
                    "location": {
                        "type": "string",
                        "description": "Job location (e.g., 'San Francisco', 'Remote')"
                    },
                    "experience_level": {
                        "type": "string",
                        "enum": ["entry", "mid", "senior", "executive"],
                        "description": "Required experience level"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_job_search
        )
        
        # Job application function
        self.register_function(
            name="apply_to_job",
            description="Apply to a specific job posting",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Unique identifier of the job posting"
                    },
                    "cover_letter_style": {
                        "type": "string",
                        "enum": ["formal", "casual", "creative"],
                        "description": "Style of cover letter to generate"
                    },
                    "custom_message": {
                        "type": "string",
                        "description": "Custom message to include in application"
                    }
                },
                "required": ["job_id"]
            },
            handler=self._handle_job_application
        )
        
        # Interview preparation function
        self.register_function(
            name="prepare_interview",
            description="Start interview preparation session",
            parameters={
                "type": "object",
                "properties": {
                    "job_role": {
                        "type": "string",
                        "description": "Job role to prepare for (e.g., 'Software Engineer')"
                    },
                    "company": {
                        "type": "string",
                        "description": "Company name (optional)"
                    },
                    "interview_type": {
                        "type": "string",
                        "enum": ["technical", "behavioral", "mixed"],
                        "description": "Type of interview preparation"
                    },
                    "language": {
                        "type": "string",
                        "description": "Interview language (e.g., 'en', 'hi', 'ta')"
                    }
                },
                "required": ["job_role"]
            },
            handler=self._handle_interview_preparation
        )
        
        # Profile update function
        self.register_function(
            name="update_profile",
            description="Update user profile information",
            parameters={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["skills", "experience", "education", "preferences"],
                        "description": "Profile field to update"
                    },
                    "value": {
                        "type": "string",
                        "description": "New value for the field"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "replace"],
                        "description": "Action to perform on the field"
                    }
                },
                "required": ["field", "value"]
            },
            handler=self._handle_profile_update
        )
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> None:
        """Register a function that can be called from voice interface."""
        function_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )
        
        self.functions[name] = function_def
        
        self.logger.info(
            "Voice function registered",
            function_name=name,
            description=description
        )
    
    async def create_assistant(
        self,
        name: str,
        system_message: str,
        first_message: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Create a Vapi.ai assistant configuration."""
        
        # Build function definitions for Vapi.ai
        functions = []
        for func_name, func_def in self.functions.items():
            functions.append({
                "name": func_name,
                "description": func_def.description,
                "parameters": func_def.parameters
            })
        
        assistant_config = {
            "name": name,
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    }
                ],
                "functions": functions,
                "maxTokens": 500,
                "temperature": 0.7
            },
            "voice": {
                "provider": self.config.tts_provider,
                "voiceId": self.config.tts_voice_id,
                "stability": self.config.tts_stability,
                "similarityBoost": self.config.tts_similarity_boost
            },
            "transcriber": {
                "provider": self.config.stt_provider,
                "model": self.config.stt_model,
                "language": self.config.stt_language
            },
            "firstMessage": first_message,
            "voicemailMessage": "I'm not available right now, but you can leave a message and I'll get back to you.",
            "endCallMessage": "Thank you for using Sovereign Career Architect. Have a great day!",
            "backgroundSound": self.config.background_sound,
            "maxDurationSeconds": self.config.max_duration,
            "silenceTimeoutSeconds": self.config.silence_timeout,
            "responseDelaySeconds": 0.4,
            "llmRequestDelaySeconds": 0.1,
            "numWordsToInterruptAssistant": 2,
            "serverUrl": self.config.webhook_url,
            "serverUrlSecret": self.config.webhook_secret
        }
        
        try:
            response = await self.client.post("/assistant", json=assistant_config)
            response.raise_for_status()
            
            assistant_data = response.json()
            
            self.logger.info(
                "Assistant created successfully",
                assistant_id=assistant_data.get("id"),
                name=name,
                language=language
            )
            
            return assistant_data
            
        except httpx.HTTPError as e:
            self.logger.error("Failed to create assistant", error=str(e))
            raise
    
    async def start_call(
        self,
        assistant_id: str,
        phone_number: Optional[str] = None,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start a voice call with the assistant."""
        
        call_config = {
            "assistantId": assistant_id,
            "metadata": metadata or {}
        }
        
        if phone_number:
            call_config["customer"] = {"number": phone_number}
        
        if customer_id:
            call_config["customerId"] = customer_id
        
        try:
            response = await self.client.post("/call", json=call_config)
            response.raise_for_status()
            
            call_data = response.json()
            call_id = call_data.get("id")
            
            # Track active call
            self.active_calls[call_id] = {
                "assistant_id": assistant_id,
                "customer_id": customer_id,
                "metadata": metadata,
                "started_at": asyncio.get_event_loop().time()
            }
            
            self.logger.info(
                "Call started successfully",
                call_id=call_id,
                assistant_id=assistant_id,
                customer_id=customer_id
            )
            
            return call_data
            
        except httpx.HTTPError as e:
            self.logger.error("Failed to start call", error=str(e))
            raise
    
    async def handle_webhook(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle webhook events from Vapi.ai."""
        
        event_type = event_data.get("type")
        call_id = event_data.get("call", {}).get("id")
        
        self.logger.info(
            "Voice webhook received",
            event_type=event_type,
            call_id=call_id
        )
        
        try:
            if event_type == VoiceEventType.FUNCTION_CALL.value:
                return await self._handle_function_call(event_data)
            
            elif event_type == VoiceEventType.CALL_STARTED.value:
                return await self._handle_call_started(event_data)
            
            elif event_type == VoiceEventType.CALL_ENDED.value:
                return await self._handle_call_ended(event_data)
            
            elif event_type == VoiceEventType.TRANSCRIPT.value:
                return await self._handle_transcript(event_data)
            
            elif event_type == VoiceEventType.ERROR.value:
                return await self._handle_error(event_data)
            
            else:
                self.logger.warning("Unhandled webhook event type", event_type=event_type)
                return {"status": "ignored", "message": f"Unhandled event type: {event_type}"}
        
        except Exception as e:
            self.logger.error("Webhook handling error", error=str(e), event_type=event_type)
            return {"status": "error", "message": str(e)}
    
    async def _handle_function_call(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle function call from voice interface."""
        
        function_call = event_data.get("functionCall", {})
        function_name = function_call.get("name")
        function_args = function_call.get("parameters", {})
        
        if function_name not in self.functions:
            error_msg = f"Unknown function: {function_name}"
            self.logger.error(error_msg, function_name=function_name)
            return {"result": error_msg}
        
        try:
            function_def = self.functions[function_name]
            result = await function_def.handler(function_args)
            
            self.logger.info(
                "Function call executed successfully",
                function_name=function_name,
                args=function_args
            )
            
            return {"result": result}
            
        except Exception as e:
            error_msg = f"Function execution failed: {str(e)}"
            self.logger.error(error_msg, function_name=function_name, error=str(e))
            return {"result": error_msg}
    
    async def _handle_call_started(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call started event."""
        call_id = event_data.get("call", {}).get("id")
        
        if call_id in self.active_calls:
            self.active_calls[call_id]["status"] = "active"
        
        return {"status": "acknowledged"}
    
    async def _handle_call_ended(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call ended event."""
        call_id = event_data.get("call", {}).get("id")
        
        if call_id in self.active_calls:
            call_info = self.active_calls[call_id]
            duration = asyncio.get_event_loop().time() - call_info["started_at"]
            
            self.logger.info(
                "Call ended",
                call_id=call_id,
                duration_seconds=duration
            )
            
            del self.active_calls[call_id]
        
        return {"status": "acknowledged"}
    
    async def _handle_transcript(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transcript event."""
        transcript = event_data.get("transcript", {})
        text = transcript.get("text", "")
        
        self.logger.info(
            "Transcript received",
            text=text[:100] + "..." if len(text) > 100 else text
        )
        
        return {"status": "acknowledged"}
    
    async def _handle_error(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error event."""
        error = event_data.get("error", {})
        error_message = error.get("message", "Unknown error")
        
        self.logger.error("Voice interface error", error_message=error_message)
        
        return {"status": "acknowledged"}
    
    # Default function handlers
    
    async def _handle_job_search(self, args: Dict[str, Any]) -> str:
        """Handle job search function call."""
        query = args.get("query", "")
        location = args.get("location", "")
        experience_level = args.get("experience_level", "")
        
        self.logger.info(
            "Job search requested via voice",
            query=query,
            location=location,
            experience_level=experience_level
        )
        
        # In a real implementation, this would integrate with the job search system
        return f"I'm searching for {query} positions{f' in {location}' if location else ''}{f' at {experience_level} level' if experience_level else ''}. Let me find the best opportunities for you."
    
    async def _handle_job_application(self, args: Dict[str, Any]) -> str:
        """Handle job application function call."""
        job_id = args.get("job_id", "")
        cover_letter_style = args.get("cover_letter_style", "formal")
        custom_message = args.get("custom_message", "")
        
        self.logger.info(
            "Job application requested via voice",
            job_id=job_id,
            cover_letter_style=cover_letter_style
        )
        
        return f"I'm preparing your application for job {job_id} with a {cover_letter_style} cover letter. I'll review your profile and submit the application for your approval."
    
    async def _handle_interview_preparation(self, args: Dict[str, Any]) -> str:
        """Handle interview preparation function call."""
        job_role = args.get("job_role", "")
        company = args.get("company", "")
        interview_type = args.get("interview_type", "mixed")
        language = args.get("language", "en")
        
        self.logger.info(
            "Interview preparation requested via voice",
            job_role=job_role,
            company=company,
            interview_type=interview_type,
            language=language
        )
        
        return f"Let's prepare for your {job_role} interview{f' at {company}' if company else ''}. I'll conduct a {interview_type} interview simulation in {language}. Are you ready to begin?"
    
    async def _handle_profile_update(self, args: Dict[str, Any]) -> str:
        """Handle profile update function call."""
        field = args.get("field", "")
        value = args.get("value", "")
        action = args.get("action", "add")
        
        self.logger.info(
            "Profile update requested via voice",
            field=field,
            value=value,
            action=action
        )
        
        return f"I'll {action} '{value}' to your {field}. Let me update your profile and confirm the changes."
    
    async def get_call_status(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an active call."""
        return self.active_calls.get(call_id)
    
    async def end_call(self, call_id: str) -> bool:
        """End an active call."""
        try:
            response = await self.client.post(f"/call/{call_id}/end")
            response.raise_for_status()
            
            if call_id in self.active_calls:
                del self.active_calls[call_id]
            
            self.logger.info("Call ended successfully", call_id=call_id)
            return True
            
        except httpx.HTTPError as e:
            self.logger.error("Failed to end call", call_id=call_id, error=str(e))
            return False
    
    async def close(self):
        """Close the voice orchestrator and cleanup resources."""
        await self.client.aclose()
        self.logger.info("Voice orchestrator closed")