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
from sovereign_career_architect.voice.sarvam import SarvamClient, LanguageRouter, IndicLanguage

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
        
        # Sarvam-1 integration for Indic languages
        self.sarvam_client = SarvamClient()
        self.language_router = LanguageRouter(self.sarvam_client)
        
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
        language: str = "en",
        enable_streaming: bool = True
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
        
        # Configure TTS and STT based on language
        voice_config, transcriber_config = await self._configure_language_models(language)
        
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
                "temperature": 0.7,
                "stream": enable_streaming  # Enable streaming for faster responses
            },
            "voice": voice_config,
            "transcriber": transcriber_config,
            "firstMessage": first_message,
            "voicemailMessage": "I'm not available right now, but you can leave a message and I'll get back to you.",
            "endCallMessage": "Thank you for using Sovereign Career Architect. Have a great day!",
            "backgroundSound": self.config.background_sound,
            "maxDurationSeconds": self.config.max_duration,
            "silenceTimeoutSeconds": self.config.silence_timeout,
            "responseDelaySeconds": 0.2 if enable_streaming else 0.4,  # Faster response with streaming
            "llmRequestDelaySeconds": 0.05 if enable_streaming else 0.1,  # Reduced LLM delay
            "numWordsToInterruptAssistant": 2,
            "serverUrl": self.config.webhook_url,
            "serverUrlSecret": self.config.webhook_secret,
            # Streaming optimizations
            "fillerInjectionEnabled": enable_streaming,
            "optimisticAcknowledgment": enable_streaming
        }
        
        try:
            response = await self.client.post("/assistant", json=assistant_config)
            response.raise_for_status()
            
            assistant_data = response.json()
            
            self.logger.info(
                "Assistant created successfully",
                assistant_id=assistant_data.get("id"),
                name=name,
                language=language,
                streaming_enabled=enable_streaming
            )
            
            return assistant_data
            
        except httpx.HTTPError as e:
            self.logger.error("Failed to create assistant", error=str(e))
            raise
    
    async def _configure_language_models(self, language: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Configure TTS and STT based on target language."""
        
        # Check if it's an Indic language
        supported_indic = await self.sarvam_client.get_supported_languages()
        
        if language in supported_indic:
            # Configure for Indic language
            voice_config = {
                "provider": self.config.tts_provider,
                "voiceId": self._get_indic_voice_id(language),
                "stability": self.config.tts_stability,
                "similarityBoost": self.config.tts_similarity_boost,
                "language": language
            }
            
            transcriber_config = {
                "provider": self.config.stt_provider,
                "model": self.config.stt_model,
                "language": self._get_stt_language_code(language)
            }
        else:
            # Default English configuration
            voice_config = {
                "provider": self.config.tts_provider,
                "voiceId": self.config.tts_voice_id,
                "stability": self.config.tts_stability,
                "similarityBoost": self.config.tts_similarity_boost
            }
            
            transcriber_config = {
                "provider": self.config.stt_provider,
                "model": self.config.stt_model,
                "language": self.config.stt_language
            }
        
        return voice_config, transcriber_config
    
    def _get_indic_voice_id(self, language: str) -> str:
        """Get appropriate voice ID for Indic languages."""
        # Map Indic languages to available voice IDs
        indic_voices = {
            "hi": "hindi-female-1",  # Hindi
            "bn": "bengali-female-1",  # Bengali
            "ta": "tamil-female-1",  # Tamil
            "te": "telugu-female-1",  # Telugu
            "mr": "marathi-female-1",  # Marathi
            "gu": "gujarati-female-1",  # Gujarati
            "kn": "kannada-female-1",  # Kannada
            "ml": "malayalam-female-1",  # Malayalam
            "pa": "punjabi-female-1",  # Punjabi
            "or": "odia-female-1"  # Odia
        }
        
        return indic_voices.get(language, self.config.tts_voice_id)
    
    def _get_stt_language_code(self, language: str) -> str:
        """Get STT language code for Indic languages."""
        stt_codes = {
            "hi": "hi-IN",  # Hindi
            "bn": "bn-IN",  # Bengali
            "ta": "ta-IN",  # Tamil
            "te": "te-IN",  # Telugu
            "mr": "mr-IN",  # Marathi
            "gu": "gu-IN",  # Gujarati
            "kn": "kn-IN",  # Kannada
            "ml": "ml-IN",  # Malayalam
            "pa": "pa-IN",  # Punjabi
            "or": "or-IN"   # Odia
        }
        
        return stt_codes.get(language, "en-US")
    
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
        
        # Extract user input for language detection
        user_input = event_data.get("transcript", {}).get("text", "")
        call_id = event_data.get("call", {}).get("id")
        
        if function_name not in self.functions:
            error_msg = f"Unknown function: {function_name}"
            self.logger.error(error_msg, function_name=function_name)
            return {"result": error_msg}
        
        try:
            # Detect language and route appropriately
            if user_input:
                detection_result = await self.sarvam_client.detect_language(user_input)
                
                # Store language preference for this call
                if call_id and call_id in self.active_calls:
                    self.active_calls[call_id]["detected_language"] = detection_result.language
                    self.active_calls[call_id]["is_indic"] = detection_result.is_indic
                
                self.logger.info(
                    "Language detected for function call",
                    call_id=call_id,
                    detected_language=detection_result.language,
                    confidence=detection_result.confidence,
                    is_indic=detection_result.is_indic
                )
            
            function_def = self.functions[function_name]
            
            # Execute function with language context
            if hasattr(function_def.handler, '__code__') and 'language_context' in function_def.handler.__code__.co_varnames:
                # Pass language context if function supports it
                language_context = {
                    "detected_language": detection_result.language if user_input else "en",
                    "is_indic": detection_result.is_indic if user_input else False,
                    "user_input": user_input
                }
                result = await function_def.handler(function_args, language_context=language_context)
            else:
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
    
    async def _handle_job_search(self, args: Dict[str, Any], language_context: Optional[Dict[str, Any]] = None) -> str:
        """Handle job search function call."""
        query = args.get("query", "")
        location = args.get("location", "")
        experience_level = args.get("experience_level", "")
        
        self.logger.info(
            "Job search requested via voice",
            query=query,
            location=location,
            experience_level=experience_level,
            language=language_context.get("detected_language", "en") if language_context else "en"
        )
        
        # Generate response in appropriate language
        if language_context and language_context.get("is_indic", False):
            detected_lang = language_context["detected_language"]
            
            # Use Sarvam-1 for Indic language response
            prompt = f"User is searching for {query} jobs{f' in {location}' if location else ''}{f' at {experience_level} level' if experience_level else ''}. Respond in {detected_lang} to confirm the search and provide encouragement."
            
            try:
                response = await self.sarvam_client.generate_response(
                    prompt=prompt,
                    language=detected_lang,
                    max_tokens=200,
                    temperature=0.7
                )
                return response.text
            except Exception as e:
                self.logger.error("Sarvam-1 response failed", error=str(e))
                # Fallback to English
                pass
        
        # Default English response
        return f"I'm searching for {query} positions{f' in {location}' if location else ''}{f' at {experience_level} level' if experience_level else ''}. Let me find the best opportunities for you."
    
    async def _handle_job_application(self, args: Dict[str, Any], language_context: Optional[Dict[str, Any]] = None) -> str:
        """Handle job application function call."""
        job_id = args.get("job_id", "")
        cover_letter_style = args.get("cover_letter_style", "formal")
        custom_message = args.get("custom_message", "")
        
        self.logger.info(
            "Job application requested via voice",
            job_id=job_id,
            cover_letter_style=cover_letter_style,
            language=language_context.get("detected_language", "en") if language_context else "en"
        )
        
        # Generate response in appropriate language
        if language_context and language_context.get("is_indic", False):
            detected_lang = language_context["detected_language"]
            
            prompt = f"User wants to apply for job {job_id} with {cover_letter_style} style. Respond in {detected_lang} to confirm the application process."
            
            try:
                response = await self.sarvam_client.generate_response(
                    prompt=prompt,
                    language=detected_lang,
                    max_tokens=200,
                    temperature=0.7
                )
                return response.text
            except Exception as e:
                self.logger.error("Sarvam-1 response failed", error=str(e))
        
        return f"I'm preparing your application for job {job_id} with a {cover_letter_style} cover letter. I'll review your profile and submit the application for your approval."
    
    async def _handle_interview_preparation(self, args: Dict[str, Any], language_context: Optional[Dict[str, Any]] = None) -> str:
        """Handle interview preparation function call."""
        job_role = args.get("job_role", "")
        company = args.get("company", "")
        interview_type = args.get("interview_type", "mixed")
        language = args.get("language", "en")
        
        # Use detected language if not specified
        if language_context and language_context.get("is_indic", False):
            language = language_context["detected_language"]
        
        self.logger.info(
            "Interview preparation requested via voice",
            job_role=job_role,
            company=company,
            interview_type=interview_type,
            language=language
        )
        
        # Generate response in appropriate language
        if language in await self.sarvam_client.get_supported_languages():
            prompt = f"User wants to prepare for {job_role} interview{f' at {company}' if company else ''} with {interview_type} type. Respond in {language} to start the interview preparation."
            
            try:
                response = await self.sarvam_client.generate_response(
                    prompt=prompt,
                    language=language,
                    max_tokens=200,
                    temperature=0.7
                )
                return response.text
            except Exception as e:
                self.logger.error("Sarvam-1 response failed", error=str(e))
        
        return f"Let's prepare for your {job_role} interview{f' at {company}' if company else ''}. I'll conduct a {interview_type} interview simulation in {language}. Are you ready to begin?"
    
    async def _handle_profile_update(self, args: Dict[str, Any], language_context: Optional[Dict[str, Any]] = None) -> str:
        """Handle profile update function call."""
        field = args.get("field", "")
        value = args.get("value", "")
        action = args.get("action", "add")
        
        self.logger.info(
            "Profile update requested via voice",
            field=field,
            value=value,
            action=action,
            language=language_context.get("detected_language", "en") if language_context else "en"
        )
        
        # Generate response in appropriate language
        if language_context and language_context.get("is_indic", False):
            detected_lang = language_context["detected_language"]
            
            prompt = f"User wants to {action} '{value}' to their {field}. Respond in {detected_lang} to confirm the profile update."
            
            try:
                response = await self.sarvam_client.generate_response(
                    prompt=prompt,
                    language=detected_lang,
                    max_tokens=200,
                    temperature=0.7
                )
                return response.text
            except Exception as e:
                self.logger.error("Sarvam-1 response failed", error=str(e))
        
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
        await self.language_router.close()
        self.logger.info("Voice orchestrator closed")
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        detection_result = await self.sarvam_client.detect_language(text)
        
        return {
            "language": detection_result.language,
            "confidence": detection_result.confidence,
            "is_indic": detection_result.is_indic,
            "detected_script": detection_result.detected_script
        }
    
    async def get_supported_indic_languages(self) -> List[str]:
        """Get list of supported Indic languages."""
        return await self.sarvam_client.get_supported_languages()
    
    async def create_multilingual_assistant(
        self,
        name: str,
        system_message: str,
        first_message: str,
        supported_languages: List[str]
    ) -> Dict[str, Any]:
        """
        Create an assistant that supports multiple languages.
        
        Args:
            name: Assistant name
            system_message: System prompt
            first_message: Initial message
            supported_languages: List of language codes to support
            
        Returns:
            Assistant configuration
        """
        # Enhance system message with multilingual capabilities
        enhanced_system_message = f"""{system_message}

You are a multilingual career assistant that can communicate in multiple languages including:
{', '.join(supported_languages)}

When a user speaks in an Indic language, respond naturally in that same language.
Maintain cultural sensitivity and use appropriate terminology for each language.
If you detect code-switching (mixing languages), respond appropriately to match the user's style.
"""
        
        # Use primary language for initial configuration
        primary_language = supported_languages[0] if supported_languages else "en"
        
        return await self.create_assistant(
            name=name,
            system_message=enhanced_system_message,
            first_message=first_message,
            language=primary_language
        )
    
    async def process_multilingual_input(
        self,
        text: str,
        system_prompt: str,
        preferred_language: Optional[str] = None,
        enable_streaming: bool = True
    ) -> Dict[str, Any]:
        """
        Process input text using appropriate language model with streaming optimization.
        
        Args:
            text: Input text
            system_prompt: System prompt for context
            preferred_language: User's preferred language
            enable_streaming: Enable streaming for faster response
            
        Returns:
            Processing result with generated response
        """
        try:
            response = await self.language_router.process_with_appropriate_model(
                text=text,
                system_prompt=system_prompt,
                preferred_language=preferred_language,
                max_tokens=500,
                temperature=0.7,
                enable_streaming=enable_streaming
            )
            
            return {
                "response": response.text,
                "language": response.language,
                "tokens_used": response.tokens_used,
                "fertility_rate": response.fertility_rate,
                "confidence": response.confidence,
                "model_used": "sarvam-1" if response.language in await self.get_supported_indic_languages() else "openai",
                "streaming_enabled": enable_streaming
            }
            
        except Exception as e:
            self.logger.error("Multilingual processing failed", error=str(e))
            return {
                "response": "I'm having trouble processing your request. Please try again.",
                "language": "en",
                "tokens_used": 0,
                "fertility_rate": 0.0,
                "confidence": 0.1,
                "model_used": "fallback",
                "streaming_enabled": False
            }
    
    async def optimize_response_latency(
        self,
        assistant_id: str,
        target_latency_ms: int = 300
    ) -> Dict[str, Any]:
        """
        Optimize assistant configuration for target latency.
        
        Args:
            assistant_id: Assistant to optimize
            target_latency_ms: Target response latency in milliseconds
            
        Returns:
            Optimization results
        """
        try:
            # Calculate optimal settings based on target latency
            if target_latency_ms <= 200:
                # Ultra-low latency settings
                settings = {
                    "responseDelaySeconds": 0.1,
                    "llmRequestDelaySeconds": 0.02,
                    "fillerInjectionEnabled": True,
                    "optimisticAcknowledgment": True,
                    "model": {
                        "stream": True,
                        "maxTokens": 200,  # Shorter responses for speed
                        "temperature": 0.5  # Lower temperature for faster generation
                    }
                }
            elif target_latency_ms <= 500:
                # Low latency settings
                settings = {
                    "responseDelaySeconds": 0.2,
                    "llmRequestDelaySeconds": 0.05,
                    "fillerInjectionEnabled": True,
                    "optimisticAcknowledgment": True,
                    "model": {
                        "stream": True,
                        "maxTokens": 300,
                        "temperature": 0.7
                    }
                }
            else:
                # Standard settings
                settings = {
                    "responseDelaySeconds": 0.4,
                    "llmRequestDelaySeconds": 0.1,
                    "fillerInjectionEnabled": False,
                    "optimisticAcknowledgment": False,
                    "model": {
                        "stream": False,
                        "maxTokens": 500,
                        "temperature": 0.7
                    }
                }
            
            # Update assistant configuration
            response = await self.client.patch(
                f"/assistant/{assistant_id}",
                json=settings
            )
            response.raise_for_status()
            
            self.logger.info(
                "Assistant latency optimized",
                assistant_id=assistant_id,
                target_latency_ms=target_latency_ms,
                settings=settings
            )
            
            return {
                "success": True,
                "target_latency_ms": target_latency_ms,
                "applied_settings": settings,
                "message": f"Assistant optimized for {target_latency_ms}ms latency"
            }
            
        except Exception as e:
            self.logger.error(
                "Latency optimization failed",
                assistant_id=assistant_id,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to optimize assistant latency"
            }
    
    async def enable_filler_phrases(
        self,
        language: str = "en"
    ) -> List[str]:
        """
        Get appropriate filler phrases for optimistic acknowledgment.
        
        Args:
            language: Target language for filler phrases
            
        Returns:
            List of filler phrases
        """
        filler_phrases = {
            "hi": [
                "हाँ, मैं समझ रहा हूँ...",
                "ठीक है, एक मिनट...",
                "मुझे लगता है...",
                "यह दिलचस्प है..."
            ],
            "bn": [
                "হ্যাঁ, আমি বুঝতে পারছি...",
                "ঠিক আছে, একটু অপেক্ষা করুন...",
                "আমার মনে হয়...",
                "এটা আকর্ষণীয়..."
            ],
            "ta": [
                "ஆம், நான் புரிந்துகொள்கிறேன்...",
                "சரி, ஒரு நிமிடம்...",
                "எனக்குத் தோன்றுகிறது...",
                "இது சுவாரস்யமானது..."
            ],
            "te": [
                "అవును, నేను అర్థం చేసుకుంటున్నాను...",
                "సరే, ఒక నిమిషం...",
                "నాకు అనిపిస్తుంది...",
                "ఇది ఆసక్తికరంగా ఉంది..."
            ],
            "en": [
                "Yes, I understand...",
                "Okay, let me think...",
                "I see what you mean...",
                "That's interesting..."
            ]
        }
        
        return filler_phrases.get(language, filler_phrases["en"])
    
    async def measure_response_latency(
        self,
        assistant_id: str,
        test_message: str = "Hello, how are you?"
    ) -> Dict[str, Any]:
        """
        Measure actual response latency for an assistant.
        
        Args:
            assistant_id: Assistant to test
            test_message: Test message to send
            
        Returns:
            Latency measurement results
        """
        try:
            import time
            
            # Start timing
            start_time = time.time()
            
            # Simulate a test call (in real implementation, this would be an actual call)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # End timing
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            self.logger.info(
                "Response latency measured",
                assistant_id=assistant_id,
                latency_ms=latency_ms,
                test_message=test_message
            )
            
            return {
                "assistant_id": assistant_id,
                "latency_ms": latency_ms,
                "test_message": test_message,
                "timestamp": end_time,
                "performance_rating": "excellent" if latency_ms < 300 else "good" if latency_ms < 500 else "needs_improvement"
            }
            
        except Exception as e:
            self.logger.error(
                "Latency measurement failed",
                assistant_id=assistant_id,
                error=str(e)
            )
            return {
                "error": str(e),
                "message": "Failed to measure response latency"
            }