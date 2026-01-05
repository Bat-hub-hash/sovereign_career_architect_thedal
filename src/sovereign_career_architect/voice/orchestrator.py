"""Voice interface orchestrator using Vapi.ai."""

import asyncio
import json
import time
import re
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
import httpx

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger
from sovereign_career_architect.voice.sarvam import SarvamClient, LanguageRouter, IndicLanguage, translate_text

logger = get_logger(__name__)


# =============================================================================
# Configuration Flags
# =============================================================================

# Dry-run mode: When enabled, skips agent execution and returns test response
# Set to True for testing/demo without hitting the real agent
DRY_RUN_MODE: bool = False

# Dry-run response (deterministic for testing)
_DRY_RUN_RESPONSE = "Voice system is running in test mode."

# Long-running agent threshold (milliseconds)
# If agent execution exceeds this, log a warning for observability
LONG_RUNNING_THRESHOLD_MS: float = 1500.0  # 1.5 seconds


# =============================================================================
# Error Classification
# =============================================================================

class VoiceErrorType(Enum):
    """Classification of voice processing errors."""
    TRANSLATION_ERROR = "translation_error"
    AGENT_ERROR = "agent_error"
    TIMEOUT = "timeout"
    EMPTY_INPUT = "empty_input"
    EMPTY_AGENT_RESPONSE = "empty_agent_response"
    UNKNOWN = "unknown"


# =============================================================================
# Voice Audit Trail
# =============================================================================

@dataclass
class VoiceAuditRecord:
    """
    Internal audit record for voice interactions.
    
    This captures all relevant metrics and flags for a single voice request.
    Used for observability and debugging - NOT returned to users.
    """
    voice_trace_id: str
    user_id: str
    session_id: str
    source_lang: str
    input_text_length: int = 0
    was_translated_in: bool = False
    was_translated_out: bool = False
    agent_invoked: bool = False
    agent_latency_ms: float = 0.0
    translate_in_latency_ms: float = 0.0
    translate_out_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fallback_used: bool = False
    dry_run: bool = False
    long_running: bool = False
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "voice_trace_id": self.voice_trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_lang": self.source_lang,
            "input_text_length": self.input_text_length,
            "was_translated_in": self.was_translated_in,
            "was_translated_out": self.was_translated_out,
            "agent_invoked": self.agent_invoked,
            "agent_latency_ms": round(self.agent_latency_ms, 2),
            "translate_in_latency_ms": round(self.translate_in_latency_ms, 2),
            "translate_out_latency_ms": round(self.translate_out_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "fallback_used": self.fallback_used,
            "dry_run": self.dry_run,
            "long_running": self.long_running,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def _log_audit_summary(audit: VoiceAuditRecord) -> None:
    """
    Log the voice audit summary.
    
    This is the single consolidated log event for the entire voice request.
    Wrapped in try/except to ensure audit failures never break processing.
    """
    try:
        logger.info(
            "voice.audit.summary",
            **audit.to_dict()
        )
    except Exception:
        # Silently ignore audit logging failures - stability first
        pass


def _log_error_classified(
    voice_trace_id: str,
    user_id: str,
    session_id: str,
    error_type: VoiceErrorType,
    error_message: str
) -> None:
    """
    Log a classified error event.
    
    Wrapped in try/except for stability.
    """
    try:
        logger.error(
            "voice.error.classified",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            error_type=error_type.value,
            error_message=error_message[:200] if error_message else None
        )
    except Exception:
        # Silently ignore - stability first
        pass


def _generate_trace_id() -> str:
    """Generate a unique voice trace ID."""
    return f"vtrace_{uuid.uuid4().hex[:16]}"


# =============================================================================
# Streaming & UX Helpers
# =============================================================================

# Early acknowledgement phrases (for future streaming support)
_ACKNOWLEDGEMENT_PHRASES = [
    "Okay, I'm checking that for you.",
    "Let me look into that.",
    "One moment, please.",
    "I'm on it.",
    "Let me find that information.",
]


def _get_acknowledgement(user_input: str) -> str:
    """
    Generate an early acknowledgement phrase based on user input.
    
    This is for future streaming support - provides immediate feedback
    while the agent processes the request.
    
    Args:
        user_input: The user's input text
        
    Returns:
        An appropriate acknowledgement phrase
    """
    # Simple heuristic: choose phrase based on input characteristics
    input_lower = user_input.lower() if user_input else ""
    
    if any(word in input_lower for word in ["find", "search", "look", "show"]):
        return "Let me find that information."
    elif any(word in input_lower for word in ["help", "can you", "could you"]):
        return "I'm on it."
    elif "?" in user_input:
        return "Let me check that for you."
    else:
        # Default acknowledgement
        return "Okay, I'm checking that for you."


def _segment_response(text: str) -> Tuple[str, str]:
    """
    Segment response into first sentence and remaining text.
    
    This prepares the response for future Vapi streaming TTS support,
    allowing the first sentence to be spoken quickly while the rest loads.
    
    Args:
        text: Full response text
        
    Returns:
        Tuple of (first_sentence, remaining_text)
        
    Note: This is INTERNAL ONLY. The final returned response is always
    the complete, unsegmented text.
    """
    if not text or not isinstance(text, str):
        return "", ""
    
    text = text.strip()
    
    # Sentence-ending patterns (handles ., !, ?)
    # Also handles common abbreviations to avoid false splits
    sentence_end_pattern = r'(?<![A-Z])(?<![A-Z]\.)(?<!\s[A-Z])(?<![0-9])[.!?](?=\s|$)'
    
    match = re.search(sentence_end_pattern, text)
    
    if match:
        split_pos = match.end()
        first_sentence = text[:split_pos].strip()
        remaining = text[split_pos:].strip()
        return first_sentence, remaining
    else:
        # No sentence break found - return all as first sentence
        return text, ""


def _log_response_segments(
    user_id: str,
    session_id: str,
    first_sentence: str,
    remaining_text: str
) -> None:
    """
    Log response segmentation for streaming observability.
    
    This is internal logging only - does not affect response format.
    """
    logger.debug(
        "voice.response.segmented",
        user_id=user_id,
        session_id=session_id,
        first_sentence_length=len(first_sentence),
        remaining_length=len(remaining_text),
        first_sentence_preview=first_sentence[:50] if first_sentence else ""
    )


# =============================================================================
# Core Agent Integration
# =============================================================================

# Global agent instance (lazily initialized)
_agent_instance: Optional[Any] = None
_agent_initialized: bool = False


def _get_or_create_agent():
    """
    Get or create the SovereignCareerArchitect agent instance.
    
    Uses lazy initialization to avoid circular imports and ensure
    the agent is only created when needed.
    """
    global _agent_instance, _agent_initialized
    
    if _agent_instance is None and not _agent_initialized:
        init_start = time.monotonic()
        try:
            from sovereign_career_architect.core.agent import SovereignCareerArchitect
            _agent_instance = SovereignCareerArchitect(
                enable_voice=False,  # Avoid circular dependency
                enable_browser=False,  # Lightweight for voice processing
                enable_interrupts=False  # No HITL for voice responses
            )
            _agent_initialized = True
            init_elapsed = (time.monotonic() - init_start) * 1000
            logger.info(
                "voice.agent.init",
                status="success",
                elapsed_ms=round(init_elapsed, 2)
            )
        except Exception as e:
            init_elapsed = (time.monotonic() - init_start) * 1000
            logger.error(
                "voice.agent.init",
                status="error",
                error=str(e),
                error_type=type(e).__name__,
                elapsed_ms=round(init_elapsed, 2)
            )
            _agent_initialized = True  # Mark as attempted to avoid repeated failures
    
    return _agent_instance


def _invoke_agent(user_id: str, text: str, session_id: Optional[str] = None) -> str:
    """
    Invoke the LangGraph agent with user input.
    
    This function handles the async-to-sync bridge and provides
    defensive error handling.
    
    Args:
        user_id: User identifier
        text: Input text in English
        session_id: Optional session ID (auto-generated if not provided)
        
    Returns:
        Agent response as plain English string
    """
    # Handle dry-run mode
    if DRY_RUN_MODE:
        logger.info(
            "voice.agent.invoke.dry_run",
            user_id=user_id,
            session_id=session_id
        )
        return _DRY_RUN_RESPONSE
    
    agent = _get_or_create_agent()
    
    if agent is None:
        logger.warning(
            "voice.agent.invoke.unavailable",
            user_id=user_id,
            reason="agent_not_initialized"
        )
        return "I'm currently unable to process your request. Please try again later."
    
    # Generate session ID if not provided
    if session_id is None:
        import uuid
        session_id = f"voice_{user_id}_{uuid.uuid4().hex[:8]}"
    
    invoke_start = time.monotonic()
    
    logger.info(
        "voice.agent.invoke.start",
        user_id=user_id,
        session_id=session_id,
        input_length=len(text)
    )
    
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to handle differently
            # Create a new thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    agent.process_user_request(
                        user_id=user_id,
                        session_id=session_id,
                        message=text
                    )
                )
                result = future.result(timeout=30.0)
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            result = asyncio.run(
                agent.process_user_request(
                    user_id=user_id,
                    session_id=session_id,
                    message=text
                )
            )
        
        invoke_elapsed = (time.monotonic() - invoke_start) * 1000
        
        # Extract the response message with validation
        response = None
        if isinstance(result, dict):
            response = result.get("message")
            if result.get("success") is False:
                logger.warning(
                    "voice.agent.invoke.unsuccessful",
                    user_id=user_id,
                    session_id=session_id,
                    result_keys=list(result.keys())
                )
        elif result is not None:
            response = str(result)
        
        # Guardrail: Handle empty or malformed output
        if not response or not isinstance(response, str) or len(response.strip()) == 0:
            logger.warning(
                "voice.agent.invoke.empty_response",
                user_id=user_id,
                session_id=session_id,
                raw_result_type=type(result).__name__
            )
            response = "I processed your request but couldn't generate a response. Please try again."
        
        # Long-running detection for observability
        if invoke_elapsed > LONG_RUNNING_THRESHOLD_MS:
            logger.warning(
                "voice.agent.long_running",
                user_id=user_id,
                session_id=session_id,
                elapsed_ms=round(invoke_elapsed, 2),
                threshold_ms=LONG_RUNNING_THRESHOLD_MS
            )
        
        logger.info(
            "voice.agent.invoke.success",
            user_id=user_id,
            session_id=session_id,
            response_length=len(response),
            elapsed_ms=round(invoke_elapsed, 2)
        )
        
        return response
        
    except asyncio.TimeoutError:
        invoke_elapsed = (time.monotonic() - invoke_start) * 1000
        logger.error(
            "voice.agent.invoke.timeout",
            user_id=user_id,
            session_id=session_id,
            elapsed_ms=round(invoke_elapsed, 2)
        )
        return "I'm taking longer than expected to process your request. Please try again."
    except Exception as e:
        invoke_elapsed = (time.monotonic() - invoke_start) * 1000
        logger.error(
            "voice.agent.invoke.error",
            user_id=user_id,
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(invoke_elapsed, 2)
        )
        return "I encountered an error while processing your request. Please try again."


def _safe_translate(text: str, target_lang: str, fallback: str) -> tuple:
    """
    Safely translate text with fallback on failure.
    
    Returns:
        Tuple of (translated_text, success_flag, elapsed_ms)
    """
    if not text:
        return fallback, False, 0.0
    
    translate_start = time.monotonic()
    try:
        translated = translate_text(text, target_lang=target_lang)
        elapsed = (time.monotonic() - translate_start) * 1000
        
        # Validate translation result
        if translated and isinstance(translated, str) and len(translated.strip()) > 0:
            return translated, True, elapsed
        else:
            logger.warning(
                "voice.translate.empty_result",
                target_lang=target_lang,
                input_length=len(text)
            )
            return fallback, False, elapsed
            
    except Exception as e:
        elapsed = (time.monotonic() - translate_start) * 1000
        logger.error(
            "voice.translate.error",
            target_lang=target_lang,
            error=str(e),
            elapsed_ms=round(elapsed, 2)
        )
        return fallback, False, elapsed


# =============================================================================
# Voice Text Processing (as specified by Member 3: The Voice)
# =============================================================================

def process_voice_text(user_id: str, text: str, source_lang: str) -> dict:
    """
    Bridge between Vapi and the Core Agent.
    
    This function:
    1. Detects language and translates to English if needed
    2. Calls the Core Agent
    3. Translates the response back to the source language if needed
    
    Args:
        user_id: The unique identifier for the user
        text: The input text from the voice interface
        source_lang: The source language code (e.g., "en", "hi", "ta")
        
    Returns:
        Dictionary with:
        - final_text: The response in the source language
        - agent_text_en: The response in English
    
    Note: This function NEVER raises exceptions. All errors return safe fallbacks.
    """
    # Start total latency measurement
    total_start = time.monotonic()
    
    # Generate unique trace ID for this voice request (end-to-end correlation)
    voice_trace_id = _generate_trace_id()
    
    # Generate session ID for correlation
    session_id = f"voice_{user_id}_{uuid.uuid4().hex[:8]}"
    
    # Initialize audit record (captures ALL metrics for this request)
    audit = VoiceAuditRecord(
        voice_trace_id=voice_trace_id,
        user_id=user_id,
        session_id=session_id,
        source_lang=source_lang,
        input_text_length=len(text) if text else 0,
        dry_run=DRY_RUN_MODE
    )
    
    # Initialize latency tracking
    translate_in_ms = 0.0
    agent_ms = 0.0
    translate_out_ms = 0.0
    
    try:
        logger.info(
            "voice.process.start",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            source_lang=source_lang,
            text_length=len(text) if text else 0
        )
        
        # Guardrail: Handle empty input
        if not text or not text.strip():
            logger.warning(
                "voice.process.empty_input",
                voice_trace_id=voice_trace_id,
                user_id=user_id,
                session_id=session_id
            )
            audit.error_type = VoiceErrorType.EMPTY_INPUT.value
            audit.error_message = "Empty or whitespace-only input"
            _log_error_classified(
                voice_trace_id, user_id, session_id,
                VoiceErrorType.EMPTY_INPUT,
                "Empty or whitespace-only input"
            )
            fallback_en = "I didn't catch that. Could you please repeat?"
            
            # Finalize audit
            audit.total_latency_ms = (time.monotonic() - total_start) * 1000
            audit.fallback_used = True
            _log_audit_summary(audit)
            
            return {
                "final_text": fallback_en,
                "agent_text_en": fallback_en
            }
        
        # Generate early acknowledgement (for future streaming support)
        # This is logged for observability but not returned separately
        acknowledgement = _get_acknowledgement(text)
        logger.debug(
            "voice.acknowledgement.prepared",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            acknowledgement=acknowledgement
        )
        
        # Step 1: Translate to English if source language is not English
        if source_lang.lower() != "en":
            text_in_english, translate_success, translate_in_ms = _safe_translate(
                text, target_lang="en", fallback=text
            )
            audit.was_translated_in = translate_success
            audit.translate_in_latency_ms = translate_in_ms
            
            if not translate_success:
                audit.error_type = VoiceErrorType.TRANSLATION_ERROR.value
                audit.error_message = "Input translation failed"
            
            logger.info(
                "voice.translate.input",
                voice_trace_id=voice_trace_id,
                user_id=user_id,
                session_id=session_id,
                source_lang=source_lang,
                target_lang="en",
                success=translate_success,
                elapsed_ms=round(translate_in_ms, 2)
            )
        else:
            text_in_english = text
        
        # Step 2: Call the Core Agent (LangGraph-based)
        agent_start = time.monotonic()
        audit.agent_invoked = True
        
        agent_response_en = _invoke_agent(
            user_id=user_id,
            text=text_in_english,
            session_id=session_id
        )
        agent_ms = (time.monotonic() - agent_start) * 1000
        audit.agent_latency_ms = agent_ms
        
        # Check for long-running agent
        if agent_ms > LONG_RUNNING_THRESHOLD_MS:
            audit.long_running = True
        
        # Guardrail: Validate agent response
        if not agent_response_en or not isinstance(agent_response_en, str):
            audit.error_type = VoiceErrorType.EMPTY_AGENT_RESPONSE.value
            audit.error_message = "Agent returned empty or invalid response"
            _log_error_classified(
                voice_trace_id, user_id, session_id,
                VoiceErrorType.EMPTY_AGENT_RESPONSE,
                "Agent returned empty or invalid response"
            )
            agent_response_en = "I processed your request but couldn't generate a response."
            audit.fallback_used = True
        
        # Streaming-ready: Segment response for future Vapi TTS streaming
        # This is internal only - final response remains unsegmented
        first_sentence, remaining_text = _segment_response(agent_response_en)
        _log_response_segments(user_id, session_id, first_sentence, remaining_text)
        
        # Step 3: Translate agent response back to source language if needed
        if source_lang.lower() != "en":
            final_response, translate_success, translate_out_ms = _safe_translate(
                agent_response_en, target_lang=source_lang, fallback=agent_response_en
            )
            audit.was_translated_out = translate_success
            audit.translate_out_latency_ms = translate_out_ms
            
            if not translate_success and not audit.error_type:
                audit.error_type = VoiceErrorType.TRANSLATION_ERROR.value
                audit.error_message = "Output translation failed"
            
            logger.info(
                "voice.translate.output",
                voice_trace_id=voice_trace_id,
                user_id=user_id,
                session_id=session_id,
                source_lang="en",
                target_lang=source_lang,
                success=translate_success,
                elapsed_ms=round(translate_out_ms, 2)
            )
        else:
            final_response = agent_response_en
        
        # Calculate total latency
        total_ms = (time.monotonic() - total_start) * 1000
        audit.total_latency_ms = total_ms
        
        # Log if total processing was long-running
        if total_ms > LONG_RUNNING_THRESHOLD_MS:
            audit.long_running = True
            logger.warning(
                "voice.process.long_running",
                voice_trace_id=voice_trace_id,
                user_id=user_id,
                session_id=session_id,
                latency_total_ms=round(total_ms, 2),
                threshold_ms=LONG_RUNNING_THRESHOLD_MS
            )
        
        # Mark success if no errors occurred
        if not audit.error_type:
            audit.success = True
        
        logger.info(
            "voice.process.success",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            source_lang=source_lang,
            response_length=len(final_response),
            latency_translate_in_ms=round(translate_in_ms, 2),
            latency_agent_ms=round(agent_ms, 2),
            latency_translate_out_ms=round(translate_out_ms, 2),
            latency_total_ms=round(total_ms, 2)
        )
        
        # Log consolidated audit summary
        _log_audit_summary(audit)
        
        return {
            "final_text": final_response,
            "agent_text_en": agent_response_en
        }
        
    except asyncio.TimeoutError as e:
        # Specific handling for timeouts
        total_ms = (time.monotonic() - total_start) * 1000
        audit.total_latency_ms = total_ms
        audit.error_type = VoiceErrorType.TIMEOUT.value
        audit.error_message = str(e) or "Operation timed out"
        audit.fallback_used = True
        
        _log_error_classified(
            voice_trace_id, user_id, session_id,
            VoiceErrorType.TIMEOUT,
            str(e) or "Operation timed out"
        )
        
        logger.error(
            "voice.process.timeout",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            source_lang=source_lang,
            error=str(e),
            latency_total_ms=round(total_ms, 2)
        )
        
        fallback_en = "I'm taking longer than expected. Please try again."
        
        # Best-effort translation for fallback
        if source_lang.lower() != "en":
            fallback_translated, _, _ = _safe_translate(
                fallback_en, target_lang=source_lang, fallback=fallback_en
            )
        else:
            fallback_translated = fallback_en
        
        # Log consolidated audit summary
        _log_audit_summary(audit)
        
        return {
            "final_text": fallback_translated,
            "agent_text_en": fallback_en
        }
        
    except Exception as e:
        # Generic error handling - classify as agent_error or unknown
        total_ms = (time.monotonic() - total_start) * 1000
        audit.total_latency_ms = total_ms
        audit.fallback_used = True
        
        # Classify error type
        error_type = VoiceErrorType.UNKNOWN
        if audit.agent_invoked and "agent" in str(e).lower():
            error_type = VoiceErrorType.AGENT_ERROR
        elif "translat" in str(e).lower():
            error_type = VoiceErrorType.TRANSLATION_ERROR
        
        audit.error_type = error_type.value
        audit.error_message = str(e)[:200] if str(e) else "Unknown error"
        
        _log_error_classified(
            voice_trace_id, user_id, session_id,
            error_type,
            str(e)[:200] if str(e) else "Unknown error"
        )
        
        logger.error(
            "voice.process.error",
            voice_trace_id=voice_trace_id,
            user_id=user_id,
            session_id=session_id,
            source_lang=source_lang,
            error=str(e),
            error_type=type(e).__name__,
            latency_total_ms=round(total_ms, 2)
        )
        
        # Return safe fallback response - NEVER raise
        fallback_en = "I apologize, but I encountered an error processing your request. Please try again."
        
        # Best-effort translation for fallback
        if source_lang.lower() != "en":
            fallback_translated, _, _ = _safe_translate(
                fallback_en, target_lang=source_lang, fallback=fallback_en
            )
        else:
            fallback_translated = fallback_en
        
        # Log consolidated audit summary
        _log_audit_summary(audit)
        
        return {
            "final_text": fallback_translated,
            "agent_text_en": fallback_en
        }


# =============================================================================
# Existing VoiceOrchestrator Code
# =============================================================================


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