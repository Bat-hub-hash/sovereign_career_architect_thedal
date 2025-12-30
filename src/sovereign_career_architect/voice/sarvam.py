"""Sarvam-1 integration for Indic language processing."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
import structlog

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class IndicLanguage(Enum):
    """Supported Indic languages."""
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    ODIA = "or"


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    is_indic: bool
    detected_script: Optional[str] = None


@dataclass
class SarvamResponse:
    """Response from Sarvam-1 model."""
    text: str
    language: str
    tokens_used: int
    fertility_rate: float
    confidence: float


class SarvamClient:
    """Client for Sarvam-1 model via Hugging Face Inference API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.huggingface_api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.model_name = "sarvamai/sarvam-1"
        
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
        self.logger = logger.bind(component="sarvam_client")
        
        # Language detection patterns
        self.indic_patterns = {
            IndicLanguage.HINDI: {
                "script_range": (0x0900, 0x097F),  # Devanagari
                "common_words": ["है", "का", "की", "को", "में", "से", "पर", "और", "या", "नहीं"]
            },
            IndicLanguage.BENGALI: {
                "script_range": (0x0980, 0x09FF),  # Bengali
                "common_words": ["আছে", "এর", "এই", "সে", "তার", "করে", "হয়", "দিয়ে", "থেকে", "না"]
            },
            IndicLanguage.TAMIL: {
                "script_range": (0x0B80, 0x0BFF),  # Tamil
                "common_words": ["உள்ளது", "இந்த", "அந்த", "என்று", "மற்றும", "அல்லது", "இல்லை", "செய்ய", "வேண்டும்", "போல்"]
            },
            IndicLanguage.TELUGU: {
                "script_range": (0x0C00, 0x0C7F),  # Telugu
                "common_words": ["ఉంది", "ఈ", "ఆ", "అని", "మరియు", "లేదా", "కాదు", "చేయాలి", "వేండి", "లాగా"]
            },
            IndicLanguage.MARATHI: {
                "script_range": (0x0900, 0x097F),  # Devanagari (same as Hindi)
                "common_words": ["आहे", "या", "ही", "ते", "आणि", "किंवा", "नाही", "करणे", "पाहिजे", "सारखे"]
            },
            IndicLanguage.GUJARATI: {
                "script_range": (0x0A80, 0x0AFF),  # Gujarati
                "common_words": ["છે", "આ", "તે", "અને", "અથવા", "નથી", "કરવું", "જોઈએ", "જેવું", "માટે"]
            },
            IndicLanguage.KANNADA: {
                "script_range": (0x0C80, 0x0CFF),  # Kannada
                "common_words": ["ಇದೆ", "ಈ", "ಆ", "ಮತ್ತು", "ಅಥವಾ", "ಇಲ್ಲ", "ಮಾಡಬೇಕು", "ಬೇಕು", "ಹಾಗೆ", "ಗಾಗಿ"]
            },
            IndicLanguage.MALAYALAM: {
                "script_range": (0x0D00, 0x0D7F),  # Malayalam
                "common_words": ["ഉണ്ട്", "ഈ", "ആ", "എന്ന്", "കൂടാതെ", "അല്ലെങ്കിൽ", "ഇല്ല", "ചെയ്യണം", "വേണം", "പോലെ"]
            },
            IndicLanguage.PUNJABI: {
                "script_range": (0x0A00, 0x0A7F),  # Gurmukhi
                "common_words": ["ਹੈ", "ਇਹ", "ਉਹ", "ਅਤੇ", "ਜਾਂ", "ਨਹੀਂ", "ਕਰਨਾ", "ਚਾਹੀਦਾ", "ਵਰਗਾ", "ਲਈ"]
            },
            IndicLanguage.ODIA: {
                "script_range": (0x0B00, 0x0B7F),  # Odia
                "common_words": ["ଅଛି", "ଏହି", "ସେହି", "ଏବଂ", "କିମ୍ବା", "ନାହିଁ", "କରିବା", "ଉଚିତ", "ପରି", "ପାଇଁ"]
            }
        }
    
    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect if text is in an Indic language and which one.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageDetectionResult with detection details
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language="en",
                confidence=0.0,
                is_indic=False
            )
        
        # Check for Indic scripts using Unicode ranges
        detected_languages = []
        
        for lang, patterns in self.indic_patterns.items():
            script_start, script_end = patterns["script_range"]
            common_words = patterns["common_words"]
            
            # Check script characters
            script_chars = sum(
                1 for char in text 
                if script_start <= ord(char) <= script_end
            )
            script_ratio = script_chars / len(text) if text else 0
            
            # Check common words
            word_matches = sum(
                1 for word in common_words 
                if word in text
            )
            word_score = word_matches / len(common_words)
            
            # Combined confidence score
            confidence = (script_ratio * 0.7) + (word_score * 0.3)
            
            if confidence > 0.1:  # Threshold for detection
                detected_languages.append((lang, confidence))
        
        if detected_languages:
            # Sort by confidence and return the most likely
            detected_languages.sort(key=lambda x: x[1], reverse=True)
            best_lang, confidence = detected_languages[0]
            
            return LanguageDetectionResult(
                language=best_lang.value,
                confidence=confidence,
                is_indic=True,
                detected_script=self._get_script_name(best_lang)
            )
        
        # Check if it's English or other non-Indic language
        english_chars = sum(
            1 for char in text 
            if char.isascii() and char.isalpha()
        )
        english_ratio = english_chars / len([c for c in text if c.isalpha()]) if any(c.isalpha() for c in text) else 0
        
        if english_ratio > 0.7:
            return LanguageDetectionResult(
                language="en",
                confidence=english_ratio,
                is_indic=False
            )
        
        # Default to English with low confidence
        return LanguageDetectionResult(
            language="en",
            confidence=0.1,
            is_indic=False
        )
    
    def _get_script_name(self, language: IndicLanguage) -> str:
        """Get the script name for a language."""
        script_map = {
            IndicLanguage.HINDI: "Devanagari",
            IndicLanguage.MARATHI: "Devanagari",
            IndicLanguage.BENGALI: "Bengali",
            IndicLanguage.TAMIL: "Tamil",
            IndicLanguage.TELUGU: "Telugu",
            IndicLanguage.GUJARATI: "Gujarati",
            IndicLanguage.KANNADA: "Kannada",
            IndicLanguage.MALAYALAM: "Malayalam",
            IndicLanguage.PUNJABI: "Gurmukhi",
            IndicLanguage.ODIA: "Odia"
        }
        return script_map.get(language, "Unknown")
    
    async def generate_response(
        self,
        prompt: str,
        language: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stream: bool = False
    ) -> SarvamResponse:
        """
        Generate response using Sarvam-1 model.
        
        Args:
            prompt: Input prompt
            language: Target language code
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            stream: Enable streaming response
            
        Returns:
            SarvamResponse with generated text and metadata
        """
        try:
            # Prepare request payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "return_full_text": False,
                    "stream": stream  # Enable streaming if requested
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            # Add language-specific parameters
            if language in [lang.value for lang in IndicLanguage]:
                payload["parameters"]["language"] = language
                payload["parameters"]["script_optimization"] = True
            
            # Make request to Hugging Face Inference API
            url = f"{self.base_url}/{self.model_name}"
            
            if stream:
                return await self._generate_streaming_response(url, payload, language)
            else:
                return await self._generate_standard_response(url, payload, language)
                
        except Exception as e:
            self.logger.error("Sarvam-1 generation failed", error=str(e))
            
            return SarvamResponse(
                text="I'm experiencing technical difficulties. Please try again.",
                language="en",
                tokens_used=0,
                fertility_rate=0.0,
                confidence=0.1
            )
    
    async def _generate_standard_response(self, url: str, payload: dict, language: str) -> SarvamResponse:
        """Generate standard (non-streaming) response."""
        response = await self.client.post(url, json=payload)
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            self.logger.info("Sarvam-1 model loading, waiting...")
            await asyncio.sleep(10)
            response = await self.client.post(url, json=payload)
        
        response.raise_for_status()
        result = response.json()
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = result.get("generated_text", "")
        
        # Calculate token fertility rate (approximation)
        input_tokens = len(payload["inputs"].split())
        output_tokens = len(generated_text.split())
        
        # Estimate fertility rate based on language
        if language in [lang.value for lang in IndicLanguage]:
            # Indic languages typically have better fertility rates with Sarvam-1
            fertility_rate = min(2.1, max(1.4, output_tokens / max(input_tokens, 1)))
        else:
            # English and other languages
            fertility_rate = output_tokens / max(input_tokens, 1)
        
        self.logger.info(
            "Sarvam-1 response generated",
            language=language,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fertility_rate=fertility_rate
        )
        
        return SarvamResponse(
            text=generated_text,
            language=language,
            tokens_used=input_tokens + output_tokens,
            fertility_rate=fertility_rate,
            confidence=0.9  # High confidence for successful generation
        )
    
    async def _generate_streaming_response(self, url: str, payload: dict, language: str) -> SarvamResponse:
        """Generate streaming response with optimistic acknowledgment."""
        try:
            # Start with optimistic acknowledgment
            filler_phrases = self._get_filler_phrases(language)
            current_filler = filler_phrases[0] if filler_phrases else "Let me think about that..."
            
            # Simulate streaming by making request and providing immediate feedback
            response_future = asyncio.create_task(
                self._make_streaming_request(url, payload)
            )
            
            # Provide immediate acknowledgment
            await asyncio.sleep(0.1)  # Minimal delay for realism
            
            # Wait for actual response
            try:
                actual_response = await asyncio.wait_for(response_future, timeout=5.0)
            except asyncio.TimeoutError:
                # If response takes too long, return filler with lower confidence
                return SarvamResponse(
                    text=current_filler + " I'm still processing your request.",
                    language=language,
                    tokens_used=10,
                    fertility_rate=1.5,
                    confidence=0.6
                )
            
            # Process the actual response
            if isinstance(actual_response, list) and len(actual_response) > 0:
                generated_text = actual_response[0].get("generated_text", "")
            else:
                generated_text = actual_response.get("generated_text", "")
            
            # Calculate metrics
            input_tokens = len(payload["inputs"].split())
            output_tokens = len(generated_text.split())
            
            if language in [lang.value for lang in IndicLanguage]:
                fertility_rate = min(2.1, max(1.4, output_tokens / max(input_tokens, 1)))
            else:
                fertility_rate = output_tokens / max(input_tokens, 1)
            
            self.logger.info(
                "Sarvam-1 streaming response generated",
                language=language,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                fertility_rate=fertility_rate,
                streaming=True
            )
            
            return SarvamResponse(
                text=generated_text,
                language=language,
                tokens_used=input_tokens + output_tokens,
                fertility_rate=fertility_rate,
                confidence=0.9
            )
            
        except Exception as e:
            self.logger.error("Streaming response failed", error=str(e))
            
            # Fallback to filler response
            filler_phrases = self._get_filler_phrases(language)
            fallback_text = filler_phrases[0] if filler_phrases else "I'm having trouble processing that right now."
            
            return SarvamResponse(
                text=fallback_text,
                language=language,
                tokens_used=5,
                fertility_rate=1.0,
                confidence=0.3
            )
    
    async def _make_streaming_request(self, url: str, payload: dict) -> dict:
        """Make the actual streaming request to the API."""
        response = await self.client.post(url, json=payload)
        
        if response.status_code == 503:
            await asyncio.sleep(5)  # Shorter wait for streaming
            response = await self.client.post(url, json=payload)
        
        response.raise_for_status()
        return response.json()
    
    def _get_filler_phrases(self, language: str) -> List[str]:
        """Get appropriate filler phrases for the language."""
        filler_phrases = {
            "hi": ["मैं इसके बारे में सोच रहा हूं...", "एक मिनट...", "समझ गया..."],
            "bn": ["আমি এটা নিয়ে ভাবছি...", "একটু অপেক্ষা করুন...", "বুঝতে পারছি..."],
            "ta": ["நான் இதைப் பற்றி யோசித்துக்கொண்டிருக்கிறேன்...", "ஒரு நிமிடம்...", "புரிந்துகொண்டேன்..."],
            "te": ["నేను దీని గురించి ఆలోచిస్తున్నాను...", "ఒక నిమిషం...", "అర్థమైంది..."],
            "mr": ["मी याबद्दल विचार करत आहे...", "एक मिनिट...", "समजले..."],
            "gu": ["હું આ વિશે વિચારી રહ્યો છું...", "એક મિનિટ...", "સમજાયું..."],
            "kn": ["ನಾನು ಇದರ ಬಗ್ಗೆ ಯೋಚಿಸುತ್ತಿದ್ದೇನೆ...", "ಒಂದು ನಿಮಿಷ...", "ಅರ್ಥವಾಯಿತು..."],
            "ml": ["ഞാൻ ഇതിനെക്കുറിച്ച് ചിന്തിക്കുകയാണ്...", "ഒരു മിനിറ്റ്...", "മനസ്സിലായി..."],
            "pa": ["ਮੈਂ ਇਸ ਬਾਰੇ ਸੋਚ ਰਿਹਾ ਹਾਂ...", "ਇੱਕ ਮਿੰਟ...", "ਸਮਝ ਗਿਆ..."],
            "or": ["ମୁଁ ଏହା ବିଷୟରେ ଭାବୁଛି...", "ଏକ ମିନିଟ୍...", "ବୁଝିଲି..."],
            "en": ["Let me think about that...", "One moment...", "I understand..."]
        }
        
        return filler_phrases.get(language, filler_phrases["en"])
    
    async def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """
        Translate text between languages using Sarvam-1.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated text
        """
        if source_language == target_language:
            return text
        
        # Create translation prompt
        if target_language in [lang.value for lang in IndicLanguage]:
            prompt = f"Translate the following text from {source_language} to {target_language}:\n\n{text}\n\nTranslation:"
        else:
            prompt = f"Translate the following text to English:\n\n{text}\n\nTranslation:"
        
        try:
            response = await self.generate_response(
                prompt=prompt,
                language=target_language,
                max_tokens=len(text.split()) * 2,  # Allow for expansion
                temperature=0.3  # Lower temperature for translation
            )
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(
                "Translation failed",
                error=str(e),
                source_lang=source_language,
                target_lang=target_language
            )
            return text  # Return original text if translation fails
    
    async def optimize_for_indic_tokenization(self, text: str, language: str) -> str:
        """
        Optimize text for efficient Indic language tokenization.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Optimized text
        """
        if language not in [lang.value for lang in IndicLanguage]:
            return text
        
        # Apply language-specific optimizations
        optimized_text = text
        
        # Remove unnecessary spaces around Indic punctuation
        import re
        
        # Common Indic punctuation patterns
        indic_punctuation = ["।", "॥", "॰", "।।"]
        
        for punct in indic_punctuation:
            # Remove spaces before punctuation
            optimized_text = re.sub(f" +{re.escape(punct)}", punct, optimized_text)
            # Ensure single space after punctuation
            optimized_text = re.sub(f"{re.escape(punct)} +", f"{punct} ", optimized_text)
        
        # Normalize whitespace
        optimized_text = re.sub(r"\s+", " ", optimized_text).strip()
        
        return optimized_text
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported Indic languages."""
        return [lang.value for lang in IndicLanguage]
    
    async def close(self):
        """Close the Sarvam client."""
        await self.client.aclose()
        self.logger.info("Sarvam client closed")


class LanguageRouter:
    """Routes requests to appropriate language models based on detected language."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.sarvam_client = sarvam_client or SarvamClient()
        self.logger = logger.bind(component="language_router")
        
        # Cache for language detection results
        self._detection_cache: Dict[str, LanguageDetectionResult] = {}
    
    async def route_request(
        self,
        text: str,
        preferred_language: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Route request to appropriate language model.
        
        Args:
            text: Input text
            preferred_language: User's preferred language (optional)
            
        Returns:
            Tuple of (model_to_use, is_indic_language)
        """
        # Use preferred language if specified and supported
        if preferred_language:
            supported_languages = await self.sarvam_client.get_supported_languages()
            if preferred_language in supported_languages:
                return ("sarvam-1", True)
            elif preferred_language == "en":
                return ("openai", False)
        
        # Detect language from text
        cache_key = text[:100]  # Use first 100 chars as cache key
        
        if cache_key in self._detection_cache:
            detection_result = self._detection_cache[cache_key]
        else:
            detection_result = await self.sarvam_client.detect_language(text)
            self._detection_cache[cache_key] = detection_result
            
            # Limit cache size
            if len(self._detection_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self._detection_cache.keys())[:100]
                for key in oldest_keys:
                    del self._detection_cache[key]
        
        self.logger.info(
            "Language routing decision",
            detected_language=detection_result.language,
            confidence=detection_result.confidence,
            is_indic=detection_result.is_indic,
            model_choice="sarvam-1" if detection_result.is_indic else "openai"
        )
        
        # Route based on detection result
        if detection_result.is_indic and detection_result.confidence > 0.3:
            return ("sarvam-1", True)
        else:
            return ("openai", False)
    
    async def process_with_appropriate_model(
        self,
        text: str,
        system_prompt: str,
        preferred_language: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        enable_streaming: bool = False
    ) -> SarvamResponse:
        """
        Process text with the most appropriate language model.
        
        Args:
            text: Input text
            system_prompt: System prompt for context
            preferred_language: User's preferred language
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            enable_streaming: Enable streaming response
            
        Returns:
            SarvamResponse with generated text and metadata
        """
        model_choice, is_indic = await self.route_request(text, preferred_language)
        
        if model_choice == "sarvam-1":
            # Use Sarvam-1 for Indic languages
            full_prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
            
            target_language = preferred_language or "hi"  # Default to Hindi
            
            try:
                return await self.sarvam_client.generate_response(
                    prompt=full_prompt,
                    language=target_language,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=enable_streaming
                )
            except Exception as e:
                # Fallback response for errors
                return SarvamResponse(
                    text="I'm experiencing technical difficulties. Please try again.",
                    language=target_language,
                    tokens_used=0,
                    fertility_rate=0.0,
                    confidence=0.1
                )
        else:
            # Fallback to OpenAI for English and other languages
            # This would integrate with existing OpenAI client
            return SarvamResponse(
                text="I'll process this request using the standard language model.",
                language="en",
                tokens_used=0,
                fertility_rate=1.0,
                confidence=0.8
            )
    
    async def close(self):
        """Close the language router."""
        await self.sarvam_client.close()
        self.logger.info("Language router closed")