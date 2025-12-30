"""Multilingual interview support with Sarvam-1 integration."""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog

from sovereign_career_architect.voice.sarvam import SarvamClient
from sovereign_career_architect.interview.generator import (
    InterviewQuestion,
    InterviewSession,
    InterviewType,
    DifficultyLevel
)
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MultilingualInterviewConfig:
    """Configuration for multilingual interviews."""
    primary_language: str
    fallback_language: str = "en"
    enable_code_switching: bool = True
    cultural_adaptation: bool = True
    translation_quality_threshold: float = 0.8


class MultilingualInterviewManager:
    """Manages multilingual interview sessions with Sarvam-1 integration."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.logger = logger.bind(component="multilingual_interview")
        self.sarvam_client = sarvam_client or SarvamClient()
        
        # Language-specific question templates
        self.language_templates = self._load_language_templates()
        
        # Cultural adaptation rules
        self.cultural_adaptations = self._load_cultural_adaptations()
        
        # Supported languages
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi", 
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "mr": "Marathi",
            "pa": "Punjabi"
        }
    
    def _load_language_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load language-specific question templates."""
        return {
            "hi": {
                "greetings": [
                    "नमस्ते! आज के interview में आपका स्वागत है।",
                    "आइए शुरू करते हैं।"
                ],
                "transitions": [
                    "अगला सवाल:",
                    "अब मैं आपसे पूछना चाहूंगा:",
                    "चलिए इस topic पर बात करते हैं:"
                ],
                "encouragements": [
                    "बहुत अच्छा जवाब!",
                    "यह interesting perspective है।",
                    "आप सही दिशा में सोच रहे हैं।"
                ],
                "clarifications": [
                    "क्या आप इसे और detail में explain कर सकते हैं?",
                    "कोई specific example दे सकते हैं?",
                    "इसके क्या benefits होंगे?"
                ]
            },
            "ta": {
                "greetings": [
                    "வணக்கம்! இன்றைய interview-க்கு உங்களை வரவேற்கிறேன்।",
                    "ஆரம்பிக்கலாம்।"
                ],
                "transitions": [
                    "அடுத்த கேள்வி:",
                    "இப்போது நான் உங்களிடம் கேட்க விரும்புகிறேன்:",
                    "இந்த topic-ல் பேசலாம்:"
                ],
                "encouragements": [
                    "மிகவும் நல்ல பதில்!",
                    "இது interesting perspective.",
                    "நீங்கள் சரியான direction-ல் யோசிக்கிறீர்கள்।"
                ],
                "clarifications": [
                    "இதை இன்னும் detail-ல் explain பண்ண முடியுமா?",
                    "ஏதாவது specific example கொடுக்க முடியுமா?",
                    "இதோட benefits என்ன?"
                ]
            },
            "te": {
                "greetings": [
                    "నమస్కారం! ఈ రోజు interview కి మిమ్మల్ని స్వాగతం చేస్తున్నాను।",
                    "మొదలు పెట్టండి।"
                ],
                "transitions": [
                    "తదుపరి ప్రశ్న:",
                    "ఇప్పుడు నేను మిమ్మల్ని అడగాలని అనుకుంటున్నాను:",
                    "ఈ topic మీద మాట్లాడుకుందాం:"
                ],
                "encouragements": [
                    "చాలా బాగా చెప్పారు!",
                    "ఇది interesting perspective.",
                    "మీరు సరైన direction లో ఆలోచిస్తున్నారు।"
                ],
                "clarifications": [
                    "దీన్ని మరింత detail లో explain చేయగలరా?",
                    "ఏదైనా specific example ఇవ్వగలరా?",
                    "దీని benefits ఏమిటి?"
                ]
            }
        }
    
    def _load_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural adaptation rules for different regions."""
        return {
            "hi": {
                "communication_style": "respectful_formal",
                "examples_preference": "indian_companies",
                "technical_terms": "mix_english_hindi",
                "time_flexibility": "moderate",
                "hierarchy_awareness": "high"
            },
            "ta": {
                "communication_style": "warm_professional", 
                "examples_preference": "south_indian_context",
                "technical_terms": "english_with_tamil_explanations",
                "time_flexibility": "moderate",
                "hierarchy_awareness": "moderate"
            },
            "te": {
                "communication_style": "respectful_friendly",
                "examples_preference": "telugu_states_context", 
                "technical_terms": "english_dominant",
                "time_flexibility": "flexible",
                "hierarchy_awareness": "moderate"
            }
        }
    
    async def translate_question(
        self,
        question: InterviewQuestion,
        target_language: str,
        config: MultilingualInterviewConfig
    ) -> InterviewQuestion:
        """
        Translate an interview question to the target language.
        
        Args:
            question: Original question
            target_language: Target language code
            config: Multilingual configuration
            
        Returns:
            Translated question
        """
        if target_language == "en" or target_language == question.language:
            return question
        
        self.logger.info(
            "Translating interview question",
            question_id=question.id,
            source_language=question.language,
            target_language=target_language
        )
        
        try:
            # Translate main question text
            translated_question = await self.sarvam_client.translate_text(
                text=question.question,
                source_language=question.language,
                target_language=target_language
            )
            
            # Translate follow-up questions
            translated_follow_ups = []
            for follow_up in question.follow_up_questions:
                translated_follow_up = await self.sarvam_client.translate_text(
                    text=follow_up,
                    source_language=question.language,
                    target_language=target_language
                )
                translated_follow_ups.append(translated_follow_up)
            
            # Translate evaluation criteria
            translated_criteria = []
            for criterion in question.evaluation_criteria:
                translated_criterion = await self.sarvam_client.translate_text(
                    text=criterion,
                    source_language=question.language,
                    target_language=target_language
                )
                translated_criteria.append(translated_criterion)
            
            # Translate sample answer points
            translated_answer_points = []
            for point in question.sample_answer_points:
                translated_point = await self.sarvam_client.translate_text(
                    text=point,
                    source_language=question.language,
                    target_language=target_language
                )
                translated_answer_points.append(translated_point)
            
            # Apply cultural adaptations
            if config.cultural_adaptation:
                translated_question = await self._apply_cultural_adaptation(
                    translated_question,
                    target_language,
                    question.category
                )
            
            # Create translated question
            translated_q = InterviewQuestion(
                id=f"{question.id}_{target_language}",
                question=translated_question,
                type=question.type,
                difficulty=question.difficulty,
                role=question.role,
                category=question.category,
                expected_duration_minutes=question.expected_duration_minutes,
                follow_up_questions=translated_follow_ups,
                evaluation_criteria=translated_criteria,
                sample_answer_points=translated_answer_points,
                tags=question.tags + [f"translated_{target_language}"],
                language=target_language
            )
            
            self.logger.info(
                "Question translated successfully",
                question_id=question.id,
                target_language=target_language
            )
            
            return translated_q
            
        except Exception as e:
            self.logger.error(
                "Failed to translate question",
                question_id=question.id,
                target_language=target_language,
                error=str(e)
            )
            
            # Fallback to original question with language marker
            fallback_question = InterviewQuestion(
                id=f"{question.id}_fallback",
                question=f"[{target_language}] {question.question}",
                type=question.type,
                difficulty=question.difficulty,
                role=question.role,
                category=question.category,
                expected_duration_minutes=question.expected_duration_minutes,
                follow_up_questions=question.follow_up_questions,
                evaluation_criteria=question.evaluation_criteria,
                sample_answer_points=question.sample_answer_points,
                tags=question.tags + ["translation_fallback"],
                language=target_language
            )
            
            return fallback_question
    
    async def _apply_cultural_adaptation(
        self,
        question_text: str,
        language: str,
        category: str
    ) -> str:
        """Apply cultural adaptations to translated questions."""
        
        adaptations = self.cultural_adaptations.get(language, {})
        
        # Apply communication style adaptations
        if adaptations.get("communication_style") == "respectful_formal":
            # Add respectful prefixes for Hindi
            if not any(prefix in question_text for prefix in ["कृपया", "आप"]):
                question_text = f"कृपया {question_text}"
        
        elif adaptations.get("communication_style") == "warm_professional":
            # Add warm tone for Tamil
            if category in ["behavioral", "leadership"]:
                question_text = question_text.replace("Tell me", "உங்கள் experience-ல்")
        
        # Apply context-specific examples
        if adaptations.get("examples_preference") == "indian_companies":
            # Replace generic company examples with Indian ones
            replacements = {
                "Google": "Infosys या TCS",
                "Amazon": "Flipkart या Zomato", 
                "Microsoft": "Wipro या HCL"
            }
            
            for original, replacement in replacements.items():
                question_text = question_text.replace(original, replacement)
        
        return question_text
    
    async def create_multilingual_session(
        self,
        role: str,
        config: MultilingualInterviewConfig,
        company: Optional[str] = None,
        interview_type: InterviewType = InterviewType.MIXED,
        difficulty: DifficultyLevel = DifficultyLevel.MID,
        duration_minutes: int = 45
    ) -> InterviewSession:
        """
        Create a multilingual interview session.
        
        Args:
            role: Target role
            config: Multilingual configuration
            company: Company name
            interview_type: Type of interview
            difficulty: Difficulty level
            duration_minutes: Session duration
            
        Returns:
            Multilingual interview session
        """
        from sovereign_career_architect.interview.generator import InterviewQuestionGenerator
        
        self.logger.info(
            "Creating multilingual interview session",
            role=role,
            primary_language=config.primary_language,
            interview_type=interview_type.value,
            difficulty=difficulty.value
        )
        
        # Create base session in English
        generator = InterviewQuestionGenerator()
        base_session = await generator.create_interview_session(
            role=role,
            company=company,
            interview_type=interview_type,
            difficulty=difficulty,
            duration_minutes=duration_minutes,
            language="en"
        )
        
        # Translate questions to target language
        if config.primary_language != "en":
            translated_questions = []
            
            for question in base_session.questions:
                translated_question = await self.translate_question(
                    question=question,
                    target_language=config.primary_language,
                    config=config
                )
                translated_questions.append(translated_question)
            
            # Update session with translated questions
            base_session.questions = translated_questions
            base_session.language = config.primary_language
        
        self.logger.info(
            "Multilingual interview session created",
            session_id=base_session.session_id,
            language=config.primary_language,
            question_count=len(base_session.questions)
        )
        
        return base_session
    
    async def generate_code_switched_response(
        self,
        english_text: str,
        target_language: str,
        context: str = "interview"
    ) -> str:
        """
        Generate code-switched response mixing English and target language.
        
        Args:
            english_text: Original English text
            target_language: Target language for mixing
            context: Context for code-switching
            
        Returns:
            Code-switched text
        """
        if target_language == "en":
            return english_text
        
        self.logger.info(
            "Generating code-switched response",
            target_language=target_language,
            context=context
        )
        
        try:
            # Use Sarvam-1 for intelligent code-switching
            prompt = f"""
            Convert this English text to natural {target_language}-English code-switched text 
            suitable for a professional interview context. Keep technical terms in English 
            but use {target_language} for conversational elements:
            
            Text: {english_text}
            
            Guidelines:
            - Keep technical terms in English
            - Use {target_language} for greetings, transitions, encouragements
            - Maintain professional tone
            - Sound natural to bilingual speakers
            """
            
            code_switched = await self.sarvam_client.generate_text(
                prompt=prompt,
                language=target_language,
                max_tokens=200
            )
            
            return code_switched.strip()
            
        except Exception as e:
            self.logger.error(
                "Failed to generate code-switched response",
                target_language=target_language,
                error=str(e)
            )
            
            # Fallback: Simple template-based code-switching
            return await self._simple_code_switch(english_text, target_language)
    
    async def _simple_code_switch(self, text: str, language: str) -> str:
        """Simple template-based code-switching fallback."""
        
        templates = self.language_templates.get(language, {})
        
        # Add language-specific greetings/transitions
        if "question" in text.lower():
            transitions = templates.get("transitions", [])
            if transitions:
                transition = transitions[0]
                text = f"{transition} {text}"
        
        # Keep technical terms in English, add language markers
        technical_terms = [
            "API", "database", "algorithm", "framework", "architecture",
            "performance", "scalability", "security", "testing", "deployment"
        ]
        
        # Simple mixing: keep technical terms, translate connectors
        if language == "hi":
            text = text.replace(" and ", " और ")
            text = text.replace(" or ", " या ")
            text = text.replace(" but ", " लेकिन ")
        elif language == "ta":
            text = text.replace(" and ", " மற்றும் ")
            text = text.replace(" or ", " அல்லது ")
        elif language == "te":
            text = text.replace(" and ", " మరియు ")
            text = text.replace(" or ", " లేదా ")
        
        return text
    
    async def adapt_interview_flow(
        self,
        session: InterviewSession,
        config: MultilingualInterviewConfig
    ) -> Dict[str, Any]:
        """
        Adapt interview flow for cultural and linguistic preferences.
        
        Args:
            session: Interview session
            config: Multilingual configuration
            
        Returns:
            Adapted flow configuration
        """
        language = config.primary_language
        adaptations = self.cultural_adaptations.get(language, {})
        
        flow_config = {
            "greeting_style": "formal",
            "transition_phrases": [],
            "encouragement_frequency": "moderate",
            "clarification_style": "direct",
            "time_flexibility": "standard",
            "technical_explanation_level": "standard"
        }
        
        # Apply language-specific adaptations
        if language in self.language_templates:
            templates = self.language_templates[language]
            flow_config["transition_phrases"] = templates.get("transitions", [])
            flow_config["encouragements"] = templates.get("encouragements", [])
            flow_config["clarifications"] = templates.get("clarifications", [])
        
        # Apply cultural adaptations
        if adaptations:
            flow_config["greeting_style"] = adaptations.get("communication_style", "formal")
            flow_config["time_flexibility"] = adaptations.get("time_flexibility", "standard")
            
            if adaptations.get("hierarchy_awareness") == "high":
                flow_config["encouragement_frequency"] = "high"
                flow_config["clarification_style"] = "gentle"
        
        self.logger.info(
            "Interview flow adapted",
            session_id=session.session_id,
            language=language,
            adaptations_applied=len(adaptations)
        )
        
        return flow_config
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    async def detect_language_preference(self, user_input: str) -> str:
        """
        Detect user's language preference from input.
        
        Args:
            user_input: User's text input
            
        Returns:
            Detected language code
        """
        try:
            detected_language = await self.sarvam_client.detect_language(user_input)
            
            if detected_language in self.supported_languages:
                return detected_language
            else:
                return "en"  # Default to English
                
        except Exception as e:
            self.logger.error(
                "Language detection failed",
                error=str(e)
            )
            return "en"  # Default fallback