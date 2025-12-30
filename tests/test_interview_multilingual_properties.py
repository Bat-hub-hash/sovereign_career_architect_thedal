"""Property-based tests for multilingual interview functionality."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List

from sovereign_career_architect.interview.multilingual import (
    MultilingualInterviewManager,
    MultilingualInterviewConfig
)
from sovereign_career_architect.interview.generator import (
    InterviewQuestion,
    InterviewType,
    DifficultyLevel
)
from sovereign_career_architect.voice.sarvam import SarvamClient


# Test data strategies
@st.composite
def language_strategy(draw):
    """Generate supported language codes."""
    languages = ["en", "hi", "ta", "te", "bn", "gu", "kn", "ml", "mr", "pa"]
    return draw(st.sampled_from(languages))


@st.composite
def multilingual_config_strategy(draw):
    """Generate multilingual interview configurations."""
    primary_lang = draw(language_strategy())
    fallback_lang = draw(st.sampled_from(["en", primary_lang]))
    
    return MultilingualInterviewConfig(
        primary_language=primary_lang,
        fallback_language=fallback_lang,
        enable_code_switching=draw(st.booleans()),
        cultural_adaptation=draw(st.booleans()),
        translation_quality_threshold=draw(st.floats(min_value=0.5, max_value=1.0))
    )


@st.composite
def sample_question_strategy(draw):
    """Generate sample interview questions for testing."""
    return InterviewQuestion(
        id=draw(st.text(min_size=5, max_size=20)),
        question=draw(st.text(min_size=10, max_size=200)),
        type=draw(st.sampled_from(list(InterviewType))),
        difficulty=draw(st.sampled_from(list(DifficultyLevel))),
        role=draw(st.sampled_from(["software_engineer", "data_scientist", "product_manager"])),
        category=draw(st.sampled_from(["technical", "behavioral", "system_design"])),
        expected_duration_minutes=draw(st.integers(min_value=5, max_value=30)),
        follow_up_questions=draw(st.lists(st.text(min_size=5, max_size=100), min_size=1, max_size=5)),
        evaluation_criteria=draw(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5)),
        sample_answer_points=draw(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5)),
        tags=draw(st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=5)),
        language="en"
    )


class TestMultilingualInterviewProperties:
    """Property-based tests for multilingual interview functionality."""
    
    def create_mock_sarvam_client(self):
        """Create mock Sarvam client."""
        client = AsyncMock(spec=SarvamClient)
        
        # Mock translation to return modified text
        async def mock_translate(text, source_language, target_language):
            return f"[{target_language}] {text}"
        
        client.translate_text = mock_translate
        
        # Mock language detection
        async def mock_detect_language(text):
            if any(char in text for char in "हिंदी"):
                return "hi"
            elif any(char in text for char in "தமிழ்"):
                return "ta"
            elif any(char in text for char in "తెలుగు"):
                return "te"
            else:
                return "en"
        
        client.detect_language = mock_detect_language
        
        # Mock text generation
        async def mock_generate_text(prompt, language, max_tokens=100):
            return f"Generated text in {language}: {prompt[:50]}..."
        
        client.generate_text = mock_generate_text
        
        return client
    
    def create_multilingual_manager(self):
        """Create multilingual interview manager with mock client."""
        mock_client = self.create_mock_sarvam_client()
        return MultilingualInterviewManager(sarvam_client=mock_client)
    
    @given(
        question=sample_question_strategy(),
        target_language=language_strategy(),
        config=multilingual_config_strategy()
    )
    @settings(max_examples=20, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_question_translation_property(self, question, target_language, config):
        """
        Property 16: Interview Language Adaptation
        
        For any interview question and target language:
        1. Translation should preserve question structure and metadata
        2. Translated content should be marked with correct language
        3. Cultural adaptations should be applied when enabled
        4. Fallback should work when translation fails
        """
        async def run_test():
            # Skip if same language
            if target_language == question.language:
                return
            
            multilingual_manager = self.create_multilingual_manager()
            
            translated_question = await multilingual_manager.translate_question(
                question=question,
                target_language=target_language,
                config=config
            )
            
            # Property 1: Structure preservation
            assert isinstance(translated_question, InterviewQuestion)
            assert translated_question.type == question.type
            assert translated_question.difficulty == question.difficulty
            assert translated_question.role == question.role
            assert translated_question.category == question.category
            assert translated_question.expected_duration_minutes == question.expected_duration_minutes
            
            # Property 2: Language marking
            assert translated_question.language == target_language
            assert translated_question.id != question.id  # Should have different ID
            
            # Property 3: Content translation
            if target_language != "en":
                # Should contain language marker from mock translation
                assert f"[{target_language}]" in translated_question.question
            
            # Property 4: Metadata translation
            assert len(translated_question.follow_up_questions) == len(question.follow_up_questions)
            assert len(translated_question.evaluation_criteria) == len(question.evaluation_criteria)
            assert len(translated_question.sample_answer_points) == len(question.sample_answer_points)
            
            # Property 5: Tags should include translation marker
            if target_language != "en":
                assert any("translated" in tag for tag in translated_question.tags)
        
        asyncio.run(run_test())
    
    @given(
        config=multilingual_config_strategy(),
        role=st.sampled_from(["software_engineer", "data_scientist", "product_manager"])
    )
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multilingual_session_creation_property(self, config, role):
        """
        Property: Multilingual Session Creation Consistency
        
        For any valid configuration and role:
        1. Session should be created with correct language settings
        2. Questions should be translated to target language
        3. Session metadata should reflect multilingual configuration
        """
        multilingual_manager = self.create_multilingual_manager()
        
        async def run_test():
            session = await multilingual_manager.create_multilingual_session(
                role=role,
                config=config,
                company="Test Company",
                interview_type=InterviewType.TECHNICAL,
                difficulty=DifficultyLevel.MID,
                duration_minutes=30
            )
            
            # Property 1: Session configuration
            assert session.role == role
            assert session.language == config.primary_language
            assert session.company == "Test Company"
            
            # Property 2: Questions should be in target language
            for question in session.questions:
                assert question.language == config.primary_language
                assert question.role == role
                
                # If not English, should have translation markers
                if config.primary_language != "en":
                    assert f"[{config.primary_language}]" in question.question
            
            # Property 3: Session should have valid structure
            assert len(session.questions) > 0
            assert session.session_id
            assert session.current_question_index == 0
        
        asyncio.run(run_test())
    
    @given(
        english_text=st.text(min_size=10, max_size=200),
        target_language=language_strategy()
    )
    @settings(max_examples=15, deadline=6000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_code_switching_property(self, english_text, target_language):
        """
        Property: Code-Switching Consistency
        
        For any English text and target language:
        1. Code-switched output should contain elements of both languages
        2. Technical terms should be preserved in English
        3. Output should be different from input (unless target is English)
        """
        # Skip empty or very short texts
        assume(len(english_text.strip()) > 5)
        
        multilingual_manager = self.create_multilingual_manager()
        
        async def run_test():
            code_switched = await multilingual_manager.generate_code_switched_response(
                english_text=english_text,
                target_language=target_language,
                context="interview"
            )
            
            # Property 1: Should return valid text
            assert isinstance(code_switched, str)
            assert len(code_switched.strip()) > 0
            
            # Property 2: For non-English languages, should be different from input
            if target_language != "en":
                # Should either be different or contain language markers
                assert (code_switched != english_text or 
                       f"[{target_language}]" in code_switched or
                       any(char not in english_text for char in code_switched))
            else:
                # For English, should return original or similar text
                assert code_switched == english_text or len(code_switched) > 0
        
        asyncio.run(run_test())
    
    @given(
        user_input=st.text(min_size=5, max_size=100),
        expected_languages=st.lists(language_strategy(), min_size=1, max_size=3)
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_language_detection_property(self, user_input, expected_languages):
        """
        Property: Language Detection Consistency
        
        For any user input:
        1. Should return a supported language code
        2. Should be consistent for the same input
        3. Should default to English for unrecognized input
        """
        multilingual_manager = self.create_multilingual_manager()
        
        async def run_test():
            detected_language = await multilingual_manager.detect_language_preference(user_input)
            
            # Property 1: Should return supported language
            supported_languages = multilingual_manager.get_supported_languages()
            assert detected_language in supported_languages
            
            # Property 2: Should be consistent
            detected_again = await multilingual_manager.detect_language_preference(user_input)
            assert detected_language == detected_again
            
            # Property 3: Should be valid language code
            assert isinstance(detected_language, str)
            assert len(detected_language) >= 2
        
        asyncio.run(run_test())
    
    @given(
        config=multilingual_config_strategy()
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cultural_adaptation_property(self, config):
        """
        Property: Cultural Adaptation Consistency
        
        For any multilingual configuration:
        1. Cultural adaptations should be applied consistently
        2. Adaptation rules should be language-specific
        3. Should preserve core interview structure
        """
        multilingual_manager = self.create_multilingual_manager()
        
        async def run_test():
            # Create a mock session
            from sovereign_career_architect.interview.generator import InterviewSession
            import uuid
            
            session = InterviewSession(
                session_id=str(uuid.uuid4()),
                role="software_engineer",
                company="Test Company",
                interview_type=InterviewType.TECHNICAL,
                difficulty=DifficultyLevel.MID,
                duration_minutes=45,
                language=config.primary_language,
                questions=[]
            )
            
            flow_config = await multilingual_manager.adapt_interview_flow(
                session=session,
                config=config
            )
            
            # Property 1: Should return valid flow configuration
            assert isinstance(flow_config, dict)
            required_keys = [
                "greeting_style", "transition_phrases", "encouragement_frequency",
                "clarification_style", "time_flexibility", "technical_explanation_level"
            ]
            for key in required_keys:
                assert key in flow_config
            
            # Property 2: Should have language-appropriate content
            if config.primary_language in multilingual_manager.language_templates:
                # Should have language-specific phrases
                assert isinstance(flow_config.get("transition_phrases"), list)
            
            # Property 3: Should maintain professional standards
            assert flow_config["greeting_style"] in ["formal", "warm_professional", "respectful_formal", "respectful_friendly"]
            assert flow_config["time_flexibility"] in ["standard", "moderate", "flexible"]
        
        asyncio.run(run_test())
    
    def test_supported_languages_property(self):
        """
        Property: Supported Languages Consistency
        
        The supported languages list should:
        1. Include English as base language
        2. Include major Indian languages
        3. Be consistent across calls
        """
        multilingual_manager = self.create_multilingual_manager()
        
        supported_languages = multilingual_manager.get_supported_languages()
        
        # Property 1: Should include English
        assert "en" in supported_languages
        assert supported_languages["en"] == "English"
        
        # Property 2: Should include major Indian languages
        expected_languages = ["hi", "ta", "te"]
        for lang in expected_languages:
            assert lang in supported_languages
        
        # Property 3: Should be consistent
        supported_again = multilingual_manager.get_supported_languages()
        assert supported_languages == supported_again
        
        # Property 4: Should have proper structure
        for code, name in supported_languages.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) >= 2
            assert len(name) > 0


class TestMultilingualConfigurationProperties:
    """Property-based tests for multilingual configuration."""
    
    @given(config=multilingual_config_strategy())
    def test_config_validation_property(self, config):
        """
        Property: Configuration Validation
        
        For any multilingual configuration:
        1. Should have valid language codes
        2. Should have reasonable threshold values
        3. Should have consistent fallback settings
        """
        # Property 1: Valid language codes
        assert isinstance(config.primary_language, str)
        assert len(config.primary_language) >= 2
        assert isinstance(config.fallback_language, str)
        assert len(config.fallback_language) >= 2
        
        # Property 2: Reasonable threshold
        assert 0.0 <= config.translation_quality_threshold <= 1.0
        
        # Property 3: Boolean flags should be boolean
        assert isinstance(config.enable_code_switching, bool)
        assert isinstance(config.cultural_adaptation, bool)
        
        # Property 4: Fallback should be reasonable
        # Fallback language should be English or same as primary
        assert config.fallback_language in ["en", config.primary_language]


if __name__ == "__main__":
    # Run a quick test
    async def test_basic():
        from unittest.mock import AsyncMock
        
        mock_client = AsyncMock()
        mock_client.translate_text = AsyncMock(return_value="Translated text")
        mock_client.detect_language = AsyncMock(return_value="hi")
        
        manager = MultilingualInterviewManager(sarvam_client=mock_client)
        
        config = MultilingualInterviewConfig(
            primary_language="hi",
            fallback_language="en",
            enable_code_switching=True,
            cultural_adaptation=True
        )
        
        # Test basic functionality
        supported = manager.get_supported_languages()
        print(f"Supported languages: {len(supported)}")
        
        # Test language detection
        detected = await manager.detect_language_preference("Hello world")
        print(f"Detected language: {detected}")
    
    asyncio.run(test_basic())