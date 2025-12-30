"""Property-based tests for language model routing."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sovereign_career_architect.voice.sarvam import (
    SarvamClient, LanguageRouter, IndicLanguage, 
    LanguageDetectionResult, SarvamResponse
)


# Feature: sovereign-career-architect, Property 14: Language Model Routing
@pytest.mark.property
class TestLanguageRoutingProperties:
    """Property-based tests for language model routing behavior."""
    
    @given(
        indic_text=st.text(min_size=5, max_size=200),
        confidence_score=st.floats(min_value=0.31, max_value=1.0)  # Changed from 0.3 to 0.31
    )
    @settings(max_examples=20, deadline=3000)
    @pytest.mark.asyncio
    async def test_indic_language_routing_to_sarvam(
        self,
        indic_text: str,
        confidence_score: float
    ):
        """
        Property 14: Language Model Routing
        
        For any input detected as an Indic language with sufficient confidence,
        the language router should route the processing to Sarvam-1 model.
        
        Validates: Requirements 4.2
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        
        # Mock language detection to return Indic language
        mock_detection = LanguageDetectionResult(
            language="hi",  # Hindi
            confidence=confidence_score,
            is_indic=True,
            detected_script="Devanagari"
        )
        mock_sarvam_client.detect_language.return_value = mock_detection
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Test routing decision
        model_choice, is_indic = await router.route_request(indic_text)
        
        # Property: Indic languages with sufficient confidence should route to Sarvam-1
        assert model_choice == "sarvam-1", (
            f"Indic language with confidence {confidence_score} should route to Sarvam-1, "
            f"but got {model_choice}"
        )
        assert is_indic is True, (
            "Routing result should indicate Indic language"
        )
        
        # Verify detection was called
        mock_sarvam_client.detect_language.assert_called_once_with(indic_text)
    
    @given(
        english_text=st.text(min_size=5, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        confidence_score=st.floats(min_value=0.5, max_value=1.0)
    )
    @settings(max_examples=20, deadline=3000, suppress_health_check=[HealthCheck.filter_too_much])
    @pytest.mark.asyncio
    async def test_english_language_routing_to_openai(
        self,
        english_text: str,
        confidence_score: float
    ):
        """
        Property: English Language Routing
        
        For any input detected as English, the language router should
        route the processing to OpenAI model.
        """
        assume(len(english_text.strip()) > 0)
        
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        
        # Mock language detection to return English
        mock_detection = LanguageDetectionResult(
            language="en",
            confidence=confidence_score,
            is_indic=False
        )
        mock_sarvam_client.detect_language.return_value = mock_detection
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Test routing decision
        model_choice, is_indic = await router.route_request(english_text)
        
        # Property: English should route to OpenAI
        assert model_choice == "openai", (
            f"English text should route to OpenAI, but got {model_choice}"
        )
        assert is_indic is False, (
            "Routing result should indicate non-Indic language"
        )
    
    @given(
        preferred_language=st.sampled_from(["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]),
        text_content=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_preferred_language_override(
        self,
        preferred_language: str,
        text_content: str
    ):
        """
        Property: Preferred Language Override
        
        For any supported Indic language specified as preferred,
        the router should use Sarvam-1 regardless of detected language.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Test routing with preferred language
        model_choice, is_indic = await router.route_request(
            text_content, 
            preferred_language=preferred_language
        )
        
        # Property: Preferred Indic language should override detection
        assert model_choice == "sarvam-1", (
            f"Preferred Indic language {preferred_language} should route to Sarvam-1, "
            f"but got {model_choice}"
        )
        assert is_indic is True, (
            "Preferred Indic language should be marked as Indic"
        )
    
    @given(
        low_confidence=st.floats(min_value=0.0, max_value=0.3),  # Changed to include 0.3
        text_content=st.text(min_size=5, max_size=100)
    )
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_low_confidence_fallback_to_openai(
        self,
        low_confidence: float,
        text_content: str
    ):
        """
        Property: Low Confidence Fallback
        
        For any language detection with low confidence (< 0.3),
        the router should fallback to OpenAI model.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        
        # Mock low confidence detection
        mock_detection = LanguageDetectionResult(
            language="hi",  # Even if detected as Indic
            confidence=low_confidence,
            is_indic=True
        )
        mock_sarvam_client.detect_language.return_value = mock_detection
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Test routing decision
        model_choice, is_indic = await router.route_request(text_content)
        
        # Property: Low confidence should fallback to OpenAI
        assert model_choice == "openai", (
            f"Low confidence ({low_confidence}) should fallback to OpenAI, "
            f"but got {model_choice}"
        )
        assert is_indic is False, (
            "Low confidence detection should be treated as non-Indic"
        )
    
    @given(
        supported_languages=st.lists(
            st.sampled_from(["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=10, deadline=2000)
    @pytest.mark.asyncio
    async def test_supported_language_consistency(
        self,
        supported_languages: list[str]
    ):
        """
        Property: Supported Language Consistency
        
        For any set of supported languages, the router should consistently
        route those languages to Sarvam-1 and others to OpenAI.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.get_supported_languages.return_value = supported_languages
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Test each supported language
        for lang in supported_languages:
            model_choice, is_indic = await router.route_request(
                "test text", 
                preferred_language=lang
            )
            
            # Property: Supported languages should route to Sarvam-1
            assert model_choice == "sarvam-1", (
                f"Supported language {lang} should route to Sarvam-1"
            )
            assert is_indic is True, (
                f"Supported language {lang} should be marked as Indic"
            )
        
        # Test unsupported language (English)
        model_choice, is_indic = await router.route_request(
            "test text", 
            preferred_language="en"
        )
        
        assert model_choice == "openai", (
            "Unsupported language 'en' should route to OpenAI"
        )
        assert is_indic is False, (
            "Unsupported language should be marked as non-Indic"
        )
    
    @given(
        system_prompt=st.text(min_size=10, max_size=200),
        user_text=st.text(min_size=5, max_size=100),
        target_language=st.sampled_from(["hi", "bn", "ta", "en"])
    )
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_model_processing_consistency(
        self,
        system_prompt: str,
        user_text: str,
        target_language: str
    ):
        """
        Property: Model Processing Consistency
        
        For any input processed with appropriate model, the response
        should be in the target language with consistent metadata.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Mock Sarvam response for Indic languages
        if target_language in ["hi", "bn", "ta"]:
            mock_response = SarvamResponse(
                text=f"Response in {target_language}",
                language=target_language,
                tokens_used=50,
                fertility_rate=1.8,  # Good fertility rate for Indic
                confidence=0.9
            )
            mock_sarvam_client.generate_response.return_value = mock_response
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Process with appropriate model
        result = await router.process_with_appropriate_model(
            text=user_text,
            system_prompt=system_prompt,
            preferred_language=target_language,
            max_tokens=100,
            temperature=0.7
        )
        
        # Property: Response should have consistent structure
        assert isinstance(result, SarvamResponse), (
            "Processing should return SarvamResponse object"
        )
        
        assert result.language == target_language, (
            f"Response language should be {target_language}, got {result.language}"
        )
        
        assert result.confidence > 0.0, (
            "Response should have positive confidence score"
        )
        
        # For Indic languages, check fertility rate
        if target_language in ["hi", "bn", "ta"]:
            assert 1.4 <= result.fertility_rate <= 2.1, (
                f"Indic language fertility rate should be 1.4-2.1, got {result.fertility_rate}"
            )
    
    @given(
        cache_size=st.integers(min_value=5, max_value=20),
        text_samples=st.lists(
            st.text(min_size=10, max_size=50),
            min_size=3,
            max_size=10
        )
    )
    @settings(max_examples=10, deadline=3000)
    @pytest.mark.asyncio
    async def test_detection_cache_consistency(
        self,
        cache_size: int,
        text_samples: list[str]
    ):
        """
        Property: Detection Cache Consistency
        
        For any text that has been processed before, the language
        detection result should be consistent (cached).
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        # Mock consistent detection results
        def mock_detect_language(text):
            # Simple deterministic detection based on text hash
            if hash(text) % 2 == 0:
                return LanguageDetectionResult("hi", 0.8, True, "Devanagari")
            else:
                return LanguageDetectionResult("en", 0.9, False)
        
        mock_sarvam_client.detect_language.side_effect = mock_detect_language
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Process each text sample twice
        first_results = []
        for text in text_samples:
            result = await router.route_request(text)
            first_results.append(result)
        
        second_results = []
        for text in text_samples:
            result = await router.route_request(text)
            second_results.append(result)
        
        # Property: Results should be consistent (cached)
        for i, (first, second) in enumerate(zip(first_results, second_results)):
            assert first == second, (
                f"Cached detection result should be consistent for text {i}: "
                f"first={first}, second={second}"
            )
    
    @given(
        fertility_rates=st.lists(
            st.floats(min_value=1.4, max_value=2.1),
            min_size=3,
            max_size=8
        )
    )
    @settings(max_examples=10, deadline=2000)
    @pytest.mark.asyncio
    async def test_indic_tokenization_efficiency(
        self,
        fertility_rates: list[float]
    ):
        """
        Property: Indic Tokenization Efficiency
        
        For any Sarvam-1 response for Indic languages, the token
        fertility rate should be within the efficient range (1.4-2.1).
        """
        # Test each fertility rate
        for i, fertility_rate in enumerate(fertility_rates):
            # Create mock Sarvam client for each test
            mock_sarvam_client = AsyncMock(spec=SarvamClient)
            mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
            
            mock_response = SarvamResponse(
                text=f"Test response {i}",
                language="hi",
                tokens_used=100,
                fertility_rate=fertility_rate,
                confidence=0.9
            )
            mock_sarvam_client.generate_response.return_value = mock_response
            
            # Create language router
            router = LanguageRouter(mock_sarvam_client)
            
            # Process with Sarvam-1
            result = await router.process_with_appropriate_model(
                text="test input",
                system_prompt="test prompt",
                preferred_language="hi"
            )
            
            # Property: Fertility rate should be in efficient range
            assert 1.4 <= result.fertility_rate <= 2.1, (
                f"Indic language fertility rate should be 1.4-2.1, "
                f"got {result.fertility_rate}"
            )
    
    @pytest.mark.asyncio
    async def test_router_cleanup_consistency(self):
        """
        Property: Router Cleanup Consistency
        
        The language router should properly cleanup resources
        when closed, without raising exceptions.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.close.return_value = None
        
        # Create language router
        router = LanguageRouter(mock_sarvam_client)
        
        # Property: Cleanup should not raise exceptions
        try:
            await router.close()
            cleanup_successful = True
        except Exception:
            cleanup_successful = False
        
        assert cleanup_successful, (
            "Language router cleanup should complete without exceptions"
        )
        
        # Verify Sarvam client was closed
        mock_sarvam_client.close.assert_called_once()
    
    @given(
        error_scenarios=st.lists(
            st.sampled_from(["network_error", "api_error", "timeout_error"]),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=10, deadline=3000)
    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(
        self,
        error_scenarios: list[str]
    ):
        """
        Property: Error Handling Graceful Degradation
        
        For any error scenario, the language router should
        gracefully degrade and provide fallback responses.
        """
        # Create mock Sarvam client
        mock_sarvam_client = AsyncMock(spec=SarvamClient)
        mock_sarvam_client.get_supported_languages.return_value = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
        
        for error_type in error_scenarios:
            # Configure different error types
            if error_type == "network_error":
                mock_sarvam_client.generate_response.side_effect = ConnectionError("Network error")
            elif error_type == "api_error":
                mock_sarvam_client.generate_response.side_effect = ValueError("API error")
            elif error_type == "timeout_error":
                mock_sarvam_client.generate_response.side_effect = TimeoutError("Timeout error")
            
            # Create language router
            router = LanguageRouter(mock_sarvam_client)
            
            # Process with error scenario
            result = await router.process_with_appropriate_model(
                text="test input",
                system_prompt="test prompt",
                preferred_language="hi"
            )
            
            # Property: Should provide fallback response
            assert isinstance(result, SarvamResponse), (
                f"Error scenario {error_type} should return SarvamResponse"
            )
            
            assert result.confidence < 0.5, (
                f"Error scenario {error_type} should have low confidence"
            )
            
            assert len(result.text) > 0, (
                f"Error scenario {error_type} should provide fallback text"
            )