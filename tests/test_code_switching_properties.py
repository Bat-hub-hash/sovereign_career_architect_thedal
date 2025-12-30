"""Property-based tests for code-switching system."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateclasses import dataclass
from typing import Dict, List, Any
import asyncio

from sovereign_career_architect.cultural.code_switching import (
    CodeSwitchingManager, SwitchingContext, SwitchingTrigger, LanguageDominance
)
from sovereign_career_architect.cultural.adapter import CulturalProfile, CommunicationStyle, CulturalDimension


# Test data strategies
@st.composite
def switching_context_strategy(draw):
    """Generate valid switching contexts."""
    return SwitchingContext(
        conversation_type=draw(st.sampled_from([
            "professional", "casual", "family", "peer"
        ])),
        emotional_state=draw(st.sampled_from([
            "neutral", "excited", "stressed", "comfortable"
        ])),
        audience_familiarity=draw(st.sampled_from([
            "stranger", "acquaintance", "friend", "family"
        ])),
        topic_domain=draw(st.sampled_from([
            "technical", "personal", "cultural", "business"
        ])),
        formality_level=draw(st.sampled_from([
            "very_formal", "formal", "informal", "very_informal"
        ]))
    )


@st.composite
def cultural_profile_strategy(draw):
    """Generate valid cultural profiles for code-switching."""
    return CulturalProfile(
        primary_culture=draw(st.sampled_from(["indian", "american", "british"])),
        secondary_cultures=draw(st.lists(
            st.sampled_from(["indian", "american", "british"]), 
            max_size=2
        )),
        communication_styles=draw(st.lists(
            st.sampled_from(list(CommunicationStyle)),
            min_size=1,
            max_size=3,
            unique=True
        )),
        cultural_dimensions={
            dim: draw(st.floats(min_value=0.0, max_value=1.0))
            for dim in CulturalDimension
        },
        language_preferences={
            "en": draw(st.floats(min_value=0.5, max_value=1.0)),
            "hi": draw(st.floats(min_value=0.0, max_value=1.0)),
            "ta": draw(st.floats(min_value=0.0, max_value=1.0)),
            "te": draw(st.floats(min_value=0.0, max_value=1.0))
        },
        context_preferences={},
        adaptation_level=draw(st.floats(min_value=0.1, max_value=1.0))
    )


class TestCodeSwitchingProperties:
    """Property-based tests for code-switching functionality."""
    
    @pytest.fixture
    def code_switching_manager(self):
        """Create code-switching manager instance."""
        return CodeSwitchingManager()
    
    @given(
        english_text=st.text(min_size=20, max_size=300),
        target_language=st.sampled_from(["hi", "ta", "te"]),
        cultural_profile=cultural_profile_strategy(),
        context=switching_context_strategy()
    )
    @settings(max_examples=30, deadline=10000)
    def test_code_switched_text_generation_validity(
        self,
        code_switching_manager,
        english_text,
        target_language,
        cultural_profile,
        context
    ):
        """
        Property 32: Code-Switched Text Generation Validity
        
        Generated code-switched text should be valid and maintain
        the original message structure while incorporating target language elements.
        """
        assume(len(english_text.strip()) > 10)  # Ensure meaningful text
        assume(english_text.count(' ') >= 3)    # Ensure multiple words
        
        async def run_test():
            code_switched_text = await code_switching_manager.generate_code_switched_text(
                english_text, target_language, cultural_profile, context
            )
            
            # Generated text should not be empty
            assert len(code_switched_text.strip()) > 0
            
            # Should maintain reasonable length relationship
            length_ratio = len(code_switched_text) / len(english_text)
            assert 0.5 <= length_ratio <= 2.0, f"Length change too dramatic: {length_ratio}"
            
            # Should contain some words from both languages
            original_words = set(english_text.lower().split())
            switched_words = set(code_switched_text.lower().split())
            
            # Should preserve some original words (technical terms, etc.)
            preserved_words = original_words & switched_words
            preservation_ratio = len(preserved_words) / len(original_words) if original_words else 0
            assert preservation_ratio >= 0.2, f"Too few words preserved: {preservation_ratio}"
            
            # Should have some new words (from target language)
            new_words = switched_words - original_words
            assert len(new_words) > 0, "No code-switching detected"
            
            # Word count should be similar (within 50%)
            original_word_count = len(english_text.split())
            switched_word_count = len(code_switched_text.split())
            word_count_ratio = switched_word_count / original_word_count
            assert 0.5 <= word_count_ratio <= 1.5, f"Word count change too dramatic: {word_count_ratio}"
            
            return code_switched_text
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        mixed_text=st.text(min_size=30, max_size=200),
        target_language=st.sampled_from(["hi", "ta", "te"]),
        cultural_profile=cultural_profile_strategy()
    )
    @settings(max_examples=25, deadline=8000)
    def test_naturalness_analysis_consistency(
        self,
        code_switching_manager,
        mixed_text,
        target_language,
        cultural_profile
    ):
        """
        Property 33: Code-Switching Naturalness Analysis Consistency
        
        Naturalness analysis should provide consistent and meaningful
        assessments of code-switched text quality.
        """
        assume(len(mixed_text.strip()) > 15)  # Ensure meaningful text
        assume(mixed_text.count(' ') >= 5)    # Ensure multiple words
        
        async def run_test():
            analysis = await code_switching_manager.analyze_code_switching_naturalness(
                mixed_text, target_language, cultural_profile
            )
            
            # Should return proper analysis structure
            assert isinstance(analysis, dict)
            required_keys = [
                "naturalness_score", "switching_frequency", "language_balance",
                "issues", "suggestions"
            ]
            assert all(key in analysis for key in required_keys)
            
            # Naturalness score should be valid
            assert isinstance(analysis["naturalness_score"], (int, float))
            assert 0.0 <= analysis["naturalness_score"] <= 1.0
            
            # Switching frequency should be valid
            assert isinstance(analysis["switching_frequency"], (int, float))
            assert 0.0 <= analysis["switching_frequency"] <= 1.0
            
            # Language balance should be proper
            assert isinstance(analysis["language_balance"], dict)
            assert "english_ratio" in analysis["language_balance"]
            assert "native_ratio" in analysis["language_balance"]
            
            english_ratio = analysis["language_balance"]["english_ratio"]
            native_ratio = analysis["language_balance"]["native_ratio"]
            
            assert isinstance(english_ratio, (int, float))
            assert isinstance(native_ratio, (int, float))
            assert 0.0 <= english_ratio <= 1.0
            assert 0.0 <= native_ratio <= 1.0
            
            # Ratios should approximately sum to 1 (allowing for rounding)
            total_ratio = english_ratio + native_ratio
            assert 0.8 <= total_ratio <= 1.2, f"Language ratios don't sum properly: {total_ratio}"
            
            # Issues and suggestions should be lists of strings
            assert isinstance(analysis["issues"], list)
            assert isinstance(analysis["suggestions"], list)
            assert all(isinstance(issue, str) for issue in analysis["issues"])
            assert all(isinstance(suggestion, str) for suggestion in analysis["suggestions"])
            
            # If naturalness is low, should have issues or suggestions
            if analysis["naturalness_score"] < 0.5:
                assert len(analysis["issues"]) > 0 or len(analysis["suggestions"]) > 0
            
            return analysis
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        context=switching_context_strategy(),
        target_language=st.sampled_from(["hi", "ta", "te"])
    )
    @settings(max_examples=40, deadline=5000)
    def test_switching_recommendations_relevance(
        self,
        code_switching_manager,
        context,
        target_language
    ):
        """
        Property 34: Code-Switching Recommendations Relevance
        
        Switching recommendations should be relevant to the context
        and target language.
        """
        recommendations = code_switching_manager.get_switching_recommendations(
            context, target_language
        )
        
        # Should provide recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert len(recommendations) <= 5  # Should be limited
        
        # All recommendations should be non-empty strings
        assert all(isinstance(rec, str) and len(rec.strip()) > 0 
                  for rec in recommendations)
        
        # Recommendations should be unique
        assert len(recommendations) == len(set(recommendations))
        
        # Should contain relevant keywords based on context
        rec_text = " ".join(recommendations).lower()
        
        # Context-specific checks
        if context.conversation_type == "professional":
            assert any(word in rec_text for word in [
                "technical", "professional", "business", "english", "clarity"
            ])
        
        if context.conversation_type == "family":
            assert any(word in rec_text for word in [
                "family", "cultural", "native", "emotional", "comfortable"
            ])
        
        if context.formality_level in ["formal", "very_formal"]:
            assert any(word in rec_text for word in [
                "formal", "respectful", "honorific", "polite"
            ])
        
        # Language-specific checks
        if target_language == "hi":
            assert any(word in rec_text for word in [
                "hindi", "ji", "respect", "emotional"
            ])
        elif target_language == "ta":
            assert any(word in rec_text for word in [
                "tamil", "cultural", "family", "flow"
            ])
        elif target_language == "te":
            assert any(word in rec_text for word in [
                "telugu", "garu", "respectful", "emotional"
            ])
    
    @given(
        english_texts=st.lists(
            st.text(min_size=20, max_size=150),
            min_size=2,
            max_size=4
        ),
        target_language=st.sampled_from(["hi", "ta", "te"]),
        cultural_profile=cultural_profile_strategy(),
        context=switching_context_strategy()
    )
    @settings(max_examples=15, deadline=15000)
    def test_switching_consistency_across_texts(
        self,
        code_switching_manager,
        english_texts,
        target_language,
        cultural_profile,
        context
    ):
        """
        Property 35: Code-Switching Consistency Across Texts
        
        Code-switching should be consistent when applied to multiple
        texts in the same context and with the same parameters.
        """
        # Filter texts to ensure they're meaningful
        meaningful_texts = [
            text for text in english_texts 
            if len(text.strip()) > 15 and text.count(' ') >= 3
        ]
        assume(len(meaningful_texts) >= 2)
        
        async def run_test():
            switched_texts = []
            
            for text in meaningful_texts:
                switched = await code_switching_manager.generate_code_switched_text(
                    text, target_language, cultural_profile, context
                )
                switched_texts.append(switched)
            
            # All texts should be processed
            assert len(switched_texts) == len(meaningful_texts)
            assert all(len(text.strip()) > 0 for text in switched_texts)
            
            # Analyze consistency patterns
            analyses = []
            for switched_text in switched_texts:
                analysis = await code_switching_manager.analyze_code_switching_naturalness(
                    switched_text, target_language, cultural_profile
                )
                analyses.append(analysis)
            
            # Check for consistent switching patterns
            switching_frequencies = [a["switching_frequency"] for a in analyses]
            naturalness_scores = [a["naturalness_score"] for a in analyses]
            
            if len(switching_frequencies) > 1:
                # Switching frequencies should be reasonably consistent
                avg_freq = sum(switching_frequencies) / len(switching_frequencies)
                freq_variance = sum((f - avg_freq) ** 2 for f in switching_frequencies) / len(switching_frequencies)
                
                # Standard deviation should be reasonable (not too high variance)
                freq_std = freq_variance ** 0.5
                assert freq_std <= 0.3, f"Switching frequency too inconsistent: {freq_std}"
                
                # Naturalness scores should be reasonably consistent
                avg_naturalness = sum(naturalness_scores) / len(naturalness_scores)
                naturalness_variance = sum((n - avg_naturalness) ** 2 for n in naturalness_scores) / len(naturalness_scores)
                naturalness_std = naturalness_variance ** 0.5
                
                assert naturalness_std <= 0.4, f"Naturalness too inconsistent: {naturalness_std}"
            
            return switched_texts, analyses
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        technical_text=st.text(min_size=30, max_size=200).filter(
            lambda x: any(tech_word in x.lower() for tech_word in [
                "software", "computer", "algorithm", "database", "network",
                "application", "system", "technology", "programming", "code"
            ])
        ),
        target_language=st.sampled_from(["hi", "ta", "te"]),
        cultural_profile=cultural_profile_strategy()
    )
    @settings(max_examples=20, deadline=10000)
    def test_technical_term_preservation(
        self,
        code_switching_manager,
        technical_text,
        target_language,
        cultural_profile
    ):
        """
        Property 36: Technical Term Preservation in Code-Switching
        
        Technical terms should be preserved in English during code-switching
        to maintain clarity and professional communication.
        """
        assume(len(technical_text.strip()) > 20)
        
        # Create professional context
        context = SwitchingContext(
            conversation_type="professional",
            emotional_state="neutral",
            audience_familiarity="acquaintance",
            topic_domain="technical",
            formality_level="formal"
        )
        
        async def run_test():
            switched_text = await code_switching_manager.generate_code_switched_text(
                technical_text, target_language, cultural_profile, context
            )
            
            # Define technical terms that should be preserved
            technical_terms = [
                "software", "hardware", "computer", "internet", "email",
                "website", "database", "algorithm", "api", "framework",
                "application", "system", "network", "server", "cloud",
                "programming", "code", "technology", "digital"
            ]
            
            # Check preservation of technical terms
            original_words = technical_text.lower().split()
            switched_words = switched_text.lower().split()
            
            original_tech_terms = [word for word in original_words 
                                 if any(tech in word for tech in technical_terms)]
            switched_tech_terms = [word for word in switched_words 
                                 if any(tech in word for tech in technical_terms)]
            
            if original_tech_terms:
                # Most technical terms should be preserved
                preserved_tech = set(original_tech_terms) & set(switched_tech_terms)
                preservation_ratio = len(preserved_tech) / len(set(original_tech_terms))
                
                assert preservation_ratio >= 0.7, f"Too many technical terms lost: {preservation_ratio}"
            
            return switched_text
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()


if __name__ == "__main__":
    pytest.main([__file__])