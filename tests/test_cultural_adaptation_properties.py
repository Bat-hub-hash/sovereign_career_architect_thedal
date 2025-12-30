"""Property-based tests for cultural adaptation system."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateclasses import dataclass
from typing import Dict, List, Any
import asyncio

from sovereign_career_architect.cultural.adapter import (
    CulturalAdapter, CulturalProfile, CulturalContext, CommunicationStyle,
    CulturalDimension
)
from sovereign_career_architect.core.models import UserProfile, PersonalInfo


# Test data strategies
@st.composite
def cultural_profile_strategy(draw):
    """Generate valid cultural profiles."""
    cultures = ["indian", "american", "british", "german"]
    primary_culture = draw(st.sampled_from(cultures))
    
    communication_styles = draw(
        st.lists(
            st.sampled_from(list(CommunicationStyle)),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    
    cultural_dimensions = {
        dim: draw(st.floats(min_value=0.0, max_value=1.0))
        for dim in CulturalDimension
    }
    
    language_preferences = {
        "en": draw(st.floats(min_value=0.5, max_value=1.0)),
        "hi": draw(st.floats(min_value=0.0, max_value=1.0)),
        "ta": draw(st.floats(min_value=0.0, max_value=1.0))
    }
    
    return CulturalProfile(
        primary_culture=primary_culture,
        secondary_cultures=draw(st.lists(st.sampled_from(cultures), max_size=2)),
        communication_styles=communication_styles,
        cultural_dimensions=cultural_dimensions,
        language_preferences=language_preferences,
        context_preferences={},
        adaptation_level=draw(st.floats(min_value=0.1, max_value=1.0))
    )


@st.composite
def cultural_context_strategy(draw):
    """Generate valid cultural contexts."""
    return CulturalContext(
        situation_type=draw(st.sampled_from([
            "interview", "networking", "application", "casual"
        ])),
        formality_level=draw(st.sampled_from([
            "very_formal", "formal", "semi_formal", "casual"
        ])),
        audience=draw(st.sampled_from([
            "recruiter", "hiring_manager", "peer", "senior_executive"
        ])),
        cultural_background=draw(st.one_of(
            st.none(),
            st.sampled_from(["indian", "american", "british", "german"])
        )),
        relationship_level=draw(st.sampled_from([
            "new", "acquaintance", "familiar", "close"
        ]))
    )


@st.composite
def user_profile_strategy(draw):
    """Generate valid user profiles."""
    return UserProfile(
        personal_info=PersonalInfo(
            name=draw(st.text(min_size=1, max_size=50)),
            email=draw(st.emails()),
            phone=draw(st.one_of(st.none(), st.text(min_size=10, max_size=15))),
            location=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
            preferred_language=draw(st.sampled_from(["en", "hi", "ta", "te"]))
        ),
        skills=[],
        experience=[],
        education=[],
        preferences={},
        documents=None
    )


class TestCulturalAdaptationProperties:
    """Property-based tests for cultural adaptation."""
    
    @pytest.fixture
    def cultural_adapter(self):
        """Create cultural adapter instance."""
        return CulturalAdapter()
    
    @given(
        user_profile=user_profile_strategy(),
        cultural_background=st.sampled_from(["indian", "american", "british", "german"]),
        communication_preferences=st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers(), st.floats()),
                max_size=5
            )
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_cultural_profile_creation_consistency(
        self,
        cultural_adapter,
        user_profile,
        cultural_background,
        communication_preferences
    ):
        """
        Property 31: Cultural Profile Creation Consistency
        
        Cultural profiles should be created consistently with valid data
        and reflect the specified cultural background.
        """
        async def run_test():
            cultural_profile = await cultural_adapter.create_cultural_profile(
                user_profile, cultural_background, communication_preferences
            )
            
            # Profile should have the specified primary culture
            assert cultural_profile.primary_culture == cultural_background
            
            # Should have valid communication styles
            assert len(cultural_profile.communication_styles) > 0
            assert all(isinstance(style, CommunicationStyle) 
                      for style in cultural_profile.communication_styles)
            
            # Should have cultural dimensions
            assert len(cultural_profile.cultural_dimensions) > 0
            assert all(0.0 <= value <= 1.0 
                      for value in cultural_profile.cultural_dimensions.values())
            
            # Should include user's preferred language
            user_lang = user_profile.personal_info.preferred_language
            assert user_lang in cultural_profile.language_preferences
            assert cultural_profile.language_preferences[user_lang] > 0.0
            
            # Adaptation level should be valid
            assert 0.0 <= cultural_profile.adaptation_level <= 1.0
            
            return cultural_profile
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        message=st.text(min_size=10, max_size=500),
        cultural_profile=cultural_profile_strategy(),
        context=cultural_context_strategy(),
        target_culture=st.one_of(
            st.none(),
            st.sampled_from(["indian", "american", "british", "german"])
        )
    )
    @settings(max_examples=30, deadline=10000)
    def test_message_adaptation_preserves_meaning(
        self,
        cultural_adapter,
        message,
        cultural_profile,
        context,
        target_culture
    ):
        """
        Property 32: Message Adaptation Meaning Preservation
        
        Adapted messages should preserve the core meaning while
        adjusting cultural elements appropriately.
        """
        assume(len(message.strip()) > 5)  # Ensure meaningful message
        
        async def run_test():
            adapted_message = await cultural_adapter.adapt_message(
                message, cultural_profile, context, target_culture
            )
            
            # Adapted message should not be empty
            assert len(adapted_message.strip()) > 0
            
            # Should preserve key content words (simplified check)
            original_words = set(message.lower().split())
            adapted_words = set(adapted_message.lower().split())
            
            # At least 30% of content words should be preserved
            # (accounting for cultural adaptations)
            content_words = original_words - {
                "the", "a", "an", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "with", "by", "from", "is", "are", "was", "were"
            }
            
            if len(content_words) > 0:
                preserved_ratio = len(content_words & adapted_words) / len(content_words)
                assert preserved_ratio >= 0.3, f"Too much content lost: {preserved_ratio}"
            
            # Length should not change dramatically (within 3x)
            length_ratio = len(adapted_message) / len(message)
            assert 0.3 <= length_ratio <= 3.0, f"Length change too dramatic: {length_ratio}"
            
            return adapted_message
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        cultural_profile=cultural_profile_strategy(),
        context=cultural_context_strategy(),
        target_culture=st.sampled_from(["indian", "american", "british", "german"])
    )
    @settings(max_examples=40, deadline=5000)
    def test_cultural_recommendations_relevance(
        self,
        cultural_adapter,
        cultural_profile,
        context,
        target_culture
    ):
        """
        Property 33: Cultural Recommendations Relevance
        
        Cultural recommendations should be relevant to the context
        and target culture.
        """
        async def run_test():
            recommendations = await cultural_adapter.get_cultural_recommendations(
                cultural_profile, context, target_culture
            )
            
            # Should provide recommendations
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            assert len(recommendations) <= 5  # Should be limited
            
            # All recommendations should be strings
            assert all(isinstance(rec, str) for rec in recommendations)
            assert all(len(rec.strip()) > 0 for rec in recommendations)
            
            # Recommendations should be unique
            assert len(recommendations) == len(set(recommendations))
            
            # Should contain relevant keywords based on context
            rec_text = " ".join(recommendations).lower()
            
            if context.situation_type == "interview":
                assert any(word in rec_text for word in [
                    "interview", "question", "answer", "professional", "confident"
                ])
            
            if context.formality_level in ["formal", "very_formal"]:
                assert any(word in rec_text for word in [
                    "formal", "respect", "polite", "professional", "appropriate"
                ])
            
            if target_culture == "indian":
                assert any(word in rec_text for word in [
                    "respect", "sir", "madam", "family", "hierarchy"
                ])
            elif target_culture == "american":
                assert any(word in rec_text for word in [
                    "direct", "confident", "individual", "achievement"
                ])
            
            return recommendations
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            assert result is not None
        finally:
            loop.close()
    
    @given(
        user_message=st.text(min_size=20, max_size=200),
        cultural_profile=cultural_profile_strategy(),
        target_culture=st.sampled_from(["indian", "american", "british", "german"])
    )
    @settings(max_examples=25, deadline=8000)
    def test_cultural_fit_analysis_consistency(
        self,
        cultural_adapter,
        user_message,
        cultural_profile,
        target_culture
    ):
        """
        Property 34: Cultural Fit Analysis Consistency
        
        Cultural fit analysis should provide consistent and meaningful
        assessments of message appropriateness.
        """
        assume(len(user_message.strip()) > 10)  # Ensure meaningful message
        
        async def run_test():
            analysis = await cultural_adapter.analyze_cultural_fit(
                user_message, cultural_profile, target_culture
            )
            
            # Should return proper analysis structure
            assert isinstance(analysis, dict)
            required_keys = [
                "overall_fit", "strengths", "areas_for_improvement", "specific_suggestions"
            ]
            assert all(key in analysis for key in required_keys)
            
            # Overall fit should be a valid score
            assert isinstance(analysis["overall_fit"], (int, float))
            assert 0.0 <= analysis["overall_fit"] <= 1.0
            
            # Lists should be proper types
            assert isinstance(analysis["strengths"], list)
            assert isinstance(analysis["areas_for_improvement"], list)
            assert isinstance(analysis["specific_suggestions"], list)
            
            # All text items should be non-empty strings
            for item_list in [analysis["strengths"], analysis["areas_for_improvement"], 
                             analysis["specific_suggestions"]]:
                assert all(isinstance(item, str) and len(item.strip()) > 0 
                          for item in item_list)
            
            # If fit is low, should provide suggestions
            if analysis["overall_fit"] < 0.6:
                assert len(analysis["areas_for_improvement"]) > 0 or \
                       len(analysis["specific_suggestions"]) > 0
            
            # If fit is high, should have strengths
            if analysis["overall_fit"] > 0.8:
                assert len(analysis["strengths"]) > 0
            
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
        messages=st.lists(
            st.text(min_size=10, max_size=100),
            min_size=2,
            max_size=5
        ),
        cultural_profile=cultural_profile_strategy(),
        context=cultural_context_strategy()
    )
    @settings(max_examples=20, deadline=10000)
    def test_adaptation_consistency_across_messages(
        self,
        cultural_adapter,
        messages,
        cultural_profile,
        context
    ):
        """
        Property 35: Adaptation Consistency Across Messages
        
        Cultural adaptation should be consistent when applied to
        multiple messages in the same context.
        """
        async def run_test():
            adapted_messages = []
            
            for message in messages:
                adapted = await cultural_adapter.adapt_message(
                    message, cultural_profile, context
                )
                adapted_messages.append(adapted)
            
            # All messages should be adapted
            assert len(adapted_messages) == len(messages)
            assert all(len(msg.strip()) > 0 for msg in adapted_messages)
            
            # Check for consistent adaptation patterns
            # (This is a simplified check - in practice, you'd look for
            # consistent use of politeness markers, formality levels, etc.)
            
            # If formal context, all adaptations should maintain formality
            if context.formality_level in ["formal", "very_formal"]:
                formal_indicators = ["dear", "sincerely", "respectfully", "please"]
                
                formal_counts = []
                for adapted in adapted_messages:
                    count = sum(1 for indicator in formal_indicators 
                              if indicator in adapted.lower())
                    formal_counts.append(count)
                
                # Should have some consistency in formality usage
                if len(formal_counts) > 1:
                    avg_formal = sum(formal_counts) / len(formal_counts)
                    # Most messages should be within 1 formal marker of average
                    within_range = sum(1 for count in formal_counts 
                                     if abs(count - avg_formal) <= 1)
                    consistency_ratio = within_range / len(formal_counts)
                    assert consistency_ratio >= 0.6, f"Inconsistent formality: {consistency_ratio}"
            
            return adapted_messages
        
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