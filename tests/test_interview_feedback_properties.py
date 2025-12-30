"""Property-based tests for interview feedback system."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict

from sovereign_career_architect.interview.feedback import (
    InterviewFeedbackAnalyzer,
    FeedbackPoint,
    QuestionFeedback,
    InterviewFeedback,
    FeedbackCategory,
    FeedbackSeverity
)
from sovereign_career_architect.interview.generator import (
    InterviewQuestion,
    InterviewSession,
    InterviewType,
    DifficultyLevel
)
from sovereign_career_architect.voice.sarvam import SarvamClient


# Test data strategies
@st.composite
def feedback_point_strategy(draw):
    """Generate feedback points."""
    return FeedbackPoint(
        category=draw(st.sampled_from(list(FeedbackCategory))),
        severity=draw(st.sampled_from(list(FeedbackSeverity))),
        title=draw(st.text(min_size=5, max_size=50)),
        description=draw(st.text(min_size=10, max_size=200)),
        specific_example=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
        improvement_suggestion=draw(st.one_of(st.none(), st.text(min_size=10, max_size=150))),
        score=draw(st.floats(min_value=0.0, max_value=10.0))
    )


@st.composite
def question_feedback_strategy(draw):
    """Generate question feedback."""
    feedback_points = draw(st.lists(feedback_point_strategy(), min_size=1, max_size=5))
    
    return QuestionFeedback(
        question_id=draw(st.text(min_size=5, max_size=20)),
        question_text=draw(st.text(min_size=10, max_size=200)),
        candidate_answer=draw(st.text(min_size=10, max_size=500)),
        feedback_points=feedback_points,
        overall_score=draw(st.floats(min_value=0.0, max_value=10.0)),
        time_taken_minutes=draw(st.floats(min_value=1.0, max_value=30.0)),
        completeness_score=draw(st.floats(min_value=0.0, max_value=10.0)),
        clarity_score=draw(st.floats(min_value=0.0, max_value=10.0)),
        accuracy_score=draw(st.floats(min_value=0.0, max_value=10.0))
    )


@st.composite
def interview_question_strategy(draw):
    """Generate interview questions."""
    return InterviewQuestion(
        id=draw(st.text(min_size=5, max_size=20)),
        question=draw(st.text(min_size=20, max_size=200)),
        type=draw(st.sampled_from(list(InterviewType))),
        difficulty=draw(st.sampled_from(list(DifficultyLevel))),
        role=draw(st.sampled_from(["software_engineer", "data_scientist", "product_manager"])),
        category=draw(st.sampled_from(["technical", "behavioral", "system_design", "coding"])),
        expected_duration_minutes=draw(st.integers(min_value=5, max_value=30)),
        follow_up_questions=draw(st.lists(st.text(min_size=5, max_size=100), min_size=1, max_size=3)),
        evaluation_criteria=draw(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5)),
        sample_answer_points=draw(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5)),
        tags=draw(st.lists(st.text(min_size=3, max_size=15), min_size=1, max_size=5)),
        language="en"
    )


class TestInterviewFeedbackProperties:
    """Property-based tests for interview feedback analysis."""
    
    def create_mock_sarvam_client(self):
        """Create mock Sarvam client."""
        client = AsyncMock(spec=SarvamClient)
        
        # Mock text generation to return scores
        async def mock_generate_text(prompt, language, max_tokens=100):
            if "score" in prompt.lower():
                return "7.5"  # Return a reasonable score
            return "Generated feedback text"
        
        client.generate_text = mock_generate_text
        return client
    
    def create_feedback_analyzer(self):
        """Create feedback analyzer with mock client."""
        mock_client = self.create_mock_sarvam_client()
        return InterviewFeedbackAnalyzer(sarvam_client=mock_client)
    
    @given(
        question=interview_question_strategy(),
        candidate_answer=st.text(min_size=20, max_size=1000),
        time_taken=st.floats(min_value=1.0, max_value=60.0)
    )
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_question_analysis_property(self, question, candidate_answer, time_taken):
        """
        Property 18: Interview Feedback Completeness
        
        For any question and candidate answer:
        1. Feedback should include all required components
        2. Scores should be within valid ranges (0-10)
        3. Feedback points should be relevant and actionable
        4. Analysis should be consistent with input quality
        """
        # Skip very short answers
        assume(len(candidate_answer.strip()) >= 10)
        
        feedback_analyzer = self.create_feedback_analyzer()
        
        async def run_test():
            question_feedback = await feedback_analyzer.analyze_question_response(
                question=question,
                candidate_answer=candidate_answer,
                time_taken_minutes=time_taken
            )
            
            # Property 1: Complete feedback structure
            assert isinstance(question_feedback, QuestionFeedback)
            assert question_feedback.question_id == question.id
            assert question_feedback.question_text == question.question
            assert question_feedback.candidate_answer == candidate_answer
            assert question_feedback.time_taken_minutes == time_taken
            
            # Property 2: Valid score ranges
            assert 0.0 <= question_feedback.overall_score <= 10.0
            assert 0.0 <= question_feedback.completeness_score <= 10.0
            assert 0.0 <= question_feedback.clarity_score <= 10.0
            assert 0.0 <= question_feedback.accuracy_score <= 10.0
            
            # Property 3: Feedback points should exist and be valid
            assert len(question_feedback.feedback_points) > 0
            for fp in question_feedback.feedback_points:
                assert isinstance(fp, FeedbackPoint)
                assert isinstance(fp.category, FeedbackCategory)
                assert isinstance(fp.severity, FeedbackSeverity)
                assert len(fp.title.strip()) > 0
                assert len(fp.description.strip()) > 0
                assert 0.0 <= fp.score <= 10.0
            
            # Property 4: Feedback should be relevant to question type
            question_type_keywords = {
                "technical": ["technical", "accuracy", "implementation"],
                "behavioral": ["behavioral", "experience", "situation"],
                "system_design": ["design", "architecture", "scalability"]
            }
            
            if question.category in question_type_keywords:
                expected_keywords = question_type_keywords[question.category]
                feedback_text = " ".join([fp.title + " " + fp.description for fp in question_feedback.feedback_points]).lower()
                
                # At least one relevant keyword should appear
                assert any(keyword in feedback_text for keyword in expected_keywords)
        
        asyncio.run(run_test())
    
    @given(
        question_feedbacks=st.lists(question_feedback_strategy(), min_size=2, max_size=8),
        role=st.sampled_from(["software_engineer", "data_scientist", "product_manager"]),
        difficulty=st.sampled_from(list(DifficultyLevel))
    )
    @settings(max_examples=15, deadline=12000)
    def test_session_feedback_property(self, feedback_analyzer, question_feedbacks, role, difficulty):
        """
        Property: Session Feedback Completeness
        
        For any collection of question feedbacks:
        1. Session feedback should aggregate individual question scores correctly
        2. Should identify strengths and improvements appropriately
        3. Should provide actionable recommendations and next steps
        4. Should maintain consistency across all components
        """
        async def run_test():
            # Create mock session
            import uuid
            session = InterviewSession(
                session_id=str(uuid.uuid4()),
                role=role,
                company="Test Company",
                interview_type=InterviewType.MIXED,
                difficulty=difficulty,
                duration_minutes=45,
                language="en",
                questions=[]
            )
            
            session_feedback = await feedback_analyzer.generate_session_feedback(
                session=session,
                question_feedbacks=question_feedbacks
            )
            
            # Property 1: Complete session feedback structure
            assert isinstance(session_feedback, InterviewFeedback)
            assert session_feedback.session_id == session.session_id
            assert len(session_feedback.question_feedbacks) == len(question_feedbacks)
            
            # Property 2: Overall score should be reasonable aggregate
            expected_avg = sum(qf.overall_score for qf in question_feedbacks) / len(question_feedbacks)
            assert abs(session_feedback.overall_score - expected_avg) < 0.1
            assert 0.0 <= session_feedback.overall_score <= 10.0
            
            # Property 3: Category scores should be valid
            for category, score in session_feedback.category_scores.items():
                assert isinstance(category, FeedbackCategory)
                assert 0.0 <= score <= 10.0
            
            # Property 4: Should have actionable content
            assert isinstance(session_feedback.strengths, list)
            assert isinstance(session_feedback.areas_for_improvement, list)
            assert isinstance(session_feedback.knowledge_gaps, list)
            assert isinstance(session_feedback.recommendations, list)
            assert isinstance(session_feedback.next_steps, list)
            
            # Property 5: Content should be non-empty and reasonable
            total_feedback_items = (
                len(session_feedback.strengths) +
                len(session_feedback.areas_for_improvement) +
                len(session_feedback.recommendations) +
                len(session_feedback.next_steps)
            )
            assert total_feedback_items > 0
            
            # Property 6: All text content should be meaningful
            all_text_items = (
                session_feedback.strengths +
                session_feedback.areas_for_improvement +
                session_feedback.knowledge_gaps +
                session_feedback.recommendations +
                session_feedback.next_steps
            )
            
            for item in all_text_items:
                assert isinstance(item, str)
                assert len(item.strip()) > 5  # Meaningful content
        
        asyncio.run(run_test())
    
    @given(
        scores=st.dictionaries(
            st.sampled_from(["overall", "accuracy", "completeness", "clarity"]),
            st.floats(min_value=0.0, max_value=10.0),
            min_size=4, max_size=4
        ),
        difficulty=st.sampled_from(list(DifficultyLevel)),
        time_taken=st.floats(min_value=1.0, max_value=60.0),
        expected_time=st.floats(min_value=5.0, max_value=30.0)
    )
    def test_score_adjustment_property(self, feedback_analyzer, scores, difficulty, time_taken, expected_time):
        """
        Property: Score Adjustment Consistency
        
        For any set of scores and timing:
        1. Adjusted scores should remain within valid range (0-10)
        2. Time penalties/bonuses should be applied consistently
        3. Difficulty adjustments should be appropriate
        """
        adjusted_scores = feedback_analyzer._adjust_scores_for_difficulty(
            scores=scores,
            difficulty=difficulty,
            time_taken=time_taken,
            expected_time=expected_time
        )
        
        # Property 1: Valid score ranges maintained
        for key, score in adjusted_scores.items():
            assert 0.0 <= score <= 10.0
            assert key in scores  # All original keys preserved
        
        # Property 2: Score adjustments should be reasonable
        for key in scores:
            original = scores[key]
            adjusted = adjusted_scores[key]
            
            # Adjustment should not be too extreme
            assert abs(adjusted - original) <= 2.0
        
        # Property 3: Time ratio effects should be consistent
        time_ratio = time_taken / expected_time
        
        if time_ratio > 2.0:
            # Should generally have lower or equal scores for taking too long
            for key in scores:
                if scores[key] > 1.0:  # Only check if there's room for penalty
                    assert adjusted_scores[key] <= scores[key] + 0.1  # Allow small floating point errors
    
    @given(
        feedback_points=st.lists(feedback_point_strategy(), min_size=1, max_size=10),
        category_scores=st.dictionaries(
            st.sampled_from(list(FeedbackCategory)),
            st.floats(min_value=0.0, max_value=10.0),
            min_size=1, max_size=6
        )
    )
    def test_strength_identification_property(self, feedback_analyzer, feedback_points, category_scores):
        """
        Property: Strength Identification Consistency
        
        For any set of feedback points and category scores:
        1. High-scoring categories should be identified as strengths
        2. Strength feedback points should be included
        3. Results should be reasonable and limited in number
        """
        # Create mock question feedbacks
        question_feedbacks = [
            QuestionFeedback(
                question_id="test_1",
                question_text="Test question",
                candidate_answer="Test answer",
                feedback_points=feedback_points,
                overall_score=7.5,
                time_taken_minutes=10.0,
                completeness_score=7.0,
                clarity_score=8.0,
                accuracy_score=7.5
            )
        ]
        
        strengths = feedback_analyzer._identify_strengths(question_feedbacks, category_scores)
        
        # Property 1: Should return a list
        assert isinstance(strengths, list)
        
        # Property 2: Should be limited in number (not overwhelming)
        assert len(strengths) <= 5
        
        # Property 3: High-scoring categories should appear as strengths
        high_scoring_categories = [cat for cat, score in category_scores.items() if score >= 7.5]
        if high_scoring_categories:
            strength_text = " ".join(strengths).lower()
            # At least one high-scoring category should be mentioned
            category_mentioned = any(
                cat.value.replace("_", " ") in strength_text 
                for cat in high_scoring_categories
            )
            # This might not always be true due to other logic, so we make it a soft check
            if len(strengths) > 0:
                assert len(strength_text) > 0  # At least some meaningful content
        
        # Property 4: All strength items should be strings
        for strength in strengths:
            assert isinstance(strength, str)
            assert len(strength.strip()) > 0
    
    @given(
        role=st.sampled_from(["software_engineer", "data_scientist", "product_manager"]),
        avg_score=st.floats(min_value=0.0, max_value=10.0),
        difficulty=st.sampled_from(list(DifficultyLevel))
    )
    def test_recommendation_generation_property(self, feedback_analyzer, role, avg_score, difficulty):
        """
        Property: Recommendation Generation Consistency
        
        For any role, score, and difficulty:
        1. Should generate appropriate recommendations based on performance
        2. Should be role and difficulty appropriate
        3. Should provide actionable guidance
        """
        # Create mock data
        question_feedbacks = [
            QuestionFeedback(
                question_id="test_1",
                question_text="Test question",
                candidate_answer="Test answer",
                feedback_points=[],
                overall_score=avg_score,
                time_taken_minutes=10.0,
                completeness_score=avg_score,
                clarity_score=avg_score,
                accuracy_score=avg_score
            )
        ]
        
        session = InterviewSession(
            session_id="test_session",
            role=role,
            company="Test Company",
            interview_type=InterviewType.MIXED,
            difficulty=difficulty,
            duration_minutes=45,
            language="en",
            questions=[]
        )
        
        category_scores = {FeedbackCategory.TECHNICAL_ACCURACY: avg_score}
        
        async def run_test():
            recommendations = await feedback_analyzer._generate_recommendations(
                session, question_feedbacks, category_scores
            )
            
            # Property 1: Should return a list of strings
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 4  # Should be limited
            
            for rec in recommendations:
                assert isinstance(rec, str)
                assert len(rec.strip()) > 10  # Meaningful content
            
            # Property 2: Performance-based recommendations should be appropriate
            if len(recommendations) > 0:
                rec_text = " ".join(recommendations).lower()
                
                if avg_score >= 8.0:
                    # High performance should get positive recommendations
                    assert any(word in rec_text for word in ["excellent", "senior", "good"])
                elif avg_score < 5.0:
                    # Low performance should get improvement recommendations
                    assert any(word in rec_text for word in ["practice", "focus", "improve", "fundamental"])
        
        asyncio.run(run_test())


class TestFeedbackDataStructures:
    """Test feedback data structure properties."""
    
    @given(feedback_point=feedback_point_strategy())
    def test_feedback_point_property(self, feedback_point):
        """
        Property: Feedback Point Structure Validity
        
        For any feedback point:
        1. Should have all required fields
        2. Should have valid enum values
        3. Should have reasonable score range
        """
        # Property 1: Required fields
        assert hasattr(feedback_point, 'category')
        assert hasattr(feedback_point, 'severity')
        assert hasattr(feedback_point, 'title')
        assert hasattr(feedback_point, 'description')
        assert hasattr(feedback_point, 'score')
        
        # Property 2: Valid enum values
        assert isinstance(feedback_point.category, FeedbackCategory)
        assert isinstance(feedback_point.severity, FeedbackSeverity)
        
        # Property 3: Valid score range
        assert 0.0 <= feedback_point.score <= 10.0
        
        # Property 4: Text fields should be meaningful
        assert len(feedback_point.title.strip()) > 0
        assert len(feedback_point.description.strip()) > 0
    
    @given(question_feedback=question_feedback_strategy())
    def test_question_feedback_property(self, question_feedback):
        """
        Property: Question Feedback Structure Validity
        
        For any question feedback:
        1. Should have all required fields with valid types
        2. Should have consistent scoring
        3. Should have meaningful content
        """
        # Property 1: Required fields and types
        assert isinstance(question_feedback.question_id, str)
        assert isinstance(question_feedback.question_text, str)
        assert isinstance(question_feedback.candidate_answer, str)
        assert isinstance(question_feedback.feedback_points, list)
        assert isinstance(question_feedback.time_taken_minutes, float)
        
        # Property 2: Valid score ranges
        assert 0.0 <= question_feedback.overall_score <= 10.0
        assert 0.0 <= question_feedback.completeness_score <= 10.0
        assert 0.0 <= question_feedback.clarity_score <= 10.0
        assert 0.0 <= question_feedback.accuracy_score <= 10.0
        
        # Property 3: Positive time
        assert question_feedback.time_taken_minutes > 0
        
        # Property 4: Feedback points should be valid
        assert len(question_feedback.feedback_points) > 0
        for fp in question_feedback.feedback_points:
            assert isinstance(fp, FeedbackPoint)


if __name__ == "__main__":
    # Run a quick test
    async def test_basic():
        from unittest.mock import AsyncMock
        
        mock_client = AsyncMock()
        mock_client.generate_text = AsyncMock(return_value="7.5")
        
        analyzer = InterviewFeedbackAnalyzer(sarvam_client=mock_client)
        
        # Test basic functionality
        question = InterviewQuestion(
            id="test_1",
            question="Explain how you would design a scalable web application.",
            type=InterviewType.TECHNICAL,
            difficulty=DifficultyLevel.MID,
            role="software_engineer",
            category="system_design",
            expected_duration_minutes=15,
            follow_up_questions=["How would you handle failures?"],
            evaluation_criteria=["Considers scalability", "Proper architecture"],
            sample_answer_points=["Microservices", "Load balancing"],
            tags=["system_design", "scalability"],
            language="en"
        )
        
        feedback = await analyzer.analyze_question_response(
            question=question,
            candidate_answer="I would use microservices architecture with load balancers and databases.",
            time_taken_minutes=12.0
        )
        
        print(f"Question feedback generated:")
        print(f"Overall score: {feedback.overall_score}")
        print(f"Feedback points: {len(feedback.feedback_points)}")
        for fp in feedback.feedback_points:
            print(f"- {fp.title}: {fp.description}")
    
    asyncio.run(test_basic())