"""Property-based tests for interview question generation."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from typing import List, Dict, Any

from sovereign_career_architect.interview.generator import (
    InterviewQuestionGenerator,
    InterviewQuestion,
    InterviewSession,
    InterviewType,
    DifficultyLevel
)


# Test data strategies
@st.composite
def role_strategy(draw):
    """Generate valid role names."""
    roles = [
        "software_engineer",
        "data_scientist", 
        "product_manager",
        "designer",
        "marketing_manager"
    ]
    return draw(st.sampled_from(roles))


@st.composite
def interview_config_strategy(draw):
    """Generate valid interview configurations."""
    return {
        "role": draw(role_strategy()),
        "interview_type": draw(st.sampled_from(list(InterviewType))),
        "difficulty": draw(st.sampled_from(list(DifficultyLevel))),
        "count": draw(st.integers(min_value=1, max_value=10)),
        "language": draw(st.sampled_from(["en", "hi", "ta", "te"]))
    }


class TestInterviewQuestionProperties:
    """Property-based tests for interview question generation."""
    
    def create_generator(self):
        """Create interview question generator."""
        return InterviewQuestionGenerator()
    
    @given(config=interview_config_strategy())
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_question_relevance_property(self, config):
        """
        Property 17: Question Relevance Generation
        
        For any valid role and interview configuration, generated questions should:
        1. Be relevant to the specified role
        2. Match the requested difficulty level
        3. Be appropriate for the interview type
        4. Include proper metadata and evaluation criteria
        """
        generator = self.create_generator()
        
        async def run_test():
            questions = await generator.generate_questions(
                role=config["role"],
                interview_type=config["interview_type"],
                difficulty=config["difficulty"],
                count=config["count"],
                language=config["language"]
            )
            
            # Property 1: Questions should be generated
            assert len(questions) == config["count"]
            
            for question in questions:
                # Property 2: Question should have valid structure
                assert isinstance(question, InterviewQuestion)
                assert question.question.strip()  # Non-empty question text
                assert question.role == config["role"]
                assert question.difficulty == config["difficulty"]
                assert question.language == config["language"]
                
                # Property 3: Question should have appropriate type
                if config["interview_type"] != InterviewType.MIXED:
                    # For specific types, questions should match or be compatible
                    if config["interview_type"] == InterviewType.TECHNICAL:
                        # Technical questions can be technical, coding, or system_design
                        assert question.type in [
                            InterviewType.TECHNICAL, 
                            InterviewType.CODING, 
                            InterviewType.SYSTEM_DESIGN
                        ] or question.category in ["technical", "system_design", "coding", "fundamentals"]
                    elif config["interview_type"] == InterviewType.BEHAVIORAL:
                        # Behavioral questions should be behavioral or have behavioral category
                        assert (question.type == InterviewType.BEHAVIORAL or 
                               question.category in ["behavioral", "leadership", "communication", "problem_solving"])
                
                # Property 4: Question should have proper metadata
                assert question.id  # Non-empty ID
                assert question.category  # Has category
                assert question.expected_duration_minutes > 0
                assert isinstance(question.follow_up_questions, list)
                assert isinstance(question.evaluation_criteria, list)
                assert isinstance(question.sample_answer_points, list)
                assert isinstance(question.tags, list)
                
                # Property 5: Evaluation criteria should be relevant
                assert len(question.evaluation_criteria) > 0
                for criterion in question.evaluation_criteria:
                    assert isinstance(criterion, str)
                    assert len(criterion.strip()) > 0
                
                # Property 6: Sample answer points should be provided
                assert len(question.sample_answer_points) > 0
                for point in question.sample_answer_points:
                    assert isinstance(point, str)
                    assert len(point.strip()) > 0
                
                # Property 7: Tags should include role and difficulty
                assert config["role"] in question.tags
                assert config["difficulty"].value in question.tags
                
                # Property 8: Duration should scale with difficulty
                if config["difficulty"] == DifficultyLevel.ENTRY:
                    assert question.expected_duration_minutes <= 25  # Allow for difficulty scaling
                elif config["difficulty"] == DifficultyLevel.EXECUTIVE:
                    assert question.expected_duration_minutes >= 8
        
        # Run async test
        asyncio.run(run_test())
    
    @given(
        role=role_strategy(),
        difficulty=st.sampled_from(list(DifficultyLevel))
    )
    @settings(max_examples=15, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_difficulty_scaling_property(self, role, difficulty):
        """
        Property: Difficulty Scaling Consistency
        
        Questions generated for higher difficulty levels should:
        1. Have longer expected durations
        2. Include more complex evaluation criteria
        3. Have appropriate complexity indicators
        """
        generator = self.create_generator()
        
        async def run_test():
            questions = await generator.generate_questions(
                role=role,
                interview_type=InterviewType.TECHNICAL,
                difficulty=difficulty,
                count=3
            )
            
            for question in questions:
                # Property 1: Duration should reflect difficulty
                if difficulty == DifficultyLevel.ENTRY:
                    assert question.expected_duration_minutes <= 20
                elif difficulty == DifficultyLevel.SENIOR:
                    assert question.expected_duration_minutes >= 8
                elif difficulty == DifficultyLevel.EXECUTIVE:
                    assert question.expected_duration_minutes >= 10
                
                # Property 2: Higher difficulty should have more evaluation criteria
                if difficulty in [DifficultyLevel.SENIOR, DifficultyLevel.EXECUTIVE]:
                    # Should have strategic thinking criteria
                    criteria_text = " ".join(question.evaluation_criteria).lower()
                    assert any(keyword in criteria_text for keyword in [
                        "strategic", "business", "leadership", "mentoring", "impact"
                    ])
                
                # Property 3: Question complexity should match difficulty
                question_text = question.question.lower()
                if difficulty == DifficultyLevel.EXECUTIVE:
                    # Executive questions should include strategic elements
                    assert any(keyword in question_text for keyword in [
                        "strategy", "leadership", "business", "scale", "trade-off"
                    ]) or "critically analyze" in question_text
        
        asyncio.run(run_test())
    
    @given(
        role=role_strategy(),
        interview_type=st.sampled_from([InterviewType.TECHNICAL, InterviewType.BEHAVIORAL])
    )
    @settings(max_examples=10, deadline=6000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_question_type_consistency_property(self, role, interview_type):
        """
        Property: Question Type Consistency
        
        Questions generated for a specific interview type should:
        1. Match the requested type or be compatible
        2. Have appropriate categories for the type
        3. Include relevant follow-up questions
        """
        generator = self.create_generator()
        
        async def run_test():
            questions = await generator.generate_questions(
                role=role,
                interview_type=interview_type,
                difficulty=DifficultyLevel.MID,
                count=3
            )
            
            for question in questions:
                if interview_type == InterviewType.TECHNICAL:
                    # Technical questions should have technical categories
                    technical_categories = [
                        "fundamentals", "system_design", "coding", "methodology", "experience"
                    ]
                    assert question.category in technical_categories
                    
                    # Should have technical follow-ups
                    follow_up_text = " ".join(question.follow_up_questions).lower()
                    assert any(keyword in follow_up_text for keyword in [
                        "implement", "optimize", "complexity", "test", "scale", "trade-off"
                    ])
                
                elif interview_type == InterviewType.BEHAVIORAL:
                    # Behavioral questions should have behavioral categories
                    behavioral_categories = [
                        "problem_solving", "leadership", "communication", "experience"
                    ]
                    assert question.category in behavioral_categories
                    
                    # Should have behavioral follow-ups
                    follow_up_text = " ".join(question.follow_up_questions).lower()
                    assert any(keyword in follow_up_text for keyword in [
                        "outcome", "learn", "different", "challenge", "team", "impact"
                    ])
        
        asyncio.run(run_test())


class TestInterviewSessionProperties:
    """Property-based tests for interview session management."""
    
    def create_generator(self):
        """Create interview question generator."""
        return InterviewQuestionGenerator()
    
    @given(
        role=role_strategy(),
        interview_type=st.sampled_from(list(InterviewType)),
        difficulty=st.sampled_from(list(DifficultyLevel)),
        duration=st.integers(min_value=15, max_value=120)
    )
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_session_creation_property(self, role, interview_type, difficulty, duration):
        """
        Property: Interview Session Creation Consistency
        
        Created interview sessions should:
        1. Have valid configuration matching inputs
        2. Generate appropriate number of questions for duration
        3. Have proper session management state
        """
        generator = self.create_generator()
        
        async def run_test():
            session = await generator.create_interview_session(
                role=role,
                company="Test Company",
                interview_type=interview_type,
                difficulty=difficulty,
                duration_minutes=duration
            )
            
            # Property 1: Session should have valid configuration
            assert session.role == role
            assert session.interview_type == interview_type
            assert session.difficulty == difficulty
            assert session.duration_minutes == duration
            assert session.company == "Test Company"
            assert session.session_id  # Non-empty session ID
            
            # Property 2: Should have appropriate number of questions
            assert len(session.questions) > 0
            if interview_type == InterviewType.MIXED:
                # Mixed interviews should have both technical and behavioral
                question_types = {q.type for q in session.questions}
                assert len(question_types) > 1  # Should have multiple types
            
            # Property 3: Session state should be properly initialized
            assert session.current_question_index == 0
            assert session.started_at is not None
            assert session.completed_at is None
            
            # Property 4: Questions should be appropriate for session
            for question in session.questions:
                assert question.role == role
                assert question.difficulty == difficulty
        
        asyncio.run(run_test())
    
    @given(
        role=role_strategy(),
        question_count=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=10, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_session_progression_property(self, role, question_count):
        """
        Property: Session Progression Consistency
        
        Session question progression should:
        1. Return questions in order
        2. Track current position correctly
        3. Handle session completion properly
        """
        generator = self.create_generator()
        
        async def run_test():
            # Create session with known question count
            session = await generator.create_interview_session(
                role=role,
                interview_type=InterviewType.TECHNICAL,
                difficulty=DifficultyLevel.MID,
                duration_minutes=60
            )
            
            # Manually set question count for testing
            session.questions = session.questions[:question_count]
            session.current_question_index = 0
            
            questions_retrieved = []
            
            # Property 1: Should retrieve all questions in order
            for i in range(question_count):
                question = await generator.get_next_question(session)
                assert question is not None
                assert question == session.questions[i]
                questions_retrieved.append(question)
                
                # Property 2: Index should be updated correctly
                assert session.current_question_index == i + 1
            
            # Property 3: Should return None when no more questions
            final_question = await generator.get_next_question(session)
            assert final_question is None
            
            # Property 4: Session completion should work correctly
            summary = await generator.complete_session(session)
            assert summary["session_id"] == session.session_id
            assert summary["questions_asked"] == question_count
            assert summary["total_questions"] == question_count
            assert summary["completion_rate"] == 1.0
            assert session.completed_at is not None
        
        asyncio.run(run_test())


class InterviewGeneratorStateMachine(RuleBasedStateMachine):
    """Stateful testing for interview generator."""
    
    def __init__(self):
        super().__init__()
        self.generator = InterviewQuestionGenerator()
        self.sessions = {}
        self.question_counts = {}
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        pass
    
    @rule(
        role=role_strategy(),
        interview_type=st.sampled_from(list(InterviewType)),
        difficulty=st.sampled_from(list(DifficultyLevel))
    )
    def create_session(self, role, interview_type, difficulty):
        """Create a new interview session."""
        async def run():
            session = await self.generator.create_interview_session(
                role=role,
                interview_type=interview_type,
                difficulty=difficulty,
                duration_minutes=45
            )
            
            self.sessions[session.session_id] = session
            self.question_counts[session.session_id] = len(session.questions)
        
        asyncio.run(run())
    
    @rule(session_id=st.data())
    def get_next_question(self, session_id):
        """Get next question from a session."""
        if not self.sessions:
            return
            
        session_id = session_id.draw(st.sampled_from(list(self.sessions.keys())))
        
        async def run():
            session = self.sessions[session_id]
            question = await self.generator.get_next_question(session)
        
        asyncio.run(run())
    
    @invariant()
    def session_consistency_invariant(self):
        """Invariant: All sessions should maintain consistency."""
        for session_id, session in self.sessions.items():
            # Session should have valid state
            assert session.current_question_index >= 0
            assert session.current_question_index <= len(session.questions)
            
            # Question count should match recorded count
            assert len(session.questions) == self.question_counts[session_id]
            
            # All questions should be valid
            for question in session.questions:
                assert isinstance(question, InterviewQuestion)
                assert question.role == session.role
                assert question.difficulty == session.difficulty


# Create the stateful test
TestInterviewGeneratorStateful = InterviewGeneratorStateMachine.TestCase


if __name__ == "__main__":
    # Run a quick test
    generator = InterviewQuestionGenerator()
    
    async def test_basic():
        questions = await generator.generate_questions(
            role="software_engineer",
            interview_type=InterviewType.TECHNICAL,
            difficulty=DifficultyLevel.MID,
            count=3
        )
        
        print(f"Generated {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q.question}")
            print(f"   Category: {q.category}, Duration: {q.expected_duration_minutes}min")
            print(f"   Follow-ups: {len(q.follow_up_questions)}")
            print()
    
    asyncio.run(test_basic())