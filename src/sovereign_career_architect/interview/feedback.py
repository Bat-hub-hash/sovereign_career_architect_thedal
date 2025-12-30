"""Interview feedback and analysis system."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import structlog

from sovereign_career_architect.interview.generator import (
    InterviewQuestion,
    InterviewSession,
    DifficultyLevel
)
from sovereign_career_architect.voice.sarvam import SarvamClient
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackCategory(Enum):
    """Categories of interview feedback."""
    TECHNICAL_ACCURACY = "technical_accuracy"
    COMMUNICATION = "communication"
    PROBLEM_SOLVING = "problem_solving"
    BEHAVIORAL = "behavioral"
    LEADERSHIP = "leadership"
    CULTURAL_FIT = "cultural_fit"


class FeedbackSeverity(Enum):
    """Severity levels for feedback points."""
    STRENGTH = "strength"
    IMPROVEMENT = "improvement"
    CRITICAL = "critical"


@dataclass
class FeedbackPoint:
    """Individual feedback point."""
    category: FeedbackCategory
    severity: FeedbackSeverity
    title: str
    description: str
    specific_example: Optional[str] = None
    improvement_suggestion: Optional[str] = None
    score: float = 0.0  # 0-10 scale


@dataclass
class QuestionFeedback:
    """Feedback for a specific question."""
    question_id: str
    question_text: str
    candidate_answer: str
    feedback_points: List[FeedbackPoint]
    overall_score: float
    time_taken_minutes: float
    completeness_score: float
    clarity_score: float
    accuracy_score: float


@dataclass
class InterviewFeedback:
    """Complete interview feedback."""
    session_id: str
    overall_score: float
    category_scores: Dict[FeedbackCategory, float]
    question_feedbacks: List[QuestionFeedback]
    strengths: List[str]
    areas_for_improvement: List[str]
    knowledge_gaps: List[str]
    recommendations: List[str]
    next_steps: List[str]
    language: str = "en"


class InterviewFeedbackAnalyzer:
    """Analyzes interview responses and generates comprehensive feedback."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.logger = logger.bind(component="interview_feedback")
        self.sarvam_client = sarvam_client or SarvamClient()
        
        # Feedback templates and criteria
        self.feedback_criteria = self._load_feedback_criteria()
        self.scoring_rubrics = self._load_scoring_rubrics()
        
    def _load_feedback_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Load feedback criteria for different question types and roles."""
        return {
            "technical": {
                "accuracy": {
                    "weight": 0.4,
                    "criteria": [
                        "Correct technical concepts",
                        "Accurate implementation details",
                        "Proper use of terminology",
                        "Understanding of fundamentals"
                    ]
                },
                "completeness": {
                    "weight": 0.3,
                    "criteria": [
                        "Addresses all parts of question",
                        "Covers edge cases",
                        "Provides comprehensive solution",
                        "Includes error handling"
                    ]
                },
                "clarity": {
                    "weight": 0.3,
                    "criteria": [
                        "Clear explanation of approach",
                        "Logical flow of ideas",
                        "Good use of examples",
                        "Easy to follow reasoning"
                    ]
                }
            },
            "behavioral": {
                "structure": {
                    "weight": 0.3,
                    "criteria": [
                        "Uses STAR method (Situation, Task, Action, Result)",
                        "Clear timeline of events",
                        "Logical flow of narrative",
                        "Specific and concrete details"
                    ]
                },
                "impact": {
                    "weight": 0.4,
                    "criteria": [
                        "Demonstrates measurable outcomes",
                        "Shows business impact",
                        "Quantifies results where possible",
                        "Explains long-term effects"
                    ]
                },
                "learning": {
                    "weight": 0.3,
                    "criteria": [
                        "Shows self-reflection",
                        "Identifies lessons learned",
                        "Demonstrates growth mindset",
                        "Applies learning to future situations"
                    ]
                }
            },
            "system_design": {
                "architecture": {
                    "weight": 0.4,
                    "criteria": [
                        "Appropriate high-level design",
                        "Considers scalability requirements",
                        "Identifies key components",
                        "Proper technology choices"
                    ]
                },
                "trade_offs": {
                    "weight": 0.3,
                    "criteria": [
                        "Discusses design alternatives",
                        "Explains pros and cons",
                        "Considers performance implications",
                        "Addresses reliability concerns"
                    ]
                },
                "details": {
                    "weight": 0.3,
                    "criteria": [
                        "Provides implementation details",
                        "Considers data flow",
                        "Addresses failure scenarios",
                        "Includes monitoring and metrics"
                    ]
                }
            }
        }
    
    def _load_scoring_rubrics(self) -> Dict[DifficultyLevel, Dict[str, Any]]:
        """Load scoring rubrics based on difficulty level."""
        return {
            DifficultyLevel.ENTRY: {
                "expectations": {
                    "technical_depth": "basic",
                    "experience_level": "0-2 years",
                    "complexity_handling": "simple problems",
                    "leadership_expected": False
                },
                "score_thresholds": {
                    "excellent": 8.0,
                    "good": 6.5,
                    "acceptable": 5.0,
                    "needs_improvement": 3.5
                }
            },
            DifficultyLevel.MID: {
                "expectations": {
                    "technical_depth": "intermediate",
                    "experience_level": "2-5 years",
                    "complexity_handling": "moderate complexity",
                    "leadership_expected": "some mentoring"
                },
                "score_thresholds": {
                    "excellent": 8.5,
                    "good": 7.0,
                    "acceptable": 5.5,
                    "needs_improvement": 4.0
                }
            },
            DifficultyLevel.SENIOR: {
                "expectations": {
                    "technical_depth": "advanced",
                    "experience_level": "5+ years",
                    "complexity_handling": "complex systems",
                    "leadership_expected": "team leadership"
                },
                "score_thresholds": {
                    "excellent": 9.0,
                    "good": 7.5,
                    "acceptable": 6.0,
                    "needs_improvement": 4.5
                }
            },
            DifficultyLevel.EXECUTIVE: {
                "expectations": {
                    "technical_depth": "strategic",
                    "experience_level": "10+ years",
                    "complexity_handling": "enterprise scale",
                    "leadership_expected": "organizational leadership"
                },
                "score_thresholds": {
                    "excellent": 9.5,
                    "good": 8.0,
                    "acceptable": 6.5,
                    "needs_improvement": 5.0
                }
            }
        }
    
    async def analyze_question_response(
        self,
        question: InterviewQuestion,
        candidate_answer: str,
        time_taken_minutes: float,
        session_context: Optional[Dict[str, Any]] = None
    ) -> QuestionFeedback:
        """
        Analyze a candidate's response to a specific question.
        
        Args:
            question: The interview question
            candidate_answer: Candidate's response
            time_taken_minutes: Time taken to answer
            session_context: Additional session context
            
        Returns:
            Detailed feedback for the question
        """
        self.logger.info(
            "Analyzing question response",
            question_id=question.id,
            question_category=question.category,
            answer_length=len(candidate_answer),
            time_taken=time_taken_minutes
        )
        
        # Determine question type for analysis
        question_type = self._determine_question_type(question)
        criteria = self.feedback_criteria.get(question_type, self.feedback_criteria["technical"])
        
        # Analyze different aspects
        accuracy_score = await self._analyze_accuracy(question, candidate_answer, criteria)
        completeness_score = await self._analyze_completeness(question, candidate_answer, criteria)
        clarity_score = await self._analyze_clarity(question, candidate_answer, criteria)
        
        # Calculate overall score
        overall_score = (
            accuracy_score * criteria["accuracy"]["weight"] +
            completeness_score * criteria["completeness"]["weight"] +
            clarity_score * criteria["clarity"]["weight"]
        )
        
        # Generate specific feedback points
        feedback_points = await self._generate_feedback_points(
            question, candidate_answer, accuracy_score, completeness_score, clarity_score
        )
        
        # Adjust scores based on difficulty and time
        adjusted_scores = self._adjust_scores_for_difficulty(
            {
                "overall": overall_score,
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "clarity": clarity_score
            },
            question.difficulty,
            time_taken_minutes,
            question.expected_duration_minutes
        )
        
        question_feedback = QuestionFeedback(
            question_id=question.id,
            question_text=question.question,
            candidate_answer=candidate_answer,
            feedback_points=feedback_points,
            overall_score=adjusted_scores["overall"],
            time_taken_minutes=time_taken_minutes,
            completeness_score=adjusted_scores["completeness"],
            clarity_score=adjusted_scores["clarity"],
            accuracy_score=adjusted_scores["accuracy"]
        )
        
        self.logger.info(
            "Question analysis completed",
            question_id=question.id,
            overall_score=adjusted_scores["overall"],
            feedback_points_count=len(feedback_points)
        )
        
        return question_feedback
    
    async def _analyze_accuracy(
        self,
        question: InterviewQuestion,
        answer: str,
        criteria: Dict[str, Any]
    ) -> float:
        """Analyze technical accuracy of the answer."""
        
        # Use LLM to evaluate accuracy
        prompt = f"""
        Evaluate the technical accuracy of this interview answer on a scale of 0-10.
        
        Question: {question.question}
        Answer: {answer}
        
        Evaluation Criteria:
        {json.dumps(criteria["accuracy"]["criteria"], indent=2)}
        
        Consider:
        - Correctness of technical concepts
        - Proper use of terminology
        - Factual accuracy
        - Understanding of fundamentals
        
        Provide a score from 0-10 where:
        - 9-10: Excellent accuracy, no significant errors
        - 7-8: Good accuracy, minor errors
        - 5-6: Acceptable accuracy, some errors
        - 3-4: Poor accuracy, major errors
        - 0-2: Very poor accuracy, fundamental misunderstandings
        
        Return only the numeric score.
        """
        
        try:
            response = await self.sarvam_client.generate_text(
                prompt=prompt,
                language="en",
                max_tokens=50
            )
            
            # Extract numeric score
            score = self._extract_score(response)
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            self.logger.error("Failed to analyze accuracy", error=str(e))
            # Fallback: simple keyword-based analysis
            return self._simple_accuracy_analysis(question, answer)
    
    async def _analyze_completeness(
        self,
        question: InterviewQuestion,
        answer: str,
        criteria: Dict[str, Any]
    ) -> float:
        """Analyze completeness of the answer."""
        
        prompt = f"""
        Evaluate how completely this answer addresses the interview question on a scale of 0-10.
        
        Question: {question.question}
        Answer: {answer}
        
        Evaluation Criteria:
        {json.dumps(criteria["completeness"]["criteria"], indent=2)}
        
        Consider:
        - Does it address all parts of the question?
        - Are important aspects covered?
        - Is the solution comprehensive?
        - Are edge cases considered?
        
        Return only the numeric score (0-10).
        """
        
        try:
            response = await self.sarvam_client.generate_text(
                prompt=prompt,
                language="en",
                max_tokens=50
            )
            
            score = self._extract_score(response)
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            self.logger.error("Failed to analyze completeness", error=str(e))
            return self._simple_completeness_analysis(question, answer)
    
    async def _analyze_clarity(
        self,
        question: InterviewQuestion,
        answer: str,
        criteria: Dict[str, Any]
    ) -> float:
        """Analyze clarity and communication quality of the answer."""
        
        prompt = f"""
        Evaluate the clarity and communication quality of this interview answer on a scale of 0-10.
        
        Question: {question.question}
        Answer: {answer}
        
        Evaluation Criteria:
        {json.dumps(criteria["clarity"]["criteria"], indent=2)}
        
        Consider:
        - Is the explanation clear and easy to follow?
        - Is there logical flow in the reasoning?
        - Are examples helpful and relevant?
        - Is the language appropriate and professional?
        
        Return only the numeric score (0-10).
        """
        
        try:
            response = await self.sarvam_client.generate_text(
                prompt=prompt,
                language="en",
                max_tokens=50
            )
            
            score = self._extract_score(response)
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            self.logger.error("Failed to analyze clarity", error=str(e))
            return self._simple_clarity_analysis(answer)
    
    async def _generate_feedback_points(
        self,
        question: InterviewQuestion,
        answer: str,
        accuracy_score: float,
        completeness_score: float,
        clarity_score: float
    ) -> List[FeedbackPoint]:
        """Generate specific feedback points for the answer."""
        
        feedback_points = []
        
        # Generate accuracy feedback
        if accuracy_score >= 8.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.TECHNICAL_ACCURACY,
                severity=FeedbackSeverity.STRENGTH,
                title="Strong Technical Accuracy",
                description="Demonstrates solid understanding of technical concepts with accurate explanations.",
                score=accuracy_score
            ))
        elif accuracy_score < 5.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.TECHNICAL_ACCURACY,
                severity=FeedbackSeverity.CRITICAL,
                title="Technical Accuracy Needs Improvement",
                description="Several technical inaccuracies or misconceptions identified.",
                improvement_suggestion="Review fundamental concepts and practice explaining technical details accurately.",
                score=accuracy_score
            ))
        
        # Generate completeness feedback
        if completeness_score >= 8.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.PROBLEM_SOLVING,
                severity=FeedbackSeverity.STRENGTH,
                title="Comprehensive Problem Solving",
                description="Provides thorough and complete solutions addressing all aspects of the problem.",
                score=completeness_score
            ))
        elif completeness_score < 5.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.PROBLEM_SOLVING,
                severity=FeedbackSeverity.IMPROVEMENT,
                title="Incomplete Solution",
                description="Answer doesn't fully address all parts of the question or misses important considerations.",
                improvement_suggestion="Practice breaking down complex problems and ensuring all requirements are addressed.",
                score=completeness_score
            ))
        
        # Generate clarity feedback
        if clarity_score >= 8.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.COMMUNICATION,
                severity=FeedbackSeverity.STRENGTH,
                title="Excellent Communication",
                description="Explains concepts clearly with good structure and logical flow.",
                score=clarity_score
            ))
        elif clarity_score < 5.0:
            feedback_points.append(FeedbackPoint(
                category=FeedbackCategory.COMMUNICATION,
                severity=FeedbackSeverity.IMPROVEMENT,
                title="Communication Clarity",
                description="Explanation could be clearer and better structured.",
                improvement_suggestion="Practice explaining technical concepts step-by-step with concrete examples.",
                score=clarity_score
            ))
        
        return feedback_points
    
    def _determine_question_type(self, question: InterviewQuestion) -> str:
        """Determine the type of question for analysis."""
        if question.category in ["system_design", "architecture"]:
            return "system_design"
        elif question.category in ["behavioral", "leadership", "communication"]:
            return "behavioral"
        else:
            return "technical"
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response."""
        import re
        
        # Look for numbers in the response
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            try:
                score = float(numbers[0])
                return score
            except ValueError:
                pass
        
        # Fallback to 5.0 if no valid number found
        return 5.0
    
    def _simple_accuracy_analysis(self, question: InterviewQuestion, answer: str) -> float:
        """Simple fallback accuracy analysis."""
        # Basic keyword matching
        answer_lower = answer.lower()
        
        # Check for technical keywords related to the question
        technical_keywords = ["algorithm", "data structure", "complexity", "performance", "scalability"]
        keyword_matches = sum(1 for keyword in technical_keywords if keyword in answer_lower)
        
        # Basic scoring based on answer length and keyword presence
        base_score = min(8.0, len(answer) / 50)  # Longer answers get higher base score
        keyword_bonus = min(2.0, keyword_matches * 0.5)
        
        return min(10.0, base_score + keyword_bonus)
    
    def _simple_completeness_analysis(self, question: InterviewQuestion, answer: str) -> float:
        """Simple fallback completeness analysis."""
        # Check if answer addresses multiple aspects
        question_words = question.question.lower().split()
        answer_words = answer.lower().split()
        
        # Count how many question keywords appear in answer
        overlap = len(set(question_words) & set(answer_words))
        overlap_ratio = overlap / len(question_words) if question_words else 0
        
        # Score based on overlap and answer length
        base_score = overlap_ratio * 6.0
        length_bonus = min(4.0, len(answer) / 100)
        
        return min(10.0, base_score + length_bonus)
    
    def _simple_clarity_analysis(self, answer: str) -> float:
        """Simple fallback clarity analysis."""
        # Basic metrics for clarity
        sentences = answer.split('.')
        avg_sentence_length = len(answer.split()) / len(sentences) if sentences else 0
        
        # Prefer moderate sentence lengths (10-25 words)
        if 10 <= avg_sentence_length <= 25:
            clarity_score = 8.0
        elif 5 <= avg_sentence_length <= 35:
            clarity_score = 6.0
        else:
            clarity_score = 4.0
        
        # Bonus for structure words
        structure_words = ["first", "second", "then", "however", "therefore", "because"]
        structure_bonus = sum(0.3 for word in structure_words if word in answer.lower())
        
        return min(10.0, clarity_score + structure_bonus)
    
    def _adjust_scores_for_difficulty(
        self,
        scores: Dict[str, float],
        difficulty: DifficultyLevel,
        time_taken: float,
        expected_time: float
    ) -> Dict[str, float]:
        """Adjust scores based on difficulty level and timing."""
        
        rubric = self.scoring_rubrics[difficulty]
        adjusted_scores = scores.copy()
        
        # Time adjustment factor
        time_ratio = time_taken / expected_time if expected_time > 0 else 1.0
        
        if time_ratio > 2.0:
            # Took too long - penalty
            time_adjustment = -1.0
        elif time_ratio < 0.5:
            # Very fast - could be good or bad
            time_adjustment = 0.5 if scores["overall"] >= 7.0 else -0.5
        else:
            # Reasonable time
            time_adjustment = 0.0
        
        # Apply adjustments
        for key in adjusted_scores:
            adjusted_scores[key] = max(0.0, min(10.0, adjusted_scores[key] + time_adjustment))
        
        return adjusted_scores
    
    async def generate_session_feedback(
        self,
        session: InterviewSession,
        question_feedbacks: List[QuestionFeedback]
    ) -> InterviewFeedback:
        """
        Generate comprehensive feedback for the entire interview session.
        
        Args:
            session: Interview session
            question_feedbacks: Feedback for individual questions
            
        Returns:
            Complete interview feedback
        """
        self.logger.info(
            "Generating session feedback",
            session_id=session.session_id,
            question_count=len(question_feedbacks)
        )
        
        # Calculate overall scores
        overall_score = sum(qf.overall_score for qf in question_feedbacks) / len(question_feedbacks)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(question_feedbacks)
        
        # Identify strengths and areas for improvement
        strengths = self._identify_strengths(question_feedbacks, category_scores)
        areas_for_improvement = self._identify_improvements(question_feedbacks, category_scores)
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(question_feedbacks, session.role)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            session, question_feedbacks, category_scores
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(
            session.difficulty, areas_for_improvement, knowledge_gaps
        )
        
        feedback = InterviewFeedback(
            session_id=session.session_id,
            overall_score=overall_score,
            category_scores=category_scores,
            question_feedbacks=question_feedbacks,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            knowledge_gaps=knowledge_gaps,
            recommendations=recommendations,
            next_steps=next_steps,
            language=session.language
        )
        
        self.logger.info(
            "Session feedback generated",
            session_id=session.session_id,
            overall_score=overall_score,
            strengths_count=len(strengths),
            improvements_count=len(areas_for_improvement)
        )
        
        return feedback
    
    def _calculate_category_scores(self, question_feedbacks: List[QuestionFeedback]) -> Dict[FeedbackCategory, float]:
        """Calculate average scores by feedback category."""
        category_scores = {}
        category_counts = {}
        
        for qf in question_feedbacks:
            for fp in qf.feedback_points:
                if fp.category not in category_scores:
                    category_scores[fp.category] = 0.0
                    category_counts[fp.category] = 0
                
                category_scores[fp.category] += fp.score
                category_counts[fp.category] += 1
        
        # Calculate averages
        for category in category_scores:
            if category_counts[category] > 0:
                category_scores[category] /= category_counts[category]
        
        return category_scores
    
    def _identify_strengths(
        self,
        question_feedbacks: List[QuestionFeedback],
        category_scores: Dict[FeedbackCategory, float]
    ) -> List[str]:
        """Identify candidate strengths."""
        strengths = []
        
        # High-scoring categories
        for category, score in category_scores.items():
            if score >= 7.5:
                strengths.append(f"Strong {category.value.replace('_', ' ')}")
        
        # Consistent high performance
        high_scores = [qf.overall_score for qf in question_feedbacks if qf.overall_score >= 8.0]
        if len(high_scores) >= len(question_feedbacks) * 0.6:
            strengths.append("Consistent high-quality responses across multiple questions")
        
        # Specific strengths from feedback points
        strength_points = []
        for qf in question_feedbacks:
            for fp in qf.feedback_points:
                if fp.severity == FeedbackSeverity.STRENGTH:
                    strength_points.append(fp.title)
        
        # Add unique strength points
        unique_strengths = list(set(strength_points))
        strengths.extend(unique_strengths[:3])  # Limit to top 3
        
        return strengths[:5]  # Return top 5 strengths
    
    def _identify_improvements(
        self,
        question_feedbacks: List[QuestionFeedback],
        category_scores: Dict[FeedbackCategory, float]
    ) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        # Low-scoring categories
        for category, score in category_scores.items():
            if score < 6.0:
                improvements.append(f"Improve {category.value.replace('_', ' ')}")
        
        # Specific improvement points
        improvement_points = []
        for qf in question_feedbacks:
            for fp in qf.feedback_points:
                if fp.severity in [FeedbackSeverity.IMPROVEMENT, FeedbackSeverity.CRITICAL]:
                    if fp.improvement_suggestion:
                        improvement_points.append(fp.improvement_suggestion)
        
        # Add unique improvement suggestions
        unique_improvements = list(set(improvement_points))
        improvements.extend(unique_improvements[:3])
        
        return improvements[:5]
    
    def _identify_knowledge_gaps(
        self,
        question_feedbacks: List[QuestionFeedback],
        role: str
    ) -> List[str]:
        """Identify specific knowledge gaps."""
        gaps = []
        
        # Low accuracy scores indicate knowledge gaps
        low_accuracy_questions = [
            qf for qf in question_feedbacks 
            if qf.accuracy_score < 5.0
        ]
        
        for qf in low_accuracy_questions:
            # Extract topic from question
            question_lower = qf.question_text.lower()
            
            if "algorithm" in question_lower or "complexity" in question_lower:
                gaps.append("Algorithms and complexity analysis")
            elif "system design" in question_lower or "architecture" in question_lower:
                gaps.append("System design and architecture")
            elif "database" in question_lower or "sql" in question_lower:
                gaps.append("Database design and optimization")
            elif "security" in question_lower:
                gaps.append("Security best practices")
        
        # Role-specific knowledge gaps
        role_gaps = {
            "software_engineer": [
                "Data structures and algorithms",
                "System design principles",
                "Software engineering best practices"
            ],
            "data_scientist": [
                "Machine learning algorithms",
                "Statistical analysis methods",
                "Data preprocessing techniques"
            ],
            "product_manager": [
                "Product strategy frameworks",
                "User research methods",
                "Metrics and analytics"
            ]
        }
        
        if role in role_gaps:
            # Add role-specific gaps if performance is low
            avg_score = sum(qf.overall_score for qf in question_feedbacks) / len(question_feedbacks)
            if avg_score < 6.0:
                gaps.extend(role_gaps[role][:2])
        
        return list(set(gaps))[:5]
    
    async def _generate_recommendations(
        self,
        session: InterviewSession,
        question_feedbacks: List[QuestionFeedback],
        category_scores: Dict[FeedbackCategory, float]
    ) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        avg_score = sum(qf.overall_score for qf in question_feedbacks) / len(question_feedbacks)
        
        # Performance-based recommendations
        if avg_score >= 8.0:
            recommendations.append("Excellent performance! Consider applying for senior-level positions.")
        elif avg_score >= 6.5:
            recommendations.append("Good performance with room for improvement in specific areas.")
        else:
            recommendations.append("Focus on fundamental concepts and practice more technical problems.")
        
        # Category-specific recommendations
        if category_scores.get(FeedbackCategory.COMMUNICATION, 0) < 6.0:
            recommendations.append("Practice explaining technical concepts clearly and concisely.")
        
        if category_scores.get(FeedbackCategory.PROBLEM_SOLVING, 0) < 6.0:
            recommendations.append("Work on structured problem-solving approaches and practice more coding problems.")
        
        # Difficulty-based recommendations
        if session.difficulty == DifficultyLevel.ENTRY and avg_score >= 7.0:
            recommendations.append("Consider preparing for mid-level positions.")
        elif session.difficulty == DifficultyLevel.SENIOR and avg_score < 6.0:
            recommendations.append("Focus on gaining more experience before applying for senior roles.")
        
        return recommendations[:4]
    
    def _generate_next_steps(
        self,
        difficulty: DifficultyLevel,
        improvements: List[str],
        knowledge_gaps: List[str]
    ) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Study recommendations based on gaps
        if knowledge_gaps:
            next_steps.append(f"Study and practice: {', '.join(knowledge_gaps[:3])}")
        
        # Practice recommendations
        if "communication" in str(improvements).lower():
            next_steps.append("Practice mock interviews focusing on clear communication")
        
        if "technical" in str(improvements).lower():
            next_steps.append("Solve more technical problems on coding platforms")
        
        # Difficulty-specific next steps
        if difficulty == DifficultyLevel.ENTRY:
            next_steps.append("Build more projects to demonstrate practical skills")
        elif difficulty == DifficultyLevel.MID:
            next_steps.append("Gain experience with system design and architecture")
        elif difficulty == DifficultyLevel.SENIOR:
            next_steps.append("Develop leadership skills and mentor junior developers")
        
        # General next steps
        next_steps.append("Schedule follow-up practice sessions to track improvement")
        
        return next_steps[:5]