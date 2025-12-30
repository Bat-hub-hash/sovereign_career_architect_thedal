"""Interview question generation system."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import structlog

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class InterviewType(Enum):
    """Types of interviews."""
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    MIXED = "mixed"
    SYSTEM_DESIGN = "system_design"
    CODING = "coding"


class DifficultyLevel(Enum):
    """Difficulty levels for interview questions."""
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


@dataclass
class InterviewQuestion:
    """Interview question with metadata."""
    id: str
    question: str
    type: InterviewType
    difficulty: DifficultyLevel
    role: str
    category: str
    expected_duration_minutes: int
    follow_up_questions: List[str]
    evaluation_criteria: List[str]
    sample_answer_points: List[str]
    tags: List[str]
    language: str = "en"


@dataclass
class InterviewSession:
    """Interview session configuration."""
    session_id: str
    role: str
    company: Optional[str]
    interview_type: InterviewType
    difficulty: DifficultyLevel
    duration_minutes: int
    language: str
    questions: List[InterviewQuestion]
    current_question_index: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class InterviewQuestionGenerator:
    """Generates role-specific interview questions with difficulty scaling."""
    
    def __init__(self):
        self.logger = logger.bind(component="interview_generator")
        
        # Question templates by role and type
        self.question_templates = self._load_question_templates()
        
        # Role-specific skills and technologies
        self.role_skills = self._load_role_skills()
        
        # Difficulty scaling factors
        self.difficulty_factors = {
            DifficultyLevel.ENTRY: {
                "complexity_multiplier": 1.0,
                "depth_level": "basic",
                "expected_experience": "0-2 years",
                "question_count_range": (3, 5)
            },
            DifficultyLevel.MID: {
                "complexity_multiplier": 1.5,
                "depth_level": "intermediate",
                "expected_experience": "2-5 years",
                "question_count_range": (5, 8)
            },
            DifficultyLevel.SENIOR: {
                "complexity_multiplier": 2.0,
                "depth_level": "advanced",
                "expected_experience": "5+ years",
                "question_count_range": (6, 10)
            },
            DifficultyLevel.EXECUTIVE: {
                "complexity_multiplier": 2.5,
                "depth_level": "expert",
                "expected_experience": "10+ years",
                "question_count_range": (4, 6)  # Fewer but more strategic questions
            }
        }
    
    def _load_question_templates(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load question templates organized by role and type."""
        return {
            "software_engineer": {
                "technical": [
                    {
                        "template": "Explain the concept of {concept} and provide an example of when you would use it.",
                        "concepts": ["polymorphism", "dependency injection", "design patterns", "microservices", "caching strategies"],
                        "category": "fundamentals",
                        "duration": 10,
                        "follow_ups": [
                            "Can you walk me through a specific implementation?",
                            "What are the trade-offs of this approach?",
                            "How would you test this?"
                        ]
                    },
                    {
                        "template": "How would you design a {system} that can handle {scale}?",
                        "systems": ["URL shortener", "chat application", "file storage system", "notification service"],
                        "scales": ["1 million users", "high availability", "real-time updates", "global distribution"],
                        "category": "system_design",
                        "duration": 20,
                        "follow_ups": [
                            "How would you handle failures?",
                            "What about data consistency?",
                            "How would you monitor this system?"
                        ]
                    },
                    {
                        "template": "Write a function to {task} in {language}.",
                        "tasks": ["reverse a linked list", "find the longest palindrome", "implement a LRU cache", "merge sorted arrays"],
                        "languages": ["Python", "JavaScript", "Java", "your preferred language"],
                        "category": "coding",
                        "duration": 15,
                        "follow_ups": [
                            "What's the time complexity?",
                            "Can you optimize this further?",
                            "How would you handle edge cases?"
                        ]
                    }
                ],
                "behavioral": [
                    {
                        "template": "Tell me about a time when you had to {situation}.",
                        "situations": [
                            "debug a complex production issue",
                            "work with a difficult team member",
                            "learn a new technology quickly",
                            "make a technical decision with limited information"
                        ],
                        "category": "problem_solving",
                        "duration": 8,
                        "follow_ups": [
                            "What was the outcome?",
                            "What would you do differently?",
                            "How did you measure success?"
                        ]
                    },
                    {
                        "template": "Describe a project where you {achievement}.",
                        "achievements": [
                            "improved system performance significantly",
                            "led a technical initiative",
                            "mentored junior developers",
                            "implemented a new feature from scratch"
                        ],
                        "category": "leadership",
                        "duration": 10,
                        "follow_ups": [
                            "What challenges did you face?",
                            "How did you measure the impact?",
                            "What did you learn from this experience?"
                        ]
                    }
                ]
            },
            "data_scientist": {
                "technical": [
                    {
                        "template": "How would you approach {problem} using {method}?",
                        "problems": [
                            "customer churn prediction",
                            "recommendation system design",
                            "A/B test analysis",
                            "time series forecasting"
                        ],
                        "methods": [
                            "machine learning",
                            "statistical analysis",
                            "deep learning",
                            "ensemble methods"
                        ],
                        "category": "methodology",
                        "duration": 15,
                        "follow_ups": [
                            "How would you validate your model?",
                            "What metrics would you use?",
                            "How would you handle missing data?"
                        ]
                    },
                    {
                        "template": "Explain the difference between {concept1} and {concept2}.",
                        "concept_pairs": [
                            ("supervised learning", "unsupervised learning"),
                            ("precision", "recall"),
                            ("bagging", "boosting"),
                            ("correlation", "causation")
                        ],
                        "category": "fundamentals",
                        "duration": 8,
                        "follow_ups": [
                            "When would you use each approach?",
                            "Can you provide a practical example?",
                            "What are the assumptions of each method?"
                        ]
                    }
                ],
                "behavioral": [
                    {
                        "template": "Tell me about a time when you had to {situation}.",
                        "situations": [
                            "explain complex findings to non-technical stakeholders",
                            "work with messy or incomplete data",
                            "choose between multiple modeling approaches",
                            "implement a model in production"
                        ],
                        "category": "communication",
                        "duration": 10,
                        "follow_ups": [
                            "How did you ensure understanding?",
                            "What was the business impact?",
                            "How did you handle pushback?"
                        ]
                    }
                ]
            },
            "product_manager": {
                "technical": [
                    {
                        "template": "How would you prioritize {scenario}?",
                        "scenarios": [
                            "feature requests from multiple stakeholders",
                            "technical debt vs new features",
                            "bug fixes vs feature development",
                            "competing product initiatives"
                        ],
                        "category": "prioritization",
                        "duration": 12,
                        "follow_ups": [
                            "What framework would you use?",
                            "How would you communicate this to stakeholders?",
                            "How would you measure success?"
                        ]
                    },
                    {
                        "template": "Design a {product} for {target_audience}.",
                        "products": [
                            "mobile app",
                            "dashboard",
                            "API",
                            "notification system"
                        ],
                        "target_audiences": [
                            "busy professionals",
                            "elderly users",
                            "developers",
                            "small business owners"
                        ],
                        "category": "product_design",
                        "duration": 20,
                        "follow_ups": [
                            "What would be your MVP?",
                            "How would you validate this idea?",
                            "What metrics would you track?"
                        ]
                    }
                ],
                "behavioral": [
                    {
                        "template": "Describe a time when you had to {challenge}.",
                        "challenges": [
                            "launch a product with limited resources",
                            "pivot a product strategy",
                            "manage conflicting stakeholder requirements",
                            "recover from a failed product launch"
                        ],
                        "category": "leadership",
                        "duration": 12,
                        "follow_ups": [
                            "What was your decision-making process?",
                            "How did you align the team?",
                            "What did you learn from this experience?"
                        ]
                    }
                ]
            }
        }
    
    def _load_role_skills(self) -> Dict[str, Dict[str, List[str]]]:
        """Load role-specific skills and technologies."""
        return {
            "software_engineer": {
                "programming_languages": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
                "frameworks": ["React", "Django", "Spring Boot", "Express.js", "Flask"],
                "databases": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"],
                "tools": ["Docker", "Kubernetes", "Git", "Jenkins", "AWS"],
                "concepts": ["OOP", "Functional Programming", "Design Patterns", "Algorithms", "Data Structures"]
            },
            "data_scientist": {
                "programming_languages": ["Python", "R", "SQL", "Scala"],
                "frameworks": ["TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy"],
                "databases": ["PostgreSQL", "MongoDB", "BigQuery", "Snowflake"],
                "tools": ["Jupyter", "Docker", "Airflow", "MLflow", "Tableau"],
                "concepts": ["Statistics", "Machine Learning", "Deep Learning", "A/B Testing", "Feature Engineering"]
            },
            "product_manager": {
                "frameworks": ["Agile", "Scrum", "Lean", "Design Thinking"],
                "tools": ["Jira", "Figma", "Analytics", "A/B Testing", "User Research"],
                "concepts": ["Product Strategy", "User Experience", "Market Research", "Roadmapping", "Metrics"]
            },
            "designer": {
                "tools": ["Figma", "Sketch", "Adobe Creative Suite", "Principle", "InVision"],
                "concepts": ["User Experience", "User Interface", "Design Systems", "Accessibility", "Usability Testing"]
            },
            "marketing_manager": {
                "tools": ["Google Analytics", "HubSpot", "Salesforce", "Mailchimp", "Social Media Platforms"],
                "concepts": ["Digital Marketing", "Content Strategy", "SEO/SEM", "Brand Management", "Customer Acquisition"]
            }
        }
    
    async def generate_questions(
        self,
        role: str,
        interview_type: InterviewType,
        difficulty: DifficultyLevel,
        count: Optional[int] = None,
        language: str = "en"
    ) -> List[InterviewQuestion]:
        """
        Generate interview questions for a specific role and difficulty.
        
        Args:
            role: Target role (e.g., "software_engineer")
            interview_type: Type of interview
            difficulty: Difficulty level
            count: Number of questions to generate (optional)
            language: Language for questions
            
        Returns:
            List of generated interview questions
        """
        self.logger.info(
            "Generating interview questions",
            role=role,
            interview_type=interview_type.value,
            difficulty=difficulty.value,
            count=count,
            language=language
        )
        
        # Determine question count if not specified
        if count is None:
            min_count, max_count = self.difficulty_factors[difficulty]["question_count_range"]
            count = random.randint(min_count, max_count)
        
        # Get templates for the role and type
        role_templates = self.question_templates.get(role, {})
        type_templates = role_templates.get(interview_type.value, [])
        
        if not type_templates:
            # Fallback to generic questions
            type_templates = self._get_generic_questions(interview_type)
            
        # If still no templates, create basic fallback
        if not type_templates:
            type_templates = [{
                "template": f"Tell me about your experience with {interview_type.value} aspects of your role.",
                "category": "general",
                "duration": 10,
                "follow_ups": [
                    "Can you provide more details?",
                    "What challenges did you face?",
                    "How did you overcome them?"
                ]
            }]
        
        questions = []
        used_templates = set()
        
        for i in range(count):
            # Select a template (avoid duplicates when possible)
            available_templates = [t for j, t in enumerate(type_templates) if j not in used_templates]
            if not available_templates:
                available_templates = type_templates  # Reset if all used
                used_templates.clear()
            
            template = random.choice(available_templates)
            used_templates.add(type_templates.index(template))
            
            # Generate question from template
            question = await self._generate_question_from_template(
                template, role, difficulty, language, i + 1, interview_type
            )
            
            questions.append(question)
        
        # Sort questions by difficulty and type for better flow
        questions.sort(key=lambda q: (q.type.value, q.expected_duration_minutes))
        
        self.logger.info(
            "Interview questions generated successfully",
            role=role,
            count=len(questions),
            difficulty=difficulty.value
        )
        
        return questions
    
    async def _generate_question_from_template(
        self,
        template: Dict[str, Any],
        role: str,
        difficulty: DifficultyLevel,
        language: str,
        question_number: int,
        interview_type: InterviewType
    ) -> InterviewQuestion:
        """Generate a specific question from a template."""
        
        # Fill in template variables
        question_text = template["template"]
        
        # Replace template variables with specific values
        for key, values in template.items():
            if key.endswith("s") and isinstance(values, list):  # e.g., "concepts", "systems"
                placeholder = "{" + key[:-1] + "}"  # Remove 's' and add braces
                if placeholder in question_text:
                    selected_value = random.choice(values)
                    question_text = question_text.replace(placeholder, selected_value)
        
        # Handle concept pairs
        if "concept_pairs" in template:
            concept1, concept2 = random.choice(template["concept_pairs"])
            question_text = question_text.replace("{concept1}", concept1)
            question_text = question_text.replace("{concept2}", concept2)
        
        # Adjust difficulty
        difficulty_factor = self.difficulty_factors[difficulty]
        base_duration = template.get("duration", 10)
        adjusted_duration = int(base_duration * difficulty_factor["complexity_multiplier"])
        
        # Add difficulty-specific modifiers
        if difficulty == DifficultyLevel.SENIOR or difficulty == DifficultyLevel.EXECUTIVE:
            if "explain" in question_text.lower():
                question_text = question_text.replace("Explain", "Critically analyze")
            elif "how would you" in question_text.lower():
                question_text += " Consider scalability, maintainability, and trade-offs."
        
        # Generate evaluation criteria based on difficulty
        evaluation_criteria = self._generate_evaluation_criteria(
            template.get("category", "general"),
            difficulty
        )
        
        # Generate sample answer points
        sample_answer_points = self._generate_sample_answer_points(
            question_text,
            template.get("category", "general"),
            difficulty
        )
        
        # Create question object
        question = InterviewQuestion(
            id=f"{role}_{difficulty.value}_{question_number}",
            question=question_text,
            type=self._determine_question_type_from_template(template, interview_type),
            difficulty=difficulty,
            role=role,
            category=template.get("category", "general"),
            expected_duration_minutes=adjusted_duration,
            follow_up_questions=template.get("follow_ups", []),
            evaluation_criteria=evaluation_criteria,
            sample_answer_points=sample_answer_points,
            tags=[role, difficulty.value, template.get("category", "general")],
            language=language
        )
        
        return question
    
    def _determine_question_type_from_template(self, template: Dict[str, Any], requested_type: InterviewType) -> InterviewType:
        """Determine the appropriate question type based on template and request."""
        category = template.get("category", "general")
        
        # Map categories to types
        if category in ["system_design", "architecture"]:
            return InterviewType.SYSTEM_DESIGN
        elif category in ["coding", "algorithms"]:
            return InterviewType.CODING
        elif category in ["behavioral", "leadership", "communication", "problem_solving"]:
            return InterviewType.BEHAVIORAL
        elif category in ["technical", "fundamentals", "methodology"]:
            return InterviewType.TECHNICAL
        else:
            # Default to requested type
            return requested_type
    
    def _generate_evaluation_criteria(self, category: str, difficulty: DifficultyLevel) -> List[str]:
        """Generate evaluation criteria based on category and difficulty."""
        base_criteria = {
            "fundamentals": [
                "Demonstrates understanding of core concepts",
                "Provides accurate technical explanations",
                "Uses appropriate terminology"
            ],
            "system_design": [
                "Considers scalability requirements",
                "Identifies potential bottlenecks",
                "Proposes appropriate technologies"
            ],
            "coding": [
                "Writes clean, readable code",
                "Handles edge cases appropriately",
                "Explains algorithmic approach"
            ],
            "problem_solving": [
                "Demonstrates analytical thinking",
                "Provides structured approach",
                "Shows learning from experience"
            ],
            "leadership": [
                "Shows leadership qualities",
                "Demonstrates team collaboration",
                "Provides measurable outcomes"
            ],
            "communication": [
                "Explains complex concepts clearly",
                "Adapts communication to audience",
                "Provides concrete examples"
            ]
        }
        
        criteria = base_criteria.get(category, base_criteria["fundamentals"]).copy()
        
        # Add difficulty-specific criteria
        if difficulty in [DifficultyLevel.SENIOR, DifficultyLevel.EXECUTIVE]:
            criteria.extend([
                "Demonstrates strategic thinking",
                "Considers business impact",
                "Shows mentoring/leadership experience"
            ])
        
        return criteria
    
    def _generate_sample_answer_points(
        self, 
        question: str, 
        category: str, 
        difficulty: DifficultyLevel
    ) -> List[str]:
        """Generate sample answer points for the question."""
        # This would typically use an LLM to generate contextual answer points
        # For now, providing category-based generic points
        
        base_points = {
            "fundamentals": [
                "Define the concept clearly",
                "Provide a practical example",
                "Explain when to use it"
            ],
            "system_design": [
                "Start with requirements gathering",
                "Design high-level architecture",
                "Consider scalability and reliability"
            ],
            "coding": [
                "Clarify requirements and constraints",
                "Write clean, well-structured code",
                "Test with various inputs"
            ],
            "problem_solving": [
                "Describe the situation clearly",
                "Explain the approach taken",
                "Share the outcome and lessons learned"
            ]
        }
        
        points = base_points.get(category, base_points["fundamentals"]).copy()
        
        # Add difficulty-specific points
        if difficulty in [DifficultyLevel.SENIOR, DifficultyLevel.EXECUTIVE]:
            points.extend([
                "Discuss trade-offs and alternatives",
                "Consider long-term implications",
                "Relate to business objectives"
            ])
        
        return points
    
    def _get_generic_questions(self, interview_type: InterviewType) -> List[Dict[str, Any]]:
        """Get generic questions when role-specific templates aren't available."""
        generic_questions = {
            InterviewType.TECHNICAL: [
                {
                    "template": "Describe your experience with {technology} and how you've used it in projects.",
                    "technologies": ["databases", "cloud platforms", "programming languages", "frameworks"],
                    "category": "experience",
                    "duration": 8,
                    "follow_ups": [
                        "What challenges did you face?",
                        "How did you overcome them?",
                        "What would you do differently?"
                    ]
                }
            ],
            InterviewType.BEHAVIORAL: [
                {
                    "template": "Tell me about a challenging project you worked on.",
                    "category": "experience",
                    "duration": 10,
                    "follow_ups": [
                        "What made it challenging?",
                        "How did you handle the challenges?",
                        "What was the outcome?"
                    ]
                }
            ]
        }
        
        return generic_questions.get(interview_type, [])
    
    async def create_interview_session(
        self,
        role: str,
        company: Optional[str] = None,
        interview_type: InterviewType = InterviewType.MIXED,
        difficulty: DifficultyLevel = DifficultyLevel.MID,
        duration_minutes: int = 45,
        language: str = "en"
    ) -> InterviewSession:
        """
        Create a complete interview session.
        
        Args:
            role: Target role
            company: Company name (optional)
            interview_type: Type of interview
            difficulty: Difficulty level
            duration_minutes: Total interview duration
            language: Interview language
            
        Returns:
            Configured interview session
        """
        import uuid
        import time
        
        session_id = str(uuid.uuid4())
        
        # Generate questions based on type
        if interview_type == InterviewType.MIXED:
            # Generate both technical and behavioral questions
            technical_questions = await self.generate_questions(
                role=role,
                interview_type=InterviewType.TECHNICAL,
                difficulty=difficulty,
                count=3,
                language=language
            )
            
            behavioral_questions = await self.generate_questions(
                role=role,
                interview_type=InterviewType.BEHAVIORAL,
                difficulty=difficulty,
                count=2,
                language=language
            )
            
            questions = technical_questions + behavioral_questions
        else:
            # Generate questions of the specified type
            question_count = max(3, duration_minutes // 10)  # Roughly 10 minutes per question
            questions = await self.generate_questions(
                role=role,
                interview_type=interview_type,
                difficulty=difficulty,
                count=question_count,
                language=language
            )
        
        # Create session
        session = InterviewSession(
            session_id=session_id,
            role=role,
            company=company,
            interview_type=interview_type,
            difficulty=difficulty,
            duration_minutes=duration_minutes,
            language=language,
            questions=questions,
            current_question_index=0,
            started_at=time.time()
        )
        
        self.logger.info(
            "Interview session created",
            session_id=session_id,
            role=role,
            company=company,
            question_count=len(questions),
            duration_minutes=duration_minutes
        )
        
        return session
    
    async def get_next_question(self, session: InterviewSession) -> Optional[InterviewQuestion]:
        """Get the next question in the interview session."""
        if session.current_question_index >= len(session.questions):
            return None
        
        question = session.questions[session.current_question_index]
        session.current_question_index += 1
        
        return question
    
    async def complete_session(self, session: InterviewSession) -> Dict[str, Any]:
        """Complete the interview session and generate summary."""
        import time
        
        session.completed_at = time.time()
        duration = session.completed_at - (session.started_at or session.completed_at)
        
        summary = {
            "session_id": session.session_id,
            "role": session.role,
            "company": session.company,
            "interview_type": session.interview_type.value,
            "difficulty": session.difficulty.value,
            "planned_duration_minutes": session.duration_minutes,
            "actual_duration_minutes": duration / 60,
            "questions_asked": session.current_question_index,
            "total_questions": len(session.questions),
            "completion_rate": session.current_question_index / len(session.questions),
            "language": session.language
        }
        
        self.logger.info(
            "Interview session completed",
            session_id=session.session_id,
            completion_rate=summary["completion_rate"],
            duration_minutes=summary["actual_duration_minutes"]
        )
        
        return summary