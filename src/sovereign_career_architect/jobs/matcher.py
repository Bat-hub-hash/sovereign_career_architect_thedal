"""Job matching system for comparing user profiles with job requirements."""

import asyncio
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from sovereign_career_architect.jobs.extractor import (
    JobRequirement,
    ExtractedRequirements,
    RequirementType,
    RequirementLevel
)
from sovereign_career_architect.core.models import UserProfile
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class MatchLevel(Enum):
    """Levels of skill matching."""
    EXACT = "exact"
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    NO_MATCH = "no_match"


@dataclass
class SkillMatch:
    """Individual skill match result."""
    requirement: JobRequirement
    user_skill: Optional[str]
    match_level: MatchLevel
    confidence: float
    gap_analysis: Optional[str] = None
    years_gap: Optional[int] = None


@dataclass
class JobPosting:
    """Job posting with extracted requirements."""
    id: str
    title: str
    company: str
    location: str
    url: Optional[str]
    extracted_requirements: ExtractedRequirements
    posted_date: Optional[str] = None
    application_deadline: Optional[str] = None


@dataclass
class MatchResult:
    """Complete job matching result."""
    job_posting: JobPosting
    overall_score: float
    category_scores: Dict[str, float]
    skill_matches: List[SkillMatch]
    missing_requirements: List[JobRequirement]
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    fit_level: str  # "excellent", "good", "fair", "poor"


class JobMatcher:
    """Matches user profiles against job requirements."""
    
    def __init__(self):
        self.logger = logger.bind(component="job_matcher")
        
        # Skill similarity mappings
        self.skill_synonyms = self._load_skill_synonyms()
        self.skill_hierarchies = self._load_skill_hierarchies()
        
        # Scoring weights
        self.scoring_weights = {
            RequirementLevel.REQUIRED: 1.0,
            RequirementLevel.PREFERRED: 0.7,
            RequirementLevel.NICE_TO_HAVE: 0.3
        }
        
        # Category weights for overall scoring
        self.category_weights = {
            "technical_skills": 0.4,
            "experience": 0.3,
            "education": 0.15,
            "soft_skills": 0.1,
            "certifications": 0.05
        }
    
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for better matching."""
        return {
            "python": ["python", "py", "python3"],
            "javascript": ["javascript", "js", "ecmascript", "node.js", "nodejs"],
            "react": ["react", "reactjs", "react.js"],
            "angular": ["angular", "angularjs", "angular.js"],
            "vue": ["vue", "vuejs", "vue.js"],
            "machine learning": ["machine learning", "ml", "artificial intelligence", "ai"],
            "deep learning": ["deep learning", "neural networks", "dl"],
            "data science": ["data science", "data analysis", "analytics"],
            "sql": ["sql", "mysql", "postgresql", "database"],
            "nosql": ["nosql", "mongodb", "cassandra", "dynamodb"],
            "aws": ["aws", "amazon web services", "amazon cloud"],
            "azure": ["azure", "microsoft azure", "microsoft cloud"],
            "docker": ["docker", "containerization", "containers"],
            "kubernetes": ["kubernetes", "k8s", "container orchestration"],
            "agile": ["agile", "scrum", "kanban", "agile methodology"],
            "project management": ["project management", "pm", "program management"]
        }
    
    def _load_skill_hierarchies(self) -> Dict[str, Dict[str, List[str]]]:
        """Load skill hierarchies for experience level matching."""
        return {
            "programming": {
                "beginner": ["basic syntax", "hello world", "simple scripts"],
                "intermediate": ["frameworks", "libraries", "apis", "databases"],
                "advanced": ["architecture", "design patterns", "optimization", "scalability"],
                "expert": ["system design", "performance tuning", "mentoring", "technical leadership"]
            },
            "data_science": {
                "beginner": ["basic statistics", "excel", "data visualization"],
                "intermediate": ["python", "pandas", "scikit-learn", "sql"],
                "advanced": ["machine learning", "deep learning", "model deployment"],
                "expert": ["research", "algorithm development", "team leadership", "strategy"]
            },
            "management": {
                "beginner": ["team coordination", "basic planning"],
                "intermediate": ["project management", "agile", "stakeholder management"],
                "advanced": ["strategic planning", "budget management", "cross-functional leadership"],
                "expert": ["organizational leadership", "vision setting", "transformation"]
            }
        }
    
    async def match_job(
        self,
        user_profile: UserProfile,
        job_posting: JobPosting
    ) -> MatchResult:
        """
        Match a user profile against a job posting.
        
        Args:
            user_profile: User's profile with skills and experience
            job_posting: Job posting with extracted requirements
            
        Returns:
            Detailed matching result
        """
        self.logger.info(
            "Matching job against user profile",
            job_title=job_posting.title,
            company=job_posting.company,
            user_skills_count=len(user_profile.skills) if user_profile.skills else 0
        )
        
        requirements = job_posting.extracted_requirements.requirements
        
        # Match individual skills/requirements
        skill_matches = []
        missing_requirements = []
        
        for requirement in requirements:
            match = await self._match_requirement(user_profile, requirement)
            skill_matches.append(match)
            
            if match.match_level == MatchLevel.NO_MATCH:
                missing_requirements.append(requirement)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(skill_matches)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores, skill_matches)
        
        # Generate insights
        strengths = self._identify_strengths(skill_matches)
        gaps = self._identify_gaps(missing_requirements, skill_matches)
        recommendations = await self._generate_recommendations(
            user_profile, job_posting, skill_matches, missing_requirements
        )
        
        # Determine fit level
        fit_level = self._determine_fit_level(overall_score)
        
        result = MatchResult(
            job_posting=job_posting,
            overall_score=overall_score,
            category_scores=category_scores,
            skill_matches=skill_matches,
            missing_requirements=missing_requirements,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
            fit_level=fit_level
        )
        
        self.logger.info(
            "Job matching completed",
            job_title=job_posting.title,
            overall_score=overall_score,
            fit_level=fit_level,
            missing_count=len(missing_requirements)
        )
        
        return result
    
    async def _match_requirement(
        self,
        user_profile: UserProfile,
        requirement: JobRequirement
    ) -> SkillMatch:
        """Match a single requirement against user profile."""
        
        # Get user skills as lowercase set for comparison
        user_skills = set()
        if user_profile.skills:
            user_skills = {skill.name.lower() for skill in user_profile.skills}
        
        # Check for exact matches
        req_keywords = [kw.lower() for kw in requirement.keywords]
        
        for req_keyword in req_keywords:
            # Direct match
            if req_keyword in user_skills:
                return SkillMatch(
                    requirement=requirement,
                    user_skill=req_keyword,
                    match_level=MatchLevel.EXACT,
                    confidence=0.95
                )
            
            # Synonym match
            synonyms = self.skill_synonyms.get(req_keyword, [])
            for synonym in synonyms:
                if synonym in user_skills:
                    return SkillMatch(
                        requirement=requirement,
                        user_skill=synonym,
                        match_level=MatchLevel.STRONG,
                        confidence=0.85
                    )
        
        # Partial matches (substring matching)
        for user_skill in user_skills:
            for req_keyword in req_keywords:
                if self._is_partial_match(user_skill, req_keyword):
                    return SkillMatch(
                        requirement=requirement,
                        user_skill=user_skill,
                        match_level=MatchLevel.PARTIAL,
                        confidence=0.6
                    )
        
        # Experience level matching
        if requirement.type == RequirementType.EXPERIENCE:
            experience_match = self._match_experience(user_profile, requirement)
            if experience_match:
                return experience_match
        
        # Education matching
        if requirement.type == RequirementType.EDUCATION:
            education_match = self._match_education(user_profile, requirement)
            if education_match:
                return education_match
        
        # No match found
        gap_analysis = self._analyze_skill_gap(user_profile, requirement)
        
        return SkillMatch(
            requirement=requirement,
            user_skill=None,
            match_level=MatchLevel.NO_MATCH,
            confidence=0.0,
            gap_analysis=gap_analysis
        )
    
    def _is_partial_match(self, user_skill: str, req_keyword: str) -> bool:
        """Check if there's a partial match between skills."""
        # Check if one is substring of the other
        if req_keyword in user_skill or user_skill in req_keyword:
            return True
        
        # Check for common words
        user_words = set(user_skill.split())
        req_words = set(req_keyword.split())
        
        # If they share significant words, consider it a partial match
        common_words = user_words & req_words
        if len(common_words) > 0 and len(common_words) >= min(len(user_words), len(req_words)) * 0.5:
            return True
        
        return False
    
    def _match_experience(self, user_profile: UserProfile, requirement: JobRequirement) -> Optional[SkillMatch]:
        """Match experience requirements."""
        if not user_profile.experience or not requirement.years_experience:
            return None
        
        # Calculate total years of experience
        total_experience = 0
        for exp in user_profile.experience:
            if hasattr(exp, 'duration_years'):
                total_experience += exp.duration_years
            elif hasattr(exp, 'start_date') and hasattr(exp, 'end_date'):
                # Calculate years from dates if available
                # This is a simplified calculation
                total_experience += 2  # Default assumption
        
        required_years = requirement.years_experience
        
        if total_experience >= required_years:
            match_level = MatchLevel.EXACT if total_experience >= required_years else MatchLevel.STRONG
            confidence = min(0.95, 0.7 + (total_experience / required_years) * 0.25)
        elif total_experience >= required_years * 0.7:
            match_level = MatchLevel.PARTIAL
            confidence = 0.6
        else:
            match_level = MatchLevel.WEAK
            confidence = 0.3
        
        years_gap = max(0, required_years - total_experience)
        
        return SkillMatch(
            requirement=requirement,
            user_skill=f"{total_experience} years experience",
            match_level=match_level,
            confidence=confidence,
            years_gap=years_gap if years_gap > 0 else None
        )
    
    def _match_education(self, user_profile: UserProfile, requirement: JobRequirement) -> Optional[SkillMatch]:
        """Match education requirements."""
        if not user_profile.education:
            return None
        
        # Simple education level matching
        user_education_levels = set()
        for edu in user_profile.education:
            if hasattr(edu, 'degree'):
                user_education_levels.add(edu.degree.lower())
        
        req_text_lower = requirement.text.lower()
        
        # Check for degree matches
        if "bachelor" in req_text_lower and any("bachelor" in edu for edu in user_education_levels):
            return SkillMatch(
                requirement=requirement,
                user_skill="Bachelor's degree",
                match_level=MatchLevel.EXACT,
                confidence=0.9
            )
        elif "master" in req_text_lower and any("master" in edu for edu in user_education_levels):
            return SkillMatch(
                requirement=requirement,
                user_skill="Master's degree",
                match_level=MatchLevel.EXACT,
                confidence=0.9
            )
        elif "phd" in req_text_lower and any("phd" in edu or "doctorate" in edu for edu in user_education_levels):
            return SkillMatch(
                requirement=requirement,
                user_skill="PhD",
                match_level=MatchLevel.EXACT,
                confidence=0.9
            )
        
        return None
    
    def _analyze_skill_gap(self, user_profile: UserProfile, requirement: JobRequirement) -> str:
        """Analyze what's missing for a requirement."""
        if requirement.type == RequirementType.TECHNICAL_SKILL:
            return f"Need to learn {requirement.text}"
        elif requirement.type == RequirementType.EXPERIENCE:
            if requirement.years_experience:
                return f"Need {requirement.years_experience} years of experience in {requirement.text}"
            return f"Need experience in {requirement.text}"
        elif requirement.type == RequirementType.EDUCATION:
            return f"Need {requirement.text}"
        elif requirement.type == RequirementType.CERTIFICATION:
            return f"Need {requirement.text} certification"
        else:
            return f"Need to develop {requirement.text}"
    
    def _calculate_category_scores(self, skill_matches: List[SkillMatch]) -> Dict[str, float]:
        """Calculate scores by requirement category."""
        category_scores = {}
        category_counts = {}
        
        for match in skill_matches:
            category = match.requirement.category
            if category not in category_scores:
                category_scores[category] = 0.0
                category_counts[category] = 0
            
            # Score based on match level and requirement importance
            match_score = self._get_match_score(match.match_level)
            weight = self.scoring_weights[match.requirement.level]
            
            category_scores[category] += match_score * weight
            category_counts[category] += weight
        
        # Calculate averages
        for category in category_scores:
            if category_counts[category] > 0:
                category_scores[category] /= category_counts[category]
        
        return category_scores
    
    def _get_match_score(self, match_level: MatchLevel) -> float:
        """Convert match level to numeric score."""
        score_map = {
            MatchLevel.EXACT: 1.0,
            MatchLevel.STRONG: 0.8,
            MatchLevel.PARTIAL: 0.5,
            MatchLevel.WEAK: 0.2,
            MatchLevel.NO_MATCH: 0.0
        }
        return score_map[match_level]
    
    def _calculate_overall_score(
        self,
        category_scores: Dict[str, float],
        skill_matches: List[SkillMatch]
    ) -> float:
        """Calculate overall matching score."""
        
        # Weight categories by importance
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            # Map categories to standard weights
            if "technical" in category or "programming" in category:
                weight = self.category_weights["technical_skills"]
            elif "experience" in category:
                weight = self.category_weights["experience"]
            elif "education" in category:
                weight = self.category_weights["education"]
            elif "soft" in category or "communication" in category:
                weight = self.category_weights["soft_skills"]
            elif "certification" in category:
                weight = self.category_weights["certifications"]
            else:
                weight = 0.1  # Default weight for other categories
            
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0
        
        # Apply penalty for missing required skills
        required_matches = [m for m in skill_matches if m.requirement.level == RequirementLevel.REQUIRED]
        missing_required = [m for m in required_matches if m.match_level == MatchLevel.NO_MATCH]
        
        if required_matches:
            required_penalty = len(missing_required) / len(required_matches) * 0.3
            overall_score = max(0.0, overall_score - required_penalty)
        
        return min(1.0, overall_score)
    
    def _identify_strengths(self, skill_matches: List[SkillMatch]) -> List[str]:
        """Identify user's strengths based on matches."""
        strengths = []
        
        # Strong matches
        strong_matches = [
            m for m in skill_matches 
            if m.match_level in [MatchLevel.EXACT, MatchLevel.STRONG]
        ]
        
        # Group by category
        category_strengths = {}
        for match in strong_matches:
            category = match.requirement.category
            if category not in category_strengths:
                category_strengths[category] = []
            category_strengths[category].append(match.requirement.text)
        
        # Generate strength statements
        for category, skills in category_strengths.items():
            if len(skills) >= 3:
                strengths.append(f"Strong {category.replace('_', ' ')} background")
            elif len(skills) >= 1:
                strengths.append(f"Good {category.replace('_', ' ')} skills")
        
        # Specific high-value skills
        high_value_matches = [
            m for m in strong_matches
            if m.requirement.level == RequirementLevel.REQUIRED
        ]
        
        if len(high_value_matches) >= 5:
            strengths.append("Meets most required qualifications")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_gaps(
        self,
        missing_requirements: List[JobRequirement],
        skill_matches: List[SkillMatch]
    ) -> List[str]:
        """Identify skill gaps and areas for improvement."""
        gaps = []
        
        # Missing required skills
        missing_required = [
            req for req in missing_requirements
            if req.level == RequirementLevel.REQUIRED
        ]
        
        for req in missing_required[:3]:  # Top 3 missing required skills
            gaps.append(f"Missing required skill: {req.text}")
        
        # Weak matches that could be improved
        weak_matches = [
            m for m in skill_matches
            if m.match_level in [MatchLevel.WEAK, MatchLevel.PARTIAL]
            and m.requirement.level == RequirementLevel.REQUIRED
        ]
        
        for match in weak_matches[:2]:  # Top 2 weak areas
            gaps.append(f"Could strengthen: {match.requirement.text}")
        
        # Experience gaps
        experience_gaps = [
            m for m in skill_matches
            if m.years_gap and m.years_gap > 0
        ]
        
        if experience_gaps:
            max_gap = max(experience_gaps, key=lambda x: x.years_gap)
            gaps.append(f"Need {max_gap.years_gap} more years of experience")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    async def _generate_recommendations(
        self,
        user_profile: UserProfile,
        job_posting: JobPosting,
        skill_matches: List[SkillMatch],
        missing_requirements: List[JobRequirement]
    ) -> List[str]:
        """Generate actionable recommendations for the user."""
        recommendations = []
        
        # Learning recommendations for missing skills
        missing_technical = [
            req for req in missing_requirements
            if req.type == RequirementType.TECHNICAL_SKILL
            and req.level in [RequirementLevel.REQUIRED, RequirementLevel.PREFERRED]
        ]
        
        for req in missing_technical[:2]:
            recommendations.append(f"Learn {req.text} to meet job requirements")
        
        # Experience building recommendations
        experience_gaps = [
            m for m in skill_matches
            if m.years_gap and m.years_gap > 0
        ]
        
        if experience_gaps:
            recommendations.append("Gain more hands-on experience through projects or internships")
        
        # Certification recommendations
        missing_certs = [
            req for req in missing_requirements
            if req.type == RequirementType.CERTIFICATION
        ]
        
        if missing_certs:
            recommendations.append(f"Consider getting {missing_certs[0].text}")
        
        # Application strategy recommendations
        overall_score = self._calculate_overall_score(
            self._calculate_category_scores(skill_matches),
            skill_matches
        )
        
        if overall_score >= 0.8:
            recommendations.append("Strong candidate - apply with confidence")
        elif overall_score >= 0.6:
            recommendations.append("Good fit - highlight your relevant experience")
        elif overall_score >= 0.4:
            recommendations.append("Consider applying but address skill gaps in cover letter")
        else:
            recommendations.append("Build more relevant skills before applying")
        
        return recommendations[:5]
    
    def _determine_fit_level(self, overall_score: float) -> str:
        """Determine overall fit level based on score."""
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    async def batch_match_jobs(
        self,
        user_profile: UserProfile,
        job_postings: List[JobPosting]
    ) -> List[MatchResult]:
        """Match user profile against multiple job postings."""
        self.logger.info(
            "Batch matching jobs",
            job_count=len(job_postings),
            user_skills_count=len(user_profile.skills) if user_profile.skills else 0
        )
        
        results = []
        for job_posting in job_postings:
            try:
                result = await self.match_job(user_profile, job_posting)
                results.append(result)
            except Exception as e:
                self.logger.error(
                    "Failed to match job",
                    job_title=job_posting.title,
                    error=str(e)
                )
                continue
        
        # Sort by overall score (best matches first)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        self.logger.info(
            "Batch matching completed",
            total_jobs=len(job_postings),
            successful_matches=len(results)
        )
        
        return results
    
    def get_match_summary(self, results: List[MatchResult]) -> Dict[str, Any]:
        """Generate summary statistics for match results."""
        if not results:
            return {"total_jobs": 0, "matches": []}
        
        fit_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for result in results:
            fit_counts[result.fit_level] += 1
        
        avg_score = sum(r.overall_score for r in results) / len(results)
        
        top_matches = results[:5]  # Top 5 matches
        
        return {
            "total_jobs": len(results),
            "average_score": avg_score,
            "fit_distribution": fit_counts,
            "top_matches": [
                {
                    "title": r.job_posting.title,
                    "company": r.job_posting.company,
                    "score": r.overall_score,
                    "fit_level": r.fit_level
                }
                for r in top_matches
            ]
        }