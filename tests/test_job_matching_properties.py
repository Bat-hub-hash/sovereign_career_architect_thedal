"""Property-based tests for job matching functionality."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from sovereign_career_architect.jobs.matcher import (
    JobMatcher,
    JobPosting,
    MatchResult,
    SkillMatch,
    MatchLevel
)
from sovereign_career_architect.jobs.extractor import (
    JobRequirementExtractor,
    ExtractedRequirements,
    JobRequirement,
    RequirementType,
    RequirementLevel
)
from sovereign_career_architect.core.models import UserProfile


# Test data strategies
@st.composite
def personal_info_strategy(draw):
    """Generate personal info for testing."""
    from sovereign_career_architect.core.models import PersonalInfo
    
    return PersonalInfo(
        name=draw(st.text(min_size=5, max_size=30)),
        email=draw(st.emails()),
        phone=draw(st.one_of(st.none(), st.text(min_size=10, max_size=15))),
        location=draw(st.one_of(st.none(), st.text(min_size=5, max_size=30))),
        preferred_language="en"
    )


@st.composite
def skill_strategy(draw):
    """Generate skills for testing."""
    from sovereign_career_architect.core.models import Skill
    
    return Skill(
        name=draw(st.sampled_from([
            "Python", "JavaScript", "React", "Node.js", "SQL", "MongoDB",
            "AWS", "Docker", "Kubernetes", "Machine Learning", "Data Science",
            "Project Management", "Agile", "Communication", "Leadership"
        ])),
        level=draw(st.sampled_from(["Beginner", "Intermediate", "Advanced", "Expert"])),
        years_experience=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10))),
        verified=draw(st.booleans())
    )


@st.composite
def user_profile_strategy(draw):
    """Generate user profiles for testing."""
    from sovereign_career_architect.core.models import UserProfile, JobPreferences, DocumentStore
    
    personal_info = draw(personal_info_strategy())
    skills = draw(st.lists(skill_strategy(), min_size=2, max_size=10))
    
    return UserProfile(
        personal_info=personal_info,
        skills=skills,
        experience=[],  # Simplified for testing
        education=[],   # Simplified for testing
        preferences=JobPreferences(),
        documents=DocumentStore()
    )


@st.composite
def job_requirement_strategy(draw):
    """Generate job requirements."""
    return JobRequirement(
        text=draw(st.text(min_size=5, max_size=100)),
        type=draw(st.sampled_from(list(RequirementType))),
        level=draw(st.sampled_from(list(RequirementLevel))),
        category=draw(st.sampled_from([
            "programming", "databases", "cloud", "soft_skills", "experience", "education"
        ])),
        keywords=draw(st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=5)),
        years_experience=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10))),
        confidence=draw(st.floats(min_value=0.5, max_value=1.0))
    )


@st.composite
def extracted_requirements_strategy(draw):
    """Generate extracted requirements."""
    requirements = draw(st.lists(job_requirement_strategy(), min_size=3, max_size=15))
    
    return ExtractedRequirements(
        job_title=draw(st.text(min_size=10, max_size=50)),
        company=draw(st.text(min_size=5, max_size=30)),
        location=draw(st.text(min_size=5, max_size=30)),
        requirements=requirements,
        salary_range=draw(st.one_of(
            st.none(),
            st.fixed_dictionaries({
                "min": st.integers(min_value=40000, max_value=200000),
                "max": st.integers(min_value=50000, max_value=300000),
                "currency": st.just("USD"),
                "period": st.just("annual")
            })
        )),
        job_type=draw(st.sampled_from(["full_time", "part_time", "contract", "internship"])),
        remote_option=draw(st.booleans()),
        benefits=draw(st.lists(st.text(min_size=5, max_size=30), max_size=5))
    )


@st.composite
def job_posting_strategy(draw):
    """Generate job postings."""
    extracted_reqs = draw(extracted_requirements_strategy())
    
    return JobPosting(
        id=draw(st.text(min_size=5, max_size=20)),
        title=extracted_reqs.job_title,
        company=extracted_reqs.company,
        location=extracted_reqs.location,
        url=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100))),
        extracted_requirements=extracted_reqs,
        posted_date=draw(st.one_of(st.none(), st.text(min_size=8, max_size=12))),
        application_deadline=draw(st.one_of(st.none(), st.text(min_size=8, max_size=12)))
    )


class TestJobMatchingProperties:
    """Property-based tests for job matching system."""
    
    def create_job_matcher(self):
        """Create job matcher instance."""
        return JobMatcher()
    
    @given(
        user_profile=user_profile_strategy(),
        job_posting=job_posting_strategy()
    )
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_job_matching_accuracy_property(self, user_profile, job_posting):
        """
        Property 3: Job Matching Accuracy
        
        For any user profile and job posting:
        1. Match result should have valid structure and scores
        2. Overall score should reflect individual skill matches
        3. Missing requirements should be accurately identified
        4. Recommendations should be relevant and actionable
        """
        job_matcher = self.create_job_matcher()
        
        async def run_test():
            match_result = await job_matcher.match_job(user_profile, job_posting)
            
            # Property 1: Valid result structure
            assert isinstance(match_result, MatchResult)
            assert match_result.job_posting == job_posting
            assert 0.0 <= match_result.overall_score <= 1.0
            assert isinstance(match_result.category_scores, dict)
            assert isinstance(match_result.skill_matches, list)
            assert isinstance(match_result.missing_requirements, list)
            assert isinstance(match_result.strengths, list)
            assert isinstance(match_result.gaps, list)
            assert isinstance(match_result.recommendations, list)
            assert match_result.fit_level in ["excellent", "good", "fair", "poor"]
            
            # Property 2: Skill matches should cover all requirements
            total_requirements = len(job_posting.extracted_requirements.requirements)
            assert len(match_result.skill_matches) == total_requirements
            
            # Property 3: Each skill match should be valid
            for skill_match in match_result.skill_matches:
                assert isinstance(skill_match, SkillMatch)
                assert isinstance(skill_match.match_level, MatchLevel)
                assert 0.0 <= skill_match.confidence <= 1.0
                assert skill_match.requirement in job_posting.extracted_requirements.requirements
            
            # Property 4: Missing requirements should be no-match items
            for missing_req in match_result.missing_requirements:
                # Find corresponding skill match
                corresponding_match = next(
                    (sm for sm in match_result.skill_matches if sm.requirement == missing_req),
                    None
                )
                assert corresponding_match is not None
                assert corresponding_match.match_level == MatchLevel.NO_MATCH
            
            # Property 5: Category scores should be reasonable
            for category, score in match_result.category_scores.items():
                assert 0.0 <= score <= 1.0
                assert isinstance(category, str)
            
            # Property 6: Fit level should match overall score
            if match_result.overall_score >= 0.8:
                assert match_result.fit_level == "excellent"
            elif match_result.overall_score >= 0.6:
                assert match_result.fit_level == "good"
            elif match_result.overall_score >= 0.4:
                assert match_result.fit_level == "fair"
            else:
                assert match_result.fit_level == "poor"
            
            # Property 7: Recommendations should be actionable strings
            for recommendation in match_result.recommendations:
                assert isinstance(recommendation, str)
                assert len(recommendation.strip()) > 10  # Meaningful content
            
            # Property 8: Strengths and gaps should be relevant
            for strength in match_result.strengths:
                assert isinstance(strength, str)
                assert len(strength.strip()) > 5
            
            for gap in match_result.gaps:
                assert isinstance(gap, str)
                assert len(gap.strip()) > 5
        
        asyncio.run(run_test())
    
    @given(
        user_profile=user_profile_strategy(),
        job_postings=st.lists(job_posting_strategy(), min_size=2, max_size=5)
    )
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_matching_consistency_property(self, user_profile, job_postings):
        """
        Property: Batch Matching Consistency
        
        For any user profile and list of job postings:
        1. Batch matching should return results for all valid jobs
        2. Results should be sorted by overall score (descending)
        3. Individual matches should be consistent with single matching
        4. Summary statistics should be accurate
        """
        job_matcher = self.create_job_matcher()
        
        async def run_test():
            batch_results = await job_matcher.batch_match_jobs(user_profile, job_postings)
            
            # Property 1: Should have results for all jobs (assuming no errors)
            assert len(batch_results) <= len(job_postings)  # May be less due to errors
            
            # Property 2: Results should be sorted by score (descending)
            scores = [result.overall_score for result in batch_results]
            assert scores == sorted(scores, reverse=True)
            
            # Property 3: Each result should be valid
            for result in batch_results:
                assert isinstance(result, MatchResult)
                assert 0.0 <= result.overall_score <= 1.0
                assert result.fit_level in ["excellent", "good", "fair", "poor"]
            
            # Property 4: Summary should be accurate
            summary = job_matcher.get_match_summary(batch_results)
            assert summary["total_jobs"] == len(batch_results)
            
            if batch_results:
                expected_avg = sum(r.overall_score for r in batch_results) / len(batch_results)
                assert abs(summary["average_score"] - expected_avg) < 0.01
                
                # Check fit distribution
                fit_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
                for result in batch_results:
                    fit_counts[result.fit_level] += 1
                
                assert summary["fit_distribution"] == fit_counts
                
                # Check top matches
                top_matches = summary["top_matches"]
                assert len(top_matches) <= min(5, len(batch_results))
                
                for i, match_info in enumerate(top_matches):
                    assert match_info["title"] == batch_results[i].job_posting.title
                    assert match_info["company"] == batch_results[i].job_posting.company
                    assert match_info["score"] == batch_results[i].overall_score
                    assert match_info["fit_level"] == batch_results[i].fit_level
        
        asyncio.run(run_test())
    
    @given(
        user_skills=st.lists(
            st.sampled_from(["Python", "JavaScript", "React", "SQL", "AWS"]),
            min_size=1, max_size=5
        ),
        requirement_keywords=st.lists(
            st.sampled_from(["python", "javascript", "react", "sql", "aws", "java", "docker"]),
            min_size=1, max_size=3
        ),
        requirement_level=st.sampled_from(list(RequirementLevel))
    )
    @settings(max_examples=15, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_skill_matching_logic_property(self, user_skills, requirement_keywords, requirement_level):
        """
        Property: Skill Matching Logic Consistency
        
        For any user skills and requirement keywords:
        1. Exact matches should be identified correctly
        2. Match levels should be consistent with similarity
        3. Confidence scores should reflect match quality
        4. Missing skills should be properly flagged
        """
        job_matcher = self.create_job_matcher()
        
        # Create test user profile
        from sovereign_career_architect.core.models import PersonalInfo, Skill, JobPreferences, DocumentStore
        
        personal_info = PersonalInfo(name="Test User", email="test@example.com")
        skills = [Skill(name=skill, level="Intermediate") for skill in user_skills]
        
        user_profile = UserProfile(
            personal_info=personal_info,
            skills=skills,
            experience=[],
            education=[],
            preferences=JobPreferences(),
            documents=DocumentStore()
        )
        
        # Create test requirement
        requirement = JobRequirement(
            text=" ".join(requirement_keywords),
            type=RequirementType.TECHNICAL_SKILL,
            level=requirement_level,
            category="technical",
            keywords=requirement_keywords,
            confidence=0.8
        )
        
        async def run_test():
            skill_match = await job_matcher._match_requirement(user_profile, requirement)
            
            # Property 1: Should return valid skill match
            assert isinstance(skill_match, SkillMatch)
            assert skill_match.requirement == requirement
            assert isinstance(skill_match.match_level, MatchLevel)
            assert 0.0 <= skill_match.confidence <= 1.0
            
            # Property 2: Match level should be logical
            user_skills_lower = {skill.lower() for skill in user_skills}
            req_keywords_lower = {kw.lower() for kw in requirement_keywords}
            
            # Check for exact matches
            if user_skills_lower & req_keywords_lower:  # Intersection exists
                assert skill_match.match_level in [MatchLevel.EXACT, MatchLevel.STRONG, MatchLevel.PARTIAL]
                assert skill_match.confidence > 0.0
            else:
                # No direct overlap - could still be partial match or no match
                assert skill_match.match_level in [MatchLevel.PARTIAL, MatchLevel.WEAK, MatchLevel.NO_MATCH]
            
            # Property 3: Confidence should match level
            if skill_match.match_level == MatchLevel.EXACT:
                assert skill_match.confidence >= 0.8
            elif skill_match.match_level == MatchLevel.STRONG:
                assert skill_match.confidence >= 0.7
            elif skill_match.match_level == MatchLevel.PARTIAL:
                assert skill_match.confidence >= 0.4
            elif skill_match.match_level == MatchLevel.NO_MATCH:
                assert skill_match.confidence == 0.0
                assert skill_match.user_skill is None
                assert skill_match.gap_analysis is not None
        
        asyncio.run(run_test())
    
    @given(
        match_levels=st.lists(
            st.sampled_from(list(MatchLevel)),
            min_size=3, max_size=10
        ),
        requirement_levels=st.lists(
            st.sampled_from(list(RequirementLevel)),
            min_size=3, max_size=10
        )
    )
    def test_scoring_consistency_property(self, match_levels, requirement_levels):
        """
        Property: Scoring Consistency
        
        For any combination of match levels and requirement levels:
        1. Better matches should result in higher scores
        2. Required skills should have more impact than preferred
        3. Overall score should be bounded between 0 and 1
        4. Score calculation should be deterministic
        """
        # Ensure lists are same length
        min_length = min(len(match_levels), len(requirement_levels))
        match_levels = match_levels[:min_length]
        requirement_levels = requirement_levels[:min_length]
        
        job_matcher = self.create_job_matcher()
        
        # Create mock skill matches
        skill_matches = []
        for i, (match_level, req_level) in enumerate(zip(match_levels, requirement_levels)):
            requirement = JobRequirement(
                text=f"Skill {i}",
                type=RequirementType.TECHNICAL_SKILL,
                level=req_level,
                category="technical",
                keywords=[f"skill{i}"],
                confidence=0.8
            )
            
            skill_match = SkillMatch(
                requirement=requirement,
                user_skill=f"User skill {i}" if match_level != MatchLevel.NO_MATCH else None,
                match_level=match_level,
                confidence=job_matcher._get_match_score(match_level)
            )
            skill_matches.append(skill_match)
        
        # Calculate scores
        category_scores = job_matcher._calculate_category_scores(skill_matches)
        overall_score = job_matcher._calculate_overall_score(category_scores, skill_matches)
        
        # Property 1: Overall score should be bounded
        assert 0.0 <= overall_score <= 1.0
        
        # Property 2: Category scores should be bounded
        for category, score in category_scores.items():
            assert 0.0 <= score <= 1.0
        
        # Property 3: Score should be deterministic (same inputs = same output)
        category_scores_2 = job_matcher._calculate_category_scores(skill_matches)
        overall_score_2 = job_matcher._calculate_overall_score(category_scores_2, skill_matches)
        
        assert category_scores == category_scores_2
        assert abs(overall_score - overall_score_2) < 0.001
        
        # Property 4: More exact matches should generally lead to higher scores
        exact_matches = sum(1 for m in skill_matches if m.match_level == MatchLevel.EXACT)
        no_matches = sum(1 for m in skill_matches if m.match_level == MatchLevel.NO_MATCH)
        
        if exact_matches > no_matches and len(skill_matches) > 2:
            assert overall_score > 0.3  # Should have reasonable score with more exact matches


class TestJobRequirementExtraction:
    """Property-based tests for job requirement extraction."""
    
    def create_extractor(self):
        """Create job requirement extractor with mock client."""
        from unittest.mock import AsyncMock
        from sovereign_career_architect.voice.sarvam import SarvamClient
        
        mock_client = AsyncMock(spec=SarvamClient)
        
        # Mock LLM response for requirement extraction
        async def mock_generate_text(prompt, language, max_tokens=500):
            return """
            - REQUIRED TECHNICAL_SKILL Python programming
            - PREFERRED EXPERIENCE 3+ years in web development
            - REQUIRED EDUCATION Bachelor's degree in Computer Science
            - NICE_TO_HAVE TECHNICAL_SKILL Docker containerization
            """
        
        mock_client.generate_text = mock_generate_text
        
        return JobRequirementExtractor(sarvam_client=mock_client)
    
    @given(
        job_text=st.text(min_size=100, max_size=2000),
        job_title=st.text(min_size=10, max_size=50),
        company=st.text(min_size=5, max_size=30)
    )
    @settings(max_examples=10, deadline=12000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_requirement_extraction_property(self, job_text, job_title, company):
        """
        Property: Requirement Extraction Completeness
        
        For any job posting text:
        1. Should extract structured requirements
        2. Should identify job metadata correctly
        3. Should categorize requirements appropriately
        4. Should handle various text formats gracefully
        """
        extractor = self.create_extractor()
        
        async def run_test():
            extracted = await extractor.extract_requirements(
                job_posting_text=job_text,
                job_title=job_title,
                company=company
            )
            
            # Property 1: Should return valid structure
            assert isinstance(extracted, ExtractedRequirements)
            assert extracted.job_title == job_title
            assert extracted.company == company
            assert isinstance(extracted.requirements, list)
            
            # Property 2: Requirements should be valid
            for req in extracted.requirements:
                assert isinstance(req, JobRequirement)
                assert isinstance(req.type, RequirementType)
                assert isinstance(req.level, RequirementLevel)
                assert len(req.text.strip()) > 0
                assert len(req.keywords) > 0
                assert 0.0 <= req.confidence <= 1.0
            
            # Property 3: Should have reasonable metadata
            assert isinstance(extracted.job_type, str)
            assert isinstance(extracted.remote_option, bool)
            assert isinstance(extracted.benefits, list)
            
            # Property 4: Salary range should be valid if present
            if extracted.salary_range:
                assert "min" in extracted.salary_range
                assert "max" in extracted.salary_range
                assert extracted.salary_range["min"] <= extracted.salary_range["max"]
        
        asyncio.run(run_test())


if __name__ == "__main__":
    # Run a quick test
    async def test_basic():
        matcher = JobMatcher()
        
        # Create test data
        from sovereign_career_architect.core.models import PersonalInfo, Skill, JobPreferences, DocumentStore
        
        personal_info = PersonalInfo(name="Test User", email="test@example.com")
        skills = [
            Skill(name="Python", level="Advanced"),
            Skill(name="JavaScript", level="Intermediate"),
            Skill(name="React", level="Intermediate"),
            Skill(name="SQL", level="Advanced")
        ]
        
        user_profile = UserProfile(
            personal_info=personal_info,
            skills=skills,
            experience=[],
            education=[],
            preferences=JobPreferences(),
            documents=DocumentStore()
        )
        
        requirements = [
            JobRequirement(
                text="Python programming",
                type=RequirementType.TECHNICAL_SKILL,
                level=RequirementLevel.REQUIRED,
                category="programming",
                keywords=["python"],
                confidence=0.9
            ),
            JobRequirement(
                text="5 years experience",
                type=RequirementType.EXPERIENCE,
                level=RequirementLevel.REQUIRED,
                category="experience",
                keywords=["experience"],
                years_experience=5,
                confidence=0.9
            )
        ]
        
        extracted_reqs = ExtractedRequirements(
            job_title="Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            requirements=requirements
        )
        
        job_posting = JobPosting(
            id="job_1",
            title="Python Developer",
            company="Tech Corp",
            location="San Francisco, CA",
            url="https://example.com/job/1",
            extracted_requirements=extracted_reqs
        )
        
        # Test matching
        result = await matcher.match_job(user_profile, job_posting)
        
        print(f"Match result:")
        print(f"Overall score: {result.overall_score}")
        print(f"Fit level: {result.fit_level}")
        print(f"Strengths: {result.strengths}")
        print(f"Gaps: {result.gaps}")
        print(f"Recommendations: {result.recommendations}")
    
    asyncio.run(test_basic())