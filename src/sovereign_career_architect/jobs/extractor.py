"""Job requirement extraction from job postings."""

import asyncio
import re
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import structlog

from sovereign_career_architect.voice.sarvam import SarvamClient
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class RequirementType(Enum):
    """Types of job requirements."""
    TECHNICAL_SKILL = "technical_skill"
    SOFT_SKILL = "soft_skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    LANGUAGE = "language"
    TOOL = "tool"
    FRAMEWORK = "framework"


class RequirementLevel(Enum):
    """Requirement importance levels."""
    REQUIRED = "required"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


@dataclass
class JobRequirement:
    """Individual job requirement."""
    text: str
    type: RequirementType
    level: RequirementLevel
    category: str
    keywords: List[str]
    years_experience: Optional[int] = None
    confidence: float = 1.0


@dataclass
class ExtractedRequirements:
    """Complete set of extracted requirements from a job posting."""
    job_title: str
    company: str
    location: str
    requirements: List[JobRequirement]
    salary_range: Optional[Dict[str, Any]] = None
    job_type: str = "full_time"
    remote_option: bool = False
    benefits: List[str] = None
    
    def __post_init__(self):
        if self.benefits is None:
            self.benefits = []


class JobRequirementExtractor:
    """Extracts structured requirements from job posting text."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.logger = logger.bind(component="job_extractor")
        self.sarvam_client = sarvam_client or SarvamClient()
        
        # Skill databases and patterns
        self.skill_patterns = self._load_skill_patterns()
        self.experience_patterns = self._load_experience_patterns()
        self.requirement_indicators = self._load_requirement_indicators()
        
    def _load_skill_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load patterns for identifying different types of skills."""
        return {
            "programming_languages": {
                "patterns": [
                    r"\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin)\b",
                    r"\b(React|Angular|Vue\.js|Node\.js|Django|Flask|Spring|Express)\b"
                ],
                "keywords": [
                    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                    "ruby", "php", "swift", "kotlin", "react", "angular", "vue", "node",
                    "django", "flask", "spring", "express"
                ]
            },
            "databases": {
                "patterns": [
                    r"\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB)\b",
                    r"\b(SQL|NoSQL|Database|DB)\b"
                ],
                "keywords": [
                    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
                    "dynamodb", "sql", "nosql", "database"
                ]
            },
            "cloud_platforms": {
                "patterns": [
                    r"\b(AWS|Azure|GCP|Google Cloud|Amazon Web Services|Microsoft Azure)\b",
                    r"\b(Docker|Kubernetes|Terraform|Ansible)\b"
                ],
                "keywords": [
                    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", 
                    "terraform", "ansible", "cloud"
                ]
            },
            "data_science": {
                "patterns": [
                    r"\b(Machine Learning|ML|Deep Learning|AI|Artificial Intelligence)\b",
                    r"\b(TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|Jupyter)\b",
                    r"\b(Statistics|Statistical Analysis|Data Analysis|Data Science)\b"
                ],
                "keywords": [
                    "machine learning", "deep learning", "ai", "tensorflow", "pytorch",
                    "scikit-learn", "pandas", "numpy", "statistics", "data science"
                ]
            },
            "soft_skills": {
                "patterns": [
                    r"\b(Leadership|Communication|Teamwork|Problem Solving|Critical Thinking)\b",
                    r"\b(Project Management|Agile|Scrum|Collaboration|Mentoring)\b"
                ],
                "keywords": [
                    "leadership", "communication", "teamwork", "problem solving",
                    "project management", "agile", "scrum", "collaboration", "mentoring"
                ]
            }
        }
    
    def _load_experience_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for extracting experience requirements."""
        return [
            {
                "pattern": r"(\d+)\+?\s*years?\s*of\s*experience",
                "type": "general_experience"
            },
            {
                "pattern": r"(\d+)\+?\s*years?\s*experience\s*(?:in|with)\s*([^,.]+)",
                "type": "specific_experience"
            },
            {
                "pattern": r"minimum\s*(\d+)\s*years?",
                "type": "minimum_experience"
            },
            {
                "pattern": r"(\d+)\+?\s*years?\s*(?:in|with)\s*([\w\s]+?)(?:\s*(?:and|or|\.|,))",
                "type": "technology_experience"
            }
        ]
    
    def _load_requirement_indicators(self) -> Dict[str, List[str]]:
        """Load indicators for requirement importance levels."""
        return {
            "required": [
                "required", "must have", "essential", "mandatory", "necessary",
                "need", "should have", "minimum", "at least"
            ],
            "preferred": [
                "preferred", "desired", "ideal", "would be great", "bonus",
                "plus", "advantage", "nice to have", "beneficial"
            ],
            "nice_to_have": [
                "nice to have", "bonus", "plus", "additional", "extra",
                "would be nice", "helpful", "beneficial"
            ]
        }
    
    async def extract_requirements(
        self,
        job_posting_text: str,
        job_title: str = "",
        company: str = "",
        location: str = ""
    ) -> ExtractedRequirements:
        """
        Extract structured requirements from job posting text.
        
        Args:
            job_posting_text: Raw job posting text
            job_title: Job title (optional)
            company: Company name (optional)
            location: Job location (optional)
            
        Returns:
            Structured requirements
        """
        self.logger.info(
            "Extracting job requirements",
            job_title=job_title,
            company=company,
            text_length=len(job_posting_text)
        )
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(job_posting_text)
        
        # Extract basic job information if not provided
        if not job_title:
            job_title = await self._extract_job_title(cleaned_text)
        if not company:
            company = await self._extract_company(cleaned_text)
        if not location:
            location = await self._extract_location(cleaned_text)
        
        # Extract requirements using multiple approaches
        requirements = []
        
        # 1. Pattern-based extraction
        pattern_requirements = await self._extract_with_patterns(cleaned_text)
        requirements.extend(pattern_requirements)
        
        # 2. LLM-based extraction for complex requirements
        llm_requirements = await self._extract_with_llm(cleaned_text, job_title)
        requirements.extend(llm_requirements)
        
        # 3. Keyword-based extraction
        keyword_requirements = await self._extract_with_keywords(cleaned_text)
        requirements.extend(keyword_requirements)
        
        # Deduplicate and merge similar requirements
        merged_requirements = self._merge_similar_requirements(requirements)
        
        # Extract additional job details
        salary_range = await self._extract_salary_range(cleaned_text)
        job_type = await self._extract_job_type(cleaned_text)
        remote_option = await self._extract_remote_option(cleaned_text)
        benefits = await self._extract_benefits(cleaned_text)
        
        extracted = ExtractedRequirements(
            job_title=job_title,
            company=company,
            location=location,
            requirements=merged_requirements,
            salary_range=salary_range,
            job_type=job_type,
            remote_option=remote_option,
            benefits=benefits
        )
        
        self.logger.info(
            "Job requirements extracted",
            job_title=job_title,
            requirements_count=len(merged_requirements),
            salary_range=salary_range is not None,
            remote_option=remote_option
        )
        
        return extracted
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize job posting text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\+\(\)\/]', ' ', text)
        
        return text.strip()
    
    async def _extract_job_title(self, text: str) -> str:
        """Extract job title from posting text."""
        # Look for common job title patterns
        title_patterns = [
            r"(?:job title|position|role):\s*([^\n\r]+)",
            r"we are looking for (?:a|an)\s*([^\n\r]+?)(?:\s*to|\s*who)",
            r"hiring\s*(?:a|an)?\s*([^\n\r]+?)(?:\s*to|\s*who|\s*for)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use first line if it looks like a title
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 100 and any(word in first_line.lower() for word in 
                                        ['engineer', 'developer', 'manager', 'analyst', 'scientist']):
            return first_line
        
        return "Unknown Position"
    
    async def _extract_company(self, text: str) -> str:
        """Extract company name from posting text."""
        company_patterns = [
            r"(?:company|organization):\s*([^\n\r]+)",
            r"join\s+([A-Z][^\s]+(?:\s+[A-Z][^\s]+)*)",
            r"at\s+([A-Z][^\s]+(?:\s+[A-Z][^\s]+)*)"
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                if len(company) < 50:  # Reasonable company name length
                    return company
        
        return "Unknown Company"
    
    async def _extract_location(self, text: str) -> str:
        """Extract job location from posting text."""
        location_patterns = [
            r"(?:location|based in|office in):\s*([^\n\r]+)",
            r"(?:remote|hybrid|on-site).*?(?:in|at)\s*([A-Z][^,\n\r]+)",
            r"([A-Z][a-z]+,\s*[A-Z]{2})",  # City, State format
            r"([A-Z][a-z]+\s*[A-Z][a-z]+)"  # City State format
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) < 100:
                    return location
        
        return "Unknown Location"
    
    async def _extract_with_patterns(self, text: str) -> List[JobRequirement]:
        """Extract requirements using regex patterns."""
        requirements = []
        
        # Extract technical skills
        for category, skill_data in self.skill_patterns.items():
            for pattern in skill_data["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    skill = match.group(0)
                    
                    # Determine requirement level from context
                    context = self._get_context(text, match.start(), match.end())
                    level = self._determine_requirement_level(context)
                    
                    # Extract experience years if mentioned
                    years_exp = self._extract_years_from_context(context)
                    
                    requirement = JobRequirement(
                        text=skill,
                        type=RequirementType.TECHNICAL_SKILL if category != "soft_skills" else RequirementType.SOFT_SKILL,
                        level=level,
                        category=category,
                        keywords=[skill.lower()],
                        years_experience=years_exp,
                        confidence=0.8
                    )
                    requirements.append(requirement)
        
        # Extract experience requirements
        for exp_pattern in self.experience_patterns:
            matches = re.finditer(exp_pattern["pattern"], text, re.IGNORECASE)
            for match in matches:
                years = int(match.group(1))
                context = self._get_context(text, match.start(), match.end())
                level = self._determine_requirement_level(context)
                
                if len(match.groups()) > 1:
                    # Specific experience
                    skill_area = match.group(2).strip()
                    requirement = JobRequirement(
                        text=f"{years} years experience in {skill_area}",
                        type=RequirementType.EXPERIENCE,
                        level=level,
                        category="experience",
                        keywords=[skill_area.lower()],
                        years_experience=years,
                        confidence=0.9
                    )
                else:
                    # General experience
                    requirement = JobRequirement(
                        text=f"{years} years of experience",
                        type=RequirementType.EXPERIENCE,
                        level=level,
                        category="experience",
                        keywords=["experience"],
                        years_experience=years,
                        confidence=0.9
                    )
                requirements.append(requirement)
        
        return requirements
    
    async def _extract_with_llm(self, text: str, job_title: str) -> List[JobRequirement]:
        """Extract requirements using LLM for complex parsing."""
        try:
            prompt = f"""
            Extract job requirements from this job posting for a {job_title} position.
            
            Job Posting:
            {text[:2000]}  # Limit text length
            
            Extract requirements in this format:
            - [REQUIRED/PREFERRED/NICE_TO_HAVE] [TECHNICAL_SKILL/SOFT_SKILL/EXPERIENCE/EDUCATION/CERTIFICATION] requirement_text
            
            Focus on:
            1. Technical skills and technologies
            2. Years of experience required
            3. Educational requirements
            4. Certifications
            5. Soft skills
            
            Example:
            - REQUIRED TECHNICAL_SKILL Python programming
            - PREFERRED EXPERIENCE 3+ years in web development
            - REQUIRED EDUCATION Bachelor's degree in Computer Science
            """
            
            response = await self.sarvam_client.generate_text(
                prompt=prompt,
                language="en",
                max_tokens=500
            )
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            self.logger.error("LLM extraction failed", error=str(e))
            return []
    
    def _parse_llm_response(self, response: str) -> List[JobRequirement]:
        """Parse LLM response into structured requirements."""
        requirements = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line.startswith('-'):
                continue
            
            # Parse format: - LEVEL TYPE requirement_text
            parts = line[1:].strip().split(' ', 2)
            if len(parts) < 3:
                continue
            
            try:
                level_str = parts[0].upper()
                type_str = parts[1].upper()
                text = parts[2]
                
                level = RequirementLevel(level_str.lower())
                req_type = RequirementType(type_str.lower())
                
                # Extract keywords from text
                keywords = [word.lower() for word in text.split() if len(word) > 2]
                
                requirement = JobRequirement(
                    text=text,
                    type=req_type,
                    level=level,
                    category=req_type.value,
                    keywords=keywords,
                    confidence=0.7
                )
                requirements.append(requirement)
                
            except (ValueError, IndexError):
                continue
        
        return requirements
    
    async def _extract_with_keywords(self, text: str) -> List[JobRequirement]:
        """Extract requirements using keyword matching."""
        requirements = []
        text_lower = text.lower()
        
        # Check for education requirements
        education_keywords = {
            "bachelor": ("Bachelor's degree", RequirementType.EDUCATION),
            "master": ("Master's degree", RequirementType.EDUCATION),
            "phd": ("PhD", RequirementType.EDUCATION),
            "degree": ("Degree", RequirementType.EDUCATION)
        }
        
        for keyword, (text_desc, req_type) in education_keywords.items():
            if keyword in text_lower:
                context = self._find_keyword_context(text, keyword)
                level = self._determine_requirement_level(context)
                
                requirement = JobRequirement(
                    text=text_desc,
                    type=req_type,
                    level=level,
                    category="education",
                    keywords=[keyword],
                    confidence=0.6
                )
                requirements.append(requirement)
        
        # Check for certification requirements
        cert_keywords = ["certification", "certified", "certificate"]
        for keyword in cert_keywords:
            if keyword in text_lower:
                context = self._find_keyword_context(text, keyword)
                level = self._determine_requirement_level(context)
                
                requirement = JobRequirement(
                    text=f"Professional certification",
                    type=RequirementType.CERTIFICATION,
                    level=level,
                    category="certification",
                    keywords=[keyword],
                    confidence=0.6
                )
                requirements.append(requirement)
                break  # Avoid duplicates
        
        return requirements
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a matched pattern."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _find_keyword_context(self, text: str, keyword: str, window: int = 100) -> str:
        """Find context around a keyword."""
        text_lower = text.lower()
        index = text_lower.find(keyword.lower())
        if index == -1:
            return ""
        
        context_start = max(0, index - window)
        context_end = min(len(text), index + len(keyword) + window)
        return text[context_start:context_end]
    
    def _determine_requirement_level(self, context: str) -> RequirementLevel:
        """Determine requirement importance level from context."""
        context_lower = context.lower()
        
        # Check for required indicators
        for indicator in self.requirement_indicators["required"]:
            if indicator in context_lower:
                return RequirementLevel.REQUIRED
        
        # Check for preferred indicators
        for indicator in self.requirement_indicators["preferred"]:
            if indicator in context_lower:
                return RequirementLevel.PREFERRED
        
        # Check for nice-to-have indicators
        for indicator in self.requirement_indicators["nice_to_have"]:
            if indicator in context_lower:
                return RequirementLevel.NICE_TO_HAVE
        
        # Default to required if no clear indicator
        return RequirementLevel.REQUIRED
    
    def _extract_years_from_context(self, context: str) -> Optional[int]:
        """Extract years of experience from context."""
        year_pattern = r"(\d+)\+?\s*years?"
        match = re.search(year_pattern, context, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _merge_similar_requirements(self, requirements: List[JobRequirement]) -> List[JobRequirement]:
        """Merge similar requirements to avoid duplicates."""
        merged = []
        seen_texts = set()
        
        for req in requirements:
            # Create a normalized key for comparison
            key = req.text.lower().strip()
            
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(req)
            else:
                # Find existing requirement and merge if confidence is higher
                for existing in merged:
                    if existing.text.lower().strip() == key:
                        if req.confidence > existing.confidence:
                            # Replace with higher confidence version
                            merged[merged.index(existing)] = req
                        break
        
        return merged
    
    async def _extract_salary_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract salary range from job posting."""
        salary_patterns = [
            r"\$(\d{1,3}(?:,\d{3})*)\s*-\s*\$(\d{1,3}(?:,\d{3})*)",
            r"\$(\d{1,3}(?:,\d{3})*)\s*to\s*\$(\d{1,3}(?:,\d{3})*)",
            r"salary.*?\$(\d{1,3}(?:,\d{3})*)",
            r"(\d{1,3}(?:,\d{3})*)\s*-\s*(\d{1,3}(?:,\d{3})*)\s*(?:per year|annually)"
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    min_salary = int(match.group(1).replace(',', ''))
                    max_salary = int(match.group(2).replace(',', ''))
                    return {
                        "min": min_salary,
                        "max": max_salary,
                        "currency": "USD",
                        "period": "annual"
                    }
                else:
                    salary = int(match.group(1).replace(',', ''))
                    return {
                        "min": salary,
                        "max": salary,
                        "currency": "USD",
                        "period": "annual"
                    }
        
        return None
    
    async def _extract_job_type(self, text: str) -> str:
        """Extract job type (full-time, part-time, contract, etc.)."""
        text_lower = text.lower()
        
        if "part-time" in text_lower or "part time" in text_lower:
            return "part_time"
        elif "contract" in text_lower or "contractor" in text_lower:
            return "contract"
        elif "intern" in text_lower or "internship" in text_lower:
            return "internship"
        elif "freelance" in text_lower:
            return "freelance"
        else:
            return "full_time"
    
    async def _extract_remote_option(self, text: str) -> bool:
        """Check if job offers remote work option."""
        text_lower = text.lower()
        remote_indicators = [
            "remote", "work from home", "wfh", "distributed", "telecommute",
            "hybrid", "flexible location"
        ]
        
        return any(indicator in text_lower for indicator in remote_indicators)
    
    async def _extract_benefits(self, text: str) -> List[str]:
        """Extract job benefits from posting."""
        benefits = []
        text_lower = text.lower()
        
        benefit_keywords = {
            "health insurance": ["health insurance", "medical insurance", "healthcare"],
            "dental insurance": ["dental", "dental insurance"],
            "vision insurance": ["vision", "vision insurance"],
            "401k": ["401k", "retirement", "pension"],
            "paid time off": ["pto", "paid time off", "vacation", "paid leave"],
            "flexible hours": ["flexible hours", "flexible schedule", "flex time"],
            "stock options": ["stock options", "equity", "shares"],
            "bonus": ["bonus", "performance bonus", "annual bonus"],
            "professional development": ["training", "professional development", "learning"],
            "gym membership": ["gym", "fitness", "wellness"]
        }
        
        for benefit, keywords in benefit_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                benefits.append(benefit)
        
        return benefits