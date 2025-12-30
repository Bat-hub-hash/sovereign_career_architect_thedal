"""Demo scenarios and test data for Sovereign Career Architect."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from sovereign_career_architect.core.models import (
    UserProfile, PersonalInfo, Skill, Experience, Education, JobPreferences,
    DocumentStore as Documents
)
from sovereign_career_architect.jobs.matcher import JobPosting, MatchResult
from sovereign_career_architect.jobs.extractor import JobRequirement, RequirementType, ExtractedRequirements


@dataclass
class DemoScenario:
    """A complete demo scenario with user profile and test data."""
    name: str
    description: str
    user_profile: UserProfile
    sample_jobs: List[JobPosting]
    sample_interactions: List[Dict[str, Any]]
    expected_outcomes: Dict[str, Any]


class DemoDataGenerator:
    """Generates realistic demo data for presentations and testing."""
    
    def __init__(self):
        self.scenarios = self._create_demo_scenarios()
    
    def _create_demo_scenarios(self) -> List[DemoScenario]:
        """Create comprehensive demo scenarios."""
        return [
            self._create_fresh_graduate_scenario(),
            self._create_experienced_developer_scenario(),
            self._create_career_changer_scenario(),
            self._create_multilingual_professional_scenario(),
            self._create_senior_executive_scenario()
        ]
    
    def _create_fresh_graduate_scenario(self) -> DemoScenario:
        """Create scenario for a fresh computer science graduate."""
        
        # User profile
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Priya Sharma",
                email="priya.sharma@email.com",
                phone="+91-9876543210",
                location="Bangalore, India",
                linkedin_url="https://linkedin.com/in/priyasharma",
                portfolio_url="https://github.com/priyasharma",
                preferred_language="en"
            ),
            skills=[
                Skill(name="Python", level="intermediate", years_experience=2),
                Skill(name="JavaScript", level="intermediate", years_experience=1.5),
                Skill(name="React", level="beginner", years_experience=1),
                Skill(name="SQL", level="intermediate", years_experience=1),
                Skill(name="Git", level="intermediate", years_experience=2),
                Skill(name="Machine Learning", level="beginner", years_experience=0.5)
            ],
            experience=[
                Experience(
                    role="Software Development Intern",
                    company="TechStart Solutions",
                    start_date=datetime(2023, 6, 1),
                    end_date=datetime(2023, 12, 31),
                    description="Developed web applications using React and Node.js. Worked on data analysis projects using Python.",
                    skills_used=["React", "Node.js", "Python", "MongoDB"]
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Technology",
                    field_of_study="Computer Science and Engineering",
                    institution="Indian Institute of Technology, Bangalore",
                    graduation_date=datetime(2024, 5, 1),
                    gpa=8.5
                )
            ],
            preferences=JobPreferences(
                desired_roles=["Software Engineer", "Full Stack Developer", "Data Analyst"],
                preferred_locations=["Bangalore", "Hyderabad", "Remote"],
                salary_range=(600000, 1200000),  # INR per annum
                work_type="hybrid",
                company_size=["startup", "medium"],
                cultural_background="indian"
            ),
            documents=Documents(
                resume_path="/demo/resumes/priya_sharma_resume.pdf",
                cover_letter_template="I am a recent computer science graduate passionate about technology...",
                portfolio_files=["/demo/portfolios/priya_projects.pdf"]
            )
        )
        
        # Sample job postings
        sample_jobs = [
            JobPosting(
                title="Junior Software Engineer",
                company="Flipkart",
                location="Bangalore, India",
                url="https://careers.flipkart.com/junior-software-engineer",
                description="Join our engineering team to build scalable e-commerce solutions. Work with React, Node.js, and microservices architecture.",
                posted_date=datetime.now() - timedelta(days=2),
                salary_range="₹8-12 LPA",
                remote_friendly=False,
                extracted_requirements=ExtractedRequirements(
                    job_title="Junior Software Engineer",
                    company="Flipkart",
                    requirements=[
                        JobRequirement(text="Bachelor's degree in Computer Science", type=RequirementType.EDUCATION, importance=0.9),
                        JobRequirement(text="Proficiency in JavaScript and React", type=RequirementType.TECHNICAL_SKILL, importance=0.8),
                        JobRequirement(text="Understanding of REST APIs", type=RequirementType.TECHNICAL_SKILL, importance=0.7),
                        JobRequirement(text="0-2 years of experience", type=RequirementType.EXPERIENCE, importance=0.6)
                    ]
                )
            ),
            JobPosting(
                title="Full Stack Developer - Entry Level",
                company="Zomato",
                location="Bangalore, India",
                url="https://careers.zomato.com/full-stack-developer",
                description="Build and maintain web applications for our food delivery platform. Work with modern tech stack including React, Python, and AWS.",
                posted_date=datetime.now() - timedelta(days=1),
                salary_range="₹6-10 LPA",
                remote_friendly=True,
                extracted_requirements=ExtractedRequirements(
                    job_title="Full Stack Developer - Entry Level",
                    company="Zomato",
                    requirements=[
                        JobRequirement(text="Python programming skills", type=RequirementType.TECHNICAL_SKILL, importance=0.8),
                        JobRequirement(text="React.js experience", type=RequirementType.TECHNICAL_SKILL, importance=0.7),
                        JobRequirement(text="Database knowledge (SQL)", type=RequirementType.TECHNICAL_SKILL, importance=0.6),
                        JobRequirement(text="Fresh graduate or 1 year experience", type=RequirementType.EXPERIENCE, importance=0.5)
                    ]
                )
            )
        ]
        
        # Sample interactions
        sample_interactions = [
            {
                "type": "job_search",
                "user_input": "I'm looking for entry-level software engineering positions in Bangalore",
                "expected_response": "I found several great opportunities for you! Based on your Python and React skills, I recommend the Full Stack Developer position at Zomato and the Junior Software Engineer role at Flipkart.",
                "actions": ["search_jobs", "match_skills", "rank_opportunities"]
            },
            {
                "type": "interview_preparation",
                "user_input": "Can you help me prepare for a technical interview at Flipkart?",
                "expected_response": "I'll create a customized interview simulation for you. Let's start with some React and JavaScript questions appropriate for your experience level.",
                "actions": ["generate_questions", "conduct_mock_interview", "provide_feedback"]
            },
            {
                "type": "application_assistance",
                "user_input": "I want to apply to the Zomato position",
                "expected_response": "I'll help you customize your application for Zomato. Let me tailor your cover letter to highlight your Python skills and full-stack experience.",
                "actions": ["customize_documents", "prepare_application", "submit_with_approval"]
            }
        ]
        
        # Expected outcomes
        expected_outcomes = {
            "job_matches": 2,
            "application_success_rate": 0.8,
            "interview_preparation_score": 0.85,
            "cultural_adaptation": "Professional Indian communication style with respectful language"
        }
        
        return DemoScenario(
            name="Fresh Graduate Success Story",
            description="Recent CS graduate from IIT Bangalore seeking first full-time role",
            user_profile=user_profile,
            sample_jobs=sample_jobs,
            sample_interactions=sample_interactions,
            expected_outcomes=expected_outcomes
        )
    
    def _create_experienced_developer_scenario(self) -> DemoScenario:
        """Create scenario for an experienced software developer."""
        
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Rajesh Kumar",
                email="rajesh.kumar@email.com",
                phone="+91-9876543211",
                location="Hyderabad, India",
                linkedin_url="https://linkedin.com/in/rajeshkumar",
                portfolio_url="https://rajeshkumar.dev",
                preferred_language="en"
            ),
            skills=[
                Skill(name="Java", level="expert", years_experience=8),
                Skill(name="Spring Boot", level="expert", years_experience=6),
                Skill(name="Microservices", level="advanced", years_experience=5),
                Skill(name="AWS", level="advanced", years_experience=4),
                Skill(name="Docker", level="advanced", years_experience=4),
                Skill(name="Kubernetes", level="intermediate", years_experience=2),
                Skill(name="System Design", level="advanced", years_experience=5)
            ],
            experience=[
                Experience(
                    title="Senior Software Engineer",
                    company="Amazon",
                    duration="3 years",
                    description="Led development of microservices architecture for e-commerce platform. Managed team of 5 engineers.",
                    technologies=["Java", "Spring Boot", "AWS", "DynamoDB", "Kafka"]
                ),
                Experience(
                    title="Software Engineer",
                    company="Infosys",
                    duration="5 years",
                    description="Developed enterprise applications for banking clients. Specialized in backend systems and API development.",
                    technologies=["Java", "Spring", "Oracle", "REST APIs"]
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Engineering",
                    field="Computer Science",
                    institution="BITS Pilani",
                    graduation_year=2016,
                    gpa=8.2
                )
            ],
            preferences=JobPreferences(
                desired_roles=["Staff Engineer", "Principal Engineer", "Engineering Manager"],
                preferred_locations=["Hyderabad", "Bangalore", "Remote"],
                salary_range=(2500000, 4500000),  # INR per annum
                work_type="remote",
                company_size=["large", "unicorn"],
                cultural_background="indian"
            ),
            documents=Documents(
                resume_path="/demo/resumes/rajesh_kumar_resume.pdf",
                cover_letter_template="As a senior software engineer with 8+ years of experience...",
                portfolio_files=["/demo/portfolios/rajesh_architecture_designs.pdf"]
            )
        )
        
        sample_jobs = [
            JobPosting(
                title="Staff Software Engineer",
                company="Google",
                location="Hyderabad, India",
                url="https://careers.google.com/staff-engineer",
                description="Lead technical initiatives across multiple teams. Design and implement large-scale distributed systems.",
                posted_date=datetime.now() - timedelta(days=3),
                salary_range="₹35-50 LPA",
                remote_friendly=True,
                extracted_requirements=ExtractedRequirements(
                    job_title="Staff Software Engineer",
                    company="Google",
                    requirements=[
                        JobRequirement(text="8+ years of software development experience", type=RequirementType.EXPERIENCE, importance=0.9),
                        JobRequirement(text="Expertise in distributed systems", type=RequirementType.TECHNICAL_SKILL, importance=0.8),
                        JobRequirement(text="Leadership and mentoring experience", type=RequirementType.SOFT_SKILL, importance=0.7),
                        JobRequirement(text="Java or similar programming language", type=RequirementType.TECHNICAL_SKILL, importance=0.8)
                    ]
                )
            )
        ]
        
        sample_interactions = [
            {
                "type": "career_advancement",
                "user_input": "I want to move to a staff engineer role at a top tech company",
                "expected_response": "Based on your experience at Amazon and technical leadership skills, you're well-positioned for staff roles. Let me find opportunities that match your distributed systems expertise.",
                "actions": ["analyze_career_progression", "identify_target_companies", "skill_gap_analysis"]
            }
        ]
        
        expected_outcomes = {
            "job_matches": 1,
            "application_success_rate": 0.9,
            "interview_preparation_score": 0.9,
            "cultural_adaptation": "Senior professional communication with technical depth"
        }
        
        return DemoScenario(
            name="Senior Engineer Career Growth",
            description="Experienced developer seeking staff-level position at top tech company",
            user_profile=user_profile,
            sample_jobs=sample_jobs,
            sample_interactions=sample_interactions,
            expected_outcomes=expected_outcomes
        )
    
    def _create_career_changer_scenario(self) -> DemoScenario:
        """Create scenario for someone changing careers to tech."""
        
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Anita Desai",
                email="anita.desai@email.com",
                phone="+91-9876543212",
                location="Mumbai, India",
                linkedin_url="https://linkedin.com/in/anitadesai",
                preferred_language="en"
            ),
            skills=[
                Skill(name="Python", level="beginner", years_experience=1),
                Skill(name="Data Analysis", level="intermediate", years_experience=1.5),
                Skill(name="Excel", level="expert", years_experience=10),
                Skill(name="Project Management", level="advanced", years_experience=8),
                Skill(name="Business Analysis", level="expert", years_experience=10),
                Skill(name="SQL", level="beginner", years_experience=0.5)
            ],
            experience=[
                Experience(
                    title="Senior Business Analyst",
                    company="HDFC Bank",
                    duration="10 years",
                    description="Led business analysis for digital banking initiatives. Managed cross-functional teams and stakeholder relationships.",
                    technologies=["Excel", "Tableau", "SQL", "Business Intelligence"]
                ),
                Experience(
                    title="Data Science Bootcamp Graduate",
                    company="UpGrad",
                    duration="6 months",
                    description="Completed intensive data science program. Built projects in Python, machine learning, and data visualization.",
                    technologies=["Python", "Pandas", "Scikit-learn", "Matplotlib"]
                )
            ],
            education=[
                Education(
                    degree="Master of Business Administration",
                    field="Finance",
                    institution="Indian Institute of Management, Mumbai",
                    graduation_year=2014,
                    gpa=8.0
                ),
                Education(
                    degree="Bachelor of Commerce",
                    field="Accounting",
                    institution="University of Mumbai",
                    graduation_year=2012,
                    gpa=7.8
                )
            ],
            preferences=JobPreferences(
                desired_roles=["Data Analyst", "Business Intelligence Analyst", "Product Analyst"],
                preferred_locations=["Mumbai", "Pune", "Remote"],
                salary_range=(1200000, 2000000),  # INR per annum
                work_type="hybrid",
                company_size=["medium", "large"],
                cultural_background="indian"
            ),
            documents=Documents(
                resume_path="/demo/resumes/anita_desai_resume.pdf",
                cover_letter_template="As a business professional transitioning to data science...",
                portfolio_files=["/demo/portfolios/anita_data_projects.pdf"]
            )
        )
        
        sample_jobs = [
            JobPosting(
                title="Business Intelligence Analyst",
                company="Swiggy",
                location="Mumbai, India",
                url="https://careers.swiggy.com/bi-analyst",
                description="Analyze business metrics and create dashboards to drive data-driven decisions. Bridge the gap between business and technology teams.",
                posted_date=datetime.now() - timedelta(days=1),
                salary_range="₹12-18 LPA",
                remote_friendly=True,
                extracted_requirements=ExtractedRequirements(
                    job_title="Business Intelligence Analyst",
                    company="Swiggy",
                    requirements=[
                        JobRequirement(text="Business analysis experience", type=RequirementType.EXPERIENCE, importance=0.8),
                        JobRequirement(text="SQL and data analysis skills", type=RequirementType.TECHNICAL_SKILL, importance=0.7),
                        JobRequirement(text="Experience with BI tools", type=RequirementType.TECHNICAL_SKILL, importance=0.6),
                        JobRequirement(text="Strong communication skills", type=RequirementType.SOFT_SKILL, importance=0.8)
                    ]
                )
            )
        ]
        
        sample_interactions = [
            {
                "type": "career_transition",
                "user_input": "I'm transitioning from banking to tech. How can I leverage my business experience?",
                "expected_response": "Your business analysis background is valuable in tech! I recommend focusing on Business Intelligence and Product Analyst roles where your domain expertise gives you an advantage.",
                "actions": ["skill_translation", "identify_transferable_skills", "career_path_planning"]
            }
        ]
        
        expected_outcomes = {
            "job_matches": 1,
            "application_success_rate": 0.75,
            "interview_preparation_score": 0.8,
            "cultural_adaptation": "Professional communication emphasizing business value"
        }
        
        return DemoScenario(
            name="Career Transition Success",
            description="Banking professional transitioning to tech data roles",
            user_profile=user_profile,
            sample_jobs=sample_jobs,
            sample_interactions=sample_interactions,
            expected_outcomes=expected_outcomes
        )
    
    def _create_multilingual_professional_scenario(self) -> DemoScenario:
        """Create scenario showcasing multilingual capabilities."""
        
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Arjun Reddy",
                email="arjun.reddy@email.com",
                phone="+91-9876543213",
                location="Chennai, India",
                linkedin_url="https://linkedin.com/in/arjunreddy",
                preferred_language="ta"  # Tamil
            ),
            skills=[
                Skill(name="React Native", level="advanced", years_experience=4),
                Skill(name="Flutter", level="intermediate", years_experience=2),
                Skill(name="iOS Development", level="advanced", years_experience=5),
                Skill(name="Android Development", level="advanced", years_experience=5),
                Skill(name="JavaScript", level="advanced", years_experience=6),
                Skill(name="Swift", level="advanced", years_experience=4)
            ],
            experience=[
                Experience(
                    title="Mobile App Developer",
                    company="Freshworks",
                    duration="4 years",
                    description="Developed mobile applications for CRM platform. Led mobile team and implemented cross-platform solutions.",
                    technologies=["React Native", "iOS", "Android", "JavaScript"]
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Engineering",
                    field="Electronics and Communication",
                    institution="Anna University",
                    graduation_year=2019,
                    gpa=8.4
                )
            ],
            preferences=JobPreferences(
                desired_roles=["Senior Mobile Developer", "Mobile Architect", "Tech Lead - Mobile"],
                preferred_locations=["Chennai", "Bangalore", "Remote"],
                salary_range=(1800000, 3000000),  # INR per annum
                work_type="hybrid",
                company_size=["startup", "medium", "large"],
                cultural_background="tamil"
            ),
            documents=Documents(
                resume_path="/demo/resumes/arjun_reddy_resume.pdf",
                cover_letter_template="Mobile development-il naan romba experienced...",  # Tamil-English mix
                portfolio_files=["/demo/portfolios/arjun_mobile_apps.pdf"]
            )
        )
        
        sample_jobs = [
            JobPosting(
                title="Senior Mobile Developer",
                company="PhonePe",
                location="Chennai, India",
                url="https://careers.phonepe.com/mobile-developer",
                description="Build and scale mobile applications for digital payments platform. Work with React Native and native iOS/Android development.",
                posted_date=datetime.now() - timedelta(days=2),
                salary_range="₹20-30 LPA",
                remote_friendly=False,
                extracted_requirements=ExtractedRequirements(
                    job_title="Senior Mobile Developer",
                    company="PhonePe",
                    requirements=[
                        JobRequirement(text="4+ years mobile development experience", type=RequirementType.EXPERIENCE, importance=0.8),
                        JobRequirement(text="React Native expertise", type=RequirementType.TECHNICAL_SKILL, importance=0.9),
                        JobRequirement(text="iOS and Android development", type=RequirementType.TECHNICAL_SKILL, importance=0.7),
                        JobRequirement(text="Fintech experience preferred", type=RequirementType.EXPERIENCE, importance=0.6)
                    ]
                )
            )
        ]
        
        sample_interactions = [
            {
                "type": "multilingual_interview",
                "user_input": "Chennai-la interview irukku, Tamil-la comfortable-aa pesalaam?",  # Tamil-English mix
                "expected_response": "Kandippa! Naan ungalukku Tamil-English mix-la interview preparation kodukka mudiyum. Mobile development questions-um Tamil-la explain pannalam.",
                "actions": ["language_detection", "code_switching", "cultural_adaptation", "interview_prep"]
            }
        ]
        
        expected_outcomes = {
            "job_matches": 1,
            "application_success_rate": 0.85,
            "interview_preparation_score": 0.9,
            "cultural_adaptation": "Tamil-English code-switching with regional cultural context"
        }
        
        return DemoScenario(
            name="Multilingual Professional Excellence",
            description="Tamil-speaking mobile developer showcasing code-switching capabilities",
            user_profile=user_profile,
            sample_jobs=sample_jobs,
            sample_interactions=sample_interactions,
            expected_outcomes=expected_outcomes
        )
    
    def _create_senior_executive_scenario(self) -> DemoScenario:
        """Create scenario for a senior executive role."""
        
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Vikram Singh",
                email="vikram.singh@email.com",
                phone="+91-9876543214",
                location="Gurgaon, India",
                linkedin_url="https://linkedin.com/in/vikramsingh",
                preferred_language="en"
            ),
            skills=[
                Skill(name="Strategic Planning", level="expert", years_experience=15),
                Skill(name="Team Leadership", level="expert", years_experience=15),
                Skill(name="Product Management", level="expert", years_experience=12),
                Skill(name="Business Development", level="expert", years_experience=15),
                Skill(name="Digital Transformation", level="advanced", years_experience=8),
                Skill(name="P&L Management", level="expert", years_experience=10)
            ],
            experience=[
                Experience(
                    title="Vice President - Engineering",
                    company="Paytm",
                    duration="5 years",
                    description="Led engineering organization of 200+ engineers. Drove digital payments platform scaling to 500M+ users.",
                    technologies=["Leadership", "Strategy", "Product", "Engineering Management"]
                ),
                Experience(
                    title="Director - Product Engineering",
                    company="Flipkart",
                    duration="6 years",
                    description="Built and scaled e-commerce platform. Led product and engineering teams through hypergrowth phase.",
                    technologies=["Product Management", "Engineering", "Strategy", "Operations"]
                )
            ],
            education=[
                Education(
                    degree="Master of Business Administration",
                    field="Strategy and Operations",
                    institution="Indian School of Business",
                    graduation_year=2009,
                    gpa=8.8
                ),
                Education(
                    degree="Bachelor of Technology",
                    field="Computer Science",
                    institution="Indian Institute of Technology, Delhi",
                    graduation_year=2007,
                    gpa=8.6
                )
            ],
            preferences=JobPreferences(
                desired_roles=["Chief Technology Officer", "Chief Product Officer", "VP Engineering"],
                preferred_locations=["Gurgaon", "Bangalore", "Mumbai"],
                salary_range=(8000000, 15000000),  # INR per annum
                work_type="office",
                company_size=["large", "unicorn"],
                cultural_background="indian"
            ),
            documents=Documents(
                resume_path="/demo/resumes/vikram_singh_resume.pdf",
                cover_letter_template="As a technology leader with 15+ years of experience...",
                portfolio_files=["/demo/portfolios/vikram_leadership_portfolio.pdf"]
            )
        )
        
        sample_jobs = [
            JobPosting(
                title="Chief Technology Officer",
                company="Razorpay",
                location="Bangalore, India",
                url="https://careers.razorpay.com/cto",
                description="Lead technology strategy and engineering organization. Drive innovation in fintech infrastructure and scale platform for global expansion.",
                posted_date=datetime.now() - timedelta(days=5),
                salary_range="₹1-2 Cr CTC",
                remote_friendly=False,
                extracted_requirements=ExtractedRequirements(
                    job_title="Chief Technology Officer",
                    company="Razorpay",
                    requirements=[
                        JobRequirement(text="15+ years technology leadership experience", type=RequirementType.EXPERIENCE, importance=0.9),
                        JobRequirement(text="Experience scaling engineering teams", type=RequirementType.EXPERIENCE, importance=0.8),
                        JobRequirement(text="Fintech or payments industry experience", type=RequirementType.EXPERIENCE, importance=0.7),
                        JobRequirement(text="Strategic vision and execution", type=RequirementType.SOFT_SKILL, importance=0.9)
                    ]
                )
            )
        ]
        
        sample_interactions = [
            {
                "type": "executive_search",
                "user_input": "I'm looking for CTO opportunities in fintech companies",
                "expected_response": "Given your leadership experience at Paytm and Flipkart, you're perfectly positioned for senior fintech roles. The CTO position at Razorpay aligns excellently with your background.",
                "actions": ["executive_matching", "strategic_positioning", "leadership_assessment"]
            }
        ]
        
        expected_outcomes = {
            "job_matches": 1,
            "application_success_rate": 0.95,
            "interview_preparation_score": 0.95,
            "cultural_adaptation": "Executive-level communication with strategic focus"
        }
        
        return DemoScenario(
            name="Executive Leadership Transition",
            description="Senior technology executive seeking CTO role in fintech",
            user_profile=user_profile,
            sample_jobs=sample_jobs,
            sample_interactions=sample_interactions,
            expected_outcomes=expected_outcomes
        )
    
    def get_scenario(self, name: str) -> Optional[DemoScenario]:
        """Get a specific demo scenario by name."""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None
    
    def get_all_scenarios(self) -> List[DemoScenario]:
        """Get all demo scenarios."""
        return self.scenarios
    
    def export_scenarios_to_json(self, file_path: str) -> None:
        """Export scenarios to JSON file for easy loading."""
        scenarios_data = []
        
        for scenario in self.scenarios:
            scenario_data = {
                "name": scenario.name,
                "description": scenario.description,
                "user_profile": scenario.user_profile.dict(),
                "sample_jobs": [job.dict() for job in scenario.sample_jobs],
                "sample_interactions": scenario.sample_interactions,
                "expected_outcomes": scenario.expected_outcomes
            }
            scenarios_data.append(scenario_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(scenarios_data, f, indent=2, default=str, ensure_ascii=False)
    
    def create_demo_presentation_script(self) -> str:
        """Create a presentation script for demos."""
        script = """
# Sovereign Career Architect - Demo Presentation Script

## Introduction (2 minutes)
"Welcome to the Sovereign Career Architect - an AI-powered autonomous job application system designed specifically for the Indian job market. Our system combines advanced AI with human oversight to provide personalized career development support."

## Key Features Overview (3 minutes)
1. **Multilingual Support**: Seamless English-Indic language mixing with cultural adaptation
2. **Autonomous Job Applications**: AI-driven job search and application with human approval
3. **Interview Simulation**: Personalized interview preparation with real-time feedback
4. **Cultural Intelligence**: Adapts communication style based on cultural context
5. **Safety Layer**: Human-in-the-loop for all high-stakes actions

## Demo Scenarios (15 minutes)

### Scenario 1: Fresh Graduate Success (3 minutes)
- **Profile**: Priya Sharma, IIT Bangalore CS graduate
- **Demo**: Job search → Interview prep → Application assistance
- **Highlight**: Entry-level matching and cultural adaptation

### Scenario 2: Multilingual Professional (4 minutes)
- **Profile**: Arjun Reddy, Tamil-speaking mobile developer
- **Demo**: Code-switching interview preparation
- **Highlight**: Tamil-English mixing and regional cultural context

### Scenario 3: Career Transition (4 minutes)
- **Profile**: Anita Desai, Banking to Tech transition
- **Demo**: Skill translation and career path planning
- **Highlight**: Transferable skills identification

### Scenario 4: Senior Executive (4 minutes)
- **Profile**: Vikram Singh, VP seeking CTO role
- **Demo**: Executive-level matching and strategic positioning
- **Highlight**: Leadership assessment and high-level communication

## Technical Architecture (5 minutes)
- **LangGraph**: Cognitive architecture with memory and reasoning
- **Mem0**: Persistent memory across sessions
- **Browser-use**: Automated job portal navigation
- **Vapi.ai**: Voice interface with multilingual support
- **Safety Layer**: Human approval for critical actions

## Q&A and Wrap-up (5 minutes)

Total Duration: 30 minutes
        """
        return script.strip()


# Create global instance for easy access
demo_generator = DemoDataGenerator()


if __name__ == "__main__":
    # Export demo data for use in presentations
    demo_generator.export_scenarios_to_json("demo_scenarios.json")
    
    # Print presentation script
    print(demo_generator.create_demo_presentation_script())