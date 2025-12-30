"""System integration tests for Sovereign Career Architect."""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from sovereign_career_architect.core.agent import SovereignCareerArchitect, create_sovereign_career_architect
from sovereign_career_architect.core.models import UserProfile, PersonalInfo
from sovereign_career_architect.demo.scenarios import demo_generator
from sovereign_career_architect.utils.performance import performance_monitor


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    async def mock_agent(self):
        """Create a mock agent for testing."""
        # Mock external dependencies
        with patch('sovereign_career_architect.memory.client.MemoryClient') as mock_memory, \
             patch('sovereign_career_architect.browser.agent.BrowserAgent') as mock_browser, \
             patch('sovereign_career_architect.voice.orchestrator.VoiceOrchestrator') as mock_voice:
            
            # Configure mocks
            mock_memory.return_value.initialize = AsyncMock()
            mock_browser.return_value.initialize = AsyncMock()
            mock_voice.return_value.initialize = AsyncMock()
            
            agent = SovereignCareerArchitect(
                enable_voice=False,  # Disable for testing
                enable_browser=False,  # Disable for testing
                enable_interrupts=False  # Disable for testing
            )
            
            await agent.initialize()
            yield agent
            
            # Cleanup
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test that the agent initializes all components correctly."""
        with patch('sovereign_career_architect.memory.client.MemoryClient') as mock_memory:
            mock_memory.return_value.initialize = AsyncMock()
            
            agent = SovereignCareerArchitect(
                enable_voice=False,
                enable_browser=False,
                enable_interrupts=False
            )
            
            await agent.initialize()
            
            # Verify core components are initialized
            assert agent.memory_client is not None
            assert agent.safety_layer is not None
            assert agent.job_matcher is not None
            assert agent.interview_generator is not None
            assert agent.cultural_adapter is not None
            assert agent.workflow is not None
            
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_user_request_processing(self, mock_agent):
        """Test processing of user requests through the system."""
        
        # Mock the workflow response
        mock_response = {
            "response": "I can help you with your job search!",
            "output_data": {"jobs_found": 5},
            "requires_approval": False,
            "next_steps": ["Review job matches", "Prepare applications"]
        }
        
        with patch.object(mock_agent.workflow, 'ainvoke', return_value=mock_response):
            result = await mock_agent.process_user_request(
                user_id="test_user",
                session_id="test_session",
                message="Help me find software engineering jobs"
            )
            
            assert result["success"] is True
            assert "message" in result
            assert "data" in result
            assert result["session_id"] == "test_session"
    
    @pytest.mark.asyncio
    async def test_job_search_functionality(self, mock_agent):
        """Test job search functionality."""
        
        search_criteria = {
            "role": "Software Engineer",
            "location": "Bangalore",
            "experience_level": "mid"
        }
        
        result = await mock_agent.start_job_search(
            user_id="test_user",
            session_id="test_session",
            search_criteria=search_criteria
        )
        
        assert result["success"] is True
        assert "search_id" in result
        assert result["criteria"] == search_criteria
    
    @pytest.mark.asyncio
    async def test_interview_simulation(self, mock_agent):
        """Test interview simulation functionality."""
        
        interview_config = {
            "job_role": "Software Engineer",
            "experience_level": "mid",
            "question_count": 3
        }
        
        # Mock interview questions
        mock_questions = [
            {"id": "q1", "question": "Tell me about yourself", "type": "behavioral"},
            {"id": "q2", "question": "Explain REST APIs", "type": "technical"},
            {"id": "q3", "question": "Describe a challenging project", "type": "behavioral"}
        ]
        
        with patch.object(mock_agent.interview_generator, 'generate_interview_questions', 
                         return_value=mock_questions):
            result = await mock_agent.start_interview_simulation(
                user_id="test_user",
                session_id="test_session",
                interview_config=interview_config
            )
            
            assert result["success"] is True
            assert "interview_id" in result
            assert len(result["questions"]) == 3
    
    @pytest.mark.asyncio
    async def test_cultural_adaptation(self, mock_agent):
        """Test cultural adaptation functionality."""
        
        user_profile = UserProfile(
            personal_info=PersonalInfo(
                name="Test User",
                email="test@example.com",
                preferred_language="hi"
            ),
            skills=[],
            experience=[],
            education=[],
            preferences={"cultural_background": "indian"},
            documents=None
        )
        
        message = "I want to apply for this job"
        context = {
            "situation_type": "professional",
            "formality_level": "formal",
            "audience": "hiring_manager"
        }
        
        result = await mock_agent.apply_cultural_adaptation(
            message=message,
            user_profile=user_profile,
            target_culture="american",
            context=context
        )
        
        assert result["success"] is True
        assert "adapted_message" in result
        assert result["original_message"] == message
    
    @pytest.mark.asyncio
    async def test_system_status(self, mock_agent):
        """Test system status reporting."""
        
        status = await mock_agent.get_system_status()
        
        assert "overall_status" in status
        assert "components" in status
        assert "active_sessions" in status
        assert "timestamp" in status
        
        # Check that core components are reported
        assert "memory" in status["components"]
        assert "safety" in status["components"]
        assert "workflow" in status["components"]
    
    @pytest.mark.asyncio
    async def test_demo_scenarios(self):
        """Test that demo scenarios are properly structured."""
        
        scenarios = demo_generator.get_all_scenarios()
        
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            # Verify scenario structure
            assert hasattr(scenario, 'name')
            assert hasattr(scenario, 'description')
            assert hasattr(scenario, 'user_profile')
            assert hasattr(scenario, 'sample_jobs')
            assert hasattr(scenario, 'sample_interactions')
            assert hasattr(scenario, 'expected_outcomes')
            
            # Verify user profile is valid
            assert scenario.user_profile.personal_info.name
            assert scenario.user_profile.personal_info.email
            
            # Verify sample jobs exist
            assert len(scenario.sample_jobs) > 0
            
            # Verify interactions exist
            assert len(scenario.sample_interactions) > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        
        # Start monitoring
        performance_monitor.start_monitoring()
        
        # Simulate some requests
        performance_monitor.track_request("test_component", "req_1")
        await asyncio.sleep(0.1)  # Simulate processing time
        performance_monitor.complete_request("test_component", "req_1", success=True)
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        
        assert "current_metrics" in summary
        assert "component_metrics" in summary
        assert "health_status" in summary
        
        # Stop monitoring
        performance_monitor.stop_monitoring_thread()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agent):
        """Test system error handling."""
        
        # Test with invalid user input
        result = await mock_agent.process_user_request(
            user_id="",  # Invalid user ID
            session_id="test_session",
            message=""   # Empty message
        )
        
        # System should handle gracefully
        assert "success" in result
        assert "error" in result or result["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_agent):
        """Test handling of concurrent requests."""
        
        # Mock workflow responses
        mock_response = {
            "response": "Request processed",
            "output_data": {},
            "requires_approval": False
        }
        
        with patch.object(mock_agent.workflow, 'ainvoke', return_value=mock_response):
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = mock_agent.process_user_request(
                    user_id=f"user_{i}",
                    session_id=f"session_{i}",
                    message=f"Request {i}"
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests completed successfully
            for result in results:
                if isinstance(result, Exception):
                    pytest.fail(f"Request failed with exception: {result}")
                assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_session_management(self, mock_agent):
        """Test session management functionality."""
        
        user_id = "test_user"
        session_id = "test_session"
        
        # First request should initialize session
        mock_response = {"response": "Hello!", "output_data": {}}
        
        with patch.object(mock_agent.workflow, 'ainvoke', return_value=mock_response):
            result1 = await mock_agent.process_user_request(
                user_id=user_id,
                session_id=session_id,
                message="Hello"
            )
            
            assert result1["success"] is True
            assert session_id in mock_agent.active_sessions
            
            # Second request should use existing session
            result2 = await mock_agent.process_user_request(
                user_id=user_id,
                session_id=session_id,
                message="How are you?"
            )
            
            assert result2["success"] is True
            
            # Verify session was updated
            session_data = mock_agent.active_sessions[session_id]
            assert session_data["message_count"] == 2
    
    def test_configuration_validation(self):
        """Test that system configuration is valid."""
        
        # Test default configuration
        agent = SovereignCareerArchitect()
        
        assert agent.enable_voice is True
        assert agent.enable_browser is True
        assert agent.enable_interrupts is True
        
        # Test custom configuration
        agent_custom = SovereignCareerArchitect(
            enable_voice=False,
            enable_browser=False,
            enable_interrupts=False
        )
        
        assert agent_custom.enable_voice is False
        assert agent_custom.enable_browser is False
        assert agent_custom.enable_interrupts is False


class TestEndToEndScenarios:
    """End-to-end scenario tests using demo data."""
    
    @pytest.mark.asyncio
    async def test_fresh_graduate_scenario(self):
        """Test the fresh graduate demo scenario."""
        
        scenario = demo_generator.get_scenario("Fresh Graduate Success Story")
        assert scenario is not None
        
        # Verify scenario has all required components
        assert scenario.user_profile.personal_info.name == "Priya Sharma"
        assert len(scenario.sample_jobs) >= 1
        assert len(scenario.sample_interactions) >= 1
        
        # Test job matching would work
        user_skills = [skill.name.lower() for skill in scenario.user_profile.skills]
        job_requirements = scenario.sample_jobs[0].extracted_requirements.requirements
        
        # Should have some skill overlap
        technical_requirements = [
            req.text.lower() for req in job_requirements 
            if "javascript" in req.text.lower() or "python" in req.text.lower()
        ]
        
        skill_overlap = any(
            any(skill in req for skill in user_skills)
            for req in technical_requirements
        )
        
        assert skill_overlap, "User skills should match job requirements"
    
    @pytest.mark.asyncio
    async def test_multilingual_scenario(self):
        """Test the multilingual professional scenario."""
        
        scenario = demo_generator.get_scenario("Multilingual Professional Excellence")
        assert scenario is not None
        
        # Verify Tamil language preference
        assert scenario.user_profile.personal_info.preferred_language == "ta"
        
        # Verify code-switching interaction
        tamil_interaction = next(
            (interaction for interaction in scenario.sample_interactions
             if "tamil" in interaction.get("user_input", "").lower()),
            None
        )
        
        assert tamil_interaction is not None
        assert "code_switching" in tamil_interaction.get("actions", [])
    
    @pytest.mark.asyncio
    async def test_senior_executive_scenario(self):
        """Test the senior executive scenario."""
        
        scenario = demo_generator.get_scenario("Executive Leadership Transition")
        assert scenario is not None
        
        # Verify executive-level profile
        assert "VP" in scenario.user_profile.experience[0].title or "Vice President" in scenario.user_profile.experience[0].title
        
        # Verify high salary expectations
        assert scenario.user_profile.preferences.salary_range[0] >= 5000000  # 50+ LPA
        
        # Verify CTO-level job posting
        cto_job = next(
            (job for job in scenario.sample_jobs if "CTO" in job.title or "Chief Technology Officer" in job.title),
            None
        )
        
        assert cto_job is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])