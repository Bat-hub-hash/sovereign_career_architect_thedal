"""Tests for job portal navigation functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio

from sovereign_career_architect.browser.navigation import JobPortalNavigator, create_job_portal_navigator
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.browser.vision import ElementIdentifier


class TestJobPortalNavigator:
    """Test cases for JobPortalNavigator."""
    
    @pytest.fixture
    def mock_browser_agent(self):
        """Create a mock browser agent."""
        agent = AsyncMock(spec=BrowserAgent)
        agent.navigate_to = AsyncMock(return_value=True)
        agent.wait_for_element = AsyncMock(return_value=True)
        agent.fill_form_field = AsyncMock(return_value=True)
        agent.click_element = AsyncMock(return_value=True)
        agent.take_screenshot = AsyncMock(return_value=b"mock_screenshot")
        agent.get_page_content = AsyncMock(return_value="<html>Mock content</html>")
        return agent
    
    @pytest.fixture
    def mock_element_identifier(self):
        """Create a mock element identifier."""
        identifier = AsyncMock(spec=ElementIdentifier)
        identifier.identify_interactive_elements = AsyncMock(return_value=[
            {
                "element_type": "link",
                "description": "Software Engineer Job",
                "coordinates": [100, 200, 400, 250],
                "importance": 9,
                "confidence": 0.8
            }
        ])
        return identifier
    
    @pytest.fixture
    def navigator(self, mock_browser_agent, mock_element_identifier):
        """Create a JobPortalNavigator with mocked dependencies."""
        return JobPortalNavigator(
            browser_agent=mock_browser_agent,
            element_identifier=mock_element_identifier
        )
    
    def test_navigator_initialization(self):
        """Test JobPortalNavigator initialization."""
        # Test default initialization
        navigator = JobPortalNavigator()
        assert navigator.browser_agent is None
        assert navigator.element_identifier is None
        assert navigator.current_portal is None
        assert navigator.current_search_results == []
        assert navigator.navigation_history == []
        
        # Test with dependencies
        mock_browser = MagicMock()
        mock_identifier = MagicMock()
        navigator_with_deps = JobPortalNavigator(
            browser_agent=mock_browser,
            element_identifier=mock_identifier
        )
        assert navigator_with_deps.browser_agent == mock_browser
        assert navigator_with_deps.element_identifier == mock_identifier
    
    def test_portal_configurations(self, navigator):
        """Test that portal configurations are properly defined."""
        expected_portals = ["linkedin", "ycombinator", "angellist"]
        
        for portal in expected_portals:
            assert portal in navigator.portal_configs
            config = navigator.portal_configs[portal]
            
            # Check required configuration fields
            assert "base_url" in config
            assert "selectors" in config
            assert isinstance(config["base_url"], str)
            assert isinstance(config["selectors"], dict)
    
    @pytest.mark.asyncio
    async def test_navigate_to_portal_success(self, navigator, mock_browser_agent):
        """Test successful navigation to a job portal."""
        result = await navigator.navigate_to_portal("linkedin")
        
        assert result is True
        assert navigator.current_portal == "linkedin"
        assert len(navigator.navigation_history) == 1
        
        # Verify browser agent was called
        mock_browser_agent.navigate_to.assert_called_once_with("https://www.linkedin.com")
        
        # Check navigation history
        history_entry = navigator.navigation_history[0]
        assert history_entry["action"] == "navigate_to_portal"
        assert history_entry["portal"] == "linkedin"
    
    @pytest.mark.asyncio
    async def test_navigate_to_portal_unknown(self, navigator):
        """Test navigation to unknown portal."""
        result = await navigator.navigate_to_portal("unknown_portal")
        
        assert result is False
        assert navigator.current_portal is None
        assert len(navigator.navigation_history) == 0
    
    @pytest.mark.asyncio
    async def test_navigate_to_portal_without_browser(self):
        """Test navigation without browser agent (mock mode)."""
        navigator = JobPortalNavigator()
        
        result = await navigator.navigate_to_portal("linkedin")
        
        assert result is True
        assert navigator.current_portal == "linkedin"
    
    @pytest.mark.asyncio
    async def test_search_jobs_success(self, navigator, mock_browser_agent):
        """Test successful job search."""
        # First navigate to a portal
        await navigator.navigate_to_portal("linkedin")
        
        # Perform job search
        jobs = await navigator.search_jobs("Software Engineer", "San Francisco")
        
        assert isinstance(jobs, list)
        assert len(jobs) > 0
        assert navigator.current_search_results == jobs
        
        # Verify browser interactions
        assert mock_browser_agent.navigate_to.call_count >= 2  # Portal + jobs page
        assert mock_browser_agent.fill_form_field.called
        
        # Check navigation history
        search_history = [h for h in navigator.navigation_history if h["action"] == "search_jobs"]
        assert len(search_history) == 1
        assert search_history[0]["query"] == "Software Engineer"
        assert search_history[0]["location"] == "San Francisco"
    
    @pytest.mark.asyncio
    async def test_search_jobs_without_portal(self, navigator):
        """Test job search without selecting a portal first."""
        jobs = await navigator.search_jobs("Software Engineer")
        
        assert jobs == []
        assert navigator.current_search_results == []
    
    @pytest.mark.asyncio
    async def test_search_jobs_mock_mode(self):
        """Test job search in mock mode (no browser agent)."""
        navigator = JobPortalNavigator()
        navigator.current_portal = "linkedin"
        
        jobs = await navigator.search_jobs("Python Developer", "Remote")
        
        assert isinstance(jobs, list)
        assert len(jobs) > 0
        
        # Verify job structure
        for job in jobs:
            assert "title" in job
            assert "company" in job
            assert "location" in job
            assert "url" in job
            assert "portal" in job
            assert job["portal"] == "linkedin"
    
    @pytest.mark.asyncio
    async def test_navigate_to_job_success(self, navigator, mock_browser_agent):
        """Test successful navigation to a specific job."""
        # Set up search results
        navigator.current_search_results = [
            {
                "title": "Software Engineer",
                "company": "TechCorp",
                "url": "https://linkedin.com/job/123"
            }
        ]
        
        result = await navigator.navigate_to_job(0)
        
        assert result is True
        mock_browser_agent.navigate_to.assert_called_with("https://linkedin.com/job/123")
        
        # Check navigation history
        job_history = [h for h in navigator.navigation_history if h["action"] == "navigate_to_job"]
        assert len(job_history) == 1
        assert job_history[0]["job_title"] == "Software Engineer"
    
    @pytest.mark.asyncio
    async def test_navigate_to_job_invalid_index(self, navigator):
        """Test navigation to job with invalid index."""
        navigator.current_search_results = [{"title": "Job 1", "url": "http://example.com"}]
        
        # Test index out of range
        result = await navigator.navigate_to_job(5)
        assert result is False
        
        # Test negative index
        result = await navigator.navigate_to_job(-1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_navigate_to_job_no_url(self, navigator):
        """Test navigation to job without URL."""
        navigator.current_search_results = [
            {"title": "Job without URL", "url": "#"}
        ]
        
        result = await navigator.navigate_to_job(0)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_job_details_with_vision(self, navigator, mock_element_identifier):
        """Test job details extraction using computer vision."""
        details = await navigator.get_job_details()
        
        assert isinstance(details, dict)
        assert "title" in details
        assert "company" in details
        assert "apply_elements" in details
        
        # Verify element identifier was called
        mock_element_identifier.identify_interactive_elements.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_job_details_mock_mode(self):
        """Test job details extraction in mock mode."""
        navigator = JobPortalNavigator()
        
        details = await navigator.get_job_details()
        
        assert isinstance(details, dict)
        assert "title" in details
        assert "company" in details
        assert "requirements" in details
        assert "benefits" in details
        assert isinstance(details["requirements"], list)
        assert isinstance(details["benefits"], list)
    
    def test_get_navigation_history(self, navigator):
        """Test navigation history retrieval."""
        # Add some mock history
        navigator.navigation_history = [
            {"action": "navigate_to_portal", "portal": "linkedin"},
            {"action": "search_jobs", "query": "Engineer"}
        ]
        
        history = navigator.get_navigation_history()
        
        assert len(history) == 2
        assert history[0]["action"] == "navigate_to_portal"
        assert history[1]["action"] == "search_jobs"
        
        # Verify it's a copy (not the original)
        history.append({"action": "test"})
        assert len(navigator.navigation_history) == 2
    
    def test_get_current_search_results(self, navigator):
        """Test current search results retrieval."""
        # Add some mock results
        navigator.current_search_results = [
            {"title": "Job 1", "company": "Company 1"},
            {"title": "Job 2", "company": "Company 2"}
        ]
        
        results = navigator.get_current_search_results()
        
        assert len(results) == 2
        assert results[0]["title"] == "Job 1"
        
        # Verify it's a copy (not the original)
        results.append({"title": "Job 3"})
        assert len(navigator.current_search_results) == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, navigator, mock_browser_agent):
        """Test error handling in navigation operations."""
        # Mock browser agent to raise exceptions
        mock_browser_agent.navigate_to.side_effect = Exception("Navigation failed")
        
        # Test portal navigation error handling
        result = await navigator.navigate_to_portal("linkedin")
        assert result is False
        
        # Reset mock for job search test
        mock_browser_agent.navigate_to.side_effect = None
        mock_browser_agent.navigate_to.return_value = True
        mock_browser_agent.fill_form_field.side_effect = Exception("Form fill failed")
        
        # Set portal for job search
        navigator.current_portal = "linkedin"
        
        # Test job search error handling
        jobs = await navigator.search_jobs("Engineer")
        assert isinstance(jobs, list)  # Should return empty list or mock results


class TestJobPortalNavigatorFactory:
    """Test cases for job portal navigator factory function."""
    
    def test_create_job_portal_navigator_default(self):
        """Test factory function with default parameters."""
        navigator = create_job_portal_navigator()
        
        assert isinstance(navigator, JobPortalNavigator)
        assert navigator.browser_agent is None
        assert navigator.element_identifier is None
    
    def test_create_job_portal_navigator_with_dependencies(self):
        """Test factory function with custom dependencies."""
        mock_browser = MagicMock()
        mock_identifier = MagicMock()
        
        navigator = create_job_portal_navigator(
            browser_agent=mock_browser,
            element_identifier=mock_identifier
        )
        
        assert isinstance(navigator, JobPortalNavigator)
        assert navigator.browser_agent == mock_browser
        assert navigator.element_identifier == mock_identifier


@pytest.mark.asyncio
async def test_job_portal_navigator_integration():
    """Integration test for job portal navigator workflow."""
    navigator = create_job_portal_navigator()
    
    # Test complete workflow
    # 1. Navigate to portal
    assert await navigator.navigate_to_portal("linkedin") is True
    
    # 2. Search for jobs
    jobs = await navigator.search_jobs("Software Engineer", "Remote")
    assert isinstance(jobs, list)
    assert len(jobs) > 0
    
    # 3. Navigate to first job
    if jobs:
        # Mock a valid URL for the first job
        jobs[0]["url"] = "https://example.com/job/1"
        navigator.current_search_results = jobs
        
        assert await navigator.navigate_to_job(0) is True
    
    # 4. Get job details
    details = await navigator.get_job_details()
    assert isinstance(details, dict)
    assert "title" in details
    
    # 5. Check navigation history
    history = navigator.get_navigation_history()
    assert len(history) >= 2  # At least portal navigation and job search
    
    # 6. Check search results
    results = navigator.get_current_search_results()
    assert len(results) > 0


@pytest.mark.asyncio
async def test_multiple_portal_navigation():
    """Test navigation across multiple job portals."""
    navigator = create_job_portal_navigator()
    
    portals = ["linkedin", "ycombinator", "angellist"]
    
    for portal in portals:
        # Navigate to portal
        assert await navigator.navigate_to_portal(portal) is True
        assert navigator.current_portal == portal
        
        # Search for jobs
        jobs = await navigator.search_jobs("Engineer")
        assert isinstance(jobs, list)
        
        # Verify portal-specific results
        for job in jobs:
            assert job["portal"] == portal
    
    # Check that history contains all portal navigations
    history = navigator.get_navigation_history()
    portal_navigations = [h for h in history if h["action"] == "navigate_to_portal"]
    assert len(portal_navigations) == len(portals)