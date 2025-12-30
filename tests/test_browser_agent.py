"""Tests for browser automation agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from sovereign_career_architect.browser.agent import BrowserAgent, create_browser_agent


class TestBrowserAgent:
    """Test cases for BrowserAgent."""
    
    @pytest.fixture
    def browser_agent(self):
        """Create a BrowserAgent instance for testing."""
        return BrowserAgent(
            headless=True,
            stealth_mode=True,
            viewport_size=(1920, 1080)
        )
    
    def test_browser_agent_initialization(self):
        """Test BrowserAgent initialization with different parameters."""
        # Test default initialization
        agent = BrowserAgent()
        assert agent.headless is True
        assert agent.stealth_mode is True
        assert agent.viewport_size == (1920, 1080)
        assert agent.user_data_dir is None
        assert not agent.is_initialized
        
        # Test custom initialization
        agent_custom = BrowserAgent(
            headless=False,
            stealth_mode=False,
            user_data_dir="/tmp/browser",
            viewport_size=(1366, 768)
        )
        assert agent_custom.headless is False
        assert agent_custom.stealth_mode is False
        assert agent_custom.user_data_dir == "/tmp/browser"
        assert agent_custom.viewport_size == (1366, 768)
    
    @pytest.mark.asyncio
    async def test_initialize_mock_fallback(self, browser_agent):
        """Test initialization falls back to mock when dependencies unavailable."""
        # Test that initialization works with mock implementation
        result = await browser_agent.initialize()
        
        assert result is True
        assert browser_agent.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_navigate_to_mock(self, browser_agent):
        """Test navigation with mock implementation."""
        # Initialize with mock
        await browser_agent.initialize()
        
        # Test navigation
        result = await browser_agent.navigate_to("https://example.com")
        
        assert result is True
        assert browser_agent.current_url == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_take_screenshot_mock(self, browser_agent):
        """Test screenshot capture with mock implementation."""
        await browser_agent.initialize()
        
        # Test screenshot without path
        screenshot = await browser_agent.take_screenshot()
        assert screenshot == b"mock_screenshot_data"
        
        # Test screenshot with path
        screenshot_with_path = await browser_agent.take_screenshot("/tmp/test.png")
        assert screenshot_with_path == b"mock_screenshot_data"
    
    @pytest.mark.asyncio
    async def test_get_page_content_mock(self, browser_agent):
        """Test page content retrieval with mock implementation."""
        await browser_agent.initialize()
        
        content = await browser_agent.get_page_content()
        assert content == "<html><body>Mock page content</body></html>"
    
    @pytest.mark.asyncio
    async def test_wait_for_element_mock(self, browser_agent):
        """Test element waiting with mock implementation."""
        await browser_agent.initialize()
        
        # Test successful wait
        result = await browser_agent.wait_for_element("button.submit")
        assert result is True
        
        # Test with custom timeout and state
        result_custom = await browser_agent.wait_for_element(
            "input[name='email']",
            timeout=5000,
            state="attached"
        )
        assert result_custom is True
    
    @pytest.mark.asyncio
    async def test_click_element_mock(self, browser_agent):
        """Test element clicking with mock implementation."""
        await browser_agent.initialize()
        
        result = await browser_agent.click_element("button.submit")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_fill_form_field_mock(self, browser_agent):
        """Test form field filling with mock implementation."""
        await browser_agent.initialize()
        
        result = await browser_agent.fill_form_field("input[name='email']", "test@example.com")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_close_mock(self, browser_agent):
        """Test browser closing with mock implementation."""
        await browser_agent.initialize()
        
        # Verify initialized state
        assert browser_agent.is_initialized is True
        
        # Close browser
        await browser_agent.close()
        
        # Verify cleaned up state
        assert browser_agent.is_initialized is False
        assert browser_agent.current_url is None
    
    @pytest.mark.asyncio
    async def test_operations_without_initialization(self):
        """Test that operations work even without explicit initialization."""
        agent = BrowserAgent()
        
        # Operations should auto-initialize
        result = await agent.navigate_to("https://example.com")
        assert result is True
        assert agent.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, browser_agent):
        """Test error handling in browser operations."""
        # Mock page to raise exceptions
        browser_agent.page = MagicMock()
        browser_agent.page.goto = AsyncMock(side_effect=Exception("Navigation failed"))
        browser_agent.page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        browser_agent.page.content = AsyncMock(side_effect=Exception("Content failed"))
        browser_agent.page.wait_for_selector = AsyncMock(side_effect=Exception("Wait failed"))
        browser_agent.page.click = AsyncMock(side_effect=Exception("Click failed"))
        browser_agent.page.fill = AsyncMock(side_effect=Exception("Fill failed"))
        
        browser_agent.is_initialized = True
        
        # Test error handling
        assert await browser_agent.navigate_to("https://example.com") is False
        assert await browser_agent.take_screenshot() is None
        assert await browser_agent.get_page_content() is None
        assert await browser_agent.wait_for_element("button") is False
        assert await browser_agent.click_element("button") is False
        assert await browser_agent.fill_form_field("input", "value") is False


class TestBrowserAgentFactory:
    """Test cases for browser agent factory function."""
    
    def test_create_browser_agent_default(self):
        """Test factory function with default parameters."""
        agent = create_browser_agent()
        
        assert isinstance(agent, BrowserAgent)
        assert agent.headless is True
        assert agent.stealth_mode is True
        assert agent.viewport_size == (1920, 1080)
        assert agent.user_data_dir is None
    
    def test_create_browser_agent_custom(self):
        """Test factory function with custom parameters."""
        agent = create_browser_agent(
            headless=False,
            stealth_mode=False,
            user_data_dir="/custom/path",
            viewport_size=(1366, 768)
        )
        
        assert isinstance(agent, BrowserAgent)
        assert agent.headless is False
        assert agent.stealth_mode is False
        assert agent.user_data_dir == "/custom/path"
        assert agent.viewport_size == (1366, 768)


@pytest.mark.asyncio
async def test_browser_agent_integration():
    """Integration test for browser agent workflow."""
    agent = create_browser_agent()
    
    try:
        # Initialize
        assert await agent.initialize() is True
        
        # Navigate
        assert await agent.navigate_to("https://example.com") is True
        
        # Take screenshot
        screenshot = await agent.take_screenshot()
        assert screenshot is not None
        
        # Get content
        content = await agent.get_page_content()
        assert content is not None
        
        # Wait for element
        assert await agent.wait_for_element("body") is True
        
        # Click element
        assert await agent.click_element("body") is True
        
        # Fill form
        assert await agent.fill_form_field("input", "test") is True
        
    finally:
        # Cleanup
        await agent.close()
        assert not agent.is_initialized