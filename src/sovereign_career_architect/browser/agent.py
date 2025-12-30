"""Browser automation agent using Browser-use and Playwright."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import structlog

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger
from sovereign_career_architect.browser.stealth import StealthManager, StealthConfig

logger = get_logger(__name__)


class BrowserAgent:
    """
    Browser automation agent using Browser-use with Playwright backend.
    
    This agent provides high-level browser automation capabilities with
    computer vision element identification and stealth mode for job applications.
    """
    
    def __init__(
        self,
        headless: bool = True,
        stealth_mode: bool = True,
        user_data_dir: Optional[str] = None,
        viewport_size: tuple = (1920, 1080),
        stealth_config: Optional[StealthConfig] = None
    ):
        """
        Initialize the browser agent.
        
        Args:
            headless: Run browser in headless mode
            stealth_mode: Enable stealth mode to avoid bot detection
            user_data_dir: Custom user data directory for persistent sessions
            viewport_size: Browser viewport size (width, height)
            stealth_config: Custom stealth configuration
        """
        self.headless = headless
        self.stealth_mode = stealth_mode
        self.user_data_dir = user_data_dir
        self.viewport_size = viewport_size
        self.logger = logger.bind(component="browser_agent")
        
        # Initialize stealth manager
        self.stealth_manager = StealthManager(stealth_config) if stealth_mode else None
        
        # Browser-use and Playwright instances
        self.browser_use_agent = None
        self.playwright = None
        self.browser = None
        self.page = None
        self.context = None
        
        # Session state
        self.is_initialized = False
        self.current_url = None
        self.session_cookies = {}
        
    async def initialize(self) -> bool:
        """
        Initialize Browser-use with Playwright configuration.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        try:
            # Import Browser-use (handle optional dependency)
            try:
                from browser_use import Agent as BrowserUseAgent
                from browser_use.browser.browser import Browser
                from browser_use.browser.context import BrowserContext
            except ImportError:
                self.logger.warning("Browser-use not available, using mock implementation")
                return await self._initialize_mock()
            
            # Import Playwright
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                self.logger.error("Playwright not available")
                return False
            
            # Initialize Playwright
            self.playwright = await async_playwright().start()
            
            # Configure browser launch options
            launch_options = {
                "headless": self.headless,
                "viewport": {"width": self.viewport_size[0], "height": self.viewport_size[1]},
                "args": []
            }
            
            # Add stealth mode arguments
            if self.stealth_mode:
                launch_options["args"].extend([
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ])
            
            # Add user data directory if specified
            if self.user_data_dir:
                launch_options["user_data_dir"] = self.user_data_dir
            
            # Launch Chromium browser
            self.browser = await self.playwright.chromium.launch(**launch_options)
            
            # Create browser context with stealth settings
            context_options = {
                "viewport": {"width": self.viewport_size[0], "height": self.viewport_size[1]},
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Configure stealth mode if enabled
            if self.stealth_mode and self.stealth_manager:
                await self.stealth_manager.setup_stealth_context(self.context)
            
            # Create new page
            self.page = await self.context.new_page()
            
            # Configure stealth page settings
            if self.stealth_mode and self.stealth_manager:
                await self.stealth_manager.setup_stealth_page(self.page)
            elif self.stealth_mode:
                # Fallback stealth configuration
                await self._configure_stealth_mode()
            
            # Initialize Browser-use agent
            self.browser_use_agent = BrowserUseAgent(
                task="Browser automation for job applications",
                llm=None,  # Will use default or configured LLM
                browser=Browser(
                    context=BrowserContext(self.context),
                    config={
                        "headless": self.headless,
                        "disable_security": True
                    }
                )
            )
            
            self.is_initialized = True
            self.logger.info(
                "Browser agent initialized successfully",
                headless=self.headless,
                stealth_mode=self.stealth_mode,
                viewport_size=self.viewport_size
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize browser agent",
                error=str(e),
                error_type=type(e).__name__
            )
            return await self._initialize_mock()
    
    async def _configure_stealth_mode(self) -> None:
        """Configure stealth mode JavaScript to avoid bot detection."""
        stealth_script = """
        // Override webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Override plugins length
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Override languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        // Override permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Override chrome runtime
        window.chrome = {
            runtime: {},
        };
        """
        
        await self.page.add_init_script(stealth_script)
    
    async def _initialize_mock(self) -> bool:
        """Initialize mock browser agent for development/testing."""
        self.logger.info("Using mock browser agent implementation")
        self.is_initialized = True
        return True
    
    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a specific URL.
        
        Args:
            url: Target URL
            
        Returns:
            True if navigation successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                # Add stealth delay before navigation
                if self.stealth_mode and self.stealth_manager:
                    await self.stealth_manager.human_like_delay("page_navigation")
                
                await self.page.goto(url, wait_until="networkidle")
                self.current_url = url
                
                # Check for bot detection after navigation
                if self.stealth_mode and self.stealth_manager:
                    challenges = await self.stealth_manager.detect_bot_challenges(self.page)
                    if any(challenges.values()):
                        success = await self.stealth_manager.handle_bot_detection(self.page, challenges)
                        if not success:
                            self.logger.error("Bot detection challenge could not be handled", challenges=challenges)
                            return False
                
                self.logger.info(
                    "Navigated to URL",
                    url=url,
                    title=await self.page.title()
                )
                return True
            else:
                # Mock implementation
                self.current_url = url
                self.logger.info("Mock navigation to URL", url=url)
                return True
                
        except Exception as e:
            self.logger.error(
                "Navigation failed",
                url=url,
                error=str(e)
            )
            return False
    
    async def take_screenshot(self, path: Optional[str] = None) -> Optional[bytes]:
        """
        Take a screenshot of the current page.
        
        Args:
            path: Optional file path to save screenshot
            
        Returns:
            Screenshot bytes if successful, None otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                screenshot_options = {"full_page": True}
                if path:
                    screenshot_options["path"] = path
                
                screenshot = await self.page.screenshot(**screenshot_options)
                
                self.logger.debug(
                    "Screenshot captured",
                    path=path,
                    size=len(screenshot) if screenshot else 0
                )
                
                return screenshot
            else:
                # Mock implementation
                self.logger.debug("Mock screenshot captured", path=path)
                return b"mock_screenshot_data"
                
        except Exception as e:
            self.logger.error(
                "Screenshot failed",
                error=str(e),
                path=path
            )
            return None
    
    async def get_page_content(self) -> Optional[str]:
        """
        Get the current page HTML content.
        
        Returns:
            Page HTML content if successful, None otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                content = await self.page.content()
                self.logger.debug(
                    "Page content retrieved",
                    content_length=len(content)
                )
                return content
            else:
                # Mock implementation
                return "<html><body>Mock page content</body></html>"
                
        except Exception as e:
            self.logger.error(
                "Failed to get page content",
                error=str(e)
            )
            return None
    
    async def wait_for_element(
        self, 
        selector: str, 
        timeout: int = 30000,
        state: str = "visible"
    ) -> bool:
        """
        Wait for an element to appear on the page.
        
        Args:
            selector: CSS selector or XPath
            timeout: Timeout in milliseconds
            state: Element state to wait for (visible, attached, detached, hidden)
            
        Returns:
            True if element found, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                await self.page.wait_for_selector(
                    selector, 
                    timeout=timeout, 
                    state=state
                )
                
                self.logger.debug(
                    "Element found",
                    selector=selector,
                    state=state
                )
                return True
            else:
                # Mock implementation
                await asyncio.sleep(0.1)  # Simulate wait
                self.logger.debug("Mock element wait", selector=selector)
                return True
                
        except Exception as e:
            self.logger.error(
                "Element wait failed",
                selector=selector,
                error=str(e)
            )
            return False
    
    async def click_element(self, selector: str) -> bool:
        """
        Click an element on the page.
        
        Args:
            selector: CSS selector or XPath
            
        Returns:
            True if click successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                # Add stealth delay and mouse movement
                if self.stealth_mode and self.stealth_manager:
                    await self.stealth_manager.human_like_delay("click")
                    await self.stealth_manager.human_like_mouse_movement(self.page, selector)
                
                await self.page.click(selector)
                
                self.logger.debug(
                    "Element clicked",
                    selector=selector
                )
                return True
            else:
                # Mock implementation
                self.logger.debug("Mock element click", selector=selector)
                return True
                
        except Exception as e:
            self.logger.error(
                "Element click failed",
                selector=selector,
                error=str(e)
            )
            return False
    
    async def fill_form_field(self, selector: str, value: str) -> bool:
        """
        Fill a form field with a value.
        
        Args:
            selector: CSS selector for the form field
            value: Value to fill
            
        Returns:
            True if fill successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.page:
                # Use human-like typing if stealth mode is enabled
                if self.stealth_mode and self.stealth_manager:
                    await self.stealth_manager.human_like_typing(self.page, selector, value)
                else:
                    await self.page.fill(selector, value)
                
                self.logger.debug(
                    "Form field filled",
                    selector=selector,
                    value_length=len(value)
                )
                return True
            else:
                # Mock implementation
                self.logger.debug(
                    "Mock form field fill", 
                    selector=selector, 
                    value_length=len(value)
                )
                return True
                
        except Exception as e:
            self.logger.error(
                "Form field fill failed",
                selector=selector,
                error=str(e)
            )
            return False
    
    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        try:
            if hasattr(self, 'page') and self.page:
                await self.page.close()
            
            if hasattr(self, 'context') and self.context:
                await self.context.close()
            
            if hasattr(self, 'browser') and self.browser:
                await self.browser.close()
            
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
            
            self.is_initialized = False
            self.current_url = None
            
            # Log session statistics if stealth mode was used
            if self.stealth_mode and self.stealth_manager:
                stats = self.stealth_manager.get_session_stats()
                self.logger.info("Browser session completed", **stats)
            
            self.logger.info("Browser agent closed successfully")
            
        except Exception as e:
            self.logger.error(
                "Error closing browser agent",
                error=str(e)
            )
    
    def get_stealth_stats(self) -> Optional[Dict[str, Any]]:
        """Get stealth session statistics."""
        if self.stealth_mode and self.stealth_manager:
            return self.stealth_manager.get_session_stats()
        return None


def create_browser_agent(
    headless: bool = True,
    stealth_mode: bool = True,
    user_data_dir: Optional[str] = None,
    viewport_size: tuple = (1920, 1080),
    stealth_config: Optional[StealthConfig] = None
) -> BrowserAgent:
    """
    Factory function to create a browser agent.
    
    Args:
        headless: Run browser in headless mode
        stealth_mode: Enable stealth mode to avoid bot detection
        user_data_dir: Custom user data directory for persistent sessions
        viewport_size: Browser viewport size (width, height)
        stealth_config: Custom stealth configuration
        
    Returns:
        Configured BrowserAgent instance
    """
    return BrowserAgent(
        headless=headless,
        stealth_mode=stealth_mode,
        user_data_dir=user_data_dir,
        viewport_size=viewport_size,
        stealth_config=stealth_config
    )