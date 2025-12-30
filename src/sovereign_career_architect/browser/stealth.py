"""Anti-bot detection and stealth measures for browser automation."""

import asyncio
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from playwright.async_api import Page, Browser, BrowserContext
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StealthConfig:
    """Configuration for stealth mode operations."""
    min_delay: float = 1.0
    max_delay: float = 3.0
    typing_delay_min: float = 0.05
    typing_delay_max: float = 0.15
    mouse_move_steps: int = 10
    viewport_width: int = 1366
    viewport_height: int = 768
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]


class StealthManager:
    """Manages stealth operations to avoid bot detection."""
    
    def __init__(self, config: Optional[StealthConfig] = None):
        self.config = config or StealthConfig()
        self.action_count = 0
        self.session_start_time = asyncio.get_event_loop().time()
        
    async def setup_stealth_context(self, context: BrowserContext) -> None:
        """Configure browser context with stealth settings."""
        logger.info("Setting up stealth browser context")
        
        # Set random user agent
        user_agent = random.choice(self.config.user_agents)
        await context.set_extra_http_headers({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        
        # Add stealth scripts
        await context.add_init_script("""
            // Override webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Override plugins
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
        """)
        
    async def setup_stealth_page(self, page: Page) -> None:
        """Configure page with stealth settings."""
        logger.info("Setting up stealth page configuration")
        
        # Set viewport
        await page.set_viewport_size({
            "width": self.config.viewport_width,
            "height": self.config.viewport_height
        })
        
        # Block unnecessary resources to reduce fingerprinting
        await page.route("**/*", self._handle_route)
        
    async def _handle_route(self, route) -> None:
        """Handle route requests to block unnecessary resources."""
        resource_type = route.request.resource_type
        
        # Block images, fonts, and other non-essential resources
        if resource_type in ["image", "font", "media"]:
            await route.abort()
        else:
            await route.continue_()
            
    async def human_like_delay(self, action_type: str = "general") -> None:
        """Add human-like delays between actions."""
        self.action_count += 1
        
        # Base delay
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        
        # Add extra delay for certain action types
        if action_type == "form_submit":
            delay += random.uniform(2.0, 5.0)
        elif action_type == "page_navigation":
            delay += random.uniform(1.0, 3.0)
        elif action_type == "click":
            delay += random.uniform(0.5, 1.5)
            
        # Add progressive delay for many actions
        if self.action_count > 10:
            delay += random.uniform(0.5, 2.0)
            
        logger.debug("Adding human-like delay", delay=delay, action_type=action_type)
        await asyncio.sleep(delay)
        
    async def human_like_typing(self, page: Page, selector: str, text: str) -> None:
        """Type text with human-like delays and patterns."""
        logger.debug("Starting human-like typing", selector=selector, text_length=len(text))
        
        element = await page.wait_for_selector(selector)
        await element.click()
        
        # Clear existing text
        await element.fill("")
        
        # Type character by character with random delays
        for char in text:
            await element.type(char)
            delay = random.uniform(self.config.typing_delay_min, self.config.typing_delay_max)
            
            # Add longer pauses for spaces (thinking time)
            if char == " ":
                delay += random.uniform(0.1, 0.3)
                
            await asyncio.sleep(delay)
            
    async def human_like_mouse_movement(self, page: Page, target_selector: str) -> None:
        """Move mouse to target with human-like path."""
        logger.debug("Starting human-like mouse movement", target=target_selector)
        
        # Get current mouse position (assume center of viewport)
        current_x = self.config.viewport_width // 2
        current_y = self.config.viewport_height // 2
        
        # Get target element position
        element = await page.wait_for_selector(target_selector)
        box = await element.bounding_box()
        
        if not box:
            logger.warning("Could not get bounding box for element", selector=target_selector)
            return
            
        target_x = box["x"] + box["width"] // 2
        target_y = box["y"] + box["height"] // 2
        
        # Move mouse in steps with slight randomness
        steps = self.config.mouse_move_steps
        for i in range(steps):
            progress = (i + 1) / steps
            
            # Add some randomness to the path
            noise_x = random.uniform(-5, 5)
            noise_y = random.uniform(-5, 5)
            
            x = current_x + (target_x - current_x) * progress + noise_x
            y = current_y + (target_y - current_y) * progress + noise_y
            
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.01, 0.03))
            
    async def detect_bot_challenges(self, page: Page) -> Dict[str, bool]:
        """Detect common bot detection challenges on the page."""
        logger.debug("Checking for bot detection challenges")
        
        challenges = {
            "captcha": False,
            "cloudflare": False,
            "rate_limit": False,
            "suspicious_activity": False
        }
        
        try:
            # Check for CAPTCHA
            captcha_selectors = [
                "[data-testid*='captcha']",
                ".captcha",
                "#captcha",
                "iframe[src*='recaptcha']",
                "iframe[src*='hcaptcha']"
            ]
            
            for selector in captcha_selectors:
                if await page.locator(selector).count() > 0:
                    challenges["captcha"] = True
                    break
                    
            # Check for Cloudflare
            if "cloudflare" in await page.title().lower():
                challenges["cloudflare"] = True
                
            # Check for rate limiting messages
            rate_limit_text = [
                "too many requests",
                "rate limit",
                "try again later",
                "temporarily blocked"
            ]
            
            page_text = await page.text_content("body") or ""
            for text in rate_limit_text:
                if text in page_text.lower():
                    challenges["rate_limit"] = True
                    break
                    
            # Check for suspicious activity warnings
            suspicious_text = [
                "suspicious activity",
                "unusual activity",
                "verify you're human",
                "automated behavior"
            ]
            
            for text in suspicious_text:
                if text in page_text.lower():
                    challenges["suspicious_activity"] = True
                    break
                    
        except Exception as e:
            logger.warning("Error detecting bot challenges", error=str(e))
            
        if any(challenges.values()):
            logger.warning("Bot detection challenges found", challenges=challenges)
            
        return challenges
        
    async def handle_bot_detection(self, page: Page, challenges: Dict[str, bool]) -> bool:
        """Handle detected bot challenges."""
        logger.info("Handling bot detection challenges", challenges=challenges)
        
        if challenges["captcha"]:
            logger.error("CAPTCHA detected - manual intervention required")
            return False
            
        if challenges["cloudflare"]:
            logger.info("Cloudflare challenge detected - waiting")
            await asyncio.sleep(random.uniform(5, 10))
            return True
            
        if challenges["rate_limit"]:
            logger.info("Rate limit detected - implementing backoff")
            backoff_time = random.uniform(30, 60)
            await asyncio.sleep(backoff_time)
            return True
            
        if challenges["suspicious_activity"]:
            logger.info("Suspicious activity warning - adding extra delays")
            await asyncio.sleep(random.uniform(10, 20))
            return True
            
        return True
        
    async def randomize_session_behavior(self) -> None:
        """Add random behaviors to make session appear more human."""
        behaviors = [
            self._random_scroll,
            self._random_pause,
            self._random_tab_switch
        ]
        
        # Randomly execute a behavior
        if random.random() < 0.3:  # 30% chance
            behavior = random.choice(behaviors)
            await behavior()
            
    async def _random_scroll(self) -> None:
        """Perform random scrolling behavior."""
        logger.debug("Performing random scroll behavior")
        # This would be implemented with page reference
        pass
        
    async def _random_pause(self) -> None:
        """Add random pause as if user is reading."""
        pause_time = random.uniform(2, 8)
        logger.debug("Adding random pause", duration=pause_time)
        await asyncio.sleep(pause_time)
        
    async def _random_tab_switch(self) -> None:
        """Simulate random tab switching behavior."""
        logger.debug("Simulating tab switch behavior")
        # This would be implemented with browser context
        pass
        
    def get_session_stats(self) -> Dict[str, any]:
        """Get current session statistics."""
        current_time = asyncio.get_event_loop().time()
        session_duration = current_time - self.session_start_time
        
        return {
            "action_count": self.action_count,
            "session_duration": session_duration,
            "actions_per_minute": self.action_count / (session_duration / 60) if session_duration > 0 else 0
        }