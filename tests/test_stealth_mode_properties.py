"""Property-based tests for stealth mode functionality."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from sovereign_career_architect.browser.stealth import StealthManager, StealthConfig


class TestStealthModeProperties:
    """Property-based tests for stealth mode activation and behavior."""
    
    @given(
        min_delay=st.floats(min_value=0.1, max_value=2.0),
        max_delay=st.floats(min_value=2.1, max_value=5.0),
        typing_delay_min=st.floats(min_value=0.01, max_value=0.1),
        typing_delay_max=st.floats(min_value=0.11, max_value=0.3),
    )
    @settings(max_examples=50)
    def test_stealth_config_consistency(self, min_delay, max_delay, typing_delay_min, typing_delay_max):
        """Property 4: Stealth Mode Activation - Configuration consistency."""
        assume(min_delay < max_delay)
        assume(typing_delay_min < typing_delay_max)
        
        config = StealthConfig(
            min_delay=min_delay,
            max_delay=max_delay,
            typing_delay_min=typing_delay_min,
            typing_delay_max=typing_delay_max
        )
        
        # Property: Configuration values should be consistent
        assert config.min_delay < config.max_delay
        assert config.typing_delay_min < config.typing_delay_max
        assert config.viewport_width > 0
        assert config.viewport_height > 0
        assert len(config.user_agents) > 0
        
        # Property: All user agents should be valid strings
        for user_agent in config.user_agents:
            assert isinstance(user_agent, str)
            assert len(user_agent) > 0
            assert "Mozilla" in user_agent  # Common browser identifier
    
    @given(
        action_counts=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=20)
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_human_like_delay_properties(self, action_counts):
        """Property 4: Stealth Mode Activation - Human-like delay consistency."""
        config = StealthConfig(min_delay=0.5, max_delay=2.0)
        stealth_manager = StealthManager(config)
        
        delays = []
        
        # Simulate multiple actions and measure delays
        for count in action_counts:
            stealth_manager.action_count = count
            
            start_time = asyncio.get_event_loop().time()
            await stealth_manager.human_like_delay("general")
            end_time = asyncio.get_event_loop().time()
            
            actual_delay = end_time - start_time
            delays.append(actual_delay)
        
        # Property: All delays should be within reasonable bounds
        for delay in delays:
            assert delay >= config.min_delay * 0.9  # Allow small timing variance
            assert delay <= config.max_delay + 5.0   # Allow for progressive delays
        
        # Property: Delays should show some variation (not all identical)
        if len(delays) > 1:
            delay_variance = max(delays) - min(delays)
            assert delay_variance > 0.1  # Should have some randomness
    
    @given(
        text_inputs=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs')), 
                   min_size=1, max_size=100),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_human_like_typing_properties(self, text_inputs):
        """Property 4: Stealth Mode Activation - Typing behavior consistency."""
        stealth_manager = StealthManager()
        
        # Mock page and element
        mock_page = AsyncMock()
        mock_element = AsyncMock()
        mock_page.wait_for_selector.return_value = mock_element
        
        for text in text_inputs:
            assume(len(text.strip()) > 0)  # Skip empty or whitespace-only strings
            
            start_time = asyncio.get_event_loop().time()
            await stealth_manager.human_like_typing(mock_page, "#test", text)
            end_time = asyncio.get_event_loop().time()
            
            typing_duration = end_time - start_time
            
            # Property: Typing duration should correlate with text length
            expected_min_duration = len(text) * stealth_manager.config.typing_delay_min
            expected_max_duration = len(text) * stealth_manager.config.typing_delay_max * 2  # Allow for space delays
            
            assert typing_duration >= expected_min_duration * 0.8  # Allow timing variance
            assert typing_duration <= expected_max_duration * 2    # Allow for extra delays
            
            # Property: Element interactions should be called correctly
            mock_element.click.assert_called()
            mock_element.fill.assert_called_with("")
            assert mock_element.type.call_count == len(text)
    
    @given(
        challenge_combinations=st.lists(
            st.sampled_from(["captcha", "cloudflare", "rate_limit", "suspicious_activity"]),
            min_size=0, max_size=4
        )
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_bot_detection_handling_properties(self, challenge_combinations):
        """Property 4: Stealth Mode Activation - Bot detection handling consistency."""
        stealth_manager = StealthManager()
        
        # Create challenge dictionary
        challenges = {
            "captcha": "captcha" in challenge_combinations,
            "cloudflare": "cloudflare" in challenge_combinations,
            "rate_limit": "rate_limit" in challenge_combinations,
            "suspicious_activity": "suspicious_activity" in challenge_combinations
        }
        
        mock_page = AsyncMock()
        
        start_time = asyncio.get_event_loop().time()
        result = await stealth_manager.handle_bot_detection(mock_page, challenges)
        end_time = asyncio.get_event_loop().time()
        
        handling_duration = end_time - start_time
        
        # Property: CAPTCHA should always return False (requires manual intervention)
        if challenges["captcha"]:
            assert result is False
        else:
            assert result is True
        
        # Property: Handling should take appropriate time based on challenge type
        if challenges["rate_limit"]:
            assert handling_duration >= 30  # Should wait at least 30 seconds
        elif challenges["cloudflare"]:
            assert handling_duration >= 5   # Should wait at least 5 seconds
        elif challenges["suspicious_activity"]:
            assert handling_duration >= 10  # Should wait at least 10 seconds
        elif not any(challenges.values()):
            assert handling_duration < 1    # No challenges should be fast
    
    @given(
        viewport_dimensions=st.tuples(
            st.integers(min_value=800, max_value=2560),
            st.integers(min_value=600, max_value=1440)
        ),
        user_agent_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_stealth_context_setup_properties(self, viewport_dimensions, user_agent_count):
        """Property 4: Stealth Mode Activation - Context setup consistency."""
        width, height = viewport_dimensions
        
        # Create custom user agents
        user_agents = [f"TestAgent{i}/1.0" for i in range(user_agent_count)]
        config = StealthConfig(
            viewport_width=width,
            viewport_height=height,
            user_agents=user_agents
        )
        
        stealth_manager = StealthManager(config)
        
        # Mock browser context
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        await stealth_manager.setup_stealth_context(mock_context)
        await stealth_manager.setup_stealth_page(mock_page)
        
        # Property: Context should be configured with headers
        mock_context.set_extra_http_headers.assert_called_once()
        headers = mock_context.set_extra_http_headers.call_args[0][0]
        
        assert "User-Agent" in headers
        assert headers["User-Agent"] in user_agents
        assert "Accept" in headers
        assert "DNT" in headers
        
        # Property: Page should be configured with viewport
        mock_page.set_viewport_size.assert_called_once()
        viewport = mock_page.set_viewport_size.call_args[0][0]
        
        assert viewport["width"] == width
        assert viewport["height"] == height
        
        # Property: Stealth script should be added
        mock_context.add_init_script.assert_called_once()
        script = mock_context.add_init_script.call_args[0][0]
        assert "webdriver" in script
        assert "navigator" in script
    
    @given(
        action_sequences=st.lists(
            st.sampled_from(["click", "type", "navigate", "form_submit", "general"]),
            min_size=1, max_size=20
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_session_behavior_properties(self, action_sequences):
        """Property 4: Stealth Mode Activation - Session behavior consistency."""
        stealth_manager = StealthManager()
        
        initial_stats = stealth_manager.get_session_stats()
        assert initial_stats["action_count"] == 0
        assert initial_stats["session_duration"] >= 0
        
        # Simulate action sequence
        for action_type in action_sequences:
            await stealth_manager.human_like_delay(action_type)
        
        final_stats = stealth_manager.get_session_stats()
        
        # Property: Action count should match sequence length
        assert final_stats["action_count"] == len(action_sequences)
        
        # Property: Session duration should increase
        assert final_stats["session_duration"] > initial_stats["session_duration"]
        
        # Property: Actions per minute should be reasonable (not too fast)
        if final_stats["session_duration"] > 0:
            actions_per_minute = final_stats["actions_per_minute"]
            assert actions_per_minute < 60  # Less than 1 action per second on average
            assert actions_per_minute >= 0
    
    @given(
        page_content=st.text(min_size=100, max_size=1000),
        challenge_indicators=st.lists(
            st.sampled_from([
                "captcha", "recaptcha", "hcaptcha", "cloudflare",
                "too many requests", "rate limit", "suspicious activity",
                "verify you're human", "automated behavior"
            ]),
            min_size=0, max_size=5
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_bot_detection_accuracy_properties(self, page_content, challenge_indicators):
        """Property 4: Stealth Mode Activation - Bot detection accuracy."""
        stealth_manager = StealthManager()
        
        # Mock page with content containing challenge indicators
        mock_page = AsyncMock()
        
        # Simulate page content with challenge indicators
        content_with_challenges = page_content
        for indicator in challenge_indicators:
            content_with_challenges += f" {indicator} "
        
        mock_page.title.return_value = asyncio.coroutine(lambda: content_with_challenges)()
        mock_page.text_content.return_value = asyncio.coroutine(lambda: content_with_challenges)()
        
        # Mock locator for CAPTCHA detection
        mock_locator = AsyncMock()
        captcha_present = any("captcha" in indicator for indicator in challenge_indicators)
        mock_locator.count.return_value = asyncio.coroutine(lambda: 1 if captcha_present else 0)()
        mock_page.locator.return_value = mock_locator
        
        challenges = await stealth_manager.detect_bot_challenges(mock_page)
        
        # Property: Detection should be accurate based on content
        if any("captcha" in indicator for indicator in challenge_indicators):
            assert challenges["captcha"] is True
        
        if any("cloudflare" in indicator for indicator in challenge_indicators):
            assert challenges["cloudflare"] is True
        
        if any(indicator in ["too many requests", "rate limit"] for indicator in challenge_indicators):
            assert challenges["rate_limit"] is True
        
        if any(indicator in ["suspicious activity", "verify you're human", "automated behavior"] 
               for indicator in challenge_indicators):
            assert challenges["suspicious_activity"] is True
        
        # Property: If no indicators present, no challenges should be detected
        if not challenge_indicators:
            assert not any(challenges.values())
    
    @pytest.mark.asyncio
    async def test_stealth_manager_initialization_properties(self):
        """Property 4: Stealth Mode Activation - Initialization consistency."""
        # Test with default config
        stealth_manager1 = StealthManager()
        assert stealth_manager1.config is not None
        assert stealth_manager1.action_count == 0
        assert stealth_manager1.session_start_time > 0
        
        # Test with custom config
        custom_config = StealthConfig(min_delay=2.0, max_delay=5.0)
        stealth_manager2 = StealthManager(custom_config)
        assert stealth_manager2.config.min_delay == 2.0
        assert stealth_manager2.config.max_delay == 5.0
        
        # Property: Different instances should have different session start times
        await asyncio.sleep(0.01)  # Small delay
        stealth_manager3 = StealthManager()
        assert stealth_manager3.session_start_time > stealth_manager1.session_start_time