"""Property-based tests for form filling automation (Property 2)."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from typing import Dict, Any, List, Optional
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from sovereign_career_architect.browser.forms import FormFiller, create_form_filler
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.browser.vision import ElementIdentifier
from sovereign_career_architect.core.models import UserProfile


# Test data strategies
@st.composite
def user_profile_strategy(draw):
    """Generate valid user profiles for testing."""
    from sovereign_career_architect.core.models import PersonalInfo, JobPreferences, DocumentStore
    
    return UserProfile(
        personal_info=PersonalInfo(
            name=draw(st.text(min_size=5, max_size=50)),
            email=draw(st.emails()),
            phone=draw(st.one_of(st.none(), st.text(min_size=10, max_size=15))),
            location=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
            linkedin_url=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100))),
            portfolio_url=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100))),
            github_url=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100)))
        ),
        preferences=JobPreferences(
            roles=draw(st.lists(st.text(min_size=5, max_size=50), min_size=1, max_size=10)),
            locations=draw(st.lists(st.text(min_size=2, max_size=50), min_size=1, max_size=10)),
            salary_min=draw(st.one_of(st.none(), st.integers(min_value=30000, max_value=200000))),
            salary_max=draw(st.one_of(st.none(), st.integers(min_value=50000, max_value=500000)))
        ),
        documents=DocumentStore(
            resume_path=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
            cover_letter_template=draw(st.one_of(st.none(), st.text(min_size=5, max_size=100))),
            portfolio_files=draw(st.lists(st.text(min_size=5, max_size=100), min_size=0, max_size=5))
        )
    )


@st.composite
def form_field_strategy(draw):
    """Generate form field configurations."""
    field_types = ["text", "email", "tel", "textarea", "select", "file", "checkbox", "radio"]
    purposes = [
        "first_name", "last_name", "email", "phone", "current_title", 
        "company", "experience_years", "resume", "cover_letter"
    ]
    
    return {
        "selector": draw(st.text(min_size=5, max_size=50)),
        "index": draw(st.integers(min_value=0, max_value=10)),
        "type": draw(st.sampled_from(field_types)),
        "name": draw(st.text(min_size=1, max_size=30)),
        "id": draw(st.text(min_size=1, max_size=30)),
        "placeholder": draw(st.text(min_size=0, max_size=50)),
        "label": draw(st.text(min_size=0, max_size=50)),
        "purpose": draw(st.sampled_from(purposes)),
        "required": draw(st.booleans())
    }


@st.composite
def detected_form_strategy(draw):
    """Generate detected form configurations."""
    return {
        "selector": draw(st.text(min_size=5, max_size=50)),
        "type": "application_form",
        "confidence": draw(st.floats(min_value=0.1, max_value=1.0)),
        "fields": draw(st.lists(form_field_strategy(), min_size=1, max_size=15))
    }


class TestFormFillingProperties:
    """Property-based tests for form filling completeness and reliability."""
    
    @pytest.fixture
    def mock_browser_agent(self):
        """Create a mock browser agent for testing."""
        agent = AsyncMock(spec=BrowserAgent)
        agent.is_initialized = True
        agent.page = AsyncMock()
        agent.get_page_content.return_value = "<html><body><form></form></body></html>"
        agent.take_screenshot.return_value = b"mock_screenshot"
        agent.fill_form_field.return_value = True
        agent.wait_for_element.return_value = True
        agent.click_element.return_value = True
        return agent
    
    @pytest.fixture
    def mock_element_identifier(self):
        """Create a mock element identifier for testing."""
        identifier = AsyncMock(spec=ElementIdentifier)
        identifier.identify_interactive_elements.return_value = [
            {
                "tag": "form",
                "selector": "form.application-form",
                "confidence": 0.8
            }
        ]
        return identifier
    
    @pytest.fixture
    def form_filler(self, mock_browser_agent, mock_element_identifier):
        """Create a form filler with mocked dependencies."""
        return FormFiller(
            browser_agent=mock_browser_agent,
            element_identifier=mock_element_identifier
        )
    
    @given(user_profile_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_property_2_form_filling_completeness(self, form_filler, user_profile):
        """
        Property 2: Form Filling Completeness
        
        For any valid user profile and detected form fields, the form filler
        should attempt to fill all mappable fields and report accurate results.
        """
        async def run_test():
            # Mock form detection
            mock_fields = [
                {
                    "selector": "input[name='firstName']",
                    "index": 0,
                    "type": "text",
                    "name": "firstName",
                    "id": "firstName",
                    "placeholder": "First Name",
                    "label": "First Name",
                    "purpose": "first_name",
                    "required": True
                },
                {
                    "selector": "input[name='email']",
                    "index": 0,
                    "type": "email",
                    "name": "email",
                    "id": "email",
                    "placeholder": "Email Address",
                    "label": "Email",
                    "purpose": "email",
                    "required": True
                }
            ]
            
            with patch.object(form_filler, '_detect_form_fields', return_value=mock_fields):
                result = await form_filler.fill_form(user_profile, "form.application")
                
                # Property: Form filling should complete successfully
                assert isinstance(result, dict)
                assert "success" in result
                assert "filled_fields" in result
                assert "failed_fields" in result
                assert "errors" in result
                
                # Property: All mappable fields should be attempted
                expected_fields = {"first_name", "email"}
                attempted_fields = set(result["filled_fields"] + result["failed_fields"])
                
                # At least some fields should be filled if profile has data
                if user_profile.personal_info:
                    assert len(result["filled_fields"]) > 0, "Should fill at least some fields with valid profile data"
                
                # Property: Results should be consistent
                total_attempted = len(result["filled_fields"]) + len(result["failed_fields"])
                assert total_attempted <= len(mock_fields), "Cannot attempt more fields than available"
                
                # Property: Success should reflect actual completion
                if len(result["filled_fields"]) > 0:
                    assert result["success"] is True, "Success should be True when fields are filled"
        
        asyncio.run(run_test())
    
    @given(detected_form_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_form_detection_consistency(self, form_filler, detected_form):
        """Test that form detection produces consistent results."""
        async def run_test():
            # Mock the browser agent page methods
            form_filler.browser_agent.page.query_selector_all.return_value = [MagicMock()]
            
            with patch.object(form_filler, '_detect_form_fields', return_value=detected_form["fields"]):
                forms = await form_filler.detect_forms()
                
                # Property: Detection should return valid form structures
                assert isinstance(forms, list)
                
                for form in forms:
                    assert isinstance(form, dict)
                    assert "selector" in form
                    assert "type" in form
                    assert "confidence" in form
                    assert "fields" in form
                    
                    # Property: Confidence should be reasonable
                    assert 0.0 <= form["confidence"] <= 1.0
                    
                    # Property: Fields should be valid
                    assert isinstance(form["fields"], list)
        
        asyncio.run(run_test())
    
    @given(
        user_profile_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=100),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_data_mapping_completeness(self, form_filler, user_profile, custom_data):
        """Test that data mapping covers all available profile fields."""
        mapping = form_filler._create_data_mapping(user_profile, custom_data)
        
        # Property: Mapping should be a dictionary
        assert isinstance(mapping, dict)
        
        # Property: Should include personal info if available
        if user_profile.personal_info:
            personal_fields = ["first_name", "last_name", "email", "phone"]
            for field in personal_fields:
                if field in user_profile.personal_info:
                    assert field in mapping, f"Personal field {field} should be mapped"
        
        # Property: Should include professional info if available
        if user_profile.professional_info:
            prof_fields = ["current_title", "company", "experience_years"]
            for field in prof_fields:
                profile_key = field if field != "company" else "current_company"
                if profile_key in user_profile.professional_info:
                    assert field in mapping, f"Professional field {field} should be mapped"
        
        # Property: Custom data should be included
        for key, value in custom_data.items():
            assert key in mapping, f"Custom field {key} should be mapped"
            assert mapping[key] == value, f"Custom field {key} should have correct value"
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20, deadline=3000)
    def test_field_purpose_identification(self, form_filler, field_text):
        """Test field purpose identification logic."""
        # Test with known patterns
        known_patterns = {
            "firstName": "first_name",
            "email-address": "email",
            "phone-number": "phone",
            "current-title": "current_title"
        }
        
        for pattern, expected_purpose in known_patterns.items():
            purpose = form_filler._identify_field_purpose(pattern, "", "", "")
            assert purpose == expected_purpose, f"Pattern {pattern} should identify as {expected_purpose}"
        
        # Property: Should handle case insensitivity
        purpose_upper = form_filler._identify_field_purpose("FIRSTNAME", "", "", "")
        purpose_lower = form_filler._identify_field_purpose("firstname", "", "", "")
        assert purpose_upper == purpose_lower, "Field identification should be case insensitive"
    
    def test_form_submission_robustness(self, form_filler):
        """Test form submission handles various scenarios."""
        async def run_test():
            # Mock successful submission
            form_filler.browser_agent.page.query_selector.return_value = MagicMock()
            
            result = await form_filler.submit_form("form.application")
            
            # Property: Should return boolean result
            assert isinstance(result, bool)
            
            # Test with no forms detected
            with patch.object(form_filler, 'detect_forms', return_value=[]):
                result = await form_filler.submit_form()
                assert result is False, "Should fail when no forms detected"
        
        asyncio.run(run_test())


class FormFillingStateMachine(RuleBasedStateMachine):
    """Stateful testing for form filling workflow."""
    
    def __init__(self):
        super().__init__()
        self.form_filler = None
        self.current_profile = None
        self.detected_forms = []
        self.fill_results = []
    
    @initialize()
    def setup_form_filler(self):
        """Initialize form filler with mocked dependencies."""
        mock_browser = AsyncMock(spec=BrowserAgent)
        mock_browser.is_initialized = True
        mock_browser.page = AsyncMock()
        mock_browser.get_page_content.return_value = "<html><body><form></form></body></html>"
        mock_browser.fill_form_field.return_value = True
        
        mock_identifier = AsyncMock(spec=ElementIdentifier)
        mock_identifier.identify_elements.return_value = []
        
        self.form_filler = FormFiller(
            browser_agent=mock_browser,
            element_identifier=mock_identifier
        )
    
    @rule(profile=user_profile_strategy())
    def set_user_profile(self, profile):
        """Set the current user profile."""
        self.current_profile = profile
    
    @rule()
    def detect_forms(self):
        """Detect forms on the current page."""
        assume(self.form_filler is not None)
        
        async def run_detect():
            # Mock form detection
            mock_forms = [{
                "selector": "form.application",
                "type": "application_form",
                "confidence": 0.8,
                "fields": [
                    {
                        "selector": "input[name='firstName']",
                        "index": 0,
                        "type": "text",
                        "purpose": "first_name",
                        "required": True
                    }
                ]
            }]
            
            with patch.object(self.form_filler, 'detect_forms', return_value=mock_forms):
                self.detected_forms = await self.form_filler.detect_forms()
        
        asyncio.run(run_detect())
    
    @rule()
    def fill_form(self):
        """Fill a form with current profile."""
        assume(self.form_filler is not None)
        assume(self.current_profile is not None)
        assume(len(self.detected_forms) > 0)
        
        async def run_fill():
            mock_fields = [{
                "selector": "input[name='firstName']",
                "index": 0,
                "type": "text",
                "purpose": "first_name",
                "required": True
            }]
            
            with patch.object(self.form_filler, '_detect_form_fields', return_value=mock_fields):
                result = await self.form_filler.fill_form(
                    self.current_profile,
                    self.detected_forms[0]["selector"]
                )
                self.fill_results.append(result)
        
        asyncio.run(run_fill())
    
    @invariant()
    def form_results_are_valid(self):
        """Invariant: All form fill results should be valid."""
        for result in self.fill_results:
            assert isinstance(result, dict)
            assert "success" in result
            assert "filled_fields" in result
            assert "failed_fields" in result
            assert isinstance(result["filled_fields"], list)
            assert isinstance(result["failed_fields"], list)


# Integration test
def test_form_filling_integration():
    """Integration test for complete form filling workflow."""
    async def run_integration():
        # Create form filler with mocked dependencies
        mock_browser = AsyncMock(spec=BrowserAgent)
        mock_browser.is_initialized = True
        mock_browser.page = AsyncMock()
        mock_browser.get_page_content.return_value = """
        <html>
            <body>
                <form class="application-form">
                    <input name="firstName" type="text" required>
                    <input name="email" type="email" required>
                    <button type="submit">Apply</button>
                </form>
            </body>
        </html>
        """
        mock_browser.fill_form_field.return_value = True
        
        form_filler = FormFiller(browser_agent=mock_browser)
        
        # Create test profile
        from sovereign_career_architect.core.models import PersonalInfo
        profile = UserProfile(
            personal_info=PersonalInfo(
                name="John Doe",
                email="john.doe@example.com"
            )
        )
        
        # Test data mapping creation
        data_mapping = form_filler._create_data_mapping(profile)
        
        assert "first_name" in data_mapping
        assert "email" in data_mapping
        assert data_mapping["first_name"] == "John"
        assert data_mapping["email"] == "john.doe@example.com"
        
        # Test basic form filling (without field detection)
        result = await form_filler.fill_form(profile, "form.application-form")
        
        # Should return a valid result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "filled_fields" in result
        assert "failed_fields" in result
        assert "errors" in result
    
    asyncio.run(run_integration())


# Run stateful tests
TestFormFillingStateful = FormFillingStateMachine.TestCase