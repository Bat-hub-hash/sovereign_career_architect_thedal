"""Tests for computer vision element identification."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from sovereign_career_architect.browser.vision import ElementIdentifier, create_element_identifier


class TestElementIdentifier:
    """Test cases for ElementIdentifier."""
    
    @pytest.fixture
    def element_identifier(self):
        """Create an ElementIdentifier instance for testing."""
        return ElementIdentifier()
    
    @pytest.fixture
    def mock_screenshot_data(self):
        """Mock screenshot data for testing."""
        return b"mock_screenshot_data_bytes"
    
    def test_element_identifier_initialization(self):
        """Test ElementIdentifier initialization."""
        # Test default initialization
        identifier = ElementIdentifier()
        assert identifier.openai_client is None  # No API key in test environment
        
        # Test with custom client
        mock_client = MagicMock()
        identifier_custom = ElementIdentifier(openai_client=mock_client)
        assert identifier_custom.openai_client == mock_client
    
    @pytest.mark.asyncio
    async def test_identify_interactive_elements_mock(self, element_identifier, mock_screenshot_data):
        """Test element identification with mock implementation."""
        elements = await element_identifier.identify_interactive_elements(
            mock_screenshot_data,
            "Identify elements for job application"
        )
        
        assert isinstance(elements, list)
        assert len(elements) == 3  # Mock returns 3 elements for job-related tasks
        
        # Verify element structure
        for element in elements:
            assert "element_type" in element
            assert "description" in element
            assert "coordinates" in element
            assert "selector_hints" in element
            assert "importance" in element
            assert "action_type" in element
            assert "confidence" in element
            
            # Verify coordinate format
            coords = element["coordinates"]
            assert isinstance(coords, list)
            assert len(coords) == 4
            assert all(isinstance(c, int) for c in coords)
            
            # Verify importance range
            assert 1 <= element["importance"] <= 10
            
            # Verify confidence range
            assert 0 <= element["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_identify_interactive_elements_non_job_task(self, element_identifier, mock_screenshot_data):
        """Test element identification with non-job related task."""
        elements = await element_identifier.identify_interactive_elements(
            mock_screenshot_data,
            "General web page analysis"
        )
        
        assert isinstance(elements, list)
        # Mock returns empty list for non-job tasks
        assert len(elements) == 0
    
    @pytest.mark.asyncio
    async def test_generate_highlighting_script(self, element_identifier):
        """Test JavaScript highlighting script generation."""
        # Test with empty elements
        script_empty = await element_identifier.generate_highlighting_script([])
        assert script_empty == ""
        
        # Test with mock elements
        mock_elements = [
            {
                "coordinates": [100, 150, 400, 180],
                "description": "Search input",
                "importance": 9
            },
            {
                "coordinates": [420, 150, 500, 180],
                "description": "Search button",
                "importance": 8
            }
        ]
        
        script = await element_identifier.generate_highlighting_script(
            mock_elements,
            highlight_color="#00ff00",
            highlight_opacity=0.5
        )
        
        assert isinstance(script, str)
        assert len(script) > 0
        assert "createHighlight" in script
        assert "removeHighlights" in script
        assert "#00ff00" in script
        assert "0.5" in script
        assert "[100, 150, 400, 180]" in script
        assert "[420, 150, 500, 180]" in script
        assert "Search input" in script
        assert "Search button" in script
    
    @pytest.mark.asyncio
    async def test_analyze_element_context_mock(self, element_identifier, mock_screenshot_data):
        """Test element context analysis with mock implementation."""
        # Test input element analysis
        analysis_input = await element_identifier.analyze_element_context(
            mock_screenshot_data,
            [100, 150, 400, 180],
            "Analyze this input field for form filling"
        )
        
        assert analysis_input["element_type"] == "input"
        assert analysis_input["recommended_action"] == "fill"
        assert "text_value" in analysis_input["data_requirements"]
        assert analysis_input["confidence"] > 0
        
        # Test button element analysis
        analysis_button = await element_identifier.analyze_element_context(
            mock_screenshot_data,
            [420, 150, 500, 180],
            "Analyze this button for clicking"
        )
        
        assert analysis_button["element_type"] == "button"
        assert analysis_button["recommended_action"] == "click"
        assert analysis_button["data_requirements"] == []
        assert analysis_button["confidence"] > 0
        
        # Test generic element analysis
        analysis_generic = await element_identifier.analyze_element_context(
            mock_screenshot_data,
            [100, 200, 300, 250],
            "Analyze this element"
        )
        
        assert analysis_generic["element_type"] == "link"
        assert analysis_generic["recommended_action"] == "click"
        assert analysis_generic["confidence"] > 0
    
    def test_parse_elements_response(self, element_identifier):
        """Test parsing of GPT-4o response for element identification."""
        # Test valid JSON response
        valid_response = '''
        Here are the identified elements:
        [
            {
                "element_type": "button",
                "description": "Submit button",
                "coordinates": [100, 200, 200, 230],
                "selector_hints": ["button[type='submit']"],
                "importance": 9,
                "action_type": "click"
            },
            {
                "element_type": "input",
                "description": "Email input",
                "coordinates": [100, 150, 300, 180],
                "selector_hints": ["input[type='email']"],
                "importance": 8,
                "action_type": "fill"
            }
        ]
        Additional text after JSON.
        '''
        
        elements = element_identifier._parse_elements_response(valid_response)
        
        assert len(elements) == 2
        assert elements[0]["element_type"] == "button"
        assert elements[0]["confidence"] == 0.8  # Default confidence
        assert elements[1]["element_type"] == "input"
        assert elements[1]["importance"] == 8
        
        # Test invalid JSON response
        invalid_response = "No JSON here, just text"
        elements_invalid = element_identifier._parse_elements_response(invalid_response)
        assert elements_invalid == []
        
        # Test malformed JSON
        malformed_response = "[{invalid json}]"
        elements_malformed = element_identifier._parse_elements_response(malformed_response)
        assert elements_malformed == []
    
    def test_parse_analysis_response(self, element_identifier):
        """Test parsing of GPT-4o response for element analysis."""
        # Test valid JSON response
        valid_response = '''
        Analysis result:
        {
            "element_type": "input",
            "recommended_action": "fill",
            "data_requirements": ["email"],
            "automation_notes": "Standard email input",
            "confidence": 0.9,
            "analysis": "This is an email input field"
        }
        '''
        
        analysis = element_identifier._parse_analysis_response(valid_response)
        
        assert analysis["element_type"] == "input"
        assert analysis["recommended_action"] == "fill"
        assert analysis["data_requirements"] == ["email"]
        assert analysis["confidence"] == 0.9
        
        # Test response without JSON
        no_json_response = "This is just text analysis without JSON structure"
        analysis_no_json = element_identifier._parse_analysis_response(no_json_response)
        
        assert analysis_no_json["element_type"] == "unknown"
        assert analysis_no_json["recommended_action"] == "click"
        assert analysis_no_json["confidence"] == 0.3
        assert analysis_no_json["analysis"] == no_json_response
    
    @pytest.mark.asyncio
    async def test_error_handling(self, element_identifier, mock_screenshot_data):
        """Test error handling in element identification."""
        # Mock OpenAI client that raises exceptions
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        element_identifier.openai_client = mock_client
        
        # Test element identification error handling
        elements = await element_identifier.identify_interactive_elements(
            mock_screenshot_data,
            "Test task"
        )
        
        # Should fall back to mock implementation
        assert isinstance(elements, list)
        
        # Test element analysis error handling
        analysis = await element_identifier.analyze_element_context(
            mock_screenshot_data,
            [100, 100, 200, 200],
            "Test analysis"
        )
        
        # Should fall back to mock implementation
        assert "element_type" in analysis
        assert "confidence" in analysis


class TestElementIdentifierFactory:
    """Test cases for element identifier factory function."""
    
    def test_create_element_identifier_default(self):
        """Test factory function with default parameters."""
        identifier = create_element_identifier()
        
        assert isinstance(identifier, ElementIdentifier)
        assert identifier.openai_client is None  # No API key in test
    
    def test_create_element_identifier_custom(self):
        """Test factory function with custom client."""
        mock_client = MagicMock()
        identifier = create_element_identifier(openai_client=mock_client)
        
        assert isinstance(identifier, ElementIdentifier)
        assert identifier.openai_client == mock_client


@pytest.mark.asyncio
async def test_element_identifier_integration():
    """Integration test for element identifier workflow."""
    identifier = create_element_identifier()
    mock_screenshot = b"test_screenshot_data"
    
    # Test full workflow
    elements = await identifier.identify_interactive_elements(
        mock_screenshot,
        "Find job application elements"
    )
    
    assert isinstance(elements, list)
    
    if elements:
        # Test highlighting script generation
        script = await identifier.generate_highlighting_script(elements)
        assert isinstance(script, str)
        assert len(script) > 0
        
        # Test element analysis
        first_element = elements[0]
        analysis = await identifier.analyze_element_context(
            mock_screenshot,
            first_element["coordinates"],
            "Analyze this element for automation"
        )
        
        assert isinstance(analysis, dict)
        assert "element_type" in analysis
        assert "recommended_action" in analysis
        assert "confidence" in analysis


@pytest.mark.asyncio
async def test_element_identifier_with_gpt4o_mock():
    """Test element identifier with mocked GPT-4o responses."""
    # Mock OpenAI client
    mock_client = AsyncMock()
    
    # Mock successful element identification response
    mock_identification_response = MagicMock()
    mock_identification_response.choices = [MagicMock()]
    mock_identification_response.choices[0].message.content = '''
    [
        {
            "element_type": "button",
            "description": "Apply Now button",
            "coordinates": [300, 400, 450, 440],
            "selector_hints": ["button.apply-btn"],
            "importance": 10,
            "action_type": "click"
        }
    ]
    '''
    
    # Mock successful analysis response
    mock_analysis_response = MagicMock()
    mock_analysis_response.choices = [MagicMock()]
    mock_analysis_response.choices[0].message.content = '''
    {
        "element_type": "button",
        "recommended_action": "click",
        "data_requirements": [],
        "automation_notes": "Primary action button",
        "confidence": 0.95,
        "analysis": "This is a prominent apply button"
    }
    '''
    
    mock_client.chat.completions.create.side_effect = [
        mock_identification_response,
        mock_analysis_response
    ]
    
    identifier = ElementIdentifier(openai_client=mock_client)
    
    # Test element identification
    elements = await identifier.identify_interactive_elements(
        b"mock_screenshot",
        "Find job application elements"
    )
    
    assert len(elements) == 1
    assert elements[0]["element_type"] == "button"
    assert elements[0]["description"] == "Apply Now button"
    assert elements[0]["importance"] == 10
    
    # Test element analysis
    analysis = await identifier.analyze_element_context(
        b"mock_screenshot",
        [300, 400, 450, 440],
        "Analyze apply button"
    )
    
    assert analysis["element_type"] == "button"
    assert analysis["recommended_action"] == "click"
    assert analysis["confidence"] == 0.95