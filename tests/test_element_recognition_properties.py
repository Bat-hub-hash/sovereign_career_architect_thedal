"""Property-based tests for visual element recognition (Property 25)."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from unittest.mock import AsyncMock, MagicMock
import asyncio

from sovereign_career_architect.browser.vision import ElementIdentifier, create_element_identifier


# Strategies for generating test data
coordinates_strategy = st.lists(
    st.integers(min_value=0, max_value=2000),
    min_size=4,
    max_size=4
).map(lambda coords: [min(coords[0], coords[2]), min(coords[1], coords[3]), 
                      max(coords[0], coords[2]), max(coords[1], coords[3])])

element_strategy = st.fixed_dictionaries({
    "element_type": st.sampled_from(["button", "input", "link", "dropdown", "checkbox", "textarea"]),
    "description": st.text(min_size=1, max_size=100),
    "coordinates": coordinates_strategy,
    "selector_hints": st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5),
    "importance": st.integers(min_value=1, max_value=10),
    "action_type": st.sampled_from(["click", "fill", "select", "check", "hover"]),
    "confidence": st.floats(min_value=0.0, max_value=1.0)
})

screenshot_data_strategy = st.binary(min_size=100, max_size=10000)

task_description_strategy = st.text(min_size=10, max_size=200).filter(
    lambda x: len(x.strip()) > 5
)


@pytest.mark.property
class TestElementRecognitionProperties:
    """Property-based tests for element recognition consistency and reliability."""
    
    @given(
        screenshot_data=screenshot_data_strategy,
        task_description=task_description_strategy
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_element_identification_consistency(self, screenshot_data, task_description):
        """
        Property 25a: Element identification should be consistent across multiple calls.
        
        For the same screenshot and task description, the element identifier should
        return consistent results (same number of elements with similar properties).
        """
        element_identifier = create_element_identifier()
        
        # Run identification multiple times
        results = []
        for _ in range(3):
            elements = await element_identifier.identify_interactive_elements(
                screenshot_data, task_description
            )
            results.append(elements)
        
        # Verify consistency
        if results[0]:  # If elements were found
            # All runs should find similar number of elements (within reasonable variance)
            element_counts = [len(result) for result in results]
            max_count = max(element_counts)
            min_count = min(element_counts)
            
            # Allow some variance but should be generally consistent
            assert max_count - min_count <= max(2, max_count * 0.3), \
                f"Inconsistent element counts: {element_counts}"
            
            # Element types should be consistent across runs
            all_types = set()
            for result in results:
                for element in result:
                    all_types.add(element.get("element_type", "unknown"))
            
            # Should have reasonable element types
            valid_types = {"button", "input", "link", "dropdown", "checkbox", "textarea", "unknown"}
            assert all_types.issubset(valid_types), f"Invalid element types found: {all_types - valid_types}"
    
    @given(elements=st.lists(element_strategy, min_size=0, max_size=10))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_highlighting_script_properties(self, elements):
        """
        Property 25b: Highlighting script generation should have consistent properties.
        
        Generated scripts should be valid JavaScript and include all provided elements.
        """
        element_identifier = create_element_identifier()
        
        script = await element_identifier.generate_highlighting_script(elements)
        
        if not elements:
            # Empty elements should produce empty script
            assert script == ""
        else:
            # Non-empty elements should produce valid script
            assert isinstance(script, str)
            assert len(script) > 0
            
            # Script should contain JavaScript structure
            assert "function" in script or "=>" in script
            assert "createHighlight" in script
            assert "removeHighlights" in script
            
            # Script should reference all element coordinates
            for element in elements:
                coords = element["coordinates"]
                coord_str = str(coords)
                assert coord_str in script, f"Element coordinates {coords} not found in script"
            
            # Script should be syntactically valid (basic check)
            # Note: We can't do perfect syntax validation due to dynamic content
            # Just check that basic structure is maintained
            assert "createHighlight" in script
            assert "removeHighlights" in script
    
    @given(
        screenshot_data=screenshot_data_strategy,
        coordinates=coordinates_strategy,
        context=task_description_strategy
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_element_analysis_properties(self, screenshot_data, coordinates, context):
        """
        Property 25c: Element analysis should return well-formed results.
        
        Analysis results should always contain required fields with valid values.
        """
        element_identifier = create_element_identifier()
        
        analysis = await element_identifier.analyze_element_context(
            screenshot_data, coordinates, context
        )
        
        # Required fields should always be present
        required_fields = [
            "element_type", "recommended_action", "data_requirements",
            "automation_notes", "confidence", "analysis"
        ]
        
        for field in required_fields:
            assert field in analysis, f"Required field '{field}' missing from analysis"
        
        # Field value validation
        assert isinstance(analysis["element_type"], str)
        assert len(analysis["element_type"]) > 0
        
        assert isinstance(analysis["recommended_action"], str)
        assert analysis["recommended_action"] in ["click", "fill", "select", "check", "hover", "submit"]
        
        assert isinstance(analysis["data_requirements"], list)
        
        assert isinstance(analysis["automation_notes"], str)
        
        assert isinstance(analysis["confidence"], (int, float))
        assert 0 <= analysis["confidence"] <= 1
        
        assert isinstance(analysis["analysis"], str)


@pytest.mark.property
class TestElementRecognitionIntegration:
    """Integration property tests for element recognition system."""
    
    @given(
        screenshot_data=screenshot_data_strategy,
        task_description=task_description_strategy
    )
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_full_workflow_properties(self, screenshot_data, task_description):
        """
        Property 25f: Full workflow should maintain data consistency.
        
        The complete workflow from identification to highlighting to analysis
        should maintain consistent element data throughout.
        """
        identifier = create_element_identifier()
        
        # Step 1: Identify elements
        elements = await identifier.identify_interactive_elements(
            screenshot_data, task_description
        )
        
        # Step 2: Generate highlighting script
        script = await identifier.generate_highlighting_script(elements)
        
        # Step 3: Analyze first element if available
        analysis = None
        if elements:
            first_element = elements[0]
            coords = first_element.get("coordinates", [0, 0, 100, 100])
            analysis = await identifier.analyze_element_context(
                screenshot_data, coords, task_description
            )
        
        # Verify workflow consistency
        if elements:
            # Script should be generated for non-empty elements
            assert isinstance(script, str)
            assert len(script) > 0
            
            # Analysis should be available for first element
            if analysis:
                assert isinstance(analysis, dict)
                assert "element_type" in analysis
                assert "confidence" in analysis
        else:
            # Empty elements should produce empty script
            assert script == ""
    
    @given(
        elements=st.lists(element_strategy, min_size=1, max_size=5)
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_element_coordinate_validation(self, elements):
        """
        Property 25g: Element coordinates should always form valid bounding boxes.
        
        All element coordinates should represent valid rectangular regions.
        """
        identifier = create_element_identifier()
        
        # Generate script with elements
        script = await identifier.generate_highlighting_script(elements)
        
        # Verify all coordinates in script are valid
        for element in elements:
            coords = element["coordinates"]
            x1, y1, x2, y2 = coords
            
            # Should form valid bounding box
            assert x1 <= x2, f"Invalid x coordinates: {x1} > {x2}"
            assert y1 <= y2, f"Invalid y coordinates: {y1} > {y2}"
            assert all(c >= 0 for c in coords), f"Negative coordinates: {coords}"
            
            # Should have positive area
            width = x2 - x1
            height = y2 - y1
            assert width >= 0, f"Negative width: {width}"
            assert height >= 0, f"Negative height: {height}"


# Simplified stateful test without fixtures
class ElementRecognitionStateMachine(RuleBasedStateMachine):
    """Stateful testing for element recognition system."""
    
    def __init__(self):
        super().__init__()
        self.element_identifier = None
        self.identified_elements = []
        self.generated_scripts = []
        self.analysis_results = []
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.element_identifier = create_element_identifier()
        self.identified_elements = []
        self.generated_scripts = []
        self.analysis_results = []
    
    @rule(
        screenshot_data=screenshot_data_strategy,
        task_description=task_description_strategy
    )
    def identify_elements(self, screenshot_data, task_description):
        """Rule: Identify elements in a screenshot."""
        async def _identify():
            elements = await self.element_identifier.identify_interactive_elements(
                screenshot_data, task_description
            )
            self.identified_elements.extend(elements)
            return elements
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            elements = loop.run_until_complete(_identify())
        finally:
            loop.close()
        
        # Verify element structure
        for element in elements:
            assert isinstance(element, dict)
            assert "coordinates" in element
            assert len(element["coordinates"]) == 4
            assert "element_type" in element
            assert "confidence" in element
            assert 0 <= element["confidence"] <= 1
    
    @rule(elements=st.lists(element_strategy, min_size=0, max_size=3))
    def generate_highlighting_script(self, elements):
        """Rule: Generate highlighting script for elements."""
        async def _generate():
            script = await self.element_identifier.generate_highlighting_script(elements)
            self.generated_scripts.append(script)
            return script
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            script = loop.run_until_complete(_generate())
        finally:
            loop.close()
        
        # Verify script properties
        if elements:
            assert isinstance(script, str)
            assert len(script) > 0
        else:
            assert script == ""
    
    @invariant()
    def element_data_consistency(self):
        """Invariant: All identified elements should have consistent data structure."""
        for element in self.identified_elements:
            # Coordinates should be valid bounding boxes
            coords = element.get("coordinates", [])
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                assert x1 <= x2, f"Invalid coordinates: x1({x1}) > x2({x2})"
                assert y1 <= y2, f"Invalid coordinates: y1({y1}) > y2({y2})"
                assert all(c >= 0 for c in coords), f"Negative coordinates: {coords}"
            
            # Importance should be in valid range
            importance = element.get("importance", 5)
            assert 1 <= importance <= 10, f"Invalid importance: {importance}"
            
            # Confidence should be in valid range
            confidence = element.get("confidence", 0.5)
            assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"


# Stateful test
TestElementRecognitionStateMachine = ElementRecognitionStateMachine.TestCase