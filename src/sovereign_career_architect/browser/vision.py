"""Computer vision element identification using GPT-4o."""

import base64
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import structlog

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class ElementIdentifier:
    """
    Computer vision element identifier using GPT-4o for visual element recognition.
    
    This component analyzes screenshots to identify interactive elements,
    generate highlighting scripts, and provide element descriptions for automation.
    """
    
    def __init__(self, openai_client=None):
        """
        Initialize the element identifier.
        
        Args:
            openai_client: Optional OpenAI client for dependency injection
        """
        self.openai_client = openai_client
        self.logger = logger.bind(component="element_identifier")
        
        # Initialize OpenAI client if not provided
        if self.openai_client is None:
            self.openai_client = self._create_openai_client()
    
    def _create_openai_client(self):
        """Create OpenAI client for GPT-4o vision."""
        try:
            if settings.openai_api_key:
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key=settings.openai_api_key)
            else:
                self.logger.warning("OpenAI API key not available, using mock implementation")
                return None
        except ImportError:
            self.logger.warning("OpenAI library not available, using mock implementation")
            return None
        except Exception as e:
            self.logger.error(
                "Failed to initialize OpenAI client",
                error=str(e)
            )
            return None
    
    async def identify_interactive_elements(
        self,
        screenshot_data: bytes,
        task_description: str = "Identify interactive elements for job application"
    ) -> List[Dict[str, Any]]:
        """
        Identify interactive elements in a screenshot using GPT-4o vision.
        
        Args:
            screenshot_data: Screenshot image data as bytes
            task_description: Description of the task context
            
        Returns:
            List of identified elements with coordinates and descriptions
        """
        self.logger.debug(
            "Identifying interactive elements",
            screenshot_size=len(screenshot_data),
            task=task_description
        )
        
        try:
            if self.openai_client:
                elements = await self._identify_with_gpt4o(screenshot_data, task_description)
            else:
                elements = await self._mock_identify_elements(task_description)
            
            self.logger.info(
                "Element identification completed",
                elements_found=len(elements),
                task=task_description
            )
            
            return elements
            
        except Exception as e:
            self.logger.error(
                "Element identification failed",
                error=str(e),
                task=task_description
            )
            return []
    
    async def _identify_with_gpt4o(
        self,
        screenshot_data: bytes,
        task_description: str
    ) -> List[Dict[str, Any]]:
        """Identify elements using GPT-4o vision API."""
        try:
            # Encode screenshot as base64
            screenshot_b64 = base64.b64encode(screenshot_data).decode('utf-8')
            
            # Prepare vision prompt
            system_prompt = """You are an expert at analyzing web page screenshots to identify interactive elements for automation.

Your task is to identify clickable and fillable elements that are relevant for job applications and career-related tasks.

For each element you identify, provide:
1. element_type: The type of element (button, input, link, dropdown, checkbox, etc.)
2. description: A clear description of what the element does
3. coordinates: Approximate bounding box coordinates as [x1, y1, x2, y2] (top-left to bottom-right)
4. selector_hints: Suggested CSS selectors or text content that could be used to locate the element
5. importance: Relevance score from 1-10 for the given task
6. action_type: Suggested action (click, fill, select, etc.)

Focus on elements like:
- Login/signup buttons and forms
- Job search inputs and filters
- Application buttons and forms
- Navigation menus and links
- Profile and settings options

Return your response as a JSON array of element objects."""
            
            user_prompt = f"""Analyze this screenshot for interactive elements relevant to: {task_description}

Please identify all clickable and fillable elements that would be useful for job application automation, including buttons, input fields, links, and form controls.

Focus on elements that are clearly visible and interactive."""
            
            # Make API call to GPT-4o
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            elements = self._parse_elements_response(response_text)
            
            return elements
            
        except Exception as e:
            self.logger.error(
                "GPT-4o element identification failed",
                error=str(e)
            )
            return await self._mock_identify_elements(task_description)
    
    def _parse_elements_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse GPT-4o response to extract element information."""
        try:
            import json
            
            # Try to find JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                elements = json.loads(json_str)
                
                # Validate and normalize element data
                normalized_elements = []
                for element in elements:
                    if isinstance(element, dict):
                        normalized_element = {
                            "element_type": element.get("element_type", "unknown"),
                            "description": element.get("description", ""),
                            "coordinates": element.get("coordinates", [0, 0, 100, 100]),
                            "selector_hints": element.get("selector_hints", []),
                            "importance": min(max(element.get("importance", 5), 1), 10),
                            "action_type": element.get("action_type", "click"),
                            "confidence": 0.8  # Default confidence for GPT-4o results
                        }
                        normalized_elements.append(normalized_element)
                
                return normalized_elements
            
            return []
            
        except Exception as e:
            self.logger.error(
                "Failed to parse elements response",
                error=str(e),
                response_preview=response_text[:200]
            )
            return []
    
    async def _mock_identify_elements(self, task_description: str) -> List[Dict[str, Any]]:
        """Mock implementation for development/testing."""
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Return mock elements based on task description
        mock_elements = []
        
        if "job" in task_description.lower() or "application" in task_description.lower():
            mock_elements = [
                {
                    "element_type": "input",
                    "description": "Job search input field",
                    "coordinates": [100, 150, 400, 180],
                    "selector_hints": ["input[placeholder*='job']", "input[name*='search']"],
                    "importance": 9,
                    "action_type": "fill",
                    "confidence": 0.9
                },
                {
                    "element_type": "button",
                    "description": "Search jobs button",
                    "coordinates": [420, 150, 500, 180],
                    "selector_hints": ["button[type='submit']", "button:contains('Search')"],
                    "importance": 8,
                    "action_type": "click",
                    "confidence": 0.9
                },
                {
                    "element_type": "link",
                    "description": "Job posting link",
                    "coordinates": [100, 250, 600, 300],
                    "selector_hints": ["a[href*='job']", ".job-title a"],
                    "importance": 10,
                    "action_type": "click",
                    "confidence": 0.8
                }
            ]
        
        self.logger.debug(
            "Mock element identification",
            elements_count=len(mock_elements),
            task=task_description
        )
        
        return mock_elements
    
    async def generate_highlighting_script(
        self,
        elements: List[Dict[str, Any]],
        highlight_color: str = "#ff0000",
        highlight_opacity: float = 0.3
    ) -> str:
        """
        Generate JavaScript code to highlight identified elements on the page.
        
        Args:
            elements: List of identified elements
            highlight_color: Color for highlighting (hex format)
            highlight_opacity: Opacity for highlight overlay
            
        Returns:
            JavaScript code to inject for highlighting
        """
        if not elements:
            return ""
        
        script_parts = [
            "// Element highlighting script generated by ElementIdentifier",
            "(() => {",
            "  const highlights = [];",
            "",
            "  function createHighlight(coords, description, importance) {",
            "    const [x1, y1, x2, y2] = coords;",
            "    const highlight = document.createElement('div');",
            "    highlight.style.position = 'absolute';",
            "    highlight.style.left = x1 + 'px';",
            "    highlight.style.top = y1 + 'px';",
            "    highlight.style.width = (x2 - x1) + 'px';",
            "    highlight.style.height = (y2 - y1) + 'px';",
            f"    highlight.style.backgroundColor = '{highlight_color}';",
            f"    highlight.style.opacity = '{highlight_opacity}';",
            "    highlight.style.border = '2px solid ' + highlight.style.backgroundColor;",
            "    highlight.style.pointerEvents = 'none';",
            "    highlight.style.zIndex = '9999';",
            "    highlight.title = description + ' (Importance: ' + importance + ')';",
            "    ",
            "    document.body.appendChild(highlight);",
            "    highlights.push(highlight);",
            "    return highlight;",
            "  }",
            "",
            "  function removeHighlights() {",
            "    highlights.forEach(h => h.remove());",
            "    highlights.length = 0;",
            "  }",
            "",
            "  // Create highlights for identified elements"
        ]
        
        # Add highlight creation for each element
        for i, element in enumerate(elements):
            coords = element.get("coordinates", [0, 0, 100, 100])
            description = element.get("description", "Interactive element").replace("'", "\\'").replace('"', '\\"')
            importance = element.get("importance", 5)
            
            script_parts.append(
                f"  createHighlight({coords}, '{description}', {importance});"
            )
        
        script_parts.extend([
            "",
            "  // Store reference for cleanup",
            "  window.elementHighlights = { remove: removeHighlights };",
            "  ",
            "  console.log('Highlighted', highlights.length, 'interactive elements');",
            "})();"
        ])
        
        script = "\n".join(script_parts)
        
        self.logger.debug(
            "Generated highlighting script",
            elements_count=len(elements),
            script_length=len(script)
        )
        
        return script
    
    async def analyze_element_context(
        self,
        screenshot_data: bytes,
        element_coordinates: List[int],
        context_description: str = "Analyze this element for automation"
    ) -> Dict[str, Any]:
        """
        Analyze a specific element in context for better automation decisions.
        
        Args:
            screenshot_data: Screenshot image data
            element_coordinates: Element bounding box [x1, y1, x2, y2]
            context_description: Description of what we want to know about the element
            
        Returns:
            Detailed analysis of the element
        """
        self.logger.debug(
            "Analyzing element context",
            coordinates=element_coordinates,
            context=context_description
        )
        
        try:
            if self.openai_client:
                analysis = await self._analyze_with_gpt4o(
                    screenshot_data, element_coordinates, context_description
                )
            else:
                analysis = await self._mock_analyze_element(element_coordinates, context_description)
            
            self.logger.debug(
                "Element context analysis completed",
                coordinates=element_coordinates
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(
                "Element context analysis failed",
                error=str(e),
                coordinates=element_coordinates
            )
            return {
                "element_type": "unknown",
                "recommended_action": "click",
                "confidence": 0.1,
                "analysis": "Analysis failed"
            }
    
    async def _analyze_with_gpt4o(
        self,
        screenshot_data: bytes,
        element_coordinates: List[int],
        context_description: str
    ) -> Dict[str, Any]:
        """Analyze element using GPT-4o vision API."""
        try:
            # Encode screenshot as base64
            screenshot_b64 = base64.b64encode(screenshot_data).decode('utf-8')
            
            x1, y1, x2, y2 = element_coordinates
            
            system_prompt = """You are an expert at analyzing web page elements for automation purposes.

Analyze the specific element highlighted in the given coordinates and provide detailed information about:
1. What type of element it is (button, input, link, etc.)
2. What action should be taken (click, fill, select, etc.)
3. What data might be needed (for form fields)
4. Any special considerations for automation
5. Confidence level in your analysis (0-1)

Return your response as a JSON object with these fields:
- element_type: Type of the element
- recommended_action: Best action to take
- data_requirements: What data is needed (for inputs)
- automation_notes: Special considerations
- confidence: Confidence score 0-1
- analysis: Detailed text analysis"""
            
            user_prompt = f"""Analyze the element at coordinates [{x1}, {y1}, {x2}, {y2}] in this screenshot.

Context: {context_description}

Focus on providing actionable information for browser automation."""
            
            # Make API call to GPT-4o
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            analysis = self._parse_analysis_response(response_text)
            
            return analysis
            
        except Exception as e:
            self.logger.error(
                "GPT-4o element analysis failed",
                error=str(e)
            )
            return await self._mock_analyze_element(element_coordinates, context_description)
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT-4o analysis response."""
        try:
            import json
            
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Ensure required fields
                return {
                    "element_type": analysis.get("element_type", "unknown"),
                    "recommended_action": analysis.get("recommended_action", "click"),
                    "data_requirements": analysis.get("data_requirements", []),
                    "automation_notes": analysis.get("automation_notes", ""),
                    "confidence": min(max(analysis.get("confidence", 0.5), 0), 1),
                    "analysis": analysis.get("analysis", response_text)
                }
            
            # Fallback if no JSON found
            return {
                "element_type": "unknown",
                "recommended_action": "click",
                "data_requirements": [],
                "automation_notes": "",
                "confidence": 0.3,
                "analysis": response_text
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to parse analysis response",
                error=str(e)
            )
            return {
                "element_type": "unknown",
                "recommended_action": "click",
                "data_requirements": [],
                "automation_notes": "Parse error",
                "confidence": 0.1,
                "analysis": response_text
            }
    
    async def _mock_analyze_element(
        self,
        element_coordinates: List[int],
        context_description: str
    ) -> Dict[str, Any]:
        """Mock implementation for element analysis."""
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Generate mock analysis based on context
        if "input" in context_description.lower() or "form" in context_description.lower():
            return {
                "element_type": "input",
                "recommended_action": "fill",
                "data_requirements": ["text_value"],
                "automation_notes": "Standard text input field",
                "confidence": 0.8,
                "analysis": "Mock analysis: This appears to be a text input field"
            }
        elif "button" in context_description.lower() or "submit" in context_description.lower():
            return {
                "element_type": "button",
                "recommended_action": "click",
                "data_requirements": [],
                "automation_notes": "Clickable button element",
                "confidence": 0.9,
                "analysis": "Mock analysis: This appears to be a clickable button"
            }
        else:
            return {
                "element_type": "link",
                "recommended_action": "click",
                "data_requirements": [],
                "automation_notes": "Clickable link element",
                "confidence": 0.7,
                "analysis": "Mock analysis: This appears to be a clickable link"
            }


def create_element_identifier(openai_client=None) -> ElementIdentifier:
    """
    Factory function to create an element identifier.
    
    Args:
        openai_client: Optional OpenAI client for dependency injection
        
    Returns:
        Configured ElementIdentifier instance
    """
    return ElementIdentifier(openai_client=openai_client)