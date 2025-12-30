"""Cultural preference adaptation system."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from sovereign_career_architect.voice.sarvam import SarvamClient
from sovereign_career_architect.core.models import UserProfile
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class CommunicationStyle(Enum):
    """Communication style preferences."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    FORMAL = "formal"
    CASUAL = "casual"
    HIERARCHICAL = "hierarchical"
    EGALITARIAN = "egalitarian"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"


class CulturalDimension(Enum):
    """Cultural dimensions based on Hofstede's model."""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    MASCULINITY = "masculinity"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"


@dataclass
class CulturalProfile:
    """User's cultural profile and preferences."""
    primary_culture: str
    secondary_cultures: List[str]
    communication_styles: List[CommunicationStyle]
    cultural_dimensions: Dict[CulturalDimension, float]  # 0-1 scale
    language_preferences: Dict[str, float]  # Language code -> preference score
    context_preferences: Dict[str, Any]
    adaptation_level: float = 0.7  # How much to adapt (0-1)


@dataclass
class CulturalContext:
    """Context for cultural adaptation."""
    situation_type: str  # "interview", "networking", "application", "casual"
    formality_level: str  # "very_formal", "formal", "semi_formal", "casual"
    audience: str  # "recruiter", "hiring_manager", "peer", "senior_executive"
    cultural_background: Optional[str] = None
    relationship_level: str = "new"  # "new", "acquaintance", "familiar", "close"


class CulturalAdapter:
    """Adapts communication based on cultural preferences and context."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.logger = logger.bind(component="cultural_adapter")
        self.sarvam_client = sarvam_client or SarvamClient()
        
        # Cultural knowledge base
        self.cultural_patterns = self._load_cultural_patterns()
        self.communication_templates = self._load_communication_templates()
        self.cultural_norms = self._load_cultural_norms()
        
    def _load_cultural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural communication patterns."""
        return {
            "indian": {
                "communication_style": [CommunicationStyle.HIERARCHICAL, CommunicationStyle.HIGH_CONTEXT],
                "greeting_patterns": {
                    "formal": ["Namaste", "Good morning/afternoon", "Respected Sir/Madam"],
                    "casual": ["Hello", "Hi", "Namaste"]
                },
                "politeness_markers": ["please", "kindly", "if you don't mind", "with due respect"],
                "relationship_building": True,
                "indirect_communication": True,
                "respect_for_authority": True,
                "family_references": True,
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.8,
                    CulturalDimension.INDIVIDUALISM: 0.3,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.7
                }
            },
            "american": {
                "communication_style": [CommunicationStyle.DIRECT, CommunicationStyle.LOW_CONTEXT],
                "greeting_patterns": {
                    "formal": ["Good morning", "Hello", "Nice to meet you"],
                    "casual": ["Hi", "Hey", "How's it going"]
                },
                "politeness_markers": ["please", "thank you", "I appreciate"],
                "relationship_building": False,
                "indirect_communication": False,
                "respect_for_authority": False,
                "family_references": False,
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.3,
                    CulturalDimension.INDIVIDUALISM: 0.9,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.4,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.3
                }
            },
            "british": {
                "communication_style": [CommunicationStyle.INDIRECT, CommunicationStyle.FORMAL],
                "greeting_patterns": {
                    "formal": ["Good morning", "Good afternoon", "How do you do"],
                    "casual": ["Hello", "Hi there", "Lovely to see you"]
                },
                "politeness_markers": ["please", "thank you", "if you would", "I wonder if"],
                "relationship_building": True,
                "indirect_communication": True,
                "respect_for_authority": True,
                "family_references": False,
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.4,
                    CulturalDimension.INDIVIDUALISM: 0.8,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.3,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.5
                }
            },
            "german": {
                "communication_style": [CommunicationStyle.DIRECT, CommunicationStyle.FORMAL],
                "greeting_patterns": {
                    "formal": ["Guten Tag", "Good morning", "Hello"],
                    "casual": ["Hallo", "Hi", "Guten Tag"]
                },
                "politeness_markers": ["bitte", "please", "danke", "thank you"],
                "relationship_building": False,
                "indirect_communication": False,
                "respect_for_authority": True,
                "family_references": False,
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.4,
                    CulturalDimension.INDIVIDUALISM: 0.7,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.7,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.8
                }
            }
        }
    
    def _load_communication_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load communication templates for different contexts."""
        return {
            "interview_responses": {
                "indian_formal": [
                    "With due respect, I would like to share that {content}",
                    "If I may humbly add, {content}",
                    "Based on my experience, I believe {content}",
                    "I am grateful for this opportunity to discuss {content}"
                ],
                "american_direct": [
                    "I believe {content}",
                    "In my experience, {content}",
                    "I'm confident that {content}",
                    "My approach to this is {content}"
                ],
                "british_polite": [
                    "I would suggest that {content}",
                    "If I may, I think {content}",
                    "I rather think {content}",
                    "It seems to me that {content}"
                ]
            },
            "networking": {
                "indian_relationship": [
                    "It's wonderful to connect with you. {content}",
                    "I hope you and your family are doing well. {content}",
                    "Thank you for taking the time to speak with me. {content}"
                ],
                "american_professional": [
                    "Great to meet you. {content}",
                    "I'm excited to connect. {content}",
                    "Thanks for your time. {content}"
                ],
                "british_courteous": [
                    "Lovely to make your acquaintance. {content}",
                    "I do hope you're well. {content}",
                    "Thank you ever so much for your time. {content}"
                ]
            },
            "follow_up": {
                "indian_respectful": [
                    "I hope this message finds you in good health. {content}",
                    "I wanted to respectfully follow up on {content}",
                    "With your kind permission, I would like to {content}"
                ],
                "american_efficient": [
                    "Following up on {content}",
                    "Quick update on {content}",
                    "Checking in about {content}"
                ],
                "british_diplomatic": [
                    "I hope you don't mind me following up on {content}",
                    "I wondered if you might have had a chance to {content}",
                    "When convenient, could we discuss {content}"
                ]
            }
        }
    
    def _load_cultural_norms(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural norms and expectations."""
        return {
            "indian": {
                "time_orientation": "flexible",
                "relationship_first": True,
                "hierarchy_respect": True,
                "indirect_feedback": True,
                "group_harmony": True,
                "elder_respect": True,
                "gift_giving": True,
                "personal_space": "close",
                "eye_contact": "respectful_avoidance_with_seniors",
                "silence_comfort": True
            },
            "american": {
                "time_orientation": "punctual",
                "relationship_first": False,
                "hierarchy_respect": False,
                "indirect_feedback": False,
                "group_harmony": False,
                "elder_respect": False,
                "gift_giving": False,
                "personal_space": "arm_length",
                "eye_contact": "direct",
                "silence_comfort": False
            },
            "british": {
                "time_orientation": "punctual",
                "relationship_first": True,
                "hierarchy_respect": True,
                "indirect_feedback": True,
                "group_harmony": True,
                "elder_respect": True,
                "gift_giving": False,
                "personal_space": "arm_length",
                "eye_contact": "polite_direct",
                "silence_comfort": True
            }
        }
    
    async def create_cultural_profile(
        self,
        user_profile: UserProfile,
        cultural_background: str,
        communication_preferences: Optional[Dict[str, Any]] = None
    ) -> CulturalProfile:
        """
        Create a cultural profile for a user.
        
        Args:
            user_profile: User's profile
            cultural_background: Primary cultural background
            communication_preferences: Additional preferences
            
        Returns:
            Cultural profile
        """
        self.logger.info(
            "Creating cultural profile",
            cultural_background=cultural_background,
            user_language=user_profile.personal_info.preferred_language
        )
        
        # Get cultural patterns
        cultural_data = self.cultural_patterns.get(cultural_background, self.cultural_patterns["american"])
        
        # Determine communication styles
        communication_styles = cultural_data["communication_style"]
        
        # Set cultural dimensions
        cultural_dimensions = cultural_data["cultural_dimensions"]
        
        # Language preferences
        language_preferences = {
            user_profile.personal_info.preferred_language: 1.0,
            "en": 0.8  # English as common professional language
        }
        
        # Context preferences from user input
        context_preferences = communication_preferences or {}
        
        cultural_profile = CulturalProfile(
            primary_culture=cultural_background,
            secondary_cultures=[],
            communication_styles=communication_styles,
            cultural_dimensions=cultural_dimensions,
            language_preferences=language_preferences,
            context_preferences=context_preferences
        )
        
        self.logger.info(
            "Cultural profile created",
            primary_culture=cultural_background,
            communication_styles=[style.value for style in communication_styles]
        )
        
        return cultural_profile
    
    async def adapt_message(
        self,
        message: str,
        cultural_profile: CulturalProfile,
        context: CulturalContext,
        target_culture: Optional[str] = None
    ) -> str:
        """
        Adapt a message based on cultural preferences and context.
        
        Args:
            message: Original message
            cultural_profile: User's cultural profile
            context: Communication context
            target_culture: Target audience culture (optional)
            
        Returns:
            Culturally adapted message
        """
        self.logger.info(
            "Adapting message",
            primary_culture=cultural_profile.primary_culture,
            situation_type=context.situation_type,
            formality_level=context.formality_level,
            target_culture=target_culture
        )
        
        # Determine adaptation strategy
        source_culture = cultural_profile.primary_culture
        target_culture = target_culture or "american"  # Default to American business culture
        
        # Get cultural patterns
        source_patterns = self.cultural_patterns.get(source_culture, {})
        target_patterns = self.cultural_patterns.get(target_culture, {})
        
        # Apply cultural adaptations
        adapted_message = message
        
        # 1. Adjust formality level
        adapted_message = await self._adjust_formality(
            adapted_message, context, source_patterns, target_patterns
        )
        
        # 2. Apply politeness markers
        adapted_message = await self._apply_politeness_markers(
            adapted_message, source_patterns, target_patterns
        )
        
        # 3. Adjust directness
        adapted_message = await self._adjust_directness(
            adapted_message, context, source_patterns, target_patterns
        )
        
        # 4. Add cultural context if needed
        adapted_message = await self._add_cultural_context(
            adapted_message, context, cultural_profile, target_patterns
        )
        
        # 5. Apply communication templates
        adapted_message = await self._apply_communication_templates(
            adapted_message, context, source_culture, target_culture
        )
        
        self.logger.info(
            "Message adaptation completed",
            original_length=len(message),
            adapted_length=len(adapted_message)
        )
        
        return adapted_message
    
    async def _adjust_formality(
        self,
        message: str,
        context: CulturalContext,
        source_patterns: Dict[str, Any],
        target_patterns: Dict[str, Any]
    ) -> str:
        """Adjust formality level of the message."""
        
        if context.formality_level == "very_formal":
            # Add formal markers
            if not message.startswith(("Dear", "Respected", "Good")):
                if context.audience == "senior_executive":
                    message = f"Respected Sir/Madam, {message}"
                else:
                    message = f"Dear Hiring Manager, {message}"
            
            # Add formal closing if not present
            if not any(closing in message.lower() for closing in ["sincerely", "regards", "respectfully"]):
                message += "\n\nWith sincere regards,"
        
        elif context.formality_level == "casual":
            # Remove overly formal language
            message = message.replace("Respected Sir/Madam", "Hi")
            message = message.replace("With sincere regards", "Best regards")
            message = message.replace("I humbly", "I")
        
        return message
    
    async def _apply_politeness_markers(
        self,
        message: str,
        source_patterns: Dict[str, Any],
        target_patterns: Dict[str, Any]
    ) -> str:
        """Apply appropriate politeness markers."""
        
        source_markers = source_patterns.get("politeness_markers", [])
        target_markers = target_patterns.get("politeness_markers", [])
        
        # If target culture uses different politeness markers, adapt
        if "kindly" in message and "kindly" not in target_markers:
            if "please" in target_markers:
                message = message.replace("kindly", "please")
        
        # Add politeness markers if target culture expects them
        if target_patterns.get("indirect_communication", False):
            if "I want" in message:
                message = message.replace("I want", "I would like")
            if "You should" in message:
                message = message.replace("You should", "You might consider")
        
        return message
    
    async def _adjust_directness(
        self,
        message: str,
        context: CulturalContext,
        source_patterns: Dict[str, Any],
        target_patterns: Dict[str, Any]
    ) -> str:
        """Adjust directness based on cultural preferences."""
        
        source_indirect = source_patterns.get("indirect_communication", False)
        target_indirect = target_patterns.get("indirect_communication", False)
        
        if source_indirect and not target_indirect:
            # Make more direct
            message = message.replace("I believe that perhaps", "I believe")
            message = message.replace("if you don't mind", "")
            message = message.replace("I was wondering if", "Can")
        
        elif not source_indirect and target_indirect:
            # Make more indirect
            message = message.replace("I need", "I would appreciate if you could")
            message = message.replace("You must", "It would be helpful if you could")
            message = message.replace("This is wrong", "This might need some adjustment")
        
        return message
    
    async def _add_cultural_context(
        self,
        message: str,
        context: CulturalContext,
        cultural_profile: CulturalProfile,
        target_patterns: Dict[str, Any]
    ) -> str:
        """Add cultural context if appropriate."""
        
        # Add relationship building elements if target culture values them
        if target_patterns.get("relationship_building", False):
            if context.situation_type == "networking" and context.relationship_level == "new":
                if not any(phrase in message.lower() for phrase in ["hope you", "pleasure to", "wonderful to"]):
                    message = f"I hope you're doing well. {message}"
        
        # Add family references if culturally appropriate
        if (target_patterns.get("family_references", False) and 
            context.situation_type in ["networking", "follow_up"]):
            if "family" not in message.lower():
                message = message.replace("I hope you're doing well.", 
                                        "I hope you and your family are doing well.")
        
        return message
    
    async def _apply_communication_templates(
        self,
        message: str,
        context: CulturalContext,
        source_culture: str,
        target_culture: str
    ) -> str:
        """Apply communication templates based on context."""
        
        template_key = f"{target_culture}_{context.formality_level.split('_')[0]}"
        
        if context.situation_type in self.communication_templates:
            templates = self.communication_templates[context.situation_type]
            
            if template_key in templates:
                # Use template if message is short enough
                if len(message) < 200:
                    template = templates[template_key][0]  # Use first template
                    return template.format(content=message)
        
        return message
    
    async def get_cultural_recommendations(
        self,
        cultural_profile: CulturalProfile,
        context: CulturalContext,
        target_culture: Optional[str] = None
    ) -> List[str]:
        """
        Get cultural recommendations for a specific context.
        
        Args:
            cultural_profile: User's cultural profile
            context: Communication context
            target_culture: Target audience culture
            
        Returns:
            List of cultural recommendations
        """
        recommendations = []
        
        target_culture = target_culture or "american"
        target_patterns = self.cultural_patterns.get(target_culture, {})
        target_norms = self.cultural_norms.get(target_culture, {})
        
        # Time-related recommendations
        if target_norms.get("time_orientation") == "punctual":
            recommendations.append("Be punctual and respect scheduled times")
        
        # Communication style recommendations
        if CommunicationStyle.DIRECT in target_patterns.get("communication_style", []):
            recommendations.append("Use direct and clear communication")
        elif CommunicationStyle.INDIRECT in target_patterns.get("communication_style", []):
            recommendations.append("Use polite and indirect communication")
        
        # Hierarchy recommendations
        if target_norms.get("hierarchy_respect", False):
            recommendations.append("Show appropriate respect for seniority and titles")
        
        # Relationship building recommendations
        if target_norms.get("relationship_first", False):
            recommendations.append("Invest time in building relationships before business discussions")
        
        # Context-specific recommendations
        if context.situation_type == "interview":
            if target_culture == "indian":
                recommendations.extend([
                    "Address interviewers with appropriate titles (Sir/Madam)",
                    "Show respect for the company and its values",
                    "Mention family support for career decisions if relevant"
                ])
            elif target_culture == "american":
                recommendations.extend([
                    "Emphasize individual achievements and contributions",
                    "Be confident and assertive about your abilities",
                    "Focus on results and measurable outcomes"
                ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def analyze_cultural_fit(
        self,
        user_message: str,
        cultural_profile: CulturalProfile,
        target_culture: str
    ) -> Dict[str, Any]:
        """
        Analyze how well a message fits the target culture.
        
        Args:
            user_message: Message to analyze
            cultural_profile: User's cultural profile
            target_culture: Target culture
            
        Returns:
            Cultural fit analysis
        """
        target_patterns = self.cultural_patterns.get(target_culture, {})
        target_norms = self.cultural_norms.get(target_culture, {})
        
        analysis = {
            "overall_fit": 0.0,
            "strengths": [],
            "areas_for_improvement": [],
            "specific_suggestions": []
        }
        
        fit_score = 0.0
        total_checks = 0
        
        # Check directness
        total_checks += 1
        if target_patterns.get("indirect_communication", False):
            if any(phrase in user_message.lower() for phrase in ["i believe", "perhaps", "might"]):
                fit_score += 1
                analysis["strengths"].append("Appropriate level of indirectness")
            else:
                analysis["areas_for_improvement"].append("Consider using more indirect language")
        else:
            if any(phrase in user_message.lower() for phrase in ["i will", "i can", "i have"]):
                fit_score += 1
                analysis["strengths"].append("Clear and direct communication")
            else:
                analysis["areas_for_improvement"].append("Be more direct and assertive")
        
        # Check politeness markers
        total_checks += 1
        politeness_markers = target_patterns.get("politeness_markers", [])
        if any(marker in user_message.lower() for marker in politeness_markers):
            fit_score += 1
            analysis["strengths"].append("Good use of politeness markers")
        else:
            analysis["areas_for_improvement"].append("Add appropriate politeness markers")
        
        # Check formality
        total_checks += 1
        formal_indicators = ["dear", "sincerely", "respectfully", "regards"]
        if any(indicator in user_message.lower() for indicator in formal_indicators):
            fit_score += 1
            analysis["strengths"].append("Appropriate formality level")
        
        # Calculate overall fit
        analysis["overall_fit"] = fit_score / total_checks if total_checks > 0 else 0.0
        
        # Generate specific suggestions
        if analysis["overall_fit"] < 0.7:
            analysis["specific_suggestions"] = await self.get_cultural_recommendations(
                cultural_profile, 
                CulturalContext(
                    situation_type="general",
                    formality_level="formal",
                    audience="professional"
                ),
                target_culture
            )
        
        return analysis