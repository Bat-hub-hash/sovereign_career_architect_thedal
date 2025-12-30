"""Code-switching support for seamless English-Indic language mixing."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import structlog

from sovereign_career_architect.voice.sarvam import SarvamClient
from sovereign_career_architect.cultural.adapter import CulturalProfile
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


class SwitchingTrigger(Enum):
    """Triggers for code-switching."""
    EMOTIONAL_EMPHASIS = "emotional_emphasis"
    TECHNICAL_TERM = "technical_term"
    CULTURAL_CONCEPT = "cultural_concept"
    RELATIONSHIP_BUILDING = "relationship_building"
    COMFORT_EXPRESSION = "comfort_expression"
    AUTHORITY_RESPECT = "authority_respect"
    HUMOR = "humor"
    CLARIFICATION = "clarification"


class LanguageDominance(Enum):
    """Language dominance in mixed speech."""
    L1_DOMINANT = "l1_dominant"  # Native language dominant
    L2_DOMINANT = "l2_dominant"  # Second language dominant
    BALANCED = "balanced"        # Balanced mixing


@dataclass
class LanguageMix:
    """Represents a language mixing pattern."""
    primary_language: str
    secondary_language: str
    mixing_ratio: float  # 0-1, where 1 is all primary language
    switching_points: List[int]  # Character positions where switching occurs
    triggers: List[SwitchingTrigger]
    dominance: LanguageDominance


@dataclass
class SwitchingContext:
    """Context for code-switching decisions."""
    conversation_type: str  # "professional", "casual", "family", "peer"
    emotional_state: str    # "neutral", "excited", "stressed", "comfortable"
    audience_familiarity: str  # "stranger", "acquaintance", "friend", "family"
    topic_domain: str      # "technical", "personal", "cultural", "business"
    formality_level: str   # "very_formal", "formal", "informal", "very_informal"


class CodeSwitchingManager:
    """Manages seamless English-Indic language code-switching."""
    
    def __init__(self, sarvam_client: Optional[SarvamClient] = None):
        self.logger = logger.bind(component="code_switching")
        self.sarvam_client = sarvam_client or SarvamClient()
        
        # Code-switching patterns and rules
        self.switching_patterns = self._load_switching_patterns()
        self.trigger_words = self._load_trigger_words()
        self.preservation_rules = self._load_preservation_rules()
        
    def _load_switching_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load code-switching patterns for different language pairs."""
        return {
            "en_hi": {
                "common_switches": {
                    # Emotional expressions
                    "really": "sach mein",
                    "actually": "asal mein", 
                    "you know": "aap jaante hain",
                    "I mean": "matlab",
                    "basically": "basically", # Often kept in English
                    
                    # Discourse markers
                    "so": "toh",
                    "but": "lekin",
                    "and": "aur",
                    "or": "ya",
                    "because": "kyunki",
                    
                    # Relationship terms
                    "sir": "sahib",
                    "madam": "madam ji",
                    "brother": "bhai",
                    "sister": "didi",
                    
                    # Emphasis
                    "very": "bahut",
                    "too much": "bahut zyada",
                    "a lot": "bahut saara"
                },
                "technical_preservation": [
                    "software", "hardware", "database", "algorithm", "API",
                    "framework", "application", "system", "network", "cloud",
                    "machine learning", "artificial intelligence", "data science"
                ],
                "cultural_switches": {
                    "respect": "izzat",
                    "family": "parivaar",
                    "tradition": "parampara",
                    "culture": "sanskriti",
                    "values": "moolya"
                }
            },
            "en_ta": {
                "common_switches": {
                    # Emotional expressions
                    "really": "unmaiyaaga",
                    "actually": "nijamaa",
                    "you know": "ungalukku theriyum",
                    "I mean": "naan solrathu",
                    
                    # Discourse markers
                    "so": "appo",
                    "but": "aana",
                    "and": "mattum",
                    "because": "yen-na",
                    
                    # Relationship terms
                    "sir": "sir",  # Often kept in English
                    "brother": "anna",
                    "sister": "akka",
                    
                    # Emphasis
                    "very": "romba",
                    "too much": "adhigam"
                },
                "technical_preservation": [
                    "software", "computer", "internet", "mobile", "technology",
                    "engineering", "project", "management", "business"
                ],
                "cultural_switches": {
                    "respect": "mariyaadhai",
                    "family": "kudumbam",
                    "tradition": "paramparai"
                }
            },
            "en_te": {
                "common_switches": {
                    # Emotional expressions
                    "really": "nijamgaa",
                    "actually": "asalu",
                    "you know": "meeku telusu",
                    "I mean": "nenu antunnaanu",
                    
                    # Discourse markers
                    "so": "anduke",
                    "but": "kaani",
                    "and": "mariyu",
                    "because": "endukante",
                    
                    # Relationship terms
                    "sir": "garu",
                    "brother": "anna",
                    "sister": "akka",
                    
                    # Emphasis
                    "very": "chaala",
                    "too much": "ekkuva"
                },
                "technical_preservation": [
                    "computer", "software", "technology", "engineering",
                    "project", "business", "management"
                ],
                "cultural_switches": {
                    "respect": "gauravam",
                    "family": "kutumbam",
                    "tradition": "sampradayam"
                }
            }
        }
    
    def _load_trigger_words(self) -> Dict[SwitchingTrigger, List[str]]:
        """Load words that trigger code-switching."""
        return {
            SwitchingTrigger.EMOTIONAL_EMPHASIS: [
                "amazing", "wonderful", "terrible", "fantastic", "awful",
                "incredible", "unbelievable", "shocking", "surprising"
            ],
            SwitchingTrigger.TECHNICAL_TERM: [
                "algorithm", "database", "framework", "API", "software",
                "hardware", "network", "cloud", "server", "application"
            ],
            SwitchingTrigger.CULTURAL_CONCEPT: [
                "family", "tradition", "culture", "values", "respect",
                "honor", "community", "festival", "ceremony", "ritual"
            ],
            SwitchingTrigger.RELATIONSHIP_BUILDING: [
                "brother", "sister", "uncle", "aunt", "friend", "colleague",
                "sir", "madam", "ji", "sahib"
            ],
            SwitchingTrigger.COMFORT_EXPRESSION: [
                "home", "comfortable", "familiar", "easy", "natural",
                "relaxed", "peaceful", "happy", "content"
            ],
            SwitchingTrigger.AUTHORITY_RESPECT: [
                "sir", "madam", "boss", "manager", "senior", "elder",
                "teacher", "professor", "doctor", "officer"
            ]
        }
    
    def _load_preservation_rules(self) -> Dict[str, List[str]]:
        """Load rules for what to preserve in each language."""
        return {
            "always_english": [
                # Technical terms
                "software", "hardware", "computer", "internet", "email",
                "website", "database", "algorithm", "API", "framework",
                "application", "system", "network", "server", "cloud",
                "machine learning", "artificial intelligence", "data science",
                
                # Business terms
                "meeting", "presentation", "project", "deadline", "budget",
                "client", "customer", "business", "company", "organization",
                "management", "strategy", "planning", "marketing", "sales",
                
                # Modern concepts
                "smartphone", "laptop", "tablet", "social media", "app",
                "download", "upload", "online", "offline", "digital"
            ],
            "prefer_native": [
                # Family terms
                "mother", "father", "brother", "sister", "son", "daughter",
                "grandmother", "grandfather", "uncle", "aunt", "cousin",
                
                # Emotional expressions
                "love", "happiness", "sadness", "anger", "fear", "joy",
                "pride", "shame", "guilt", "excitement", "worry",
                
                # Cultural concepts
                "tradition", "culture", "values", "respect", "honor",
                "community", "festival", "ceremony", "prayer", "blessing",
                
                # Food and daily life
                "food", "meal", "breakfast", "lunch", "dinner", "tea",
                "home", "house", "room", "kitchen", "garden"
            ]
        }
    
    async def generate_code_switched_text(
        self,
        english_text: str,
        target_language: str,
        cultural_profile: CulturalProfile,
        context: SwitchingContext
    ) -> str:
        """
        Generate code-switched text mixing English and target language.
        
        Args:
            english_text: Original English text
            target_language: Target language for mixing
            cultural_profile: User's cultural profile
            context: Switching context
            
        Returns:
            Code-switched text
        """
        if target_language == "en":
            return english_text
        
        self.logger.info(
            "Generating code-switched text",
            target_language=target_language,
            context_type=context.conversation_type,
            text_length=len(english_text)
        )
        
        # Determine switching strategy
        switching_strategy = self._determine_switching_strategy(
            cultural_profile, context, target_language
        )
        
        # Identify switching points and triggers
        switching_points = await self._identify_switching_points(
            english_text, target_language, context
        )
        
        # Apply code-switching
        code_switched_text = await self._apply_code_switching(
            english_text, target_language, switching_points, switching_strategy
        )
        
        # Post-process for naturalness
        final_text = await self._post_process_switching(
            code_switched_text, target_language, context
        )
        
        self.logger.info(
            "Code-switched text generated",
            original_length=len(english_text),
            final_length=len(final_text),
            switching_points=len(switching_points)
        )
        
        return final_text
    
    def _determine_switching_strategy(
        self,
        cultural_profile: CulturalProfile,
        context: SwitchingContext,
        target_language: str
    ) -> Dict[str, Any]:
        """Determine the code-switching strategy based on context."""
        
        strategy = {
            "mixing_ratio": 0.7,  # Default: 70% English, 30% target language
            "dominance": LanguageDominance.L2_DOMINANT,  # English dominant
            "switching_frequency": "moderate",
            "preserve_technical": True,
            "cultural_emphasis": False
        }
        
        # Adjust based on conversation type
        if context.conversation_type == "professional":
            strategy["mixing_ratio"] = 0.8  # More English
            strategy["preserve_technical"] = True
        elif context.conversation_type == "casual":
            strategy["mixing_ratio"] = 0.6  # More native language
            strategy["switching_frequency"] = "high"
        elif context.conversation_type == "family":
            strategy["mixing_ratio"] = 0.4  # Native language dominant
            strategy["dominance"] = LanguageDominance.L1_DOMINANT
            strategy["cultural_emphasis"] = True
        
        # Adjust based on formality
        if context.formality_level in ["very_formal", "formal"]:
            strategy["mixing_ratio"] = 0.9  # Mostly English
            strategy["switching_frequency"] = "low"
        elif context.formality_level == "very_informal":
            strategy["switching_frequency"] = "high"
            strategy["cultural_emphasis"] = True
        
        # Adjust based on emotional state
        if context.emotional_state in ["excited", "stressed"]:
            strategy["switching_frequency"] = "high"
            strategy["cultural_emphasis"] = True
        
        # Adjust based on audience familiarity
        if context.audience_familiarity in ["friend", "family"]:
            strategy["mixing_ratio"] = 0.5  # Balanced
            strategy["dominance"] = LanguageDominance.BALANCED
        
        return strategy
    
    async def _identify_switching_points(
        self,
        text: str,
        target_language: str,
        context: SwitchingContext
    ) -> List[Dict[str, Any]]:
        """Identify points in text where code-switching should occur."""
        
        switching_points = []
        words = text.split()
        
        # Get switching patterns for language pair
        lang_pair = f"en_{target_language}"
        patterns = self.switching_patterns.get(lang_pair, {})
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check for trigger words
            triggers = []
            for trigger_type, trigger_words in self.trigger_words.items():
                if word_lower in trigger_words:
                    triggers.append(trigger_type)
            
            # Check for common switches
            if word_lower in patterns.get("common_switches", {}):
                switching_points.append({
                    "position": i,
                    "word": word,
                    "replacement": patterns["common_switches"][word_lower],
                    "triggers": triggers,
                    "confidence": 0.8
                })
            
            # Check for cultural switches
            elif word_lower in patterns.get("cultural_switches", {}):
                if context.topic_domain == "cultural" or SwitchingTrigger.CULTURAL_CONCEPT in triggers:
                    switching_points.append({
                        "position": i,
                        "word": word,
                        "replacement": patterns["cultural_switches"][word_lower],
                        "triggers": triggers,
                        "confidence": 0.9
                    })
            
            # Check for emotional emphasis
            elif SwitchingTrigger.EMOTIONAL_EMPHASIS in triggers:
                if context.emotional_state != "neutral":
                    # Use Sarvam-1 to translate emotional words
                    try:
                        translation = await self.sarvam_client.translate_text(
                            word, "en", target_language
                        )
                        switching_points.append({
                            "position": i,
                            "word": word,
                            "replacement": translation,
                            "triggers": triggers,
                            "confidence": 0.7
                        })
                    except Exception:
                        pass  # Skip if translation fails
        
        return switching_points
    
    async def _apply_code_switching(
        self,
        text: str,
        target_language: str,
        switching_points: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> str:
        """Apply code-switching based on identified points and strategy."""
        
        words = text.split()
        result_words = words.copy()
        
        # Sort switching points by position (reverse order for safe replacement)
        switching_points.sort(key=lambda x: x["position"], reverse=True)
        
        # Apply switches based on strategy
        switches_applied = 0
        max_switches = int(len(words) * (1 - strategy["mixing_ratio"]))
        
        for switch_point in switching_points:
            if switches_applied >= max_switches:
                break
            
            position = switch_point["position"]
            replacement = switch_point["replacement"]
            confidence = switch_point["confidence"]
            
            # Apply switch based on confidence and strategy
            if confidence >= 0.7:
                # Check preservation rules
                word_lower = switch_point["word"].lower()
                
                if word_lower in self.preservation_rules["always_english"]:
                    continue  # Don't switch technical terms
                
                if (strategy["preserve_technical"] and 
                    SwitchingTrigger.TECHNICAL_TERM in switch_point["triggers"]):
                    continue  # Preserve technical terms in professional context
                
                # Apply the switch
                result_words[position] = replacement
                switches_applied += 1
        
        return " ".join(result_words)
    
    async def _post_process_switching(
        self,
        text: str,
        target_language: str,
        context: SwitchingContext
    ) -> str:
        """Post-process code-switched text for naturalness."""
        
        # Add language-specific discourse markers
        if target_language == "hi":
            # Add Hindi discourse markers
            text = self._add_hindi_markers(text, context)
        elif target_language == "ta":
            # Add Tamil discourse markers
            text = self._add_tamil_markers(text, context)
        elif target_language == "te":
            # Add Telugu discourse markers
            text = self._add_telugu_markers(text, context)
        
        # Ensure proper punctuation and spacing
        text = self._fix_punctuation(text)
        
        # Add cultural greetings/closings if appropriate
        if context.conversation_type in ["casual", "family"]:
            text = self._add_cultural_elements(text, target_language, context)
        
        return text
    
    def _add_hindi_markers(self, text: str, context: SwitchingContext) -> str:
        """Add Hindi discourse markers for natural flow."""
        
        # Add "ji" for respect in formal contexts
        if context.formality_level in ["formal", "very_formal"]:
            text = re.sub(r'\b(sir|madam)\b', r'\1 ji', text, flags=re.IGNORECASE)
        
        # Add "na" for casual confirmation
        if context.conversation_type == "casual":
            text = re.sub(r'\bright\?', 'hai na?', text)
            text = re.sub(r'\bokay\?', 'theek hai na?', text)
        
        # Add "yaar" for friendly contexts
        if context.audience_familiarity in ["friend", "family"]:
            text = re.sub(r'\bfriend\b', 'yaar', text)
        
        return text
    
    def _add_tamil_markers(self, text: str, context: SwitchingContext) -> str:
        """Add Tamil discourse markers for natural flow."""
        
        # Add "nga" for casual contexts
        if context.conversation_type == "casual":
            text = re.sub(r'\bright\?', 'sari nga?', text)
        
        # Add respectful markers
        if context.formality_level in ["formal", "very_formal"]:
            text = re.sub(r'\byes\b', 'aamam sir', text, flags=re.IGNORECASE)
        
        return text
    
    def _add_telugu_markers(self, text: str, context: SwitchingContext) -> str:
        """Add Telugu discourse markers for natural flow."""
        
        # Add "gaa" for emphasis
        if context.emotional_state == "excited":
            text = re.sub(r'\breally\b', 'nijamgaa', text, flags=re.IGNORECASE)
        
        # Add respectful "garu"
        if context.formality_level in ["formal", "very_formal"]:
            text = re.sub(r'\b(sir|madam)\b', r'\1 garu', text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation and spacing in mixed text."""
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _add_cultural_elements(
        self, 
        text: str, 
        target_language: str, 
        context: SwitchingContext
    ) -> str:
        """Add cultural elements like greetings or blessings."""
        
        if context.conversation_type == "family":
            if target_language == "hi":
                if not any(greeting in text.lower() for greeting in ["namaste", "pranam"]):
                    text = f"Namaste! {text}"
            elif target_language == "ta":
                if not any(greeting in text.lower() for greeting in ["vanakkam", "namaste"]):
                    text = f"Vanakkam! {text}"
            elif target_language == "te":
                if not any(greeting in text.lower() for greeting in ["namaskaram", "namaste"]):
                    text = f"Namaskaram! {text}"
        
        return text
    
    async def analyze_code_switching_naturalness(
        self,
        mixed_text: str,
        target_language: str,
        cultural_profile: CulturalProfile
    ) -> Dict[str, Any]:
        """
        Analyze how natural the code-switching sounds.
        
        Args:
            mixed_text: Code-switched text
            target_language: Target language
            cultural_profile: User's cultural profile
            
        Returns:
            Naturalness analysis
        """
        analysis = {
            "naturalness_score": 0.0,
            "switching_frequency": 0.0,
            "language_balance": {},
            "issues": [],
            "suggestions": []
        }
        
        words = mixed_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return analysis
        
        # Count language distribution
        english_words = 0
        native_words = 0
        
        lang_pair = f"en_{target_language}"
        patterns = self.switching_patterns.get(lang_pair, {})
        native_words_set = set(patterns.get("common_switches", {}).values())
        native_words_set.update(patterns.get("cultural_switches", {}).values())
        
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in native_words_set:
                native_words += 1
            else:
                english_words += 1
        
        # Calculate metrics
        analysis["language_balance"] = {
            "english_ratio": english_words / total_words,
            "native_ratio": native_words / total_words
        }
        
        analysis["switching_frequency"] = native_words / total_words
        
        # Assess naturalness
        naturalness_score = 0.0
        
        # Check for balanced mixing (not too much of either language)
        english_ratio = analysis["language_balance"]["english_ratio"]
        if 0.3 <= english_ratio <= 0.8:
            naturalness_score += 0.3
        else:
            analysis["issues"].append("Language balance seems unnatural")
        
        # Check for appropriate technical term preservation
        technical_terms = self.preservation_rules["always_english"]
        preserved_technical = sum(1 for word in words if word.lower() in technical_terms)
        if preserved_technical > 0:
            naturalness_score += 0.2
            analysis["suggestions"].append("Good preservation of technical terms")
        
        # Check for cultural appropriateness
        cultural_switches = patterns.get("cultural_switches", {})
        cultural_usage = sum(1 for word in words if word in cultural_switches.values())
        if cultural_usage > 0:
            naturalness_score += 0.2
        
        # Check for discourse markers
        common_switches = patterns.get("common_switches", {})
        discourse_markers = ["toh", "lekin", "aur", "ya", "kyunki"]  # Hindi examples
        marker_usage = sum(1 for word in words if word in discourse_markers)
        if marker_usage > 0:
            naturalness_score += 0.3
        
        analysis["naturalness_score"] = min(1.0, naturalness_score)
        
        # Generate suggestions
        if analysis["naturalness_score"] < 0.6:
            analysis["suggestions"].extend([
                "Consider more balanced language mixing",
                "Add more discourse markers for natural flow",
                "Use cultural terms where appropriate"
            ])
        
        return analysis
    
    def get_switching_recommendations(
        self,
        context: SwitchingContext,
        target_language: str
    ) -> List[str]:
        """Get recommendations for effective code-switching."""
        
        recommendations = []
        
        # Context-based recommendations
        if context.conversation_type == "professional":
            recommendations.extend([
                "Keep technical terms in English for clarity",
                "Use native language for relationship building",
                "Maintain professional tone with appropriate honorifics"
            ])
        elif context.conversation_type == "casual":
            recommendations.extend([
                "Feel free to mix languages naturally",
                "Use native language for emotional expressions",
                "Add discourse markers for natural flow"
            ])
        
        # Formality-based recommendations
        if context.formality_level in ["formal", "very_formal"]:
            recommendations.append("Use respectful terms and honorifics in native language")
        else:
            recommendations.append("Use casual expressions and friendly terms")
        
        # Language-specific recommendations
        if target_language == "hi":
            recommendations.extend([
                "Use 'ji' for respect and politeness",
                "Switch to Hindi for emotional emphasis",
                "Keep English for technical and business terms"
            ])
        elif target_language == "ta":
            recommendations.extend([
                "Use Tamil for cultural and family contexts",
                "Keep English for modern concepts and technology",
                "Add Tamil discourse markers for natural flow"
            ])
        elif target_language == "te":
            recommendations.extend([
                "Use 'garu' for respectful address",
                "Switch to Telugu for emotional expressions",
                "Maintain English for professional terminology"
            ])
        
        return recommendations[:5]