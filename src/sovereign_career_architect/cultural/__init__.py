"""Cultural and language adaptation system."""

from .adapter import (
    CulturalAdapter,
    CulturalProfile,
    CommunicationStyle,
    CulturalContext
)
from .code_switching import (
    CodeSwitchingManager,
    LanguageMix,
    SwitchingContext
)

__all__ = [
    "CulturalAdapter",
    "CulturalProfile",
    "CommunicationStyle", 
    "CulturalContext",
    "CodeSwitchingManager",
    "LanguageMix",
    "SwitchingContext"
]