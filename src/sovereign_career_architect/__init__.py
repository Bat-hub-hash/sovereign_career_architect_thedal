"""
Sovereign Career Architect: An autonomous AI agent for career navigation.

This package implements a cognitive architecture using LangGraph for autonomous
career guidance, job search automation, and interview preparation with voice
interaction in multiple Indic languages.
"""

__version__ = "0.1.0"
__author__ = "AI-VERSE Team"
__email__ = "team@aiverse.com"

from sovereign_career_architect.core.agent import SovereignCareerAgent
from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.memory.client import MemoryClient
from sovereign_career_architect.browser.agent import BrowserAgent
from sovereign_career_architect.voice.orchestrator import VoiceOrchestrator

__all__ = [
    "SovereignCareerAgent",
    "AgentState", 
    "MemoryClient",
    "BrowserAgent",
    "VoiceOrchestrator",
]