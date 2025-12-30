"""Browser automation components for web interaction."""

from sovereign_career_architect.browser.agent import BrowserAgent, create_browser_agent
from sovereign_career_architect.browser.vision import ElementIdentifier, create_element_identifier
from sovereign_career_architect.browser.navigation import JobPortalNavigator, create_job_portal_navigator
from sovereign_career_architect.browser.forms import FormFiller, create_form_filler

__all__ = [
    "BrowserAgent", "create_browser_agent",
    "ElementIdentifier", "create_element_identifier", 
    "JobPortalNavigator", "create_job_portal_navigator",
    "FormFiller", "create_form_filler"
]