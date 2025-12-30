"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any, Dict

import structlog
from rich.logging import RichHandler

from sovereign_career_architect.config import settings


def configure_logging() -> None:
    """Configure structured logging with rich output."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a log context for function calls."""
    return {
        "function": func_name,
        "parameters": {k: v for k, v in kwargs.items() if not k.startswith("_")},
    }


def log_agent_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a log context for agent state."""
    current_plan = state.get("current_plan")
    plan_steps = 0
    if current_plan and hasattr(current_plan, 'steps'):
        plan_steps = len(current_plan.steps)
    elif isinstance(current_plan, list):
        plan_steps = len(current_plan)
    
    return {
        "agent_state": {
            "messages_count": len(state.get("messages", [])),
            "current_plan_steps": plan_steps,
            "retry_count": state.get("retry_count", 0),
            "has_user_profile": bool(state.get("user_profile")),
        }
    }