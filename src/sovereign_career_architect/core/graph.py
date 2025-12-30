"""LangGraph workflow orchestration for the Sovereign Career Architect."""

from typing import Dict, Any, Optional
import operator
from typing_extensions import Annotated

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel

from sovereign_career_architect.core.state import AgentState
from sovereign_career_architect.core.nodes.profiler import create_profiler_node
from sovereign_career_architect.core.nodes.planner import create_planner_node
from sovereign_career_architect.core.nodes.executor import create_executor_node
from sovereign_career_architect.core.nodes.reviewer import create_reviewer_node
from sovereign_career_architect.core.nodes.archivist import create_archivist_node
from sovereign_career_architect.core.nodes.supervisor import create_supervisor_node
from sovereign_career_architect.memory.client import MemoryClient
from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


def create_career_architect_graph(
    planning_model: Optional[BaseChatModel] = None,
    review_model: Optional[BaseChatModel] = None,
    extraction_model: Optional[BaseChatModel] = None,
    routing_model: Optional[BaseChatModel] = None,
    memory_client: Optional[MemoryClient] = None,
    browser_agent=None,
    api_clients: Optional[Dict[str, Any]] = None,
    enable_checkpoints: bool = True
) -> StateGraph:
    """
    Create the complete LangGraph workflow for the Sovereign Career Architect.
    
    Args:
        planning_model: Language model for plan generation
        review_model: Language model for output review
        extraction_model: Language model for fact extraction
        routing_model: Language model for supervisor routing
        memory_client: Memory client for persistent storage
        browser_agent: Browser automation agent
        api_clients: Dictionary of API clients
        enable_checkpoints: Whether to enable state checkpointing
        
    Returns:
        Configured StateGraph workflow
    """
    logger.info("Creating Sovereign Career Architect graph")
    
    # Create node instances
    profiler_node = create_profiler_node(memory_client=memory_client)
    planner_node = create_planner_node(reasoning_model=planning_model)
    executor_node = create_executor_node(
        browser_agent=browser_agent,
        api_clients=api_clients
    )
    reviewer_node = create_reviewer_node(review_model=review_model)
    archivist_node = create_archivist_node(
        memory_client=memory_client,
        extraction_model=extraction_model
    )
    supervisor_node = create_supervisor_node(routing_model=routing_model)
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("archivist", archivist_node)
    workflow.add_node("supervisor", supervisor_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor to all other nodes
    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "profiler": "profiler",
            "planner": "planner",
            "executor": "executor",
            "reviewer": "reviewer",
            "archivist": "archivist",
            "end": END
        }
    )
    
    # Add edges back to supervisor from all nodes
    workflow.add_edge("profiler", "supervisor")
    workflow.add_edge("planner", "supervisor")
    workflow.add_edge("executor", "supervisor")
    workflow.add_edge("reviewer", "supervisor")
    workflow.add_edge("archivist", "supervisor")
    
    # Add checkpointing if enabled
    if enable_checkpoints:
        memory = MemorySaver()
        workflow = workflow.compile(checkpointer=memory)
    else:
        workflow = workflow.compile()
    
    logger.info("Sovereign Career Architect graph created successfully")
    return workflow


def _route_from_supervisor(state: AgentState) -> str:
    """
    Route from supervisor node based on the next_node decision.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute
    """
    next_node = getattr(state, "next_node", "end") or "end"
    
    logger.debug(
        "Routing from supervisor",
        next_node=next_node,
        session_id=str(getattr(state, "session_id", "unknown"))
    )
    
    return next_node


def create_simple_career_architect_graph(
    planning_model: Optional[BaseChatModel] = None,
    memory_client: Optional[MemoryClient] = None
) -> StateGraph:
    """
    Create a simplified version of the career architect graph for testing.
    
    This version uses a direct linear flow without the supervisor for simpler
    testing and development scenarios.
    
    Args:
        planning_model: Language model for plan generation
        memory_client: Memory client for persistent storage
        
    Returns:
        Simplified StateGraph workflow
    """
    logger.info("Creating simplified Career Architect graph")
    
    # Create node instances with minimal dependencies
    profiler_node = create_profiler_node(memory_client=memory_client)
    planner_node = create_planner_node(reasoning_model=planning_model)
    executor_node = create_executor_node()
    reviewer_node = create_reviewer_node()
    archivist_node = create_archivist_node(memory_client=memory_client)
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("archivist", archivist_node)
    
    # Set entry point
    workflow.set_entry_point("profiler")
    
    # Add linear edges for simple flow
    workflow.add_edge("profiler", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reviewer")
    
    # Add conditional edge from reviewer
    workflow.add_conditional_edges(
        "reviewer",
        _route_from_reviewer,
        {
            "retry": "planner",  # Retry planning if unsatisfactory
            "continue": "archivist",  # Continue to archival if satisfactory
            "end": END  # End if no more steps
        }
    )
    
    # End after archival
    workflow.add_edge("archivist", END)
    
    # Compile without checkpointing for simplicity
    workflow = workflow.compile()
    
    logger.info("Simplified Career Architect graph created successfully")
    return workflow


def _route_from_reviewer(state: AgentState) -> str:
    """
    Route from reviewer node based on critique and plan status.
    
    Args:
        state: Current agent state
        
    Returns:
        Next routing decision
    """
    # Check if we have a critique
    if not state.critique:
        return "end"
    
    # Check if critique indicates retry needed
    if "unsatisfactory" in state.critique.lower() and state.retry_count < 3:
        return "retry"
    
    # Check if we have more steps to execute
    if (state.current_plan and 
        state.current_step_index < len(state.current_plan.steps) - 1):
        return "continue"
    
    # Default to archival
    return "continue"


# Interrupt configuration for human-in-the-loop workflows
INTERRUPT_BEFORE_NODES = [
    "executor"  # Interrupt before high-stakes actions
]


def create_career_architect_with_interrupts(
    planning_model: Optional[BaseChatModel] = None,
    review_model: Optional[BaseChatModel] = None,
    extraction_model: Optional[BaseChatModel] = None,
    routing_model: Optional[BaseChatModel] = None,
    memory_client: Optional[MemoryClient] = None,
    browser_agent=None,
    api_clients: Optional[Dict[str, Any]] = None
) -> StateGraph:
    """
    Create the career architect graph with human-in-the-loop interrupts.
    
    This version includes interrupt points before high-stakes actions like
    job applications for user approval.
    
    Args:
        planning_model: Language model for plan generation
        review_model: Language model for output review
        extraction_model: Language model for fact extraction
        routing_model: Language model for supervisor routing
        memory_client: Memory client for persistent storage
        browser_agent: Browser automation agent
        api_clients: Dictionary of API clients
        
    Returns:
        StateGraph with interrupt capabilities
    """
    logger.info("Creating Career Architect graph with interrupts")
    
    # Create the base graph
    workflow = create_career_architect_graph(
        planning_model=planning_model,
        review_model=review_model,
        extraction_model=extraction_model,
        routing_model=routing_model,
        memory_client=memory_client,
        browser_agent=browser_agent,
        api_clients=api_clients,
        enable_checkpoints=True  # Required for interrupts
    )
    
    # Add interrupt configuration
    workflow.interrupt_before = INTERRUPT_BEFORE_NODES
    
    logger.info("Career Architect graph with interrupts created successfully")
    return workflow