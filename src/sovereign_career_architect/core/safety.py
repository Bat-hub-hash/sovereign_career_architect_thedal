"""Human-in-the-loop safety layer for high-stakes actions."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json
import structlog

from sovereign_career_architect.core.models import ActionResult
from sovereign_career_architect.core.state import AgentState

logger = structlog.get_logger(__name__)


class ActionRisk(Enum):
    """Risk levels for actions requiring human approval."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionCategory(Enum):
    """Categories of actions for classification."""
    NAVIGATION = "navigation"
    FORM_FILLING = "form_filling"
    SUBMISSION = "submission"
    FILE_UPLOAD = "file_upload"
    ACCOUNT_MODIFICATION = "account_modification"
    PAYMENT = "payment"
    COMMUNICATION = "communication"
    DATA_DELETION = "data_deletion"


@dataclass
class ActionClassification:
    """Classification of an action for safety assessment."""
    category: ActionCategory
    risk_level: ActionRisk
    description: str
    reasoning: str
    requires_approval: bool = False
    timeout_seconds: int = 300  # 5 minutes default
    
    def __post_init__(self):
        # Auto-set approval requirement based on risk level
        if self.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]:
            self.requires_approval = True


@dataclass
class ActionSummary:
    """Human-readable summary of an action for approval."""
    title: str
    description: str
    consequences: List[str]
    risks: List[str]
    benefits: List[str]
    affected_systems: List[str]
    estimated_duration: str
    reversible: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "consequences": self.consequences,
            "risks": self.risks,
            "benefits": self.benefits,
            "affected_systems": self.affected_systems,
            "estimated_duration": self.estimated_duration,
            "reversible": self.reversible
        }


@dataclass
class ApprovalRequest:
    """Request for human approval of an action."""
    id: str
    timestamp: datetime
    action_type: str
    classification: ActionClassification
    summary: ActionSummary
    context: Dict[str, Any]
    state_snapshot: Dict[str, Any]
    approved: Optional[bool] = None
    approval_timestamp: Optional[datetime] = None
    approver_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "classification": {
                "category": self.classification.category.value,
                "risk_level": self.classification.risk_level.value,
                "description": self.classification.description,
                "reasoning": self.classification.reasoning,
                "requires_approval": self.classification.requires_approval,
                "timeout_seconds": self.classification.timeout_seconds
            },
            "summary": self.summary.to_dict(),
            "context": self.context,
            "state_snapshot": self.state_snapshot,
            "approved": self.approved,
            "approval_timestamp": self.approval_timestamp.isoformat() if self.approval_timestamp else None,
            "approver_notes": self.approver_notes
        }


@dataclass
class AuditLogEntry:
    """Entry in the audit log for action tracking."""
    id: str
    timestamp: datetime
    action_type: str
    classification: ActionClassification
    summary: ActionSummary
    approved: bool
    executed: bool
    result: Optional[ActionResult]
    duration_seconds: Optional[float]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "classification": {
                "category": self.classification.category.value,
                "risk_level": self.classification.risk_level.value,
                "description": self.classification.description,
                "reasoning": self.classification.reasoning
            },
            "summary": self.summary.to_dict(),
            "approved": self.approved,
            "executed": self.executed,
            "result": self.result.to_dict() if self.result else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error
        }


class ActionClassifier:
    """Classifies actions to determine risk level and approval requirements."""
    
    def __init__(self):
        self.high_risk_patterns = [
            "submit application",
            "upload resume",
            "send message",
            "accept offer",
            "decline offer",
            "withdraw application",
            "delete account",
            "change password",
            "update profile",
            "make payment"
        ]
        
        self.critical_risk_patterns = [
            "accept job offer",
            "sign contract",
            "authorize payment",
            "delete all data",
            "close account permanently"
        ]
    
    def classify_action(self, action_type: str, context: Dict[str, Any]) -> ActionClassification:
        """Classify an action to determine its risk level and requirements."""
        action_lower = action_type.lower()
        
        # Determine category
        category = self._determine_category(action_type, context)
        
        # Determine risk level
        risk_level = self._determine_risk_level(action_lower, context)
        
        # Generate description and reasoning
        description = self._generate_description(action_type, context)
        reasoning = self._generate_reasoning(action_type, context, risk_level)
        
        return ActionClassification(
            category=category,
            risk_level=risk_level,
            description=description,
            reasoning=reasoning
        )
    
    def _determine_category(self, action_type: str, context: Dict[str, Any]) -> ActionCategory:
        """Determine the category of an action."""
        action_lower = action_type.lower()
        
        if "navigate" in action_lower or "visit" in action_lower:
            return ActionCategory.NAVIGATION
        elif "fill" in action_lower or "enter" in action_lower:
            return ActionCategory.FORM_FILLING
        elif "submit" in action_lower or "apply" in action_lower:
            return ActionCategory.SUBMISSION
        elif "upload" in action_lower or "attach" in action_lower:
            return ActionCategory.FILE_UPLOAD
        elif "message" in action_lower or "email" in action_lower:
            return ActionCategory.COMMUNICATION
        elif "payment" in action_lower or "pay" in action_lower:
            return ActionCategory.PAYMENT
        elif "delete" in action_lower or "remove" in action_lower:
            return ActionCategory.DATA_DELETION
        elif "update" in action_lower or "modify" in action_lower:
            return ActionCategory.ACCOUNT_MODIFICATION
        else:
            return ActionCategory.NAVIGATION  # Default
    
    def _determine_risk_level(self, action_lower: str, context: Dict[str, Any]) -> ActionRisk:
        """Determine the risk level of an action."""
        # Check for critical risk patterns
        for pattern in self.critical_risk_patterns:
            if pattern in action_lower:
                return ActionRisk.CRITICAL
        
        # Check for high risk patterns
        for pattern in self.high_risk_patterns:
            if pattern in action_lower:
                return ActionRisk.HIGH
        
        # Check context for risk indicators
        if context.get("involves_payment", False):
            return ActionRisk.CRITICAL
        elif context.get("irreversible", False):
            return ActionRisk.HIGH
        elif context.get("affects_application", False):
            return ActionRisk.HIGH
        elif context.get("modifies_profile", False):
            return ActionRisk.MEDIUM
        else:
            return ActionRisk.LOW
    
    def _generate_description(self, action_type: str, context: Dict[str, Any]) -> str:
        """Generate a human-readable description of the action."""
        base_description = f"Execute action: {action_type}"
        
        if context.get("target_url"):
            base_description += f" on {context['target_url']}"
        
        if context.get("form_data"):
            base_description += f" with form data containing {len(context['form_data'])} fields"
        
        return base_description
    
    def _generate_reasoning(self, action_type: str, context: Dict[str, Any], risk_level: ActionRisk) -> str:
        """Generate reasoning for the risk classification."""
        reasons = []
        
        if risk_level == ActionRisk.CRITICAL:
            reasons.append("Action has irreversible consequences")
        elif risk_level == ActionRisk.HIGH:
            reasons.append("Action affects job applications or account settings")
        elif risk_level == ActionRisk.MEDIUM:
            reasons.append("Action modifies user data or preferences")
        else:
            reasons.append("Action is low-risk navigation or data retrieval")
        
        if context.get("involves_payment"):
            reasons.append("Involves financial transaction")
        
        if context.get("affects_application"):
            reasons.append("Affects job application status")
        
        if context.get("modifies_profile"):
            reasons.append("Modifies user profile information")
        
        return "; ".join(reasons)


class ActionSummarizer:
    """Generates human-readable summaries of actions for approval."""
    
    def generate_summary(self, action_type: str, context: Dict[str, Any], classification: ActionClassification) -> ActionSummary:
        """Generate a comprehensive summary of an action."""
        title = self._generate_title(action_type, context)
        description = self._generate_detailed_description(action_type, context)
        consequences = self._identify_consequences(action_type, context)
        risks = self._identify_risks(action_type, context, classification)
        benefits = self._identify_benefits(action_type, context)
        affected_systems = self._identify_affected_systems(action_type, context)
        estimated_duration = self._estimate_duration(action_type, context)
        reversible = self._is_reversible(action_type, context)
        
        return ActionSummary(
            title=title,
            description=description,
            consequences=consequences,
            risks=risks,
            benefits=benefits,
            affected_systems=affected_systems,
            estimated_duration=estimated_duration,
            reversible=reversible
        )
    
    def _generate_title(self, action_type: str, context: Dict[str, Any]) -> str:
        """Generate a concise title for the action."""
        if "submit" in action_type.lower():
            return f"Submit Job Application"
        elif "upload" in action_type.lower():
            return f"Upload Resume/Documents"
        elif "message" in action_type.lower():
            return f"Send Message to Recruiter"
        else:
            return action_type.title()
    
    def _generate_detailed_description(self, action_type: str, context: Dict[str, Any]) -> str:
        """Generate a detailed description of what the action will do."""
        description = f"The system will {action_type.lower()}"
        
        if context.get("target_url"):
            description += f" on the website {context['target_url']}"
        
        if context.get("form_data"):
            fields = list(context["form_data"].keys())
            description += f" using the following information: {', '.join(fields[:5])}"
            if len(fields) > 5:
                description += f" and {len(fields) - 5} other fields"
        
        return description
    
    def _identify_consequences(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """Identify potential consequences of the action."""
        consequences = []
        
        if "submit" in action_type.lower():
            consequences.extend([
                "Job application will be submitted to the employer",
                "Your contact information will be shared with the employer",
                "Application status will change to 'submitted'"
            ])
        elif "upload" in action_type.lower():
            consequences.extend([
                "Resume/documents will be uploaded to the platform",
                "Files will be associated with your profile",
                "Employers may access uploaded documents"
            ])
        elif "message" in action_type.lower():
            consequences.extend([
                "Message will be sent to the recipient",
                "Your contact information may be shared",
                "Conversation history will be recorded"
            ])
        
        return consequences
    
    def _identify_risks(self, action_type: str, context: Dict[str, Any], classification: ActionClassification) -> List[str]:
        """Identify potential risks of the action."""
        risks = []
        
        if classification.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]:
            risks.append("Action cannot be easily undone")
        
        if "submit" in action_type.lower():
            risks.extend([
                "Application may be rejected if information is incorrect",
                "Multiple applications to same company may be flagged"
            ])
        elif "upload" in action_type.lower():
            risks.extend([
                "Incorrect documents may harm application chances",
                "Personal information in documents will be exposed"
            ])
        
        return risks
    
    def _identify_benefits(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """Identify potential benefits of the action."""
        benefits = []
        
        if "submit" in action_type.lower():
            benefits.extend([
                "Progress toward job search goals",
                "Opportunity to be considered for the position",
                "Automated application saves time and effort"
            ])
        elif "upload" in action_type.lower():
            benefits.extend([
                "Profile becomes more complete and attractive",
                "Employers can better assess qualifications",
                "Automated process ensures consistency"
            ])
        
        return benefits
    
    def _identify_affected_systems(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """Identify systems that will be affected by the action."""
        systems = []
        
        if context.get("target_url"):
            domain = context["target_url"].split("//")[-1].split("/")[0]
            systems.append(domain)
        
        systems.extend([
            "User profile database",
            "Application tracking system",
            "Audit logging system"
        ])
        
        return systems
    
    def _estimate_duration(self, action_type: str, context: Dict[str, Any]) -> str:
        """Estimate how long the action will take."""
        if "submit" in action_type.lower():
            return "2-5 minutes"
        elif "upload" in action_type.lower():
            return "1-3 minutes"
        elif "message" in action_type.lower():
            return "30 seconds - 1 minute"
        else:
            return "Less than 1 minute"
    
    def _is_reversible(self, action_type: str, context: Dict[str, Any]) -> bool:
        """Determine if the action can be easily reversed."""
        irreversible_actions = ["submit", "send", "delete", "accept", "decline"]
        return not any(keyword in action_type.lower() for keyword in irreversible_actions)


class SafetyLayer:
    """Human-in-the-loop safety layer for managing high-stakes actions."""
    
    def __init__(self, approval_callback: Optional[Callable] = None):
        self.classifier = ActionClassifier()
        self.summarizer = ActionSummarizer()
        self.approval_callback = approval_callback
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.logger = logger.bind(component="safety_layer")
    
    async def evaluate_action(self, action_type: str, context: Dict[str, Any], state: AgentState) -> Tuple[bool, Optional[ApprovalRequest]]:
        """
        Evaluate an action and determine if it requires human approval.
        
        Args:
            action_type: Type of action being performed
            context: Context information about the action
            state: Current agent state
            
        Returns:
            Tuple of (can_proceed, approval_request)
        """
        # Classify the action
        classification = self.classifier.classify_action(action_type, context)
        
        self.logger.info(
            "Action classified",
            action_type=action_type,
            category=classification.category.value,
            risk_level=classification.risk_level.value,
            requires_approval=classification.requires_approval
        )
        
        # If no approval required, proceed immediately
        if not classification.requires_approval:
            return True, None
        
        # Generate action summary
        summary = self.summarizer.generate_summary(action_type, context, classification)
        
        # Create approval request
        approval_request = ApprovalRequest(
            id=f"approval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(timezone.utc),
            action_type=action_type,
            classification=classification,
            summary=summary,
            context=context,
            state_snapshot=self._create_state_snapshot(state)
        )
        
        # Store pending approval
        self.pending_approvals[approval_request.id] = approval_request
        
        self.logger.info(
            "Approval required for high-stakes action",
            approval_id=approval_request.id,
            action_type=action_type,
            risk_level=classification.risk_level.value
        )
        
        return False, approval_request
    
    async def request_approval(self, approval_request: ApprovalRequest) -> bool:
        """
        Request human approval for an action.
        
        Args:
            approval_request: The approval request to process
            
        Returns:
            True if approved, False if denied
        """
        if self.approval_callback:
            # Use custom approval callback
            try:
                approved = await self.approval_callback(approval_request)
                return await self._process_approval_response(approval_request.id, approved)
            except Exception as e:
                self.logger.error("Approval callback failed", error=str(e))
                return False
        else:
            # Default: wait for manual approval via API
            return await self._wait_for_approval(approval_request)
    
    async def _wait_for_approval(self, approval_request: ApprovalRequest) -> bool:
        """Wait for manual approval via external interface."""
        timeout = approval_request.classification.timeout_seconds
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(
            "Waiting for manual approval",
            approval_id=approval_request.id,
            timeout_seconds=timeout
        )
        
        while True:
            # Check if approval has been provided
            current_request = self.pending_approvals.get(approval_request.id)
            if current_request and current_request.approved is not None:
                return current_request.approved
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                self.logger.warning(
                    "Approval request timed out",
                    approval_id=approval_request.id,
                    elapsed_seconds=elapsed
                )
                return False
            
            # Wait before checking again
            await asyncio.sleep(1)
    
    async def _process_approval_response(self, approval_id: str, approved: bool, notes: Optional[str] = None) -> bool:
        """Process an approval response."""
        if approval_id not in self.pending_approvals:
            self.logger.error("Approval ID not found", approval_id=approval_id)
            return False
        
        approval_request = self.pending_approvals[approval_id]
        approval_request.approved = approved
        approval_request.approval_timestamp = datetime.now(timezone.utc)
        approval_request.approver_notes = notes
        
        self.logger.info(
            "Approval response processed",
            approval_id=approval_id,
            approved=approved,
            notes=notes
        )
        
        return approved
    
    def provide_approval(self, approval_id: str, approved: bool, notes: Optional[str] = None) -> bool:
        """
        Provide approval response for a pending request.
        
        Args:
            approval_id: ID of the approval request
            approved: Whether the action is approved
            notes: Optional notes from the approver
            
        Returns:
            True if approval was processed successfully
        """
        if approval_id not in self.pending_approvals:
            return False
        
        approval_request = self.pending_approvals[approval_id]
        approval_request.approved = approved
        approval_request.approval_timestamp = datetime.now(timezone.utc)
        approval_request.approver_notes = notes
        
        return True
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [req for req in self.pending_approvals.values() if req.approved is None]
    
    def log_action_execution(
        self,
        approval_request: ApprovalRequest,
        executed: bool,
        result: Optional[ActionResult] = None,
        duration_seconds: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Log the execution of an approved action."""
        audit_entry = AuditLogEntry(
            id=approval_request.id,
            timestamp=datetime.now(timezone.utc),
            action_type=approval_request.action_type,
            classification=approval_request.classification,
            summary=approval_request.summary,
            approved=approval_request.approved or False,
            executed=executed,
            result=result,
            duration_seconds=duration_seconds,
            error=error
        )
        
        self.audit_log.append(audit_entry)
        
        # Remove from pending approvals
        if approval_request.id in self.pending_approvals:
            del self.pending_approvals[approval_request.id]
        
        self.logger.info(
            "Action execution logged",
            approval_id=approval_request.id,
            executed=executed,
            success=result.success if result else None,
            error=error
        )
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[AuditLogEntry]:
        """Get audit log entries."""
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log.copy()
    
    def _create_state_snapshot(self, state: AgentState) -> Dict[str, Any]:
        """Create a snapshot of the current agent state."""
        return {
            "messages_count": len(state.get("messages", [])),
            "current_plan": str(state.get("current_plan", "")),
            "retry_count": state.get("retry_count", 0),
            "user_profile": bool(state.get("user_profile")),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }