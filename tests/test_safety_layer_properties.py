"""Property-based tests for human-in-the-loop safety layer."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any

from sovereign_career_architect.core.safety import (
    SafetyLayer, ActionClassifier, ActionSummarizer, ActionRisk, ActionCategory,
    ActionClassification, ActionSummary, ApprovalRequest
)
from sovereign_career_architect.core.models import ActionResult
from sovereign_career_architect.core.state import AgentState


class TestSafetyLayerProperties:
    """Property-based tests for safety layer functionality."""
    
    @given(
        action_types=st.lists(
            st.sampled_from([
                "navigate to job page", "fill application form", "submit application",
                "upload resume", "send message", "accept offer", "decline offer",
                "update profile", "delete account", "make payment", "sign contract"
            ]),
            min_size=1, max_size=20
        )
    )
    @settings(max_examples=30)
    def test_action_classification_consistency(self, action_types):
        """Property 20: High-Stakes Action Approval - Classification consistency."""
        classifier = ActionClassifier()
        
        for action_type in action_types:
            context = {"target_url": "https://example.com", "form_data": {"field1": "value1"}}
            
            # Classify the same action multiple times
            classification1 = classifier.classify_action(action_type, context)
            classification2 = classifier.classify_action(action_type, context)
            
            # Property: Classification should be deterministic
            assert classification1.category == classification2.category
            assert classification1.risk_level == classification2.risk_level
            assert classification1.description == classification2.description
            assert classification1.reasoning == classification2.reasoning
            assert classification1.requires_approval == classification2.requires_approval
            
            # Property: High-risk actions should require approval
            if classification1.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]:
                assert classification1.requires_approval is True
            
            # Property: Critical actions should have higher timeout
            if classification1.risk_level == ActionRisk.CRITICAL:
                assert classification1.timeout_seconds >= 300
    
    @given(
        contexts=st.lists(
            st.dictionaries(
                st.sampled_from(["target_url", "form_data", "involves_payment", "irreversible", "affects_application"]),
                st.one_of(
                    st.text(min_size=1, max_size=100),
                    st.booleans(),
                    st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50))
                )
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=20)
    def test_risk_assessment_properties(self, contexts):
        """Property 20: High-Stakes Action Approval - Risk assessment accuracy."""
        classifier = ActionClassifier()
        
        for context in contexts:
            # Test different action types with the same context
            test_actions = [
                ("navigate to page", ActionRisk.LOW),
                ("fill form", ActionRisk.MEDIUM),
                ("submit application", ActionRisk.HIGH),
                ("accept job offer", ActionRisk.CRITICAL)
            ]
            
            for action_type, expected_min_risk in test_actions:
                classification = classifier.classify_action(action_type, context)
                
                # Property: Risk level should be at least the expected minimum
                risk_levels = [ActionRisk.LOW, ActionRisk.MEDIUM, ActionRisk.HIGH, ActionRisk.CRITICAL]
                expected_index = risk_levels.index(expected_min_risk)
                actual_index = risk_levels.index(classification.risk_level)
                
                # Context can increase risk but not decrease below action baseline
                assert actual_index >= expected_index
                
                # Property: Payment context should increase risk
                if context.get("involves_payment", False):
                    assert classification.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]
                
                # Property: Irreversible context should increase risk
                if context.get("irreversible", False):
                    assert classification.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]
    
    @given(
        action_summaries=st.lists(
            st.tuples(
                st.sampled_from(["submit application", "upload resume", "send message", "navigate page"]),
                st.dictionaries(
                    st.sampled_from(["target_url", "form_data", "recipient"]),
                    st.text(min_size=1, max_size=100)
                )
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=20)
    def test_action_summary_completeness(self, action_summaries):
        """Property 21: Action Summary Clarity - Summary completeness."""
        classifier = ActionClassifier()
        summarizer = ActionSummarizer()
        
        for action_type, context in action_summaries:
            classification = classifier.classify_action(action_type, context)
            summary = summarizer.generate_summary(action_type, context, classification)
            
            # Property: Summary should have all required fields
            assert isinstance(summary.title, str) and len(summary.title) > 0
            assert isinstance(summary.description, str) and len(summary.description) > 0
            assert isinstance(summary.consequences, list)
            assert isinstance(summary.risks, list)
            assert isinstance(summary.benefits, list)
            assert isinstance(summary.affected_systems, list)
            assert isinstance(summary.estimated_duration, str) and len(summary.estimated_duration) > 0
            assert isinstance(summary.reversible, bool)
            
            # Property: High-risk actions should have more detailed information
            if classification.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]:
                assert len(summary.consequences) > 0
                assert len(summary.risks) > 0
                
            # Property: Summary should be serializable
            summary_dict = summary.to_dict()
            assert isinstance(summary_dict, dict)
            assert all(key in summary_dict for key in [
                "title", "description", "consequences", "risks", 
                "benefits", "affected_systems", "estimated_duration", "reversible"
            ])
    
    @given(
        approval_scenarios=st.lists(
            st.tuples(
                st.sampled_from(["submit application", "upload resume", "accept offer", "navigate page"]),
                st.dictionaries(
                    st.sampled_from(["involves_payment", "affects_application", "irreversible"]),
                    st.booleans()
                ),
                st.booleans()  # approval decision
            ),
            min_size=1, max_size=15
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_approval_workflow_properties(self, approval_scenarios):
        """Property 20: High-Stakes Action Approval - Workflow consistency."""
        safety_layer = SafetyLayer()
        
        for action_type, context, approval_decision in approval_scenarios:
            # Create mock state
            mock_state = {
                "messages": ["test message"],
                "current_plan": "test plan",
                "retry_count": 0,
                "user_profile": {"name": "Test User"}
            }
            
            # Evaluate action
            can_proceed, approval_request = await safety_layer.evaluate_action(
                action_type, context, mock_state
            )
            
            # Property: Low-risk actions should proceed immediately
            classification = safety_layer.classifier.classify_action(action_type, context)
            if classification.risk_level == ActionRisk.LOW:
                assert can_proceed is True
                assert approval_request is None
            
            # Property: High-risk actions should require approval
            if classification.risk_level in [ActionRisk.HIGH, ActionRisk.CRITICAL]:
                assert can_proceed is False
                assert approval_request is not None
                
                # Property: Approval request should be complete
                assert approval_request.id is not None
                assert approval_request.timestamp is not None
                assert approval_request.action_type == action_type
                assert approval_request.classification is not None
                assert approval_request.summary is not None
                assert approval_request.context == context
                assert approval_request.state_snapshot is not None
                
                # Test approval process
                success = safety_layer.provide_approval(
                    approval_request.id, approval_decision
                )
                assert success is True
                
                # Property: Approval should be recorded
                updated_request = safety_layer.pending_approvals[approval_request.id]
                assert updated_request.approved == approval_decision
                assert updated_request.approval_timestamp is not None
    
    @given(
        execution_results=st.lists(
            st.tuples(
                st.booleans(),  # executed
                st.booleans(),  # success
                st.floats(min_value=0.1, max_value=60.0),  # duration
                st.one_of(st.none(), st.text(min_size=1, max_size=100))  # error
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=20)
    def test_audit_logging_properties(self, execution_results):
        """Property 24: Audit Trail Completeness - Logging consistency."""
        safety_layer = SafetyLayer()
        
        for executed, success, duration, error in execution_results:
            # Create mock approval request
            classification = ActionClassification(
                category=ActionCategory.SUBMISSION,
                risk_level=ActionRisk.HIGH,
                description="Test action",
                reasoning="Test reasoning"
            )
            
            summary = ActionSummary(
                title="Test Action",
                description="Test description",
                consequences=["Test consequence"],
                risks=["Test risk"],
                benefits=["Test benefit"],
                affected_systems=["Test system"],
                estimated_duration="1 minute",
                reversible=False
            )
            
            approval_request = ApprovalRequest(
                id=f"test_{len(safety_layer.audit_log)}",
                timestamp=datetime.now(timezone.utc),
                action_type="test_action",
                classification=classification,
                summary=summary,
                context={"test": "context"},
                state_snapshot={"test": "state"}
            )
            approval_request.approved = True
            
            # Create mock result
            result = ActionResult(
                success=success,
                message="Test result",
                data={"test": "data"}
            ) if executed else None
            
            # Log execution
            safety_layer.log_action_execution(
                approval_request=approval_request,
                executed=executed,
                result=result,
                duration_seconds=duration,
                error=error
            )
            
            # Property: Audit log should contain the entry
            audit_entries = safety_layer.get_audit_log()
            assert len(audit_entries) > 0
            
            latest_entry = audit_entries[-1]
            assert latest_entry.id == approval_request.id
            assert latest_entry.executed == executed
            assert latest_entry.duration_seconds == duration
            assert latest_entry.error == error
            
            if executed and result:
                assert latest_entry.result is not None
                assert latest_entry.result.success == success
            
            # Property: Audit entry should be serializable
            entry_dict = latest_entry.to_dict()
            assert isinstance(entry_dict, dict)
            assert "timestamp" in entry_dict
            assert "action_type" in entry_dict
            assert "executed" in entry_dict
    
    @given(
        concurrent_requests=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_concurrent_approval_handling(self, concurrent_requests):
        """Property 20: High-Stakes Action Approval - Concurrent request handling."""
        safety_layer = SafetyLayer()
        
        # Create multiple concurrent approval requests
        tasks = []
        for i in range(concurrent_requests):
            action_type = f"test_action_{i}"
            context = {"request_id": i, "affects_application": True}
            mock_state = {"messages": [f"message_{i}"]}
            
            task = safety_layer.evaluate_action(action_type, context, mock_state)
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        # Property: All high-risk requests should require approval
        approval_requests = []
        for can_proceed, approval_request in results:
            if approval_request is not None:
                approval_requests.append(approval_request)
                assert can_proceed is False
        
        # Property: All approval requests should have unique IDs
        approval_ids = [req.id for req in approval_requests]
        assert len(approval_ids) == len(set(approval_ids))
        
        # Property: All requests should be in pending approvals
        pending = safety_layer.get_pending_approvals()
        assert len(pending) == len(approval_requests)
        
        # Approve all requests
        for approval_request in approval_requests:
            success = safety_layer.provide_approval(approval_request.id, True)
            assert success is True
    
    @given(
        timeout_scenarios=st.lists(
            st.tuples(
                st.sampled_from([ActionRisk.HIGH, ActionRisk.CRITICAL]),
                st.integers(min_value=1, max_value=10)  # timeout multiplier
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=10)
    def test_timeout_configuration_properties(self, timeout_scenarios):
        """Property 20: High-Stakes Action Approval - Timeout handling."""
        classifier = ActionClassifier()
        
        for risk_level, timeout_multiplier in timeout_scenarios:
            # Create action that results in the specified risk level
            if risk_level == ActionRisk.CRITICAL:
                action_type = "accept job offer"
                context = {"involves_payment": True}
            else:
                action_type = "submit application"
                context = {"affects_application": True}
            
            classification = classifier.classify_action(action_type, context)
            
            # Property: Classification should match expected risk level
            assert classification.risk_level == risk_level
            
            # Property: Timeout should be reasonable for risk level
            if risk_level == ActionRisk.CRITICAL:
                assert classification.timeout_seconds >= 300  # At least 5 minutes
            elif risk_level == ActionRisk.HIGH:
                assert classification.timeout_seconds >= 180  # At least 3 minutes
            
            # Property: Timeout should be positive
            assert classification.timeout_seconds > 0
    
    @pytest.mark.asyncio
    async def test_safety_layer_initialization_properties(self):
        """Property 20: High-Stakes Action Approval - Initialization consistency."""
        # Test with default initialization
        safety_layer1 = SafetyLayer()
        assert safety_layer1.classifier is not None
        assert safety_layer1.summarizer is not None
        assert safety_layer1.approval_callback is None
        assert len(safety_layer1.pending_approvals) == 0
        assert len(safety_layer1.audit_log) == 0
        
        # Test with custom approval callback
        mock_callback = AsyncMock(return_value=True)
        safety_layer2 = SafetyLayer(approval_callback=mock_callback)
        assert safety_layer2.approval_callback == mock_callback
        
        # Property: Different instances should be independent
        mock_state = {"messages": ["test"]}
        await safety_layer1.evaluate_action("submit application", {"test": True}, mock_state)
        await safety_layer2.evaluate_action("upload resume", {"test": True}, mock_state)
        
        pending1 = safety_layer1.get_pending_approvals()
        pending2 = safety_layer2.get_pending_approvals()
        
        # Should have different pending requests
        if pending1 and pending2:
            assert pending1[0].id != pending2[0].id