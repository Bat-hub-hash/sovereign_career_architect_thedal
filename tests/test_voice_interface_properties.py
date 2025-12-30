"""Property-based tests for voice interface functionality."""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from sovereign_career_architect.voice.orchestrator import (
    VoiceOrchestrator, VoiceConfig, FunctionDefinition, VoiceEventType
)


class TestVoiceInterfaceProperties:
    """Property-based tests for voice interface functionality."""
    
    @given(
        function_names=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')), 
                   min_size=3, max_size=20),
            min_size=1, max_size=10, unique=True
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_function_registration_properties(self, function_names):
        """Property 15: Voice Command Mapping - Function registration consistency."""
        orchestrator = VoiceOrchestrator()
        
        # Register functions
        for func_name in function_names:
            assume(func_name.isidentifier())  # Valid Python identifier
            
            handler = AsyncMock(return_value=f"Result for {func_name}")
            
            orchestrator.register_function(
                name=func_name,
                description=f"Test function {func_name}",
                parameters={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Test parameter"}
                    }
                },
                handler=handler
            )
        
        # Property: All functions should be registered
        for func_name in function_names:
            assert func_name in orchestrator.functions
            
            func_def = orchestrator.functions[func_name]
            assert func_def.name == func_name
            assert isinstance(func_def.description, str)
            assert isinstance(func_def.parameters, dict)
            assert callable(func_def.handler)
        
        # Property: Function count should match
        # Account for default functions that are pre-registered
        expected_count = len(function_names) + len(orchestrator.functions) - len(function_names)
        assert len(orchestrator.functions) >= len(function_names)
    
    @given(
        webhook_events=st.lists(
            st.dictionaries(
                st.sampled_from(["type", "call", "functionCall", "transcript"]),
                st.one_of(
                    st.sampled_from([e.value for e in VoiceEventType]),
                    st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=50)),
                    st.text(min_size=1, max_size=100)
                )
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_webhook_handling_properties(self, webhook_events):
        """Property 15: Voice Command Mapping - Webhook handling consistency."""
        orchestrator = VoiceOrchestrator()
        
        for event_data in webhook_events:
            # Ensure event has required structure
            if "type" not in event_data:
                event_data["type"] = VoiceEventType.TRANSCRIPT.value
            
            result = await orchestrator.handle_webhook(event_data)
            
            # Property: Webhook handling should always return a response
            assert isinstance(result, dict)
            assert "status" in result or "result" in result
            
            # Property: Valid event types should be handled gracefully
            if event_data["type"] in [e.value for e in VoiceEventType]:
                assert result.get("status") in ["acknowledged", "ignored", "error"] or "result" in result
    
    @given(
        assistant_configs=st.lists(
            st.tuples(
                st.text(min_size=3, max_size=50),  # name
                st.text(min_size=10, max_size=200),  # system_message
                st.text(min_size=5, max_size=100),  # first_message
                st.sampled_from(["en", "hi", "ta", "es", "fr"])  # language
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_assistant_creation_properties(self, assistant_configs):
        """Property 15: Voice Command Mapping - Assistant creation consistency."""
        orchestrator = VoiceOrchestrator()
        
        # Mock the HTTP client
        with patch.object(orchestrator.client, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"id": "test_assistant_id", "name": "test"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            for name, system_message, first_message, language in assistant_configs:
                assume(len(name.strip()) > 0)
                assume(len(system_message.strip()) > 0)
                assume(len(first_message.strip()) > 0)
                
                assistant_data = await orchestrator.create_assistant(
                    name=name,
                    system_message=system_message,
                    first_message=first_message,
                    language=language
                )
                
                # Property: Assistant creation should return valid data
                assert isinstance(assistant_data, dict)
                assert "id" in assistant_data
                
                # Property: API should be called with correct structure
                mock_post.assert_called()
                call_args = mock_post.call_args[1]["json"]
                
                assert call_args["name"] == name
                assert call_args["model"]["messages"][0]["content"] == system_message
                assert call_args["firstMessage"] == first_message
                assert isinstance(call_args["model"]["functions"], list)
    
    @given(
        function_calls=st.lists(
            st.tuples(
                st.sampled_from(["search_jobs", "apply_to_job", "prepare_interview", "update_profile"]),
                st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.one_of(st.text(min_size=1, max_size=50), st.booleans())
                )
            ),
            min_size=1, max_size=8
        )
    )
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_function_call_handling_properties(self, function_calls):
        """Property 15: Voice Command Mapping - Function call handling consistency."""
        orchestrator = VoiceOrchestrator()
        
        for function_name, function_args in function_calls:
            event_data = {
                "type": VoiceEventType.FUNCTION_CALL.value,
                "functionCall": {
                    "name": function_name,
                    "parameters": function_args
                },
                "call": {"id": "test_call_id"}
            }
            
            result = await orchestrator.handle_webhook(event_data)
            
            # Property: Function calls should return results
            assert isinstance(result, dict)
            assert "result" in result
            assert isinstance(result["result"], str)
            
            # Property: Valid function names should execute successfully
            if function_name in orchestrator.functions:
                # Should not contain error messages for valid functions
                assert "Unknown function" not in result["result"]
                assert "execution failed" not in result["result"].lower()
    
    @given(
        call_configs=st.lists(
            st.tuples(
                st.text(min_size=10, max_size=30),  # assistant_id
                st.one_of(st.none(), st.text(min_size=10, max_size=15)),  # phone_number
                st.one_of(st.none(), st.text(min_size=5, max_size=20)),  # customer_id
                st.one_of(st.none(), st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20)))  # metadata
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_call_management_properties(self, call_configs):
        """Property 15: Voice Command Mapping - Call management consistency."""
        orchestrator = VoiceOrchestrator()
        
        # Mock the HTTP client
        with patch.object(orchestrator.client, 'post') as mock_post:
            for assistant_id, phone_number, customer_id, metadata in call_configs:
                mock_response = AsyncMock()
                call_id = f"call_{len(orchestrator.active_calls)}"
                mock_response.json.return_value = {"id": call_id, "status": "started"}
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                call_data = await orchestrator.start_call(
                    assistant_id=assistant_id,
                    phone_number=phone_number,
                    customer_id=customer_id,
                    metadata=metadata
                )
                
                # Property: Call should be tracked
                assert call_id in orchestrator.active_calls
                
                call_info = orchestrator.active_calls[call_id]
                assert call_info["assistant_id"] == assistant_id
                assert call_info["customer_id"] == customer_id
                assert call_info["metadata"] == metadata
                assert "started_at" in call_info
                
                # Property: Call data should be returned
                assert isinstance(call_data, dict)
                assert call_data["id"] == call_id
                
                # Property: Call status should be retrievable
                status = await orchestrator.get_call_status(call_id)
                assert status is not None
                assert status["assistant_id"] == assistant_id
    
    @pytest.mark.asyncio
    async def test_voice_orchestrator_initialization_properties(self):
        """Property 15: Voice Command Mapping - Initialization consistency."""
        # Test with default config
        orchestrator1 = VoiceOrchestrator()
        assert orchestrator1.config is not None
        assert isinstance(orchestrator1.functions, dict)
        assert len(orchestrator1.functions) > 0  # Should have default functions
        assert isinstance(orchestrator1.active_calls, dict)
        
        # Test with custom config
        custom_config = VoiceConfig(
            api_key="test_key",
            webhook_url="https://example.com/webhook",
            voice_id="custom_voice"
        )
        orchestrator2 = VoiceOrchestrator(custom_config)
        assert orchestrator2.config.api_key == "test_key"
        assert orchestrator2.config.webhook_url == "https://example.com/webhook"
        assert orchestrator2.config.voice_id == "custom_voice"
        
        # Property: Different instances should be independent
        assert orchestrator1.active_calls is not orchestrator2.active_calls
        assert orchestrator1.functions is not orchestrator2.functions
    
    @given(
        event_sequences=st.lists(
            st.sampled_from([
                VoiceEventType.CALL_STARTED.value,
                VoiceEventType.TRANSCRIPT.value,
                VoiceEventType.FUNCTION_CALL.value,
                VoiceEventType.CALL_ENDED.value
            ]),
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_event_sequence_handling_properties(self, event_sequences):
        """Property 15: Voice Command Mapping - Event sequence handling."""
        orchestrator = VoiceOrchestrator()
        call_id = "test_call_sequence"
        
        # Initialize call tracking
        orchestrator.active_calls[call_id] = {
            "assistant_id": "test_assistant",
            "customer_id": "test_customer",
            "metadata": {},
            "started_at": asyncio.get_event_loop().time()
        }
        
        results = []
        for event_type in event_sequences:
            event_data = {
                "type": event_type,
                "call": {"id": call_id}
            }
            
            # Add specific data for function calls
            if event_type == VoiceEventType.FUNCTION_CALL.value:
                event_data["functionCall"] = {
                    "name": "search_jobs",
                    "parameters": {"query": "software engineer"}
                }
            elif event_type == VoiceEventType.TRANSCRIPT.value:
                event_data["transcript"] = {"text": "Hello, I need help with job search"}
            
            result = await orchestrator.handle_webhook(event_data)
            results.append(result)
        
        # Property: All events should be handled successfully
        for result in results:
            assert isinstance(result, dict)
            assert "status" in result or "result" in result
        
        # Property: Call should be removed after CALL_ENDED
        if VoiceEventType.CALL_ENDED.value in event_sequences:
            # Find the last occurrence of CALL_ENDED
            last_call_ended_index = len(event_sequences) - 1 - event_sequences[::-1].index(VoiceEventType.CALL_ENDED.value)
            
            # If CALL_ENDED was the last event, call should be removed
            if last_call_ended_index == len(event_sequences) - 1:
                assert call_id not in orchestrator.active_calls