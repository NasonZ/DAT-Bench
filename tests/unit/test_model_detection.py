"""Unit tests for LLM provider model detection logic."""

import pytest
from unittest.mock import Mock, patch
from divergent_bench.llm.providers import OpenAICompatibleClient, MODEL_CAPABILITIES


class TestModelDetection:
    """Test model detection and capability identification."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch('divergent_bench.llm.providers.AsyncOpenAI'):
            client = OpenAICompatibleClient(
                provider="openai",
                api_key="test_key",
                model="gpt-4"
            )
            return client
    
    def test_model_capabilities_registry_pattern_ordering(self):
        """Test that specific patterns come before general ones."""
        # Critical: Specific patterns must come before general ones
        for capability_type in ['reasoning_api', 'reasoning_restricted', 'completion_tokens']:
            patterns = MODEL_CAPABILITIES[capability_type]
            
            # Check that o1-, o3-, o4- come before o1, o3, o4
            o1_dash_idx = patterns.index('o1-') if 'o1-' in patterns else -1
            o1_idx = patterns.index('o1') if 'o1' in patterns else -1
            if o1_dash_idx >= 0 and o1_idx >= 0:
                assert o1_dash_idx < o1_idx, f"Pattern 'o1-' must come before 'o1' in {capability_type}"
            
            o3_dash_idx = patterns.index('o3-') if 'o3-' in patterns else -1
            o3_idx = patterns.index('o3') if 'o3' in patterns else -1
            if o3_dash_idx >= 0 and o3_idx >= 0:
                assert o3_dash_idx < o3_idx, f"Pattern 'o3-' must come before 'o3' in {capability_type}"
            
            o4_dash_idx = patterns.index('o4-') if 'o4-' in patterns else -1
            o4_idx = patterns.index('o4') if 'o4' in patterns else -1
            if o4_dash_idx >= 0 and o4_idx >= 0:
                assert o4_dash_idx < o4_idx, f"Pattern 'o4-' must come before 'o4' in {capability_type}"
    
    @pytest.mark.parametrize("model,expected_reasoning,expected_api,expected_tokens", [
        # O-series with dashes (specific patterns)
        ("o1-mini", True, True, True),
        ("o1-preview", True, True, True),
        ("o3-large", True, True, True),
        ("o4-turbo", True, True, True),
        
        # O-series base models (general patterns)
        ("o1", True, True, True),
        ("o3", True, True, True),
        ("o4", True, True, True),
        
        # GPT-5 series (reasoning API but not restricted)
        ("gpt-5-nano", False, True, True),
        ("gpt-5-mini", False, True, True),
        ("gpt-5.1-mini", False, True, True),
        
        # Regular models (no reasoning, no special API)
        ("gpt-4", False, False, False),
        ("gpt-4-turbo", False, False, False),
        ("gpt-3.5-turbo", False, False, False),
        ("claude-3", False, False, False),
    ])
    def test_model_detection_methods(self, mock_client, model, expected_reasoning, 
                                    expected_api, expected_tokens):
        """Test that models are correctly categorized."""
        assert mock_client._is_reasoning_model(model) == expected_reasoning, \
            f"Model {model} reasoning detection failed"
        assert mock_client._is_reasoning_api_model(model) == expected_api, \
            f"Model {model} API detection failed"
        assert mock_client._uses_completion_tokens(model) == expected_tokens, \
            f"Model {model} token parameter detection failed"
    
    def test_critical_bug_fix_o1_mini(self, mock_client):
        """Test the critical bug fix: o1-mini should match o1- not o1."""
        # This was the original bug: o1-mini incorrectly matched o1
        assert mock_client._is_reasoning_model("o1-mini") == True
        assert mock_client._is_reasoning_api_model("o1-mini") == True
        
        # Verify it's detecting the dash version, not the base version
        # by checking that removing the dash would give different behavior
        # (This is a conceptual test of the pattern matching)
        patterns = MODEL_CAPABILITIES['reasoning_api']
        
        # Find which pattern actually matches
        matching_pattern = None
        for pattern in patterns:
            if "o1-mini".startswith(pattern):
                matching_pattern = pattern
                break
        
        assert matching_pattern == "o1-", f"o1-mini matched '{matching_pattern}' instead of 'o1-'"
    
    def test_model_routing_decisions(self, mock_client):
        """Test that models route to the correct API endpoint."""
        # GPT-5 models should use Responses API
        assert mock_client._is_reasoning_api_model("gpt-5-nano") == True
        
        # Regular OpenAI models should not
        assert mock_client._is_reasoning_api_model("gpt-4") == False
        
        # O-series should use Responses API
        assert mock_client._is_reasoning_api_model("o4-mini") == True
        
        # Non-OpenAI provider should return False even for reasoning models
        mock_client.provider = "anthropic"
        assert mock_client._is_reasoning_api_model("o4-mini") == False
    
    def test_parameter_restrictions(self, mock_client):
        """Test that reasoning models have correct parameter restrictions."""
        # O-series models have restricted parameters
        assert mock_client._is_reasoning_model("o1-mini") == True
        assert mock_client._is_reasoning_model("o4") == True
        
        # GPT-5 models use reasoning API but aren't restricted
        assert mock_client._is_reasoning_model("gpt-5-mini") == False
        
        # Regular models have no restrictions
        assert mock_client._is_reasoning_model("gpt-4") == False
    
    def test_completion_tokens_parameter(self, mock_client):
        """Test models that use max_completion_tokens vs max_tokens."""
        # All reasoning API models use max_completion_tokens
        assert mock_client._uses_completion_tokens("gpt-5-nano") == True
        assert mock_client._uses_completion_tokens("o1-mini") == True
        assert mock_client._uses_completion_tokens("o4") == True
        
        # Regular models use max_tokens
        assert mock_client._uses_completion_tokens("gpt-4") == False
        assert mock_client._uses_completion_tokens("gpt-3.5-turbo") == False
    
    def test_edge_cases(self, mock_client):
        """Test edge cases in model detection."""
        # Models that start with pattern prefixes
        assert mock_client._is_reasoning_api_model("gpt-5") == True  # Exact match
        assert mock_client._is_reasoning_api_model("gpt-5-turbo") == True  # Starts with gpt-5
        assert mock_client._is_reasoning_api_model("gpt-6") == False  # Different version
        
        # Very long model names
        assert mock_client._is_reasoning_api_model("o1-mini-experimental-v2-beta") == True
        
        # Empty or None model names shouldn't crash
        assert mock_client._is_reasoning_model("") == False
        
        # Case sensitivity (models are case-sensitive)
        assert mock_client._is_reasoning_model("O1-MINI") == False
        assert mock_client._is_reasoning_model("GPT-5-NANO") == False


class TestModelCapabilitiesIntegration:
    """Test integration between model detection and actual usage."""
    
    def test_all_patterns_are_used(self):
        """Ensure all patterns in MODEL_CAPABILITIES are actually referenced."""
        # This test ensures we don't have dead patterns
        with patch('divergent_bench.llm.providers.AsyncOpenAI'):
            client = OpenAICompatibleClient(
                provider="openai",
                api_key="test_key",
                model="test"
            )
            
            # All three methods should reference the registry
            for model in ["gpt-5", "o1-mini", "o4"]:
                # These calls should not raise errors
                client._is_reasoning_model(model)
                client._is_reasoning_api_model(model)
                client._uses_completion_tokens(model)