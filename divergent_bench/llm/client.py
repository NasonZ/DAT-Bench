"""
LLM Client Protocol and Base Classes

Defines the interface that all LLM clients must implement for AGE.
"""

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None  # For tool messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Standard response format from LLM."""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # For reasoning summaries and other metadata
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)
        
    def to_message(self) -> LLMMessage:
        """Convert response to message for conversation history."""
        return LLMMessage(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls
        )


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients in AGE."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        model: Optional[str] = None,
        output_type: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Conversation history
            model: Override default model
            output_type: Pydantic model for structured output
            tools: Available function definitions
            temperature: Sampling temperature
            max_tokens: Maximum response length
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse for standard responses, or
            instance of output_type for structured outputs
        """
        ...
        
    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Check if this client supports structured outputs."""
        ...
        
    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if this client supports tool calling."""
        ...


class BaseLLMClient(ABC):
    """Base implementation with common functionality."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ):
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.config = kwargs
        
    def _normalize_messages(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]]
    ) -> List[Dict[str, str]]:
        """Convert messages to dict format."""
        normalized = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                normalized.append(msg.to_dict())
            else:
                normalized.append(msg)
        return normalized
        
    @abstractmethod
    async def generate(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """Generate response - must be implemented by subclasses."""
        pass