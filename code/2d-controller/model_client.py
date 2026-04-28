"""
model_client.py - Unified client for OpenAI and Anthropic models

This module provides a unified interface for calling both OpenAI (GPT) 
and Anthropic (Claude) models with the same API.
"""

import os
from typing import List, Dict, Optional


class ModelClient:
    """Unified client for OpenAI and Claude models."""
    
    # Model name mappings to actual API model strings
    MODEL_MAP = {
        # OpenAI models
        "gpt-5.2": "gpt-5.2",  # Adjust to actual model name when available
        "gpt-4.1": "gpt-4.1",  # Adjust to actual model name when available
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        
        # Claude models - using exact model strings from Anthropic
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        
        # Legacy Claude 3.5 models
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    }
    
    def __init__(self, model_name: str):
        """
        Initialize the client for a specific model.
        
        Args:
            model_name: Short model name (e.g., "claude-opus-4.5" or "gpt-4o")
        """
        self.model_name = model_name
        self.is_claude = model_name.startswith("claude")
        
        # Get actual API model string
        self.api_model = self.MODEL_MAP.get(model_name, model_name)
        
        # Initialize appropriate client
        if self.is_claude:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set for Claude models")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set for OpenAI models")
            self.client = openai.OpenAI(api_key=api_key)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: Optional[str] = None
    ) -> str:
        """
        Generate a response using the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt (optional, for Claude)
        
        Returns:
            Generated text response
        """
        if self.is_claude:
            return self._generate_claude(messages, max_tokens, temperature, system)
        else:
            return self._generate_openai(messages, max_tokens, temperature, system)
    
    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system: Optional[str]
    ) -> str:
        """Generate using OpenAI API."""
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        response = self.client.chat.completions.create(
            model=self.api_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _generate_claude(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system: Optional[str]
    ) -> str:
        """Generate using Anthropic Claude API."""
        # Claude API separates system prompt from messages
        # and doesn't use 'system' role in messages array
        filtered_messages = [
            msg for msg in messages 
            if msg.get("role") != "system"
        ]
        
        # If no explicit system prompt, extract from messages
        if not system:
            system_msgs = [msg["content"] for msg in messages if msg.get("role") == "system"]
            system = "\n\n".join(system_msgs) if system_msgs else None
        
        kwargs = {
            "model": self.api_model,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            kwargs["system"] = system
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def __str__(self):
        return f"ModelClient({self.model_name} -> {self.api_model})"


# Example usage and testing
if __name__ == "__main__":
    # Test with GPT model
    print("Testing GPT-4o:")
    gpt_client = ModelClient("gpt-4o")
    response = gpt_client.generate(
        messages=[{"role": "user", "content": "Say 'Hello from GPT' in exactly 3 words."}],
        max_tokens=50
    )
    print(f"GPT Response: {response}\n")
    
    # Test with Claude model
    print("Testing Claude Sonnet 4.5:")
    claude_client = ModelClient("claude-sonnet-4.5")
    response = claude_client.generate(
        messages=[{"role": "user", "content": "Say 'Hello from Claude' in exactly 3 words."}],
        max_tokens=50
    )
    print(f"Claude Response: {response}")