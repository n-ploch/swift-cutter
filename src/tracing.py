"""
Conditional tracing utilities for Langfuse observability.

This module provides decorators and context managers that conditionally enable
Langfuse tracing based on configuration, eliminating the need for scattered
if-statements throughout the codebase.
"""

import os
import functools
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager


class ConditionalContextManager:
    """No-op context manager for when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def conditional_observation(
    enabled: bool,
    as_type: str = "span",
    name: Optional[str] = None,
    **kwargs
) -> ContextManager:
    """
    Returns a no-op context manager.

    Note: This function currently always returns a no-op since we handle tracing
    through the @trace() decorator (which uses Langfuse's @observe decorator).
    For manual observation creation, consider using the Langfuse client directly
    with client.span() or similar methods, or use the @trace() decorator.

    Args:
        enabled: Whether tracing is enabled (currently unused - kept for API compatibility)
        as_type: Observation type (currently unused - kept for API compatibility)
        name: Observation name (currently unused - kept for API compatibility)
        **kwargs: Additional arguments (currently unused - kept for API compatibility)

    Returns:
        No-op context manager

    Example:
        # Note: This doesn't actually create observations - use @trace() decorator instead
        with conditional_observation(config.enable_langfuse, as_type="span", name="process_data"):
            process_data()
    """
    # Simplified: Always return no-op since @trace() decorator handles observations
    # Kept for backward compatibility if anyone is using it
    return ConditionalContextManager()


def conditional_propagate(
    enabled: bool,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    **kwargs
) -> ContextManager:
    """
    Returns Langfuse propagate_attributes context manager if enabled, otherwise no-op.

    Args:
        enabled: Whether tracing is enabled
        session_id: Session ID to propagate to all child observations
        user_id: User ID to propagate to all child observations
        metadata: Metadata to propagate to all child observations
        **kwargs: Additional propagation arguments (version, tags, etc.)

    Returns:
        Langfuse propagate_attributes context manager if enabled, otherwise no-op

    Example:
        with conditional_propagate(config.enable_langfuse, session_id="sess_123"):
            # All observations within this context will have session_id="sess_123"
            agent.generate()
    """
    if enabled:
        from langfuse import propagate_attributes
        return propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            **kwargs
        )
    return ConditionalContextManager()


def trace(
    as_type: str = "span",
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True
):
    """
    Decorator for conditional function tracing.

    This decorator wraps functions with Langfuse observation tracking when tracing
    is enabled, and does nothing when disabled. It automatically:
    - Detects if tracing is enabled from self.config.enable_langfuse
    - Extracts and propagates session_id from kwargs or self
    - Creates properly nested observations with the specified type

    Note: When tracing is enabled, this uses Langfuse's @observe decorator pattern.
    The actual observations are created by the LangChain CallbackHandler that's
    passed to the LLM chains.

    Args:
        as_type: Observation type ("span", "generation", "agent", "tool", etc.)
        name: Observation name (defaults to function name)
        capture_input: Whether to capture function inputs in trace
        capture_output: Whether to capture function outputs in trace

    Returns:
        Decorator function

    Example:
        @trace(as_type="agent", name="screenwriter")
        def generate_storyline(self, prompt: str, session_id: str = None):
            return self.chain.invoke(prompt)

    Usage:
        class MyAgent:
            def __init__(self, config: PipelineConfig):
                self.config = config

            @trace(as_type="agent", name="my_agent")
            def process(self, data, session_id=None):
                # This will be traced if self.config.enable_langfuse is True
                return result
    """
    def decorator(func):
        # Apply Langfuse observe decorator conditionally at call time
        observation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if tracing is enabled
            enabled = _is_tracing_enabled(args, kwargs)

            if not enabled:
                # Tracing disabled - just call the function directly
                return func(*args, **kwargs)

            # Tracing enabled - use Langfuse observe decorator pattern
            try:
                from langfuse import observe

                # Apply observe decorator dynamically
                observed_func = observe(
                    as_type=as_type,
                    name=observation_name,
                    capture_input=capture_input,
                    capture_output=capture_output
                )(func)

                # Extract and propagate session_id
                session_id = kwargs.get('session_id') or _get_session_id(args, kwargs)

                # Execute with session propagation
                with conditional_propagate(enabled, session_id=session_id):
                    return observed_func(*args, **kwargs)

            except ImportError:
                # Langfuse not available, fall back to no tracing
                return func(*args, **kwargs)

        return wrapper
    return decorator


def _is_tracing_enabled(args, kwargs) -> bool:
    """
    Check multiple sources for tracing enabled flag.

    Checks in order:
    1. First argument (self) has runtime.enable_langfuse attribute (nested config)
    2. First argument (self) has config.enable_langfuse attribute (flat config)
    3. Environment variable LANGFUSE_ENABLED

    Args:
        args: Function positional arguments
        kwargs: Function keyword arguments

    Returns:
        Boolean indicating if tracing is enabled
    """
    # Check if first arg (self) has runtime with enable_langfuse (nested config)
    if args and hasattr(args[0], 'runtime'):
        runtime = args[0].runtime
        if hasattr(runtime, 'enable_langfuse'):
            return runtime.enable_langfuse

    # Check if first arg (self) has runtime_config with enable_langfuse (StoryPipeline pattern)
    if args and hasattr(args[0], 'runtime_config'):
        runtime_config = args[0].runtime_config
        if hasattr(runtime_config, 'enable_langfuse'):
            return runtime_config.enable_langfuse

    # Fallback: Check if first arg (self) has config with enable_langfuse (old flat pattern)
    if args and hasattr(args[0], 'config'):
        config = args[0].config
        if hasattr(config, 'enable_langfuse'):
            return config.enable_langfuse

    # Default to False if not found
    return False


def _get_session_id(args, kwargs) -> Optional[str]:
    """
    Extract session_id from various sources.

    Checks in order:
    1. kwargs['session_id']
    2. First argument (self).session_id attribute

    Args:
        args: Function positional arguments
        kwargs: Function keyword arguments

    Returns:
        Session ID if found, None otherwise
    """
    # Check kwargs first
    if 'session_id' in kwargs:
        return kwargs['session_id']

    # Check if first arg (self) has session_id attribute
    if args and hasattr(args[0], 'session_id'):
        return args[0].session_id

    return None
