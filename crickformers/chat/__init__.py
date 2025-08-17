# Purpose: Knowledge Graph Chat System - Interactive cricket analytics via LLM
# Author: WicketWise Team, Last Modified: 2025-08-16

"""
Knowledge Graph Chat System

This module provides an interactive chat interface that allows users to query
the cricket knowledge graph using natural language. The system uses OpenAI's
function calling to safely execute graph queries and return formatted responses.

Components:
- kg_query_engine: Safe NetworkX query execution with guardrails
- kg_chat_agent: OpenAI LLM integration with function calling
- function_tools: OpenAI function definitions for graph queries
"""

from .kg_query_engine import KGQueryEngine
from .kg_chat_agent import KGChatAgent

__all__ = ['KGQueryEngine', 'KGChatAgent']
