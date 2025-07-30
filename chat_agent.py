# Purpose: Chat agent wrapper for OpenAI API with cricket analysis tools
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import tool registry
from tool_registry import TOOL_REGISTRY, get_tool_function, get_all_tool_schemas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

class ChatAgent:
    """Cricket analysis chat agent using OpenAI API with function calling."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the chat agent.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment
            model: OpenAI model to use (default: gpt-4o)
        """
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE:
            # Get API key from parameter, environment, or session
            if api_key:
                self.api_key = api_key
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")
            
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"ChatAgent initialized with model: {model}")
            else:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        else:
            self.api_key = None
            logger.warning("OpenAI library not available. ChatAgent will use fallback mode.")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for cricket analysis."""
        return """You are a professional cricket analyst AI assistant. You have access to specialized cricket analysis tools:

1. get_form_vector - Analyze player form and performance metrics
2. predict_ball_outcome - Predict cricket ball outcomes with uncertainty
3. query_kg_relationship - Query knowledge graph for player relationships and matchups
4. get_odds_delta - Calculate differences between market and model odds

When users ask about:
- Player form, performance, or statistics → use get_form_vector
- Predictions, outcomes, or "what will happen" → use predict_ball_outcome
- Player matchups, head-to-head records, or comparisons → use query_kg_relationship
- Betting odds, value opportunities, or market analysis → use get_odds_delta

Always provide clear, insightful analysis based on the tool results. If no specific tool is needed, provide general cricket knowledge and guidance."""

    def ask_llm_with_tools(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send user input to OpenAI with tools and handle function calling.
        
        Args:
            user_input: User's question or request
            context: Additional context about match state, etc.
            
        Returns:
            Dictionary containing:
            - answer: Final LLM response
            - tool_used: Name of tool used (if any)
            - tool_result: Result from tool execution (if any)
            - raw_logs: Raw API response logs
            - success: Whether the request was successful
        """
        if context is None:
            context = {}
        
        # Initialize response structure
        response = {
            "answer": "",
            "tool_used": None,
            "tool_result": None,
            "raw_logs": [],
            "success": False,
            "error": None
        }
        
        # Check if OpenAI client is available
        if not self.client:
            return self._fallback_response(user_input, context, response)
        
        try:
            # Get all tool schemas for OpenAI
            tool_schemas = get_all_tool_schemas()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self._format_user_input(user_input, context)}
            ]
            
            # Log the request
            request_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "request",
                "messages": messages,
                "tools": [tool["name"] for tool in tool_schemas],
                "model": self.model
            }
            response["raw_logs"].append(request_log)
            
            # Make OpenAI API call
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto"
            )
            
            # Log the response
            response_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "response",
                "response": completion.model_dump() if hasattr(completion, 'model_dump') else str(completion)
            }
            response["raw_logs"].append(response_log)
            
            # Process the response
            message = completion.choices[0].message
            
            # Check if tools were called
            if message.tool_calls:
                # Handle function calls
                tool_responses = []
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args, context)
                    
                    # Store tool information
                    response["tool_used"] = tool_name
                    response["tool_result"] = tool_result
                    
                    # Add tool response to conversation
                    tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result)
                    })
                
                # Send tool results back to OpenAI for final response
                messages.append(message)
                messages.extend(tool_responses)
                
                final_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                
                response["answer"] = final_completion.choices[0].message.content
                
                # Log final response
                final_log = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "final_response",
                    "response": final_completion.model_dump() if hasattr(final_completion, 'model_dump') else str(final_completion)
                }
                response["raw_logs"].append(final_log)
                
            else:
                # No tools called, use direct response
                response["answer"] = message.content
            
            response["success"] = True
            return response
            
        except Exception as e:
            logger.error(f"Error in ask_llm_with_tools: {str(e)}")
            response["error"] = str(e)
            return self._fallback_response(user_input, context, response)
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool from the registry."""
        try:
            # Get the tool function
            tool_function = get_tool_function(tool_name)
            
            # Add context to tool arguments if not already present
            if "context" not in tool_args and context:
                tool_args["context"] = context
            
            # Execute the tool
            result = tool_function(**tool_args)
            
            # Log tool execution
            logger.info(f"Tool '{tool_name}' executed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name,
                "tool_args": tool_args
            }
    
    def _format_user_input(self, user_input: str, context: Dict[str, Any]) -> str:
        """Format user input with context information."""
        if not context:
            return user_input
        
        context_info = []
        
        # Add match context if available
        if "current_data" in context:
            data = context["current_data"]
            context_info.append(f"Current match: {data.get('batter', 'Unknown')} vs {data.get('bowler', 'Unknown')}")
            context_info.append(f"Over {data.get('over', 0)}, Ball {data.get('ball', 0)}")
        
        # Add match state if available
        if "match_state" in context:
            state = context["match_state"]
            if state.get("progress"):
                context_info.append(f"Match progress: {state['progress']:.1f}%")
        
        # Add ball information if available
        if "ball_number" in context and "total_balls" in context:
            context_info.append(f"Ball {context['ball_number']}/{context['total_balls']}")
        
        if context_info:
            formatted_input = f"User query: {user_input}\n\nMatch context:\n" + "\n".join(f"- {info}" for info in context_info)
            return formatted_input
        
        return user_input
    
    def _fallback_response(self, user_input: str, context: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback response when OpenAI API is not available."""
        # Import the existing chat tools for fallback
        try:
            from chat_tools import handle_chat_query
            fallback_answer = handle_chat_query(user_input, context)
            
            response["answer"] = f"**Fallback Mode - OpenAI API Unavailable**\n\n{fallback_answer}"
            response["tool_used"] = "fallback_handler"
            response["success"] = True
            
            # Log fallback usage
            fallback_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "fallback",
                "reason": "OpenAI API not available",
                "input": user_input,
                "context": context
            }
            response["raw_logs"].append(fallback_log)
            
        except Exception as e:
            response["answer"] = f"I apologize, but I'm unable to process your request right now. Please ensure the OpenAI API is properly configured. Error: {str(e)}"
            response["error"] = str(e)
        
        return response

# Global agent instance
_agent_instance = None

def get_chat_agent(api_key: Optional[str] = None, model: str = "gpt-4o") -> ChatAgent:
    """Get or create a chat agent instance."""
    global _agent_instance
    
    if _agent_instance is None:
        _agent_instance = ChatAgent(api_key=api_key, model=model)
    
    return _agent_instance

def ask_llm_with_tools(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to ask LLM with tools using the global agent."""
    agent = get_chat_agent()
    return agent.ask_llm_with_tools(user_input, context)

def reset_chat_agent():
    """Reset the global chat agent instance."""
    global _agent_instance
    _agent_instance = None

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("=== Chat Agent Demo ===\n")
    
    # Test queries
    test_queries = [
        "What is Virat Kohli's form?",
        "Predict the outcome of the next ball",
        "What is the matchup between Virat Kohli and Pat Cummins?",
        "Calculate odds delta between 1.5 and 1.8 home odds"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Query {i} ---")
        print(f"Question: {query}")
        
        result = ask_llm_with_tools(query)
        
        print(f"Success: {result['success']}")
        print(f"Tool Used: {result['tool_used']}")
        print(f"Answer: {result['answer'][:200]}...")
        
        if result['error']:
            print(f"Error: {result['error']}")
        
        print("\n" + "="*50 + "\n")
