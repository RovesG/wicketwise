# Purpose: Dynamic Knowledge Graph Query Engine - LLM creates and executes queries
# Author: WicketWise Team, Last Modified: 2025-08-26

import ast
import logging
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class DynamicKGQueryEngine:
    """
    Allows LLM to create and execute dynamic NetworkX queries on the cricket knowledge graph.
    
    Features:
    - Safe execution of NetworkX queries
    - Query validation and sanitization
    - Support for complex graph algorithms
    - Access to both graph structure and node/edge data
    """
    
    def __init__(self, kg_path: str = "models/unified_cricket_kg.pkl"):
        """Initialize with knowledge graph"""
        self.kg_path = Path(kg_path)
        self.kg: Optional[nx.Graph] = None
        self.safe_functions = self._get_safe_functions()
        self._load_kg()
    
    def _load_kg(self):
        """Load the knowledge graph"""
        try:
            with open(self.kg_path, 'rb') as f:
                self.kg = pickle.load(f)
            logger.info(f"Loaded KG for dynamic queries: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to load KG: {e}")
            raise
    
    def _get_safe_functions(self) -> Dict[str, Any]:
        """Get safe NetworkX and utility functions for query execution"""
        return {
            # NetworkX graph functions
            'nodes': lambda: list(self.kg.nodes()),
            'edges': lambda: list(self.kg.edges()),
            'neighbors': lambda node: list(self.kg.neighbors(node)),
            'degree': lambda node=None: dict(self.kg.degree()) if node is None else self.kg.degree(node),
            'shortest_path': lambda source, target: nx.shortest_path(self.kg, source, target),
            'shortest_path_length': lambda source, target: nx.shortest_path_length(self.kg, source, target),
            'connected_components': lambda: list(nx.connected_components(self.kg.to_undirected())),
            'clustering': lambda node=None: nx.clustering(self.kg.to_undirected(), node),
            'pagerank': lambda: nx.pagerank(self.kg),
            'betweenness_centrality': lambda: nx.betweenness_centrality(self.kg),
            'closeness_centrality': lambda: nx.closeness_centrality(self.kg),
            'eigenvector_centrality': lambda: nx.eigenvector_centrality(self.kg),
            
            # Node/edge data access
            'get_node_data': lambda node: dict(self.kg.nodes[node]) if node in self.kg.nodes else {},
            'get_edge_data': lambda source, target: dict(self.kg.edges[source, target]) if self.kg.has_edge(source, target) else {},
            'find_nodes_by_type': lambda node_type: [n for n, d in self.kg.nodes(data=True) if d.get('type') == node_type],
            'find_nodes_by_attribute': lambda attr, value: [n for n, d in self.kg.nodes(data=True) if d.get(attr) == value],
            
            # Filtering and searching
            'filter_nodes': self._filter_nodes,
            'filter_edges': self._filter_edges,
            'search_nodes': self._search_nodes,
            'get_subgraph': self._get_subgraph,
            
            # Aggregation functions
            'count_nodes_by_type': self._count_nodes_by_type,
            'aggregate_node_stats': self._aggregate_node_stats,
            'get_performance_stats': self._get_performance_stats,
            
            # Utility functions
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'sorted': sorted,
            'set': set,
            'list': list,
            'dict': dict,
        }
    
    def _filter_nodes(self, condition_func: str) -> List[str]:
        """Filter nodes based on a condition function"""
        try:
            # Parse and compile the condition function safely
            condition = compile(condition_func, '<string>', 'eval')
            results = []
            
            for node, data in self.kg.nodes(data=True):
                # Create safe evaluation context
                context = {
                    'node': node,
                    'data': data,
                    'type': data.get('type', ''),
                    'name': data.get('name', ''),
                    **{k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
                }
                
                try:
                    if eval(condition, {"__builtins__": {}}, context):
                        results.append(node)
                except:
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error in filter_nodes: {e}")
            return []
    
    def _filter_edges(self, condition_func: str) -> List[tuple]:
        """Filter edges based on a condition function"""
        try:
            condition = compile(condition_func, '<string>', 'eval')
            results = []
            
            for source, target, data in self.kg.edges(data=True):
                context = {
                    'source': source,
                    'target': target,
                    'data': data,
                    **{k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
                }
                
                try:
                    if eval(condition, {"__builtins__": {}}, context):
                        results.append((source, target))
                except:
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error in filter_edges: {e}")
            return []
    
    def _search_nodes(self, search_term: str, attributes: List[str] = None) -> List[str]:
        """Search nodes by text in specified attributes"""
        if attributes is None:
            attributes = ['name', 'id']
        
        search_term = search_term.lower()
        results = []
        
        for node, data in self.kg.nodes(data=True):
            for attr in attributes:
                if attr in data and search_term in str(data[attr]).lower():
                    results.append(node)
                    break
        
        return results
    
    def _get_subgraph(self, nodes: List[str]) -> Dict[str, Any]:
        """Get subgraph containing specified nodes"""
        try:
            subgraph = self.kg.subgraph(nodes)
            return {
                'nodes': list(subgraph.nodes()),
                'edges': list(subgraph.edges()),
                'node_count': subgraph.number_of_nodes(),
                'edge_count': subgraph.number_of_edges(),
                'node_data': dict(subgraph.nodes(data=True)),
                'edge_data': dict(subgraph.edges(data=True))
            }
        except Exception as e:
            logger.error(f"Error creating subgraph: {e}")
            return {}
    
    def _count_nodes_by_type(self) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node, data in self.kg.nodes(data=True):
            node_type = data.get('type', 'unknown')
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _aggregate_node_stats(self, node_type: str, stat_field: str) -> Dict[str, float]:
        """Aggregate statistics for nodes of a specific type"""
        values = []
        for node, data in self.kg.nodes(data=True):
            if data.get('type') == node_type and stat_field in data:
                try:
                    values.append(float(data[stat_field]))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def _get_performance_stats(self, player_node: str) -> Dict[str, Any]:
        """Get performance statistics for a player"""
        if player_node not in self.kg.nodes:
            return {}
        
        data = self.kg.nodes[player_node]
        stats = {}
        
        # Extract numerical performance metrics
        for key, value in data.items():
            if isinstance(value, (int, float)) and any(term in key.lower() for term in 
                ['runs', 'wickets', 'average', 'strike', 'economy', 'balls', 'matches']):
                stats[key] = value
        
        return stats
    
    def execute_query(self, query_code: str, description: str = "") -> Dict[str, Any]:
        """
        Execute a dynamic NetworkX query created by the LLM
        
        Args:
            query_code: Python code string using NetworkX functions
            description: Human-readable description of what the query does
            
        Returns:
            Query results with metadata
        """
        try:
            # Validate query safety
            if not self._is_query_safe(query_code):
                return {
                    'error': 'Query contains unsafe operations',
                    'description': description,
                    'query': query_code
                }
            
            # Create execution context with safe functions
            context = {
                'kg': self.kg,
                'nx': nx,  # NetworkX module
                **self.safe_functions,
                '__builtins__': {}  # Disable built-in functions for security
            }
            
            # Execute the query
            result = eval(query_code, context)
            
            # Format result for JSON serialization
            formatted_result = self._format_result(result)
            
            return {
                'success': True,
                'result': formatted_result,
                'description': description,
                'query': query_code,
                'result_type': type(result).__name__
            }
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                'error': str(e),
                'description': description,
                'query': query_code
            }
    
    def _is_query_safe(self, query_code: str) -> bool:
        """Check if query code is safe to execute"""
        # List of dangerous operations to block (more targeted)
        dangerous_operations = [
            '__import__', 'reload', 'execfile', 'open(', 'file(',
            'subprocess', 'os.', 'sys.', 'globals(', 'locals(',
            'vars(', 'delattr', 'setattr', 'help(', 'quit(', 'exit('
        ]
        
        # Check for dangerous patterns
        for dangerous in dangerous_operations:
            if dangerous in query_code:
                return False
        
        # Block dangerous dunder methods
        if '__' in query_code and not any(safe in query_code for safe in ['__len__', '__str__', '__repr__']):
            return False
        
        # Block direct exec/eval calls (but allow in safe contexts)
        if any(pattern in query_code for pattern in ['exec(', 'compile(']):
            return False
        
        return True
    
    def _format_result(self, result: Any) -> Any:
        """Format result for JSON serialization"""
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        elif isinstance(result, (list, tuple)):
            return [self._format_result(item) for item in result]
        elif isinstance(result, dict):
            return {str(k): self._format_result(v) for k, v in result.items()}
        elif isinstance(result, set):
            return list(result)
        elif hasattr(result, '__dict__'):
            return str(result)
        else:
            return str(result)
    
    def get_query_examples(self) -> List[Dict[str, str]]:
        """Get example queries for the LLM to learn from"""
        return [
            {
                'description': 'Find all players who have scored more than 1000 runs',
                'query': 'filter_nodes("type == \'player\' and data.get(\'total_runs\', 0) > 1000")'
            },
            {
                'description': 'Get the top 5 venues by number of matches',
                'query': 'sorted([(n, degree(n)) for n in find_nodes_by_type("venue")], key=lambda x: x[1], reverse=True)[:5]'
            },
            {
                'description': 'Find players connected to a specific venue',
                'query': 'list(neighbors("venue_wankhede_stadium"))'
            },
            {
                'description': 'Calculate average strike rate for all batsmen',
                'query': 'aggregate_node_stats("player", "strike_rate")'
            },
            {
                'description': 'Find shortest path between two players',
                'query': 'shortest_path("player_virat_kohli", "player_ms_dhoni")'
            },
            {
                'description': 'Get centrality scores for all players',
                'query': '{n: betweenness_centrality()[n] for n in find_nodes_by_type("player")}'
            }
        ]


def get_dynamic_query_function_tool():
    """Get OpenAI function tool for dynamic query execution"""
    return {
        "type": "function",
        "function": {
            "name": "execute_dynamic_kg_query",
            "description": "Execute a custom NetworkX query on the cricket knowledge graph. Create Python code using NetworkX functions to analyze the graph structure and data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_code": {
                        "type": "string",
                        "description": "Python code using NetworkX functions and safe operations. Available functions: nodes(), edges(), neighbors(node), degree(node), shortest_path(source, target), find_nodes_by_type(type), filter_nodes(condition), get_node_data(node), etc."
                    },
                    "description": {
                        "type": "string", 
                        "description": "Human-readable description of what the query does"
                    }
                },
                "required": ["query_code", "description"]
            }
        }
    }
