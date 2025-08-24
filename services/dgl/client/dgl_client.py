# Purpose: DGL client for external system integration
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Client

Provides a client interface for external systems to interact with the DGL service:
- Proposal evaluation requests
- Exposure monitoring
- Rules management
- Audit trail access
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    BetProposal, GovernanceDecision, DecisionType, RuleId,
    ExposureSnapshot, AuditRecord
)


logger = logging.getLogger(__name__)


class DGLClientConfig(BaseModel):
    """Configuration for DGL client"""
    base_url: str = Field(default="http://localhost:8001", description="DGL service base URL")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker pattern")
    circuit_breaker_threshold: int = Field(default=5, description="Failure threshold for circuit breaker")


class DGLClientError(Exception):
    """Base exception for DGL client errors"""
    pass


class DGLServiceUnavailableError(DGLClientError):
    """Raised when DGL service is unavailable"""
    pass


class DGLValidationError(DGLClientError):
    """Raised when request validation fails"""
    pass


class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self.last_failure_time and (datetime.now() - self.last_failure_time).seconds > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise DGLServiceUnavailableError("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.threshold:
                self.state = "OPEN"
            
            raise e


class DGLClient:
    """
    Client for interacting with the DGL service
    
    Provides high-level methods for:
    - Evaluating bet proposals
    - Monitoring exposure
    - Managing rules
    - Accessing audit trails
    """
    
    def __init__(self, config: Optional[DGLClientConfig] = None):
        """
        Initialize DGL client
        
        Args:
            config: Client configuration (uses defaults if None)
        """
        self.config = config or DGLClientConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds
        )
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold
        ) if self.config.enable_circuit_breaker else None
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        logger.info(f"DGL client initialized for {self.config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close the client connection"""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check DGL service health
        
        Returns:
            Health status information
            
        Raises:
            DGLServiceUnavailableError: If service is unavailable
        """
        try:
            response = await self._make_request("GET", "/healthz")
            return response
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise DGLServiceUnavailableError(f"DGL service health check failed: {str(e)}")
    
    async def evaluate_proposal(self, proposal: BetProposal) -> GovernanceDecision:
        """
        Evaluate a single bet proposal
        
        Args:
            proposal: Bet proposal to evaluate
            
        Returns:
            Governance decision
            
        Raises:
            DGLValidationError: If proposal is invalid
            DGLServiceUnavailableError: If service is unavailable
        """
        try:
            proposal_data = proposal.model_dump()
            response = await self._make_request("POST", "/governance/evaluate", json=proposal_data)
            return GovernanceDecision(**response)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise DGLValidationError(f"Invalid proposal: {e.response.text}")
            raise DGLServiceUnavailableError(f"Service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Proposal evaluation failed: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to evaluate proposal: {str(e)}")
    
    async def evaluate_batch_proposals(self, proposals: List[BetProposal]) -> Dict[str, Any]:
        """
        Evaluate multiple proposals in batch
        
        Args:
            proposals: List of proposals to evaluate
            
        Returns:
            Batch evaluation results
            
        Raises:
            DGLValidationError: If any proposal is invalid
            DGLServiceUnavailableError: If service is unavailable
        """
        try:
            batch_data = {
                "proposals": [p.model_dump() for p in proposals],
                "options": {}
            }
            response = await self._make_request("POST", "/governance/evaluate/batch", json=batch_data)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise DGLValidationError(f"Invalid batch request: {e.response.text}")
            raise DGLServiceUnavailableError(f"Service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to evaluate batch: {str(e)}")
    
    async def validate_proposal(self, proposal: BetProposal) -> Dict[str, Any]:
        """
        Validate a proposal without making a decision
        
        Args:
            proposal: Proposal to validate
            
        Returns:
            Validation results
        """
        try:
            proposal_data = proposal.model_dump()
            response = await self._make_request("POST", "/governance/validate", json=proposal_data)
            return response
        except Exception as e:
            logger.error(f"Proposal validation failed: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to validate proposal: {str(e)}")
    
    async def get_current_exposure(self) -> ExposureSnapshot:
        """
        Get current exposure snapshot
        
        Returns:
            Current exposure information
        """
        try:
            response = await self._make_request("GET", "/exposure/current")
            return ExposureSnapshot(**response)
        except Exception as e:
            logger.error(f"Failed to get current exposure: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to get exposure: {str(e)}")
    
    async def get_exposure_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed exposure breakdown
        
        Returns:
            Exposure breakdown by market, match, side
        """
        try:
            response = await self._make_request("GET", "/exposure/breakdown")
            return response
        except Exception as e:
            logger.error(f"Failed to get exposure breakdown: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to get exposure breakdown: {str(e)}")
    
    async def get_exposure_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current exposure alerts
        
        Args:
            severity: Optional severity filter (INFO, WARNING, CRITICAL)
            
        Returns:
            List of active alerts
        """
        try:
            params = {"severity": severity} if severity else {}
            response = await self._make_request("GET", "/exposure/alerts", params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to get exposure alerts: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to get alerts: {str(e)}")
    
    async def get_rules_configuration(self) -> Dict[str, Any]:
        """
        Get current rules configuration
        
        Returns:
            Complete rules configuration
        """
        try:
            response = await self._make_request("GET", "/rules/config")
            return response
        except Exception as e:
            logger.error(f"Failed to get rules configuration: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to get rules config: {str(e)}")
    
    async def test_rules(self, rule_ids: List[RuleId], test_proposal: BetProposal) -> Dict[str, Any]:
        """
        Test specific rules against a proposal
        
        Args:
            rule_ids: Rules to test
            test_proposal: Proposal to test against
            
        Returns:
            Rule test results
        """
        try:
            test_data = {
                "rule_ids": [rule_id.value for rule_id in rule_ids],
                "test_proposal": test_proposal.model_dump(),
                "options": {}
            }
            response = await self._make_request("POST", "/rules/test", json=test_data)
            return response
        except Exception as e:
            logger.error(f"Failed to test rules: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to test rules: {str(e)}")
    
    async def search_audit_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Search audit records
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum records to return
            
        Returns:
            Audit search results
        """
        try:
            params = {"limit": limit}
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self._make_request("GET", "/audit/records", params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to search audit records: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to search audit: {str(e)}")
    
    async def get_compliance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
        try:
            params = {}
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self._make_request("GET", "/audit/compliance/report", params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to get compliance report: {str(e)}")
            raise DGLServiceUnavailableError(f"Failed to get compliance report: {str(e)}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and circuit breaker
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response data
            
        Raises:
            DGLServiceUnavailableError: If request fails after retries
        """
        start_time = datetime.now()
        
        async def _request():
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        
        # Use circuit breaker if enabled
        if self.circuit_breaker:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.circuit_breaker.call, lambda: asyncio.run(_request())
                )
            except DGLServiceUnavailableError:
                raise
            except Exception as e:
                raise DGLServiceUnavailableError(f"Request failed: {str(e)}")
        else:
            # Retry logic without circuit breaker
            last_exception = None
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    result = await _request()
                    break
                except Exception as e:
                    last_exception = e
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                    else:
                        self.error_count += 1
                        raise DGLServiceUnavailableError(f"Request failed after {self.config.max_retries + 1} attempts: {str(e)}")
        
        # Update performance metrics
        response_time = (datetime.now() - start_time).total_seconds()
        self.request_count += 1
        self.total_response_time += response_time
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get client performance statistics
        
        Returns:
            Performance metrics
        """
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        error_rate = (
            (self.error_count / self.request_count) * 100 
            if self.request_count > 0 else 0.0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_pct": error_rate,
            "avg_response_time_seconds": avg_response_time,
            "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "DISABLED",
            "base_url": self.config.base_url
        }
    
    async def ping(self) -> bool:
        """
        Simple connectivity test
        
        Returns:
            True if service is reachable, False otherwise
        """
        try:
            await self.health_check()
            return True
        except:
            return False


# Convenience functions for common operations

async def quick_evaluate(proposal: BetProposal, base_url: str = "http://localhost:8001") -> GovernanceDecision:
    """
    Quick proposal evaluation with default client
    
    Args:
        proposal: Proposal to evaluate
        base_url: DGL service URL
        
    Returns:
        Governance decision
    """
    config = DGLClientConfig(base_url=base_url)
    async with DGLClient(config) as client:
        return await client.evaluate_proposal(proposal)


async def quick_exposure_check(base_url: str = "http://localhost:8001") -> ExposureSnapshot:
    """
    Quick exposure check with default client
    
    Args:
        base_url: DGL service URL
        
    Returns:
        Current exposure
    """
    config = DGLClientConfig(base_url=base_url)
    async with DGLClient(config) as client:
        return await client.get_current_exposure()
