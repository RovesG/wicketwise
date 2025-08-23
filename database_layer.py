#!/usr/bin/env python3
"""
Optimized Database Layer with Graph Database and Vector Storage
Neo4j for knowledge graphs, Qdrant for embeddings, InfluxDB for time-series

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import numpy as np
from pathlib import Path

# Database clients
try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

# Local imports
from unified_configuration import get_config
from service_container import BaseService

logger = logging.getLogger(__name__)
config = get_config()

# ==================== DATA MODELS ====================

@dataclass
class PlayerNode:
    """Player node in knowledge graph"""
    id: str
    name: str
    role: str
    team: Optional[str] = None
    batting_stats: Dict[str, float] = None
    bowling_stats: Dict[str, float] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class MatchNode:
    """Match node in knowledge graph"""
    id: str
    date: str
    venue: str
    home_team: str
    away_team: str
    competition: str
    total_runs: int = 0
    total_wickets: int = 0
    created_at: datetime = None

@dataclass
class VenueNode:
    """Venue node in knowledge graph"""
    id: str
    name: str
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    matches_played: int = 0

@dataclass
class PlayerEmbedding:
    """Player embedding vector"""
    player_id: str
    embedding: List[float]
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

# ==================== DATABASE INTERFACES ====================

class DatabaseInterface(ABC):
    """Abstract database interface"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from database"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        pass

class GraphDatabaseInterface(DatabaseInterface):
    """Interface for graph database operations"""
    
    @abstractmethod
    async def create_player(self, player: PlayerNode) -> str:
        """Create player node"""
        pass
    
    @abstractmethod
    async def create_match(self, match: MatchNode) -> str:
        """Create match node"""
        pass
    
    @abstractmethod
    async def create_relationship(self, from_id: str, to_id: str, relationship: str, properties: Dict = None) -> None:
        """Create relationship between nodes"""
        pass
    
    @abstractmethod
    async def query_players(self, filters: Dict[str, Any]) -> List[PlayerNode]:
        """Query players with filters"""
        pass

class VectorDatabaseInterface(DatabaseInterface):
    """Interface for vector database operations"""
    
    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create vector collection"""
        pass
    
    @abstractmethod
    async def upsert_embeddings(self, collection_name: str, embeddings: List[PlayerEmbedding]) -> None:
        """Upsert embeddings"""
        pass
    
    @abstractmethod
    async def search_similar(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar vectors"""
        pass

class TimeSeriesDatabaseInterface(DatabaseInterface):
    """Interface for time-series database operations"""
    
    @abstractmethod
    async def write_metrics(self, bucket: str, measurements: List[Dict[str, Any]]) -> None:
        """Write time-series metrics"""
        pass
    
    @abstractmethod
    async def query_metrics(self, bucket: str, query: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Query time-series metrics"""
        pass

# ==================== NEO4J IMPLEMENTATION ====================

class Neo4jGraphDatabase(GraphDatabaseInterface):
    """Neo4j graph database implementation"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    async def connect(self) -> None:
        """Connect to Neo4j"""
        if not NEO4J_AVAILABLE:
            raise RuntimeError("Neo4j client not available. Install with: pip install neo4j")
        
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )
        
        # Test connection
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 as test")
            await result.consume()
        
        logger.info("✅ Connected to Neo4j")
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4j"""
        if self.driver:
            await self.driver.close()
            logger.info("🔌 Disconnected from Neo4j")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j health"""
        if not self.driver:
            return {"status": "disconnected"}
        
        try:
            async with self.driver.session() as session:
                result = await session.run("CALL dbms.components() YIELD name, versions")
                components = await result.data()
                
                # Get database stats
                stats_result = await session.run("""
                    MATCH (n) 
                    RETURN labels(n) as labels, count(n) as count
                """)
                stats = await stats_result.data()
                
                return {
                    "status": "healthy",
                    "components": components,
                    "node_counts": {item['labels'][0] if item['labels'] else 'unlabeled': item['count'] for item in stats}
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def create_player(self, player: PlayerNode) -> str:
        """Create player node in Neo4j"""
        async with self.driver.session() as session:
            query = """
                CREATE (p:Player {
                    id: $id,
                    name: $name,
                    role: $role,
                    team: $team,
                    batting_stats: $batting_stats,
                    bowling_stats: $bowling_stats,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                RETURN p.id as id
            """
            
            result = await session.run(query, {
                "id": player.id,
                "name": player.name,
                "role": player.role,
                "team": player.team,
                "batting_stats": json.dumps(player.batting_stats or {}),
                "bowling_stats": json.dumps(player.bowling_stats or {})
            })
            
            record = await result.single()
            return record["id"]
    
    async def create_match(self, match: MatchNode) -> str:
        """Create match node in Neo4j"""
        async with self.driver.session() as session:
            query = """
                CREATE (m:Match {
                    id: $id,
                    date: $date,
                    venue: $venue,
                    home_team: $home_team,
                    away_team: $away_team,
                    competition: $competition,
                    total_runs: $total_runs,
                    total_wickets: $total_wickets,
                    created_at: datetime()
                })
                RETURN m.id as id
            """
            
            result = await session.run(query, asdict(match))
            record = await result.single()
            return record["id"]
    
    async def create_relationship(self, from_id: str, to_id: str, relationship: str, properties: Dict = None) -> None:
        """Create relationship between nodes"""
        async with self.driver.session() as session:
            query = f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                CREATE (a)-[r:{relationship}]->(b)
                SET r += $properties
                RETURN r
            """
            
            await session.run(query, {
                "from_id": from_id,
                "to_id": to_id,
                "properties": properties or {}
            })
    
    async def query_players(self, filters: Dict[str, Any]) -> List[PlayerNode]:
        """Query players with filters"""
        async with self.driver.session() as session:
            # Build dynamic query
            where_clauses = []
            params = {}
            
            for key, value in filters.items():
                if key == "name_contains":
                    where_clauses.append("p.name CONTAINS $name_contains")
                    params["name_contains"] = value
                elif key == "role":
                    where_clauses.append("p.role = $role")
                    params["role"] = value
                elif key == "team":
                    where_clauses.append("p.team = $team")
                    params["team"] = value
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                MATCH (p:Player)
                WHERE {where_clause}
                RETURN p
                LIMIT 100
            """
            
            result = await session.run(query, params)
            records = await result.data()
            
            players = []
            for record in records:
                player_data = record["p"]
                players.append(PlayerNode(
                    id=player_data["id"],
                    name=player_data["name"],
                    role=player_data["role"],
                    team=player_data.get("team"),
                    batting_stats=json.loads(player_data.get("batting_stats", "{}")),
                    bowling_stats=json.loads(player_data.get("bowling_stats", "{}"))
                ))
            
            return players

# ==================== QDRANT IMPLEMENTATION ====================

class QdrantVectorDatabase(VectorDatabaseInterface):
    """Qdrant vector database implementation"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
    
    async def connect(self) -> None:
        """Connect to Qdrant"""
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.client = QdrantClient(host=self.host, port=self.port)
        
        # Test connection
        collections = self.client.get_collections()
        logger.info(f"✅ Connected to Qdrant ({len(collections.collections)} collections)")
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant"""
        if self.client:
            self.client.close()
            logger.info("🔌 Disconnected from Qdrant")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            collections = self.client.get_collections()
            return {
                "status": "healthy",
                "collections": [c.name for c in collections.collections],
                "collection_count": len(collections.collections)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create vector collection in Qdrant"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"✅ Created Qdrant collection: {collection_name}")
        except Exception as e:
            if "already exists" not in str(e):
                raise
    
    async def upsert_embeddings(self, collection_name: str, embeddings: List[PlayerEmbedding]) -> None:
        """Upsert embeddings to Qdrant"""
        points = []
        
        for i, embedding in enumerate(embeddings):
            points.append(models.PointStruct(
                id=i,
                vector=embedding.embedding,
                payload={
                    "player_id": embedding.player_id,
                    "metadata": embedding.metadata or {},
                    "timestamp": embedding.timestamp.isoformat() if embedding.timestamp else None
                }
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"✅ Upserted {len(embeddings)} embeddings to {collection_name}")
    
    async def search_similar(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar vectors in Qdrant"""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "player_id": result.payload["player_id"],
                "similarity": result.score,
                "metadata": result.payload.get("metadata", {})
            }
            for result in results
        ]

# ==================== INFLUXDB IMPLEMENTATION ====================

class InfluxDBTimeSeriesDatabase(TimeSeriesDatabaseInterface):
    """InfluxDB time-series database implementation"""
    
    def __init__(self, url: str, token: str, org: str):
        self.url = url
        self.token = token
        self.org = org
        self.client = None
        self.write_api = None
        self.query_api = None
    
    async def connect(self) -> None:
        """Connect to InfluxDB"""
        if not INFLUXDB_AVAILABLE:
            raise RuntimeError("InfluxDB client not available. Install with: pip install influxdb-client")
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        # Test connection
        health = self.client.health()
        if health.status != "pass":
            raise RuntimeError(f"InfluxDB health check failed: {health.message}")
        
        logger.info("✅ Connected to InfluxDB")
    
    async def disconnect(self) -> None:
        """Disconnect from InfluxDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Disconnected from InfluxDB")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check InfluxDB health"""
        if not self.client:
            return {"status": "disconnected"}
        
        try:
            health = self.client.health()
            return {
                "status": "healthy" if health.status == "pass" else "unhealthy",
                "message": health.message,
                "version": health.version
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def write_metrics(self, bucket: str, measurements: List[Dict[str, Any]]) -> None:
        """Write time-series metrics to InfluxDB"""
        points = []
        
        for measurement in measurements:
            point = Point(measurement["measurement"])
            
            # Add tags
            for tag_key, tag_value in measurement.get("tags", {}).items():
                point = point.tag(tag_key, tag_value)
            
            # Add fields
            for field_key, field_value in measurement.get("fields", {}).items():
                point = point.field(field_key, field_value)
            
            # Add timestamp
            if "timestamp" in measurement:
                point = point.time(measurement["timestamp"], WritePrecision.NS)
            
            points.append(point)
        
        self.write_api.write(bucket=bucket, record=points)
        logger.info(f"✅ Wrote {len(measurements)} metrics to {bucket}")
    
    async def query_metrics(self, bucket: str, query: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Query time-series metrics from InfluxDB"""
        flux_query = f"""
            from(bucket: "{bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> {query}
        """
        
        result = self.query_api.query(flux_query)
        
        records = []
        for table in result:
            for record in table.records:
                records.append({
                    "time": record.get_time(),
                    "measurement": record.get_measurement(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "tags": {k: v for k, v in record.values.items() if k.startswith("_") == False and k not in ["_time", "_measurement", "_field", "_value"]}
                })
        
        return records

# ==================== DATABASE SERVICE ====================

class DatabaseService(BaseService):
    """Unified database service managing all database connections"""
    
    def __init__(self):
        super().__init__("database_service")
        self.graph_db: Optional[GraphDatabaseInterface] = None
        self.vector_db: Optional[VectorDatabaseInterface] = None
        self.timeseries_db: Optional[TimeSeriesDatabaseInterface] = None
        self.connections_established = 0
    
    async def _start_implementation(self) -> None:
        """Start database service"""
        # Initialize Neo4j (Graph Database)
        neo4j_config = config.performance.database
        if neo4j_config.get("neo4j_enabled", True):
            self.graph_db = Neo4jGraphDatabase(
                uri=neo4j_config.get("neo4j_uri", "bolt://localhost:7687"),
                username=neo4j_config.get("neo4j_username", "neo4j"),
                password=neo4j_config.get("neo4j_password", "password")
            )
            try:
                await self.graph_db.connect()
                self.connections_established += 1
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
        
        # Initialize Qdrant (Vector Database)
        qdrant_config = config.performance.database
        if qdrant_config.get("qdrant_enabled", True):
            self.vector_db = QdrantVectorDatabase(
                host=qdrant_config.get("qdrant_host", "localhost"),
                port=qdrant_config.get("qdrant_port", 6333)
            )
            try:
                await self.vector_db.connect()
                self.connections_established += 1
            except Exception as e:
                logger.warning(f"Qdrant connection failed: {e}")
        
        # Initialize InfluxDB (Time Series)
        influx_config = config.performance.database
        if influx_config.get("influxdb_enabled", True):
            self.timeseries_db = InfluxDBTimeSeriesDatabase(
                url=influx_config.get("influxdb_url", "http://localhost:8086"),
                token=influx_config.get("influxdb_token", "your-token"),
                org=influx_config.get("influxdb_org", "wicketwise")
            )
            try:
                await self.timeseries_db.connect()
                self.connections_established += 1
            except Exception as e:
                logger.warning(f"InfluxDB connection failed: {e}")
        
        logger.info(f"✅ Database service started ({self.connections_established} connections)")
    
    async def _stop_implementation(self) -> None:
        """Stop database service"""
        if self.graph_db:
            await self.graph_db.disconnect()
        
        if self.vector_db:
            await self.vector_db.disconnect()
        
        if self.timeseries_db:
            await self.timeseries_db.disconnect()
        
        logger.info("✅ Database service stopped")
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get database health metrics"""
        health = {
            "connections_established": self.connections_established,
            "graph_db": None,
            "vector_db": None,
            "timeseries_db": None
        }
        
        if self.graph_db:
            health["graph_db"] = await self.graph_db.health_check()
        
        if self.vector_db:
            health["vector_db"] = await self.vector_db.health_check()
        
        if self.timeseries_db:
            health["timeseries_db"] = await self.timeseries_db.health_check()
        
        return health
    
    # Convenience methods
    async def create_player(self, player: PlayerNode) -> str:
        """Create player in graph database"""
        if not self.graph_db:
            raise RuntimeError("Graph database not available")
        return await self.graph_db.create_player(player)
    
    async def search_similar_players(self, embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar players using vector database"""
        if not self.vector_db:
            raise RuntimeError("Vector database not available")
        return await self.vector_db.search_similar("players", embedding, limit)
    
    async def log_performance_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Log performance metric to time-series database"""
        if not self.timeseries_db:
            return  # Silently fail if not available
        
        measurement = {
            "measurement": "performance",
            "tags": {"metric": metric_name, **(tags or {})},
            "fields": {"value": value},
            "timestamp": datetime.utcnow()
        }
        
        await self.timeseries_db.write_metrics("wicketwise", [measurement])

# ==================== MOCK IMPLEMENTATIONS ====================

class MockGraphDatabase(GraphDatabaseInterface):
    """Mock graph database for development"""
    
    def __init__(self):
        self.players = {}
        self.matches = {}
        self.relationships = []
    
    async def connect(self) -> None:
        logger.info("✅ Connected to Mock Graph Database")
    
    async def disconnect(self) -> None:
        logger.info("🔌 Disconnected from Mock Graph Database")
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "players": len(self.players),
            "matches": len(self.matches),
            "relationships": len(self.relationships)
        }
    
    async def create_player(self, player: PlayerNode) -> str:
        self.players[player.id] = player
        return player.id
    
    async def create_match(self, match: MatchNode) -> str:
        self.matches[match.id] = match
        return match.id
    
    async def create_relationship(self, from_id: str, to_id: str, relationship: str, properties: Dict = None) -> None:
        self.relationships.append({
            "from": from_id,
            "to": to_id,
            "type": relationship,
            "properties": properties or {}
        })
    
    async def query_players(self, filters: Dict[str, Any]) -> List[PlayerNode]:
        results = list(self.players.values())
        
        # Apply filters
        if "name_contains" in filters:
            results = [p for p in results if filters["name_contains"].lower() in p.name.lower()]
        
        if "role" in filters:
            results = [p for p in results if p.role == filters["role"]]
        
        return results[:100]  # Limit results

# Example usage
if __name__ == "__main__":
    async def main():
        """Example usage of database layer"""
        
        # Initialize database service
        db_service = DatabaseService()
        
        try:
            await db_service.start()
            
            # Example: Create a player
            player = PlayerNode(
                id="player_001",
                name="Virat Kohli",
                role="batsman",
                team="RCB",
                batting_stats={"average": 37.25, "strike_rate": 131.97}
            )
            
            if db_service.graph_db:
                player_id = await db_service.create_player(player)
                logger.info(f"Created player: {player_id}")
            
            # Example: Log performance metric
            await db_service.log_performance_metric("api_response_time", 0.125, {"endpoint": "players"})
            
            # Check health
            health = await db_service._get_health_metrics()
            logger.info(f"Database health: {health}")
            
        finally:
            await db_service.stop()
    
    # Run example
    asyncio.run(main())
