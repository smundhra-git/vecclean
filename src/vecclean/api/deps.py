"""
FastAPI dependency injection for VecClean.

Provides singleton instances of configuration, pipeline, and other
shared resources with advanced dependency management features including
database connections, health monitoring, metrics collection, and
hot-reloading capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Depends, HTTPException, status, Request

from vecclean.core.config import Config, load_config
from vecclean.core.pipeline import Pipeline
from vecclean.core.types import ConfigurationError


logger = logging.getLogger(__name__)


class DependencyStatus(Enum):
    """Status of a dependency."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class DependencyMetrics:
    """Metrics for dependency usage."""
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    version: str = "1.0.0"
    
    def update_access(self, response_time: float = 0.0, is_error: bool = False):
        """Update access metrics."""
        self.last_access_time = datetime.now()
        self.access_count += 1
        if is_error:
            self.error_count += 1
        
        # Update average response time
        if response_time > 0:
            self.average_response_time = (
                (self.average_response_time * (self.access_count - 1) + response_time) / 
                self.access_count
            )


@dataclass
class DatabaseConnection:
    """Database connection wrapper."""
    connection_string: str
    pool_size: int = 10
    max_overflow: int = 20
    pool: Optional[Any] = None
    status: DependencyStatus = DependencyStatus.UNINITIALIZED
    
    async def connect(self):
        """Establish database connection."""
        try:
            self.status = DependencyStatus.INITIALIZING
            
            # Parse connection string and determine database type
            if self.connection_string.startswith('sqlite'):
                await self._connect_sqlite()
            elif self.connection_string.startswith('postgresql'):
                await self._connect_postgresql()
            else:
                # Default to SQLite for simplicity
                await self._connect_sqlite()
                
            self.status = DependencyStatus.READY
            logger.info(f"Database connected: {self.connection_string}")
        except Exception as e:
            self.status = DependencyStatus.ERROR
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def _connect_sqlite(self):
        """Connect to SQLite database."""
        import aiosqlite
        import sqlite3
        
        # Extract database path from connection string
        db_path = self.connection_string.replace('sqlite:///', '').replace('sqlite://', '')
        if not db_path:
            db_path = ':memory:'
        
        # Create connection pool (simulated for SQLite)
        self.pool = {
            'db_path': db_path,
            'connections': [],
            'max_connections': self.pool_size
        }
        
        # Test connection
        async with aiosqlite.connect(db_path) as db:
            await db.execute("SELECT 1")
            
    async def _connect_postgresql(self):
        """Connect to PostgreSQL database."""
        try:
            import asyncpg
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=self.pool_size,
                max_inactive_connection_lifetime=300
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
        except ImportError:
            logger.warning("asyncpg not installed, falling back to sqlite")
            await self._connect_sqlite()
    
    async def disconnect(self):
        """Close database connection."""
        try:
            self.status = DependencyStatus.SHUTTING_DOWN
            if self.pool:
                if isinstance(self.pool, dict):
                    # SQLite cleanup
                    self.pool.clear()
                else:
                    # PostgreSQL pool cleanup
                    await self.pool.close()
            self.status = DependencyStatus.UNINITIALIZED
            logger.info("Database disconnected")
        except Exception as e:
            logger.error(f"Database disconnection error: {e}")
    
    async def execute_query(self, query: str, params: tuple = None):
        """Execute a database query."""
        if self.status != DependencyStatus.READY:
            raise RuntimeError("Database not connected")
            
        if isinstance(self.pool, dict):
            # SQLite execution
            import aiosqlite
            async with aiosqlite.connect(self.pool['db_path']) as db:
                if params:
                    cursor = await db.execute(query, params)
                else:
                    cursor = await db.execute(query)
                await db.commit()
                return await cursor.fetchall()
        else:
            # PostgreSQL execution
            async with self.pool.acquire() as conn:
                if params:
                    return await conn.fetch(query, *params)
                else:
                    return await conn.fetch(query)


class DependencyManager:
    """Advanced dependency manager with lifecycle and metrics."""
    
    def __init__(self):
        self.dependencies: Dict[str, Any] = {}
        self.metrics: Dict[str, DependencyMetrics] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.shutdown_hooks: List[Callable] = []
        self._lock = threading.RLock()
        self._file_watchers: Dict[str, Any] = {}
        self._contexts: Dict[str, Dict[str, Any]] = {}
    
    def register_dependency(
        self, 
        name: str, 
        factory: Callable, 
        singleton: bool = True,
        health_check: Optional[Callable] = None,
        shutdown_hook: Optional[Callable] = None,
        version: str = "1.0.0"
    ):
        """Register a dependency with lifecycle management."""
        with self._lock:
            self.dependencies[name] = {
                "factory": factory,
                "instance": None,
                "singleton": singleton,
                "status": DependencyStatus.UNINITIALIZED
            }
            self.metrics[name] = DependencyMetrics(version=version)
            
            if health_check:
                self.health_checks[name] = health_check
            
            if shutdown_hook:
                self.shutdown_hooks.append(shutdown_hook)
    
    async def get_dependency(self, name: str, context: Optional[str] = None) -> Any:
        """Get dependency instance with metrics tracking."""
        start_time = time.time()
        error_occurred = False
        
        try:
            with self._lock:
                if name not in self.dependencies:
                    raise ValueError(f"Dependency '{name}' not registered")
                
                dep_info = self.dependencies[name]
                
                # Check context-specific instance
                if context and context in self._contexts:
                    context_key = f"{name}:{context}"
                    if context_key in self._contexts[context]:
                        instance = self._contexts[context][context_key]
                        # Update context access time
                        self._contexts[context]["_last_access_time"] = datetime.now()
                        self._update_metrics(name, start_time, False)
                        return instance
                
                # Return singleton instance if available
                if dep_info["singleton"] and dep_info["instance"] is not None:
                    self._update_metrics(name, start_time, False)
                    return dep_info["instance"]
                
                # Create new instance
                dep_info["status"] = DependencyStatus.INITIALIZING
                instance = await self._create_instance(dep_info["factory"])
                dep_info["status"] = DependencyStatus.READY
                
                if dep_info["singleton"]:
                    dep_info["instance"] = instance
                
                # Store in context if specified
                if context:
                    if context not in self._contexts:
                        self._contexts[context] = {
                            "_created_at": datetime.now(),
                            "_last_access_time": datetime.now()
                        }
                    self._contexts[context][f"{name}:{context}"] = instance
                    self._contexts[context]["_last_access_time"] = datetime.now()
                
                self._update_metrics(name, start_time, False)
                return instance
                
        except Exception as e:
            error_occurred = True
            if name in self.dependencies:
                self.dependencies[name]["status"] = DependencyStatus.ERROR
            self._update_metrics(name, start_time, True)
            logger.error(f"Failed to get dependency '{name}': {e}")
            raise
    
    def _update_metrics(self, name: str, start_time: float, is_error: bool):
        """Update dependency metrics."""
        if name in self.metrics:
            response_time = time.time() - start_time
            self.metrics[name].update_access(response_time, is_error)
    
    async def _create_instance(self, factory: Callable) -> Any:
        """Create dependency instance."""
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        else:
            return factory()
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Run health checks for all dependencies."""
        health_status = {}
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                health_status[name] = {"status": "healthy", "details": result}
            except Exception as e:
                health_status[name] = {"status": "unhealthy", "error": str(e)}
        
        return health_status
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all dependencies."""
        metrics_data = {}
        for name, metrics in self.metrics.items():
            metrics_data[name] = {
                "creation_time": metrics.creation_time.isoformat(),
                "last_access_time": metrics.last_access_time.isoformat(),
                "access_count": metrics.access_count,
                "error_count": metrics.error_count,
                "error_rate": metrics.error_count / max(metrics.access_count, 1),
                "average_response_time": metrics.average_response_time,
                "version": metrics.version,
                "uptime": (datetime.now() - metrics.creation_time).total_seconds()
            }
        return metrics_data
    
    async def shutdown_all(self):
        """Gracefully shutdown all dependencies."""
        logger.info("Starting graceful dependency shutdown")
        
        # Run shutdown hooks
        for hook in self.shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(f"Shutdown hook failed: {e}")
        
        # Clear instances
        with self._lock:
            for name, dep_info in self.dependencies.items():
                dep_info["instance"] = None
                dep_info["status"] = DependencyStatus.UNINITIALIZED
            self._contexts.clear()
        
        logger.info("Dependency shutdown completed")
    
    def watch_config_file(self, file_path: Path, callback: Callable):
        """Watch configuration file for changes."""
        try:
            import watchdog.observers
            import watchdog.events
            
            class ConfigHandler(watchdog.events.FileSystemEventHandler):
                def on_modified(self, event):
                    if not event.is_directory and Path(event.src_path) == file_path:
                        asyncio.create_task(callback())
            
            observer = watchdog.observers.Observer()
            observer.schedule(ConfigHandler(), str(file_path.parent), recursive=False)
            observer.start()
            
            self._file_watchers[str(file_path)] = observer
            logger.info(f"Started watching config file: {file_path}")
            
        except ImportError:
            logger.warning("Watchdog not installed, file watching disabled")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")


# Global dependency manager
dependency_manager = DependencyManager()

# Global instances for singleton pattern (legacy support)
_config_instance: Optional[Config] = None
_pipeline_instance: Optional[Pipeline] = None
_database_connections: Dict[str, DatabaseConnection] = {}


@lru_cache()
def get_config_path() -> Optional[Path]:
    """
    Get the configuration file path.
    
    Checks environment variables and default locations.
    """
    import os
    
    # Check environment variable first
    config_path_env = os.getenv("VECCLEAN_CONFIG_PATH")
    if config_path_env:
        path = Path(config_path_env)
        if path.exists():
            return path
        else:
            logger.warning(f"Config path from environment not found: {path}")
    
    # Check default locations
    default_paths = [
        Path("configs/default.yaml"),
        Path("/etc/vecclean/config.yaml"),
        Path.home() / ".vecclean" / "config.yaml",
    ]
    
    for path in default_paths:
        if path.exists():
            return path
    
    logger.info("No configuration file found, using defaults")
    return None


# Enhanced dependency factories
async def create_config() -> Config:
    """Factory function for creating configuration."""
    config_path = get_config_path()
    config = load_config(config_path)
    logger.info(f"Configuration created from {config_path or 'defaults'}")
    return config


async def create_pipeline() -> Pipeline:
    """Factory function for creating pipeline."""
    config = await dependency_manager.get_dependency("config")
    pipeline = Pipeline(config)
    logger.info("Pipeline created successfully")
    return pipeline


async def create_database_connection(connection_name: str = "default") -> DatabaseConnection:
    """Factory function for creating database connection."""
    import os
    
    # Get connection string from environment or config
    conn_string = os.getenv(f"DATABASE_URL_{connection_name.upper()}", 
                           os.getenv("DATABASE_URL", "sqlite:///vecclean.db"))
    
    db_conn = DatabaseConnection(connection_string=conn_string)
    await db_conn.connect()
    return db_conn


async def create_embedding_pipeline() -> Pipeline:
    """Factory function for creating embedding-focused pipeline."""
    config = await dependency_manager.get_dependency("config", context="embedding")
    
    # Customize config for embedding-only pipeline
    embedding_config = config.model_copy(deep=True)
    embedding_config.chunking.chunk_size = min(embedding_config.chunking.chunk_size, 512)
    embedding_config.embedding.batch_size = min(embedding_config.embedding.batch_size, 64)
    embedding_config.cleaning.remove_boilerplate = False  # Keep more text for embeddings
    embedding_config.dedup.sentence_dedup = False  # Less aggressive dedup for embeddings
    
    pipeline = Pipeline(embedding_config)
    logger.info("Embedding-focused pipeline created successfully")
    return pipeline


# Health check functions
async def config_health_check() -> Dict[str, Any]:
    """Health check for configuration."""
    try:
        config = await dependency_manager.get_dependency("config")
        return {
            "version": getattr(config, 'version', '1.0.0'),
            "loaded_from": str(get_config_path() or 'defaults'),
            "status": "healthy"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def pipeline_health_check() -> Dict[str, Any]:
    """Health check for pipeline."""
    try:
        pipeline = await dependency_manager.get_dependency("pipeline")
        
        # Test pipeline components
        test_text = "This is a test."
        test_result = await pipeline.process_text(test_text, "health_check")
        
        return {
            "status": "healthy",
            "components": {
                "text_processor": "available",
                "embedder": "available" if pipeline._embedding_model else "not_loaded",
                "chunker": "available",
                "normalizer": "available"
            },
            "test_processing": "success" if test_result.status.value == "completed" else "failed"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def database_health_check() -> Dict[str, Any]:
    """Health check for database connections."""
    try:
        health_status = {}
        for name, db_conn in _database_connections.items():
            health_status[name] = {
                "status": db_conn.status.value,
                "connection_string": db_conn.connection_string.split('@')[-1],  # Hide credentials
                "pool_size": db_conn.pool_size
            }
        return health_status
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Shutdown hooks
async def config_shutdown():
    """Shutdown hook for configuration."""
    global _config_instance
    _config_instance = None
    get_config_path.cache_clear()
    logger.info("Configuration cleared")


async def pipeline_shutdown():
    """Shutdown hook for pipeline."""
    global _pipeline_instance
    _pipeline_instance = None
    logger.info("Pipeline cleared")


async def database_shutdown():
    """Shutdown hook for database connections."""
    for name, db_conn in _database_connections.items():
        await db_conn.disconnect()
    _database_connections.clear()
    logger.info("Database connections closed")


# Register dependencies with the manager
def initialize_dependencies():
    """Initialize all dependencies with the dependency manager."""
    
    # Register core dependencies
    dependency_manager.register_dependency(
        name="config",
        factory=create_config,
        health_check=config_health_check,
        shutdown_hook=config_shutdown,
        version="1.0.0"
    )
    
    dependency_manager.register_dependency(
        name="pipeline",
        factory=create_pipeline,
        health_check=pipeline_health_check,
        shutdown_hook=pipeline_shutdown,
        version="1.0.0"
    )
    
    dependency_manager.register_dependency(
        name="embedding_pipeline",
        factory=create_embedding_pipeline,
        health_check=pipeline_health_check,
        shutdown_hook=pipeline_shutdown,
        version="1.0.0"
    )
    
    # Register database connection
    dependency_manager.register_dependency(
        name="database",
        factory=lambda: create_database_connection("default"),
        health_check=database_health_check,
        shutdown_hook=database_shutdown,
        version="1.0.0"
    )
    
    # Set up config file watching for hot-reloading
    config_path = get_config_path()
    if config_path:
        dependency_manager.watch_config_file(config_path, reload_dependencies)


async def reload_dependencies():
    """Reload dependencies when configuration changes."""
    logger.info("Configuration file changed, reloading dependencies")
    
    # Clear config cache
    get_config_path.cache_clear()
    
    # Reset specific dependencies that depend on config
    with dependency_manager._lock:
        for name in ["config", "pipeline", "embedding_pipeline"]:
            if name in dependency_manager.dependencies:
                dependency_manager.dependencies[name]["instance"] = None
                dependency_manager.dependencies[name]["status"] = DependencyStatus.UNINITIALIZED
    
    logger.info("Dependencies marked for reload")


async def get_config() -> Config:
    """
    Get the application configuration with enhanced dependency management.
    
    Uses the dependency manager for metrics, health monitoring, and hot-reloading.
    
    Returns:
        Configuration instance
        
    Raises:
        HTTPException: If configuration loading fails
    """
    try:
        return await dependency_manager.get_dependency("config")
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {e}"
        )


async def get_pipeline(context: Optional[str] = None) -> Pipeline:
    """
    Get the processing pipeline with context-aware dependency injection.
    
    Args:
        context: Optional context for pipeline specialization
        
    Returns:
        Pipeline instance
        
    Raises:
        HTTPException: If pipeline initialization fails
    """
    try:
        pipeline_name = "embedding_pipeline" if context == "embedding" else "pipeline"
        return await dependency_manager.get_dependency(pipeline_name, context)
    except Exception as e:
        logger.error(f"Failed to get pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize processing pipeline"
        )


async def get_database(connection_name: str = "default") -> DatabaseConnection:
    """
    Get database connection with connection pooling.
    
    Args:
        connection_name: Name of the database connection
        
    Returns:
        Database connection instance
        
    Raises:
        HTTPException: If database connection fails
    """
    try:
        if connection_name not in _database_connections:
            db_conn = await create_database_connection(connection_name)
            _database_connections[connection_name] = db_conn
        
        return _database_connections[connection_name]
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )


async def get_pipeline_with_context(request: Request) -> Pipeline:
    """
    Get pipeline with context extracted from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Context-appropriate pipeline instance
    """
    # Extract context from request headers, path, or query params
    context = request.headers.get("X-Processing-Context")
    if not context:
        context = request.query_params.get("context")
    
    return await get_pipeline(context)


def reload_config() -> None:
    """
    Reload configuration and pipeline.
    
    Forces reloading of configuration and recreation of pipeline.
    Useful for development and configuration updates.
    """
    global _config_instance, _pipeline_instance
    
    logger.info("Reloading configuration and pipeline")
    
    # Clear cached instances
    _config_instance = None
    _pipeline_instance = None
    
    # Clear LRU cache
    get_config_path.cache_clear()
    
    logger.info("Configuration and pipeline cleared for reload")


async def get_embedding_model(pipeline: Pipeline = Depends(get_pipeline)):
    """
    Get the embedding model instance from the pipeline.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        Embedding model instance or None if not loaded
    """
    if hasattr(pipeline, '_embedding_model') and pipeline._embedding_model:
        return pipeline._embedding_model
    
    # Initialize embedding model if not already loaded
    from vecclean.core.embedding import LocalSentenceTransformerEmbedding
    config = pipeline.config
    
    embedding_model = LocalSentenceTransformerEmbedding(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_embeddings=config.embedding.cache_embeddings
    )
    
    # Cache in pipeline for reuse
    pipeline._embedding_model = embedding_model
    return embedding_model


async def validate_request_size(content_length: Optional[int] = None) -> None:
    """
    Validate request size limits.
    
    Args:
        content_length: Content length from request headers
        
    Raises:
        HTTPException: If request is too large
    """
    config = await get_config()
    max_size = config.processing.max_file_size_mb * 1024 * 1024
    
    if content_length and content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request too large. Maximum size: {max_size // 1024 // 1024}MB"
        )


async def validate_batch_size(file_count: int) -> None:
    """
    Validate batch processing limits.
    
    Args:
        file_count: Number of files in batch
        
    Raises:
        HTTPException: If batch is too large
    """
    config = await get_config()
    max_batch = config.processing.batch_size
    
    if file_count > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large. Maximum files: {max_batch}"
        )


class RateLimiter:
    """
    Enhanced rate limiter with Redis support and in-memory fallback.
    """
    
    def __init__(self, requests_per_minute: int = 60, redis_url: Optional[str] = None):
        self.requests_per_minute = requests_per_minute
        self.redis_client = None
        self.in_memory_requests = {}
        
        # Try to connect to Redis
        if redis_url:
            try:
                import aioredis
                self.redis_client = aioredis.from_url(redis_url)
                logger.info("Connected to Redis for rate limiting")
            except ImportError:
                logger.warning("aioredis not installed, using in-memory rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using in-memory fallback")
    
    async def check_rate_limit(self, client_id: str) -> None:
        """
        Check if client has exceeded rate limit using Redis or in-memory storage.
        """
        if self.redis_client:
            await self._check_rate_limit_redis(client_id)
        else:
            await self._check_rate_limit_memory(client_id)
    
    async def _check_rate_limit_redis(self, client_id: str) -> None:
        """Check rate limit using Redis."""
        import time
        
        current_time = int(time.time())
        window_start = current_time - 60  # 1 minute window
        
        # Use Redis sorted set to track requests
        key = f"rate_limit:{client_id}"
        
        # Remove old entries
        await self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_count = await self.redis_client.zcard(key)
        
        if current_count >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        await self.redis_client.zadd(key, {str(current_time): current_time})
        await self.redis_client.expire(key, 60)
    
    async def _check_rate_limit_memory(self, client_id: str) -> None:
        """Check rate limit using in-memory storage."""
        import time
        
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        # Clean old entries
        self.in_memory_requests = {
            key: value for key, value in self.in_memory_requests.items()
            if key[1] >= minute_window - 1
        }
        
        # Count requests in current window
        key = (client_id, minute_window)
        current_count = self.in_memory_requests.get(key, 0)
        
        if current_count >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Increment counter
        self.in_memory_requests[key] = current_count + 1


class AuthManager:
    """
    Comprehensive authentication and authorization manager with JWT and API keys.
    """
    
    def __init__(self):
        self.api_keys = {}  # Will be loaded from configuration
        self.jwt_secret = None
        self.jwt_algorithm = "HS256"
        self._load_auth_config()
    
    def _load_auth_config(self):
        """Load authentication configuration from environment and config."""
        import os
        
        # Load JWT secret
        self.jwt_secret = os.getenv("VECCLEAN_JWT_SECRET", "default-secret-change-in-production")
        
        # Load API keys from environment or file
        api_keys_str = os.getenv("VECCLEAN_API_KEYS", "")
        if api_keys_str:
            # Format: key1:admin,key2:user,key3:readonly
            for key_pair in api_keys_str.split(","):
                if ":" in key_pair:
                    key, role = key_pair.strip().split(":", 1)
                    self.api_keys[key] = {
                        "role": role,
                        "permissions": self._get_role_permissions(role),
                        "created_at": time.time()
                    }
        
        # Add default API key for development
        if not self.api_keys:
            self.api_keys["dev-key-12345"] = {
                "role": "admin",
                "permissions": {"read": True, "write": True, "admin": True},
                "created_at": time.time()
            }
            logger.warning("Using default API key for development. Change in production!")
    
    def _get_role_permissions(self, role: str) -> Dict[str, bool]:
        """Get permissions for a role."""
        role_permissions = {
            "admin": {"read": True, "write": True, "admin": True, "delete": True},
            "user": {"read": True, "write": True, "admin": False, "delete": False},
            "readonly": {"read": True, "write": False, "admin": False, "delete": False},
            "guest": {"read": True, "write": False, "admin": False, "delete": False}
        }
        return role_permissions.get(role, role_permissions["guest"])
    
    async def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return user info."""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            return {
                "valid": True,
                "role": key_info["role"],
                "permissions": key_info["permissions"],
                "auth_method": "api_key"
            }
        return {"valid": False, "error": "Invalid API key"}
    
    async def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info."""
        try:
            import jwt
            
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return {"valid": False, "error": "Token expired"}
            
            return {
                "valid": True,
                "user_id": payload.get("user_id"),
                "role": payload.get("role", "user"),
                "permissions": self._get_role_permissions(payload.get("role", "user")),
                "auth_method": "jwt"
            }
        except ImportError:
            return {"valid": False, "error": "JWT library not available"}
        except Exception as e:
            return {"valid": False, "error": f"Invalid token: {str(e)}"}
    
    async def create_jwt_token(self, user_id: str, role: str = "user", expires_in: int = 3600) -> str:
        """Create a JWT token."""
        try:
            import jwt
            
            payload = {
                "user_id": user_id,
                "role": role,
                "iat": time.time(),
                "exp": time.time() + expires_in
            }
            
            return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        except ImportError:
            raise RuntimeError("JWT library not available")
    
    async def get_user_permissions(self, api_key: str = None, jwt_token: str = None) -> Dict[str, Any]:
        """Get user permissions from API key or JWT token."""
        if api_key:
            return await self.verify_api_key(api_key)
        elif jwt_token:
            return await self.verify_jwt_token(jwt_token)
        else:
            return {
                "valid": False,
                "permissions": {"read": False, "write": False, "admin": False},
                "auth_method": "none"
            }


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(client_ip: str = None) -> None:
    """
    Rate limiting dependency.
    
    Args:
        client_ip: Client IP address for rate limiting
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    if client_ip:
        await _rate_limiter.check_rate_limit(client_ip)


# Global auth manager instance
_auth_manager = AuthManager()


async def verify_authentication(
    api_key: Optional[str] = None, 
    jwt_token: Optional[str] = None,
    request: Optional[Request] = None
) -> dict:
    """
    Enhanced authentication dependency supporting multiple methods.
    
    Args:
        api_key: Optional API key from request
        jwt_token: Optional JWT token from request
        request: FastAPI request object for extracting auth info
        
    Returns:
        User permissions and authentication info
        
    Raises:
        HTTPException: If authentication fails
    """
    # Extract authentication from request headers if not provided
    if request and not api_key and not jwt_token:
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            jwt_token = auth_header.split(" ", 1)[1]
    
    # Try JWT first, then API key
    if jwt_token:
        result = await _auth_manager.verify_jwt_token(jwt_token)
        if result["valid"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result["error"]
            )
    
    if api_key:
        result = await _auth_manager.verify_api_key(api_key)
        if result["valid"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    
    # Allow unauthenticated access for development (configurable)
    import os
    if os.getenv("VECCLEAN_ALLOW_ANONYMOUS", "false").lower() == "true":
        return {
            "valid": True,
            "role": "guest",
            "permissions": {"read": True, "write": False, "admin": False},
            "auth_method": "anonymous"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


async def require_admin_access(auth_info: dict = Depends(verify_authentication)) -> None:
    """
    Dependency that requires admin access.
    
    Args:
        auth_info: Authentication info from verify_authentication
        
    Raises:
        HTTPException: If user doesn't have admin access
    """
    if not auth_info.get("permissions", {}).get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )


async def require_write_access(auth_info: dict = Depends(verify_authentication)) -> None:
    """
    Dependency that requires write access.
    """
    if not auth_info.get("permissions", {}).get("write", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write access required"
        )


# Enhanced health check dependencies
async def check_system_health() -> dict:
    """
    Comprehensive system health check with detailed component status.
    """
    checks = {}
    
    try:
        # Check configuration
        config = await get_config()
        checks["config"] = {
            "status": "ok",
            "chunking_strategy": config.chunking.strategy,
            "embedding_model": config.embedding.model_name
        }
    except Exception as e:
        checks["config"] = {"status": "error", "error": str(e)}
    
    try:
        # Check pipeline initialization
        pipeline = await get_pipeline()
        test_result = await pipeline.process_text("test", "health_check")
        checks["pipeline"] = {
            "status": "ok",
            "test_processing": "success" if test_result.status.value == "completed" else "failed",
            "cache_info": pipeline.get_cache_info()
        }
    except Exception as e:
        checks["pipeline"] = {"status": "error", "error": str(e)}
    
    # Check database connectivity
    try:
        db_conn = await get_database()
        await db_conn.execute_query("SELECT 1")
        checks["database"] = {"status": "ok", "connection_status": db_conn.status.value}
    except Exception as e:
        checks["database"] = {"status": "error", "error": str(e)}
    
    # Check authentication system
    try:
        test_auth = await _auth_manager.verify_api_key("invalid-key")
        checks["authentication"] = {
            "status": "ok",
            "api_keys_loaded": len(_auth_manager.api_keys),
            "jwt_enabled": _auth_manager.jwt_secret is not None
        }
    except Exception as e:
        checks["authentication"] = {"status": "error", "error": str(e)}
    
    # Check rate limiting
    try:
        await _rate_limiter._check_rate_limit_memory("health_check")
        checks["rate_limiting"] = {
            "status": "ok",
            "redis_available": _rate_limiter.redis_client is not None,
            "requests_per_minute": _rate_limiter.requests_per_minute
        }
    except Exception as e:
        checks["rate_limiting"] = {"status": "error", "error": str(e)}
    
    return checks


# Utility functions for dependency management

def reset_dependencies() -> None:
    """
    Reset all dependency instances.
    
    Useful for testing and development.
    """
    global _config_instance, _pipeline_instance
    
    _config_instance = None
    _pipeline_instance = None
    get_config_path.cache_clear()
    
    logger.info("All dependencies reset")


def get_dependency_info() -> dict:
    """
    Get information about current dependency instances.
    
    Returns:
        Dependency status information
    """
    return {
        "config_loaded": _config_instance is not None,
        "pipeline_initialized": _pipeline_instance is not None,
        "config_cache_info": get_config_path.cache_info()._asdict(),
    }


# Enhanced dependency management functions

async def get_dependency_metrics() -> Dict[str, Any]:
    """
    Get comprehensive metrics for all dependencies.
    
    Returns:
        Metrics data for monitoring and debugging
    """
    return {
        "metrics": dependency_manager.get_metrics(),
        "health": await dependency_manager.health_check_all(),
        "system": {
            "active_dependencies": len(dependency_manager.dependencies),
            "total_contexts": len(dependency_manager._contexts),
            "file_watchers": len(dependency_manager._file_watchers)
        }
    }


async def get_dependency_status() -> Dict[str, Any]:
    """
    Get current status of all dependencies.
    
    Returns:
        Status information for each dependency
    """
    status_info = {}
    
    for name, dep_info in dependency_manager.dependencies.items():
        metrics = dependency_manager.metrics.get(name)
        status_info[name] = {
            "status": dep_info["status"].value,
            "singleton": dep_info["singleton"],
            "has_instance": dep_info["instance"] is not None,
            "access_count": metrics.access_count if metrics else 0,
            "error_count": metrics.error_count if metrics else 0,
            "version": metrics.version if metrics else "unknown",
            "uptime": (datetime.now() - metrics.creation_time).total_seconds() if metrics else 0
        }
    
    return status_info


async def reload_specific_dependency(dependency_name: str) -> Dict[str, Any]:
    """
    Reload a specific dependency.
    
    Args:
        dependency_name: Name of dependency to reload
        
    Returns:
        Reload result information
    """
    try:
        with dependency_manager._lock:
            if dependency_name not in dependency_manager.dependencies:
                raise ValueError(f"Dependency '{dependency_name}' not found")
            
            # Clear instance and mark for reload
            dep_info = dependency_manager.dependencies[dependency_name]
            old_instance = dep_info["instance"]
            dep_info["instance"] = None
            dep_info["status"] = DependencyStatus.UNINITIALIZED
            
            # Clear from all contexts
            for context_deps in dependency_manager._contexts.values():
                context_key = f"{dependency_name}:{context_deps}"
                if context_key in context_deps:
                    del context_deps[context_key]
        
        logger.info(f"Dependency '{dependency_name}' marked for reload")
        
        return {
            "status": "success",
            "dependency": dependency_name,
            "message": f"Dependency '{dependency_name}' will be reloaded on next access",
            "previous_instance": old_instance is not None
        }
        
    except Exception as e:
        logger.error(f"Failed to reload dependency '{dependency_name}': {e}")
        return {
            "status": "error",
            "dependency": dependency_name,
            "error": str(e)
        }


async def create_dependency_context(context_name: str, config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new dependency context with configuration overrides.
    
    Args:
        context_name: Name for the new context
        config_overrides: Configuration overrides for this context
        
    Returns:
        Context creation result
    """
    try:
        if context_name not in dependency_manager._contexts:
            dependency_manager._contexts[context_name] = {}
        
        # Apply configuration overrides to context-specific instances
        if config_overrides:
            logger.info(f"Configuration overrides for context '{context_name}': {config_overrides}")
            
            # Store override configuration for this context
            base_config = await dependency_manager.get_dependency("config")
            context_config = base_config.model_copy(deep=True)
            
            # Apply overrides using nested key access
            for key, value in config_overrides.items():
                if "." in key:
                    # Handle nested keys like "chunking.chunk_size"
                    keys = key.split(".")
                    target = context_config
                    for k in keys[:-1]:
                        target = getattr(target, k)
                    setattr(target, keys[-1], value)
                else:
                    # Handle top-level keys
                    if hasattr(context_config, key):
                        setattr(context_config, key, value)
            
            # Create context-specific pipeline with overridden config
            context_pipeline = Pipeline(context_config)
            dependency_manager._contexts[context_name][f"pipeline:{context_name}"] = context_pipeline
            dependency_manager._contexts[context_name]["_config_overrides"] = config_overrides
            dependency_manager._contexts[context_name]["_created_at"] = datetime.now()
        
        return {
            "status": "success",
            "context": context_name,
            "message": f"Context '{context_name}' created successfully",
            "overrides": config_overrides or {}
        }
        
    except Exception as e:
        logger.error(f"Failed to create context '{context_name}': {e}")
        return {
            "status": "error",
            "context": context_name,
            "error": str(e)
        }


async def cleanup_unused_contexts() -> Dict[str, Any]:
    """
    Clean up unused dependency contexts to free memory.
    
    Returns:
        Cleanup operation results
    """
    try:
        cleaned_contexts = 0
        total_instances = 0
        
        with dependency_manager._lock:
            # Identify contexts that haven't been accessed recently
            cutoff_time = datetime.now() - timedelta(hours=1)  # Cleanup after 1 hour
            
            contexts_to_remove = []
            for context_name, context_deps in dependency_manager._contexts.items():
                # Check context last access time
                last_access = context_deps.get("_last_access_time", datetime.now())
                context_age = last_access < cutoff_time
                
                if context_age and context_name != "default":  # Don't remove default context
                    total_instances += len(context_deps)
                    contexts_to_remove.append(context_name)
            
            # Remove old contexts
            for context_name in contexts_to_remove:
                del dependency_manager._contexts[context_name]
                cleaned_contexts += 1
                logger.info(f"Cleaned up unused context: {context_name}")
        
        return {
            "status": "success",
            "cleaned_contexts": cleaned_contexts,
            "freed_instances": total_instances,
            "remaining_contexts": len(dependency_manager._contexts)
        }
        
    except Exception as e:
        logger.error(f"Context cleanup failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def export_dependency_config() -> Dict[str, Any]:
    """
    Export current dependency configuration for backup or analysis.
    
    Returns:
        Serializable dependency configuration
    """
    try:
        config_export = {
            "dependencies": {},
            "metrics": dependency_manager.get_metrics(),
            "contexts": list(dependency_manager._contexts.keys()),
            "export_time": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        for name, dep_info in dependency_manager.dependencies.items():
            config_export["dependencies"][name] = {
                "singleton": dep_info["singleton"],
                "status": dep_info["status"].value,
                "has_health_check": name in dependency_manager.health_checks,
                "has_shutdown_hook": any(True for hook in dependency_manager.shutdown_hooks)  # Simplified
            }
        
        return config_export
        
    except Exception as e:
        logger.error(f"Failed to export dependency config: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Enhanced system lifecycle management
async def startup_dependencies():
    """
    Initialize all dependencies during application startup.
    """
    logger.info("Starting dependency initialization")
    
    try:
        # Initialize the dependency manager
        initialize_dependencies()
        
        # Pre-warm critical dependencies
        await dependency_manager.get_dependency("config")
        await dependency_manager.get_dependency("pipeline")
        
        logger.info("Dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"Dependency startup failed: {e}")
        raise


async def shutdown_dependencies():
    """
    Gracefully shutdown all dependencies during application shutdown.
    """
    logger.info("Starting dependency shutdown")
    
    try:
        await dependency_manager.shutdown_all()
        
        # Stop file watchers
        for watcher in dependency_manager._file_watchers.values():
            if hasattr(watcher, 'stop'):
                watcher.stop()
        
        logger.info("Dependencies shutdown completed")
        
    except Exception as e:
        logger.error(f"Dependency shutdown failed: {e}")


# Startup initialization
initialize_dependencies() 