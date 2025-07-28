"""
Logging utilities for VecClean.

Provides structured logging setup with consistent formatting,
rotation, performance monitoring, and distributed system support.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, TypeVar, ParamSpec, List
from datetime import datetime, timezone
import threading

# Type hints for decorators
P = ParamSpec('P')
T = TypeVar('T')


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Converts log records to JSON format for better machine readability
    and integration with log aggregation systems.
    """
    
    def __init__(
        self, 
        include_extra: bool = True,
        timestamp_format: str = "iso",
        service_name: Optional[str] = None
    ):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra: Include extra fields from log record
            timestamp_format: Timestamp format ("iso", "epoch", "custom")
            service_name: Service name to include in logs
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format
        self.service_name = service_name or "vecclean"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }
        
        # Add location info
        if record.pathname:
            log_data.update({
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            })
        
        # Add process/thread info
        log_data.update({
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add stack info if present
        if record.stack_info:
            log_data["stack"] = record.stack_info
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info', 'message'
                }:
                    # Only include JSON-serializable values
                    try:
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False)
    
    def _format_timestamp(self, created: float) -> str:
        """Format timestamp according to configured format."""
        if self.timestamp_format == "epoch":
            return str(created)
        elif self.timestamp_format == "iso":
            dt = datetime.fromtimestamp(created, tz=timezone.utc)
            return dt.isoformat()
        else:
            # Default to ISO format
            dt = datetime.fromtimestamp(created, tz=timezone.utc)
            return dt.isoformat()


class PerformanceFilter(logging.Filter):
    """
    Filter to add performance metrics to log records.
    
    Tracks timing and memory usage for performance monitoring.
    """
    
    def __init__(self):
        """Initialize performance filter."""
        super().__init__()
        self._start_times: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to log record."""
        # Add timing information
        current_time = time.perf_counter()
        
        # Add memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            record.memory_rss = memory_info.rss
            record.memory_vms = memory_info.vms
            record.memory_percent = process.memory_percent()
        except ImportError:
            pass
        
        # Add CPU usage if available
        try:
            import psutil
            record.cpu_percent = psutil.cpu_percent()
        except ImportError:
            pass
        
        return True


class LogRotationHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced rotating file handler with compression and cleanup.
    
    Extends the standard rotating handler with automatic compression
    of old log files and intelligent cleanup policies.
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 10 * 1024 * 1024,  # 10MB
        backupCount: int = 5,
        compress: bool = True,
        cleanup_days: int = 30,
        **kwargs
    ):
        """
        Initialize rotating handler.
        
        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Maximum file size before rotation
            backupCount: Number of backup files to keep
            compress: Whether to compress rotated files
            cleanup_days: Days after which to delete old logs
            **kwargs: Additional arguments for parent class
        """
        super().__init__(filename, mode, maxBytes, backupCount, **kwargs)
        self.compress = compress
        self.cleanup_days = cleanup_days
    
    def doRollover(self) -> None:
        """Perform log rotation with optional compression."""
        super().doRollover()
        
        if self.compress:
            self._compress_rotated_files()
        
        self._cleanup_old_files()
    
    def _compress_rotated_files(self) -> None:
        """Compress rotated log files."""
        import gzip
        import shutil
        
        for i in range(1, self.backupCount + 1):
            backup_file = f"{self.baseFilename}.{i}"
            compressed_file = f"{backup_file}.gz"
            
            if os.path.exists(backup_file) and not os.path.exists(compressed_file):
                try:
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(backup_file)
                except Exception as e:
                    # Log compression failure but don't crash
                    print(f"Failed to compress {backup_file}: {e}", file=sys.stderr)
    
    def _cleanup_old_files(self) -> None:
        """Clean up old log files based on age."""
        if self.cleanup_days <= 0:
            return
        
        cutoff_time = time.time() - (self.cleanup_days * 24 * 60 * 60)
        log_dir = os.path.dirname(self.baseFilename)
        base_name = os.path.basename(self.baseFilename)
        
        try:
            for filename in os.listdir(log_dir):
                if filename.startswith(base_name) and filename != base_name:
                    file_path = os.path.join(log_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
        except Exception as e:
            print(f"Failed to cleanup old logs: {e}", file=sys.stderr)


class LogAggregator:
    """
    Log aggregator for distributed systems.
    
    Collects and forwards logs to centralized logging systems
    like ELK stack, Splunk, or cloud logging services.
    """
    
    def __init__(
        self,
        aggregator_type: str = "elasticsearch",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_prefix: str = "vecclean",
        batch_size: int = 100,
        flush_interval: int = 30,
    ):
        """
        Initialize log aggregator.
        
        Args:
            aggregator_type: Type of aggregator ("elasticsearch", "splunk", "cloudwatch")
            endpoint: Aggregator endpoint URL
            api_key: API key for authentication
            index_prefix: Index/source prefix for logs
            batch_size: Number of logs to batch before sending
            flush_interval: Seconds between forced flushes
        """
        self.aggregator_type = aggregator_type
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_prefix = index_prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._batch: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        
        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
    
    def add_log(self, log_record: Dict[str, Any]) -> None:
        """
        Add a log record to the batch.
        
        Args:
            log_record: Structured log record
        """
        with self._lock:
            self._batch.append(log_record)
            
            if len(self._batch) >= self.batch_size:
                self._flush_batch()
    
    def _flush_batch(self) -> None:
        """Flush current batch to aggregator."""
        if not self._batch:
            return
        
        batch_to_send = self._batch.copy()
        self._batch.clear()
        self._last_flush = time.time()
        
        # Send in background to avoid blocking
        threading.Thread(
            target=self._send_logs,
            args=(batch_to_send,),
            daemon=True
        ).start()
    
    def _flush_worker(self) -> None:
        """Background worker for periodic flushing."""
        while True:
            time.sleep(self.flush_interval)
            
            with self._lock:
                if time.time() - self._last_flush >= self.flush_interval:
                    self._flush_batch()
    
    def _send_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Send logs to the aggregator."""
        if not self.endpoint:
            return
        
        try:
            if self.aggregator_type == "elasticsearch":
                self._send_to_elasticsearch(logs)
            elif self.aggregator_type == "splunk":
                self._send_to_splunk(logs)
            elif self.aggregator_type == "cloudwatch":
                self._send_to_cloudwatch(logs)
            else:
                print(f"Unknown aggregator type: {self.aggregator_type}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to send logs to {self.aggregator_type}: {e}", file=sys.stderr)
    
    def _send_to_elasticsearch(self, logs: List[Dict[str, Any]]) -> None:
        """Send logs to Elasticsearch."""
        import requests
        
        # Prepare bulk indexing payload
        bulk_data = []
        for log in logs:
            index_name = f"{self.index_prefix}-{datetime.now().strftime('%Y-%m-%d')}"
            
            # Index metadata
            index_meta = {
                "index": {
                    "_index": index_name,
                    "_type": "_doc"
                }
            }
            bulk_data.append(json.dumps(index_meta))
            bulk_data.append(json.dumps(log))
        
        payload = '\n'.join(bulk_data) + '\n'
        
        headers = {'Content-Type': 'application/x-ndjson'}
        if self.api_key:
            headers['Authorization'] = f'ApiKey {self.api_key}'
        
        response = requests.post(
            f"{self.endpoint}/_bulk",
            data=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
    
    def _send_to_splunk(self, logs: List[Dict[str, Any]]) -> None:
        """Send logs to Splunk."""
        import requests
        
        for log in logs:
            event_data = {
                "event": log,
                "source": self.index_prefix,
                "sourcetype": "json",
                "index": "main"
            }
            
            headers = {'Authorization': f'Splunk {self.api_key}'}
            
            response = requests.post(
                f"{self.endpoint}/services/collector/event",
                json=event_data,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
    
    def _send_to_cloudwatch(self, logs: List[Dict[str, Any]]) -> None:
        """Send logs to AWS CloudWatch."""
        try:
            import boto3
            
            client = boto3.client('logs')
            
            log_events = []
            for log in logs:
                log_events.append({
                    'timestamp': int(time.time() * 1000),
                    'message': json.dumps(log)
                })
            
            client.put_log_events(
                logGroupName=f'/aws/vecclean/{self.index_prefix}',
                logStreamName=f'stream-{datetime.now().strftime("%Y-%m-%d")}',
                logEvents=log_events
            )
        except ImportError:
            print("boto3 not available for CloudWatch logging", file=sys.stderr)


class AggregatorHandler(logging.Handler):
    """
    Logging handler that forwards to log aggregator.
    
    Integrates with the LogAggregator to send structured logs
    to distributed logging systems.
    """
    
    def __init__(self, aggregator: LogAggregator):
        """
        Initialize aggregator handler.
        
        Args:
            aggregator: Log aggregator instance
        """
        super().__init__()
        self.aggregator = aggregator
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to aggregator."""
        try:
            # Convert record to dictionary
            log_data = {
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info', 'message'
                }:
                    try:
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
            
            self.aggregator.add_log(log_data)
            
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timing: bool = True,
    json_format: bool = False,
    rotation_enabled: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    service_name: Optional[str] = None,
    aggregator_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup application logging with advanced features.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string (ignored if json_format=True)
        include_timing: Include timing information
        json_format: Use JSON structured logging
        rotation_enabled: Enable log rotation
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        service_name: Service name for structured logging
        aggregator_config: Configuration for log aggregation
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Determine formatter
    if json_format:
        formatter = JSONFormatter(
            service_name=service_name or "vecclean",
            include_extra=True
        )
    else:
        if format_string is None:
            if include_timing:
                format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            else:
                format_string = "%(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format_string)
    
    # Add performance filter
    perf_filter = PerformanceFilter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(perf_filter)
    root_logger.addHandler(console_handler)
    
    # File handler with optional rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation_enabled:
            file_handler = LogRotationHandler(
                filename=str(log_path),
                maxBytes=max_file_size,
                backupCount=backup_count,
                compress=True,
                cleanup_days=30
            )
        else:
            file_handler = logging.FileHandler(log_path)
        
        file_handler.setFormatter(formatter)
        file_handler.addFilter(perf_filter)
        root_logger.addHandler(file_handler)
    
    # Log aggregation
    if aggregator_config:
        try:
            aggregator = LogAggregator(**aggregator_config)
            agg_handler = AggregatorHandler(aggregator)
            agg_handler.setFormatter(JSONFormatter(service_name=service_name))
            root_logger.addHandler(agg_handler)
        except Exception as e:
            print(f"Failed to setup log aggregation: {e}", file=sys.stderr)
    
    # Set library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def performance_logger(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for logging function performance.
    
    Args:
        logger: Logger to use (default: function's module logger)
        level: Logging level
        include_args: Include function arguments in log
        include_result: Include function result in log
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.perf_counter()
            
            # Log function entry
            log_data = {
                "function": func.__name__,
                "module": func.__module__,
                "event": "function_start"
            }
            
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            func_logger.log(level, "Function started", extra=log_data)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                log_data.update({
                    "event": "function_complete",
                    "duration_seconds": duration,
                    "status": "success"
                })
                
                if include_result:
                    log_data["result"] = str(result)[:1000]  # Limit size
                
                func_logger.log(level, f"Function completed in {duration:.3f}s", extra=log_data)
                
                return result
                
            except Exception as e:
                # Log exception
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                log_data.update({
                    "event": "function_error",
                    "duration_seconds": duration,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                func_logger.log(logging.ERROR, f"Function failed after {duration:.3f}s", extra=log_data)
                raise
        
        return wrapper
    return decorator


def memory_logger(
    logger: Optional[logging.Logger] = None,
    threshold_mb: float = 100.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for logging memory usage.
    
    Args:
        logger: Logger to use
        threshold_mb: Memory threshold in MB to trigger logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_logger = logger or logging.getLogger(func.__module__)
            
            try:
                import psutil
                process = psutil.Process()
                
                # Memory before
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                result = func(*args, **kwargs)
                
                # Memory after
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                mem_diff = mem_after - mem_before
                
                if mem_after > threshold_mb or abs(mem_diff) > threshold_mb / 10:
                    log_data = {
                        "function": func.__name__,
                        "memory_before_mb": mem_before,
                        "memory_after_mb": mem_after,
                        "memory_diff_mb": mem_diff,
                        "event": "memory_usage"
                    }
                    
                    func_logger.info(
                        f"Memory usage: {mem_after:.1f}MB ({mem_diff:+.1f}MB)",
                        extra=log_data
                    )
                
                return result
                
            except ImportError:
                # psutil not available, just run function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions for common use cases

def get_performance_logger(name: str) -> logging.Logger:
    """Get a logger configured for performance monitoring."""
    logger = logging.getLogger(f"performance.{name}")
    return logger


def get_structured_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger that automatically includes extra structured fields."""
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create a custom logger that adds extra fields
        original_makeRecord = logger.makeRecord
        
        def makeRecord(*args, **kwargs):
            record = original_makeRecord(*args, **kwargs)
            for key, value in extra_fields.items():
                setattr(record, key, value)
            return record
        
        logger.makeRecord = makeRecord
    
    return logger 