"""
Timing utilities for VecClean.

Provides performance monitoring and timing utilities for
tracking processing performance, identifying bottlenecks,
and detecting performance regressions.
"""

from __future__ import annotations

import functools
import statistics
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, ParamSpec, Tuple
from pathlib import Path
import json

# Type hints for decorators
P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class TimingEntry:
    """Single timing measurement entry."""
    name: str
    start_time: float
    end_time: float
    duration: float
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class MemoryEntry:
    """Memory usage measurement entry."""
    name: str
    timestamp: float
    rss_bytes: int
    vms_bytes: int
    percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def rss_mb(self) -> float:
        """RSS memory in MB."""
        return self.rss_bytes / (1024 * 1024)
    
    @property
    def vms_mb(self) -> float:
        """VMS memory in MB."""
        return self.vms_bytes / (1024 * 1024)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    operation: str
    mean_duration: float
    std_duration: float
    mean_memory: float
    std_memory: float
    sample_count: int
    last_updated: float
    
    def is_regression(
        self, 
        duration: float, 
        memory: Optional[float] = None,
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Check if performance represents a regression.
        
        Args:
            duration: Current duration
            memory: Current memory usage
            threshold_std: Number of standard deviations for regression
            
        Returns:
            Regression analysis results
        """
        duration_z_score = (duration - self.mean_duration) / max(self.std_duration, 0.001)
        is_duration_regression = duration_z_score > threshold_std
        
        memory_regression = False
        memory_z_score = 0.0
        if memory is not None and self.std_memory > 0:
            memory_z_score = (memory - self.mean_memory) / max(self.std_memory, 0.001)
            memory_regression = memory_z_score > threshold_std
        
        return {
            "operation": self.operation,
            "is_regression": is_duration_regression or memory_regression,
            "duration_regression": is_duration_regression,
            "memory_regression": memory_regression,
            "duration_z_score": duration_z_score,
            "memory_z_score": memory_z_score,
            "current_duration": duration,
            "baseline_duration": self.mean_duration,
            "current_memory": memory,
            "baseline_memory": self.mean_memory,
            "threshold_std": threshold_std
        }


class HierarchicalTimer:
    """
    Hierarchical timer for nested timing measurements.
    
    Supports nested operations and provides detailed timing
    breakdown with parent-child relationships.
    """
    
    def __init__(self):
        """Initialize hierarchical timer."""
        self._entries: Dict[str, TimingEntry] = {}
        self._stack: List[str] = []
        self._current_id = 0
        self._lock = threading.Lock()
    
    def _generate_id(self) -> str:
        """Generate unique timing entry ID."""
        with self._lock:
            self._current_id += 1
            return f"timing_{self._current_id}"
    
    @contextmanager
    def time_operation(self, name: str, **metadata):
        """
        Context manager for hierarchical timing.
        
        Args:
            name: Operation name
            **metadata: Additional metadata to store
        """
        entry_id = self._generate_id()
        start_time = time.perf_counter()
        
        # Set up hierarchy
        parent_id = self._stack[-1] if self._stack else None
        
        entry = TimingEntry(
            name=name,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            parent=parent_id,
            metadata=metadata
        )
        
        # Add to parent's children
        if parent_id and parent_id in self._entries:
            self._entries[parent_id].children.append(entry_id)
        
        self._entries[entry_id] = entry
        self._stack.append(entry_id)
        
        try:
            yield entry
        finally:
            end_time = time.perf_counter()
            entry.end_time = end_time
            entry.duration = end_time - start_time
            self._stack.pop()
    
    def get_tree(self, root_only: bool = True) -> Dict[str, Any]:
        """
        Get timing tree structure.
        
        Args:
            root_only: Only return root-level operations
            
        Returns:
            Hierarchical timing data
        """
        def build_tree(entry_id: str) -> Dict[str, Any]:
            entry = self._entries[entry_id]
            tree = {
                "name": entry.name,
                "duration": entry.duration,
                "duration_ms": entry.duration_ms,
                "start_time": entry.start_time,
                "end_time": entry.end_time,
                "metadata": entry.metadata,
                "children": []
            }
            
            for child_id in entry.children:
                tree["children"].append(build_tree(child_id))
            
            return tree
        
        if root_only:
            # Find root entries (no parent)
            roots = []
            for entry_id, entry in self._entries.items():
                if entry.parent is None:
                    roots.append(build_tree(entry_id))
            return {"roots": roots}
        else:
            # Return all entries
            return {entry_id: build_tree(entry_id) for entry_id in self._entries}
    
    def get_flat_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get flattened timing statistics by operation name."""
        stats = defaultdict(list)
        
        for entry in self._entries.values():
            stats[entry.name].append(entry.duration)
        
        result = {}
        for name, durations in stats.items():
            if durations:
                result[name] = {
                    "count": len(durations),
                    "total_duration": sum(durations),
                    "mean_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0.0
                }
        
        return result
    
    def reset(self) -> None:
        """Reset all timing data."""
        with self._lock:
            self._entries.clear()
            self._stack.clear()
            self._current_id = 0


class Timer:
    """
    Enhanced timer with memory tracking and performance analysis.
    
    Tracks operation timing and provides statistics for
    performance analysis and optimization.
    """
    
    def __init__(self) -> None:
        """Initialize timer."""
        self._times: Dict[str, List[float]] = defaultdict(list)
        self._memory_usage: Dict[str, List[MemoryEntry]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        self._start_memory: Dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def time_operation(self, name: str, track_memory: bool = True):
        """
        Context manager for timing operations with memory tracking.
        
        Args:
            name: Operation name for tracking
            track_memory: Whether to track memory usage
        """
        start_time = time.perf_counter()
        start_memory = None
        
        if track_memory:
            start_memory = self._get_memory_info(name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            with self._lock:
                self._times[name].append(duration)
            
            if track_memory and start_memory:
                end_memory = self._get_memory_info(name)
                self._memory_usage[name].append(end_memory)
    
    def _get_memory_info(self, name: str) -> Optional[MemoryEntry]:
        """Get current memory information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return MemoryEntry(
                name=name,
                timestamp=time.time(),
                rss_bytes=memory_info.rss,
                vms_bytes=memory_info.vms,
                percent=process.memory_percent()
            )
        except ImportError:
            return None
    
    def start(self, name: str, track_memory: bool = True) -> None:
        """Start timing an operation."""
        with self._lock:
            self._start_times[name] = time.perf_counter()
            if track_memory:
                self._start_memory[name] = self._get_memory_info(name)
    
    def stop(self, name: str) -> float:
        """
        Stop timing an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Duration in seconds
        """
        end_time = time.perf_counter()
        
        with self._lock:
            if name not in self._start_times:
                raise ValueError(f"Timer '{name}' was not started")
            
            start_time = self._start_times[name]
            duration = end_time - start_time
            
            self._times[name].append(duration)
            
            # Memory tracking
            if name in self._start_memory:
                end_memory = self._get_memory_info(name)
                if end_memory:
                    self._memory_usage[name].append(end_memory)
                del self._start_memory[name]
            
            del self._start_times[name]
            return duration
    
    def get_time(self, name: str) -> float:
        """Get total time for an operation."""
        times = self._times.get(name, [])
        return sum(times)
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation."""
        times = self._times.get(name, [])
        return statistics.mean(times) if times else 0.0
    
    def get_count(self, name: str) -> int:
        """Get call count for an operation."""
        return len(self._times.get(name, []))
    
    def get_total_time(self) -> float:
        """Get total time across all operations."""
        return sum(sum(times) for times in self._times.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics."""
        stats = {}
        
        for name, times in self._times.items():
            if times:
                memory_entries = self._memory_usage.get(name, [])
                memory_stats = {}
                
                if memory_entries:
                    rss_values = [entry.rss_mb for entry in memory_entries]
                    memory_stats = {
                        "mean_memory_mb": statistics.mean(rss_values),
                        "max_memory_mb": max(rss_values),
                        "min_memory_mb": min(rss_values),
                        "memory_std_mb": statistics.stdev(rss_values) if len(rss_values) > 1 else 0.0
                    }
                
                stats[name] = {
                    "total_time": sum(times),
                    "count": len(times),
                    "average_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
                    **memory_stats
                }
        
        stats["_summary"] = {
            "total_time": self.get_total_time(),
            "operation_count": len(self._times),
        }
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {}
        
        for name, entries in self._memory_usage.items():
            if entries:
                rss_values = [entry.rss_mb for entry in entries]
                vms_values = [entry.vms_mb for entry in entries]
                percent_values = [entry.percent for entry in entries]
                
                stats[name] = {
                    "count": len(entries),
                    "rss_mean_mb": statistics.mean(rss_values),
                    "rss_max_mb": max(rss_values),
                    "rss_min_mb": min(rss_values),
                    "rss_std_mb": statistics.stdev(rss_values) if len(rss_values) > 1 else 0.0,
                    "vms_mean_mb": statistics.mean(vms_values),
                    "percent_mean": statistics.mean(percent_values),
                    "percent_max": max(percent_values)
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all timing data."""
        with self._lock:
            self._times.clear()
            self._memory_usage.clear()
            self._start_times.clear()
            self._start_memory.clear()


class PerformanceRegressor:
    """
    Performance regression detection system.
    
    Maintains baselines and detects when performance
    significantly deviates from historical norms.
    """
    
    def __init__(
        self, 
        baseline_file: Optional[str] = None,
        min_samples: int = 10,
        max_samples: int = 1000
    ):
        """
        Initialize performance regressor.
        
        Args:
            baseline_file: File to save/load baselines
            min_samples: Minimum samples before creating baseline
            max_samples: Maximum samples to keep in rolling window
        """
        self.baseline_file = Path(baseline_file) if baseline_file else None
        self.min_samples = min_samples
        self.max_samples = max_samples
        
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.Lock()
        
        self._load_baselines()
    
    def add_measurement(
        self, 
        operation: str, 
        duration: float, 
        memory: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add a performance measurement.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            memory: Memory usage in bytes
            
        Returns:
            Regression analysis if baseline exists
        """
        with self._lock:
            # Add to samples
            sample = {"duration": duration, "memory": memory, "timestamp": time.time()}
            self._samples[operation].append(sample)
            
            # Update baseline if we have enough samples
            if len(self._samples[operation]) >= self.min_samples:
                self._update_baseline(operation)
            
            # Check for regression
            if operation in self._baselines:
                return self._baselines[operation].is_regression(duration, memory)
        
        return None
    
    def _update_baseline(self, operation: str) -> None:
        """Update baseline for an operation."""
        samples = list(self._samples[operation])
        
        durations = [s["duration"] for s in samples]
        memories = [s["memory"] for s in samples if s["memory"] is not None]
        
        baseline = PerformanceBaseline(
            operation=operation,
            mean_duration=statistics.mean(durations),
            std_duration=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            mean_memory=statistics.mean(memories) if memories else 0.0,
            std_memory=statistics.stdev(memories) if len(memories) > 1 else 0.0,
            sample_count=len(samples),
            last_updated=time.time()
        )
        
        self._baselines[operation] = baseline
        self._save_baselines()
    
    def get_baseline(self, operation: str) -> Optional[PerformanceBaseline]:
        """Get baseline for an operation."""
        return self._baselines.get(operation)
    
    def get_all_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Get all baselines."""
        return self._baselines.copy()
    
    def reset_baseline(self, operation: str) -> None:
        """Reset baseline for an operation."""
        with self._lock:
            if operation in self._baselines:
                del self._baselines[operation]
            if operation in self._samples:
                self._samples[operation].clear()
            self._save_baselines()
    
    def _save_baselines(self) -> None:
        """Save baselines to file."""
        if not self.baseline_file:
            return
        
        try:
            data = {}
            for operation, baseline in self._baselines.items():
                data[operation] = {
                    "operation": baseline.operation,
                    "mean_duration": baseline.mean_duration,
                    "std_duration": baseline.std_duration,
                    "mean_memory": baseline.mean_memory,
                    "std_memory": baseline.std_memory,
                    "sample_count": baseline.sample_count,
                    "last_updated": baseline.last_updated
                }
            
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save baselines: {e}")
    
    def _load_baselines(self) -> None:
        """Load baselines from file."""
        if not self.baseline_file or not self.baseline_file.exists():
            return
        
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
            
            for operation, baseline_data in data.items():
                baseline = PerformanceBaseline(**baseline_data)
                self._baselines[operation] = baseline
                
        except Exception as e:
            print(f"Failed to load baselines: {e}")


# Decorators for performance monitoring

def time_it(
    operation_name: Optional[str] = None,
    timer: Optional[Timer] = None,
    track_memory: bool = True,
    log_results: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for timing function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        timer: Timer instance to use
        track_memory: Whether to track memory usage
        log_results: Whether to log timing results
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal timer
            if timer is None:
                timer = Timer()
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with timer.time_operation(name, track_memory=track_memory):
                result = func(*args, **kwargs)
            
            if log_results:
                import logging
                logger = logging.getLogger(func.__module__)
                stats = timer.get_stats()
                if name in stats:
                    op_stats = stats[name]
                    logger.info(
                        f"Function {name} completed in {op_stats['average_time']:.3f}s "
                        f"(count: {op_stats['count']})"
                    )
            
            return result
        
        return wrapper
    return decorator


def monitor_performance(
    regressor: Optional[PerformanceRegressor] = None,
    operation_name: Optional[str] = None,
    alert_on_regression: bool = True
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for performance monitoring with regression detection.
    
    Args:
        regressor: Performance regressor instance
        operation_name: Name for the operation
        alert_on_regression: Whether to log alerts on regression
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal regressor
            if regressor is None:
                regressor = PerformanceRegressor()
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Get memory before
            start_memory = None
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except ImportError:
                pass
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                end_memory = None
                if start_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss
                    except ImportError:
                        pass
                
                # Check for regression
                regression_info = regressor.add_measurement(name, duration, end_memory)
                
                if regression_info and regression_info["is_regression"] and alert_on_regression:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    
                    logger.warning(
                        f"Performance regression detected in {name}: "
                        f"duration={duration:.3f}s "
                        f"(baseline={regression_info['baseline_duration']:.3f}s, "
                        f"z-score={regression_info['duration_z_score']:.2f})"
                    )
                
                return result
                
            except Exception:
                # Still record the timing for failed operations
                end_time = time.perf_counter()
                duration = end_time - start_time
                regressor.add_measurement(name, duration)
                raise
        
        return wrapper
    return decorator


# Global instances for convenience
_global_timer = Timer()
_global_hierarchical_timer = HierarchicalTimer()
_global_regressor = PerformanceRegressor()


def get_global_timer() -> Timer:
    """Get the global timer instance."""
    return _global_timer


def get_global_hierarchical_timer() -> HierarchicalTimer:
    """Get the global hierarchical timer instance."""
    return _global_hierarchical_timer


def get_global_regressor() -> PerformanceRegressor:
    """Get the global performance regressor instance."""
    return _global_regressor


# Convenience functions

def time_function(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Tuple[T, float]:
    """
    Time a function call and return result with duration.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, duration_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return result, duration


def benchmark_function(
    func: Callable[P, T], 
    *args: P.args, 
    iterations: int = 100, 
    warmup: int = 10,
    **kwargs: P.kwargs
) -> Dict[str, Any]:
    """
    Benchmark a function with multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark statistics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        "iterations": iterations,
        "warmup": warmup,
        "total_time": sum(times),
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "times": times
    } 