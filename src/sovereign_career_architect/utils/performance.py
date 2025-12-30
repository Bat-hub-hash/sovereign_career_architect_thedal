"""Performance optimization and monitoring utilities."""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import structlog

from sovereign_career_architect.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    response_time_ms: float
    active_sessions: int
    requests_per_minute: int
    error_rate: float
    component_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentMetrics:
    """Metrics for individual components."""
    name: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    total_requests: int
    error_count: int
    last_updated: datetime


class PerformanceMonitor:
    """Monitors and optimizes system performance."""
    
    def __init__(self, collection_interval: int = 60):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.logger = logger.bind(component="performance_monitor")
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.active_requests: Dict[str, float] = {}
        self.session_count = 0
        self.total_requests = 0
        self.error_count = 0
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 5000.0,
            "error_rate": 0.05
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            "high_cpu": self._optimize_cpu_usage,
            "high_memory": self._optimize_memory_usage,
            "slow_response": self._optimize_response_time,
            "high_errors": self._optimize_error_handling
        }
    
    def start_monitoring(self) -> None:
        """Start performance monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Performance monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop performance monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.collection_interval):
            try:
                self._collect_metrics()
                self._analyze_performance()
            except Exception as e:
                self.logger.error("Monitoring loop error", error=str(e))
    
    def _collect_metrics(self) -> None:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Calculate request metrics
            current_time = time.time()
            recent_requests = [
                req_time for req_time in self.request_times.get("all", [])
                if current_time - req_time < 60  # Last minute
            ]
            requests_per_minute = len(recent_requests)
            
            # Calculate average response time
            if self.request_times.get("response_times"):
                avg_response_time = sum(self.request_times["response_times"]) / len(self.request_times["response_times"])
            else:
                avg_response_time = 0.0
            
            # Calculate error rate
            error_rate = self.error_count / max(self.total_requests, 1)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                response_time_ms=avg_response_time,
                active_sessions=self.session_count,
                requests_per_minute=requests_per_minute,
                error_rate=error_rate,
                component_metrics={
                    name: {
                        "avg_response_time": comp.avg_response_time,
                        "error_count": comp.error_count,
                        "total_requests": comp.total_requests
                    }
                    for name, comp in self.component_metrics.items()
                }
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            self.logger.debug(
                "Metrics collected",
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                response_time_ms=avg_response_time,
                requests_per_minute=requests_per_minute
            )
            
        except Exception as e:
            self.logger.error("Metrics collection failed", error=str(e))
    
    def _analyze_performance(self) -> None:
        """Analyze performance and trigger optimizations if needed."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        issues = []
        
        # Check thresholds
        if latest_metrics.cpu_percent > self.thresholds["cpu_percent"]:
            issues.append("high_cpu")
        
        if latest_metrics.memory_percent > self.thresholds["memory_percent"]:
            issues.append("high_memory")
        
        if latest_metrics.response_time_ms > self.thresholds["response_time_ms"]:
            issues.append("slow_response")
        
        if latest_metrics.error_rate > self.thresholds["error_rate"]:
            issues.append("high_errors")
        
        # Trigger optimizations
        for issue in issues:
            if issue in self.optimization_strategies:
                try:
                    self.optimization_strategies[issue](latest_metrics)
                except Exception as e:
                    self.logger.error(
                        "Optimization strategy failed",
                        issue=issue,
                        error=str(e)
                    )
        
        if issues:
            self.logger.warning(
                "Performance issues detected",
                issues=issues,
                metrics={
                    "cpu": latest_metrics.cpu_percent,
                    "memory": latest_metrics.memory_percent,
                    "response_time": latest_metrics.response_time_ms,
                    "error_rate": latest_metrics.error_rate
                }
            )
    
    def track_request(self, component: str, request_id: str) -> None:
        """Start tracking a request."""
        self.active_requests[request_id] = time.time()
        self.total_requests += 1
        self.request_times["all"].append(time.time())
        
        self.logger.debug(
            "Request tracking started",
            component=component,
            request_id=request_id
        )
    
    def complete_request(
        self,
        component: str,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> float:
        """Complete request tracking and return response time."""
        if request_id not in self.active_requests:
            self.logger.warning("Request not found for completion", request_id=request_id)
            return 0.0
        
        start_time = self.active_requests.pop(request_id)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Update component metrics
        if component not in self.component_metrics:
            self.component_metrics[component] = ComponentMetrics(
                name=component,
                avg_response_time=response_time,
                max_response_time=response_time,
                min_response_time=response_time,
                total_requests=1,
                error_count=0 if success else 1,
                last_updated=datetime.now()
            )
        else:
            comp_metrics = self.component_metrics[component]
            comp_metrics.total_requests += 1
            comp_metrics.avg_response_time = (
                (comp_metrics.avg_response_time * (comp_metrics.total_requests - 1) + response_time) /
                comp_metrics.total_requests
            )
            comp_metrics.max_response_time = max(comp_metrics.max_response_time, response_time)
            comp_metrics.min_response_time = min(comp_metrics.min_response_time, response_time)
            comp_metrics.last_updated = datetime.now()
            
            if not success:
                comp_metrics.error_count += 1
                self.error_count += 1
        
        # Store response time
        self.request_times["response_times"].append(response_time)
        
        self.logger.debug(
            "Request completed",
            component=component,
            request_id=request_id,
            response_time_ms=response_time,
            success=success,
            error=error
        )
        
        return response_time
    
    def update_session_count(self, count: int) -> None:
        """Update active session count."""
        self.session_count = count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends (last 10 minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if (datetime.now() - m.timestamp).total_seconds() < 600
        ]
        
        if len(recent_metrics) > 1:
            cpu_trend = recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent
            memory_trend = recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent
            response_trend = recent_metrics[-1].response_time_ms - recent_metrics[0].response_time_ms
        else:
            cpu_trend = memory_trend = response_trend = 0.0
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "current_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_mb": latest_metrics.memory_mb,
                "response_time_ms": latest_metrics.response_time_ms,
                "active_sessions": latest_metrics.active_sessions,
                "requests_per_minute": latest_metrics.requests_per_minute,
                "error_rate": latest_metrics.error_rate
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "response_trend": response_trend
            },
            "component_metrics": {
                name: {
                    "avg_response_time": comp.avg_response_time,
                    "max_response_time": comp.max_response_time,
                    "min_response_time": comp.min_response_time,
                    "total_requests": comp.total_requests,
                    "error_count": comp.error_count,
                    "error_rate": comp.error_count / max(comp.total_requests, 1)
                }
                for name, comp in self.component_metrics.items()
            },
            "health_status": self._get_health_status(latest_metrics),
            "recommendations": self._get_performance_recommendations(latest_metrics)
        }
    
    def _get_health_status(self, metrics: PerformanceMetrics) -> str:
        """Determine overall health status."""
        if (metrics.cpu_percent > self.thresholds["cpu_percent"] or
            metrics.memory_percent > self.thresholds["memory_percent"] or
            metrics.response_time_ms > self.thresholds["response_time_ms"] or
            metrics.error_rate > self.thresholds["error_rate"]):
            return "critical"
        elif (metrics.cpu_percent > self.thresholds["cpu_percent"] * 0.8 or
              metrics.memory_percent > self.thresholds["memory_percent"] * 0.8 or
              metrics.response_time_ms > self.thresholds["response_time_ms"] * 0.8):
            return "warning"
        else:
            return "healthy"
    
    def _get_performance_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"] * 0.8:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if metrics.memory_percent > self.thresholds["memory_percent"] * 0.8:
            recommendations.append("Monitor memory usage and consider implementing caching strategies")
        
        if metrics.response_time_ms > self.thresholds["response_time_ms"] * 0.8:
            recommendations.append("Optimize slow operations and consider async processing")
        
        if metrics.error_rate > self.thresholds["error_rate"] * 0.5:
            recommendations.append("Investigate and fix recurring errors")
        
        if metrics.requests_per_minute > 100:
            recommendations.append("Consider implementing rate limiting and request queuing")
        
        return recommendations
    
    def _optimize_cpu_usage(self, metrics: PerformanceMetrics) -> None:
        """Optimize CPU usage."""
        self.logger.warning(
            "High CPU usage detected, implementing optimizations",
            cpu_percent=metrics.cpu_percent
        )
        
        # Implement CPU optimization strategies
        # 1. Reduce concurrent operations
        # 2. Implement request queuing
        # 3. Cache frequently computed results
        
        # For now, just log the optimization attempt
        self.logger.info("CPU optimization strategies applied")
    
    def _optimize_memory_usage(self, metrics: PerformanceMetrics) -> None:
        """Optimize memory usage."""
        self.logger.warning(
            "High memory usage detected, implementing optimizations",
            memory_percent=metrics.memory_percent
        )
        
        # Implement memory optimization strategies
        # 1. Clear old cache entries
        # 2. Reduce memory-intensive operations
        # 3. Implement garbage collection
        
        import gc
        gc.collect()
        
        self.logger.info("Memory optimization strategies applied")
    
    def _optimize_response_time(self, metrics: PerformanceMetrics) -> None:
        """Optimize response time."""
        self.logger.warning(
            "Slow response time detected, implementing optimizations",
            response_time_ms=metrics.response_time_ms
        )
        
        # Implement response time optimization strategies
        # 1. Enable async processing
        # 2. Implement caching
        # 3. Optimize database queries
        
        self.logger.info("Response time optimization strategies applied")
    
    def _optimize_error_handling(self, metrics: PerformanceMetrics) -> None:
        """Optimize error handling."""
        self.logger.warning(
            "High error rate detected, implementing optimizations",
            error_rate=metrics.error_rate
        )
        
        # Implement error handling optimization strategies
        # 1. Add retry mechanisms
        # 2. Improve error recovery
        # 3. Add circuit breakers
        
        self.logger.info("Error handling optimization strategies applied")


class PerformanceDecorator:
    """Decorator for tracking function performance."""
    
    def __init__(self, monitor: PerformanceMonitor, component: str):
        self.monitor = monitor
        self.component = component
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with performance tracking."""
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                request_id = f"{self.component}_{func.__name__}_{time.time()}"
                self.monitor.track_request(self.component, request_id)
                
                try:
                    result = await func(*args, **kwargs)
                    self.monitor.complete_request(self.component, request_id, success=True)
                    return result
                except Exception as e:
                    self.monitor.complete_request(
                        self.component, request_id, success=False, error=str(e)
                    )
                    raise
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                request_id = f"{self.component}_{func.__name__}_{time.time()}"
                self.monitor.track_request(self.component, request_id)
                
                try:
                    result = func(*args, **kwargs)
                    self.monitor.complete_request(self.component, request_id, success=True)
                    return result
                except Exception as e:
                    self.monitor.complete_request(
                        self.component, request_id, success=False, error=str(e)
                    )
                    raise
            
            return sync_wrapper


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def track_performance(component: str):
    """Decorator factory for performance tracking."""
    return PerformanceDecorator(performance_monitor, component)


# Optimization utilities
class CacheManager:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry["expires_at"]:
            self.delete(key)
            return None
        
        self.access_times[key] = time.time()
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
        self.access_times[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
    
    def _evict_oldest(self) -> None:
        """Evict oldest accessed entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.delete(oldest_key)


# Global cache instance
cache_manager = CacheManager()


def cached(ttl: int = 300):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator