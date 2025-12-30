"""Main FastAPI application for Sovereign Career Architect."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

from sovereign_career_architect.config import settings
from sovereign_career_architect.utils.logging import configure_logging, get_logger
from sovereign_career_architect.api.routes import all_routers
from sovereign_career_architect.api.models import ErrorResponse
from sovereign_career_architect.core.safety import SafetyLayer
from sovereign_career_architect.browser.agent import BrowserAgent

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Global application state
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Sovereign Career Architect API")
    
    try:
        # Initialize safety layer
        safety_layer = SafetyLayer()
        app_state["safety_layer"] = safety_layer
        
        # Initialize browser agent
        browser_agent = BrowserAgent(
            headless=settings.browser_headless,
            stealth_mode=settings.browser_stealth_mode
        )
        app_state["browser_agent"] = browser_agent
        
        # Initialize browser agent
        await browser_agent.initialize()
        
        logger.info("Application startup completed successfully")
        
        # Update global references in routes module
        import sovereign_career_architect.api.routes as routes_module
        routes_module.safety_layer = safety_layer
        routes_module.browser_agent = browser_agent
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sovereign Career Architect API")
    
    try:
        # Cleanup browser agent
        if "browser_agent" in app_state:
            await app_state["browser_agent"].close()
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error("Application shutdown error", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Sovereign Career Architect API",
        description="AI-powered autonomous job application system with human oversight",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    for router in all_routers:
        app.include_router(router, prefix="/api/v1")
    
    # Add root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Sovereign Career Architect API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled"
        }
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if settings.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration
            )
            
            return response
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                duration_seconds=duration
            )
            raise


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                timestamp=datetime.now(timezone.utc)
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                details={"validation_errors": exc.errors()},
                timestamp=datetime.now(timezone.utc)
            ).dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(
            "Starlette HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="StarletteHTTPException",
                message=exc.detail or "Internal server error",
                timestamp=datetime.now(timezone.utc)
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            error=str(exc),
            error_type=type(exc).__name__,
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details={"error_type": type(exc).__name__} if settings.debug else None,
                timestamp=datetime.now(timezone.utc)
            ).dict()
        )


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    from datetime import datetime, timezone
    
    uvicorn.run(
        "sovereign_career_architect.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None  # Use our custom logging
    )