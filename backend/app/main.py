"""
IFB Service Intelligence - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from app.config import settings
from app.api import forecasting, analytics, franchise, inventory, data
from app.database import engine, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    logger.info("Starting IFB Service Intelligence API...")

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

    # Load ML models (if they exist)
    try:
        from app.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        forecaster.load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load ML models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down IFB Service Intelligence API...")


# Initialize FastAPI app
app = FastAPI(
    title="IFB Service Intelligence API",
    description="""
    AI-powered forecasting and analytics API for IFB's service ecosystem.

    ## Features

    * **30/60/90-day demand forecasting** for service volumes, spare parts, warranty claims
    * **Inventory optimization** with ABC analysis
    * **Franchise performance tracking** and scoring
    * **Revenue leakage identification** and optimization
    * **Location intelligence** across branches and regions
    * **Real-time analytics** with AI-generated insights

    ## Authentication

    Currently using API key authentication. Include your API key in the header:
    `X-API-Key: your-api-key`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "IFB Service Intelligence API",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IFB Service Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include API routers
app.include_router(
    forecasting.router,
    prefix="/api/v1/forecasting",
    tags=["Forecasting"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)

app.include_router(
    franchise.router,
    prefix="/api/v1/franchise",
    tags=["Franchise"]
)

app.include_router(
    inventory.router,
    prefix="/api/v1/inventory",
    tags=["Inventory"]
)

app.include_router(
    data.router,
    prefix="/api/v1/data",
    tags=["Data Management"]
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
