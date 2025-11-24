"""Router initialization for the FATRAG application."""

from app.api.documents import router as documents_router
from app.api.files import router as files_router


def include_routers(app):
    """Include all API routers in the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.include_router(documents_router)
    app.include_router(files_router)
    print("✅ Included API routers: documents, files")
