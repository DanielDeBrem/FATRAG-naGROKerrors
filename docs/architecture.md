# FATRAG Architecture Overview

## Current State Analysis

### FastAPI Backend Structure
The application is built with FastAPI and runs on port 8020. The main entry point is `main.py` which creates a single FastAPI application instance with numerous endpoints:

- **Routers/Endpoints**: Currently all endpoints are defined in `main.py` (over 1000 lines) without clear separation:
  - Document management (`/admin/docs/*`)
  - Client/project management (`/admin/clients/*`, `/admin/projects/*`)
  - LLM configuration (`/admin/config`)
  - Analysis jobs (`/admin/jobs/*`)
  - Query interface (`/query`)
  - Admin UI serving

- **Middleware**: Uses SlowAPIMiddleware for rate limiting and custom TimeoutMiddleware

- **Services**: Business logic is scattered across the main file or separate files like:
  - `clients_projects.py` - Client/project CRUD
  - `organogram_service.py` - Organogram management
  - `tax_calculator.py` - Tax calculations
  - `analysis.py` - Document analysis
  - Various job processing scripts in `scripts/`

### Data Storage
- **MySQL**: Extensive use of MySQL with tables for clients, projects, documents, jobs, feedback, etc.
- **Vector Store**: ChromaDB for embeddings storage
- **File System**: Documents stored in `fatrag_data/uploads/`

### LLM Integration
- **Ollama**: Primary LLM provider with support for multiple models (llama3.1:8b, gemma2:2b)
- **Load Balancing**: Multi-GPU worker routing via OLLAMA_WORKER_PORTS
- **Cloud Routing**: Support for cloud-hosted Ollama via OLLAMA_CLOUD_ROUTED

### Frontend
- **Admin Interface**: HTML/JS interface served from `static/admin/`
- **No Streamlit**: Current frontend is pure HTML/CSS/JS, no Streamlit integration

## Weaknesses Identified

### Architecture Issues
1. **No Clear Layer Separation**: All logic mixed in `main.py`
2. **No Dependency Injection**: Global variables and direct imports
3. **No Abstraction Layer**: Direct Ollama calls scattered throughout
4. **Massive Single File**: `main.py` is over 1000+ lines

### Quality Issues
5. **No Tests**: No unit or integration tests visible
6. **Poor Error Handling**: Basic try/catch blocks, inconsistent responses
7. **No Logging Strategy**: Print statements instead of proper logging
8. **No Configuration Management**: Environment variables scattered

### Code Organization Issues
9. **Mixed Concerns**: HTTP logic, business logic, and data access in one file
10. **Hard Dependencies**: Direct DB calls in endpoints
11. **No Type Safety**: Limited Pydantic model usage
12. **Ad-hoc Patterns**: Inconsistent API design patterns

## Proposed Architecture

### Directory Structure
```
app/
├── api/                    # HTTP layer
│   ├── documents.py       # Document upload/management
│   ├── advisory.py        # Query and advice endpoints
│   ├── __init__.py
│   └── dependencies.py    # Dependency injection
├── core/                  # Cross-cutting concerns
│   ├── config.py         # Settings management
│   ├── logging.py        # Logging configuration
│   ├── exceptions.py     # Custom exceptions
│   └── utils.py          # Utility functions
├── models/                # Data models
│   ├── domain/           # Domain entities
│   │   ├── document.py
│   │   ├── advisory.py
│   │   └── __init__.py
│   ├── dto/              # Data transfer objects
│   │   ├── upload.py
│   │   └── advisory.py
│   └── __init__.py
├── services/             # Business logic layer
│   ├── document_ingest.py     # Document processing
│   ├── document_classifier.py # Content classification
│   ├── document_extractors.py # Text/data extraction
│   ├── embeddings.py          # Vector operations
│   ├── advisory.py           # Advice generation
│   ├── ollama_client.py      # LLM abstracted client
│   └── __init__.py
├── repositories/         # Data access layer
│   ├── documents.py      # Document persistence
│   ├── embeddings.py     # Vector storage
│   ├── jobs.py          # Job tracking
│   └── __init__.py
└── main.py              # FastAPI application

frontend/
└── streamlit_app.py     # Streamlit UI

scripts/                 # Existing analysis scripts
docs/                    # Documentation
tests/                   # Test suite
```

### Design Principles
1. **Dependency Injection**: Use dependency injection for services and repositories
2. **SOLID Principles**: Single responsibility, open/closed, etc.
3. **Clean Architecture**: Clear separation between layers
4. **Type Safety**: Extensive Pydantic usage
5. **Error Handling**: Custom exceptions and consistent error responses
6. **Configuration**: Centralized settings management
7. **Logging**: Structured logging throughout
8. **Testing**: Unit and integration tests for each layer
