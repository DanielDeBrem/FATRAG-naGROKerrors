# FATRAG - Financial Advisory Tool

A comprehensive Financial Advisory Tool using RAG (Retrieval-Augmented Generation) architecture with Milvus vector database, MySQL for metadata, Ollama for LLM capabilities, and a full web GUI for financial and fiscal professionals.

## 🚀 Features

- **Document Processing**: Upload and process PDF, DOCX, XLSX files
- **Vector Search**: Advanced semantic search with Milvus
- **AI-Powered Responses**: Dutch-language responses using Ollama models
- **Financial Focus**: Specialized for financial and compliance queries
- **Admin Dashboard**: Complete system management interface
- **Real-time Monitoring**: Health checks and system statistics
- **Docker Ready**: Complete containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web GUI       │    │   FastAPI       │    │   Background    │
│   (Tailwind +   │◄──►│   Backend       │◄──►│   Tasks         │
│   Jinja2)       │    │   (Port 8050)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MySQL         │    │   Milvus        │    │   Ollama        │
│   (Metadata)    │    │   (Vectors)     │    │   (LLM/Embed)   │
│   fatrag_db     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Database**: MySQL (metadata), Milvus (vectors)
- **AI/ML**: Ollama integration
- **Frontend**: Tailwind CSS, Jinja2 templates
- **Containerization**: Docker, Docker Compose
- **Task Queue**: Redis, Celery
- **Monitoring**: Health checks, logging

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ free disk space

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FATRAG
   ```

2. **Set up environment:**
   ```bash
   cp .env.docker .env
   # Edit .env with your secure passwords
   nano .env
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   - Main interface: http://localhost:8050
   - Admin panel: http://localhost:8050/admin
   - Health check: http://localhost:8050/health
   - API docs: http://localhost:8050/docs

### Local Development

1. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start external services:**
   - MySQL server running on localhost:3306
   - Milvus running on localhost:19530
   - Ollama running on localhost:11434

4. **Run the application:**
   ```bash
   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8050
   ```

## 📖 Documentation

- **Docker Setup**: [README_DOCKER.md](README_DOCKER.md)
- **API Documentation**: http://localhost:8050/docs
- **Implementation Plan**: [.cheetah/plans/fatrag/plan.md](.cheetah/plans/fatrag/plan.md)
- **Todo List**: [.cheetah/plans/fatrag/todo.md](.cheetah/plans/fatrag/todo.md)

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=mysql+pymysql://fatrag:password@localhost:3306/fatrag

# Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM Service
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_GENERATION_MODEL=llama2

# Application
SECRET_KEY=your_secret_key_here
APP_PORT=8050
LOG_LEVEL=INFO

# Dutch Language Settings
DEFAULT_LANGUAGE=nl
SUMMARY_MAX_LENGTH=500
PII_DETECTION_ENABLED=true
```

## 📊 Usage

### For Financial Professionals

1. **Upload Documents**: Use the admin panel to upload financial documents
2. **Ask Questions**: Use the main interface to ask financial questions in Dutch
3. **Review Results**: Get AI-powered answers with source citations
4. **Provide Feedback**: Rate responses to improve system performance

### For Administrators

1. **Monitor System**: Check health status and system statistics
2. **Manage Documents**: Upload, categorize, and organize documents
3. **Manage Models**: Pull and configure LLM models
4. **View Analytics**: Track usage and performance metrics

## 🔍 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload document
- `GET /api/documents/` - List documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document

### Queries
- `POST /api/queries/ask` - Ask question
- `GET /api/queries/` - List queries
- `POST /api/queries/{id}/feedback` - Submit feedback

### Admin
- `GET /admin/stats` - System statistics
- `GET /admin/documents` - Manage documents
- `GET /admin/models` - List models
- `POST /admin/models/pull` - Pull model

### Health
- `GET /health/` - Comprehensive health check
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

## 🌍 Dutch Language Support

FATRAG is specifically designed for Dutch financial professionals:

- **Dutch Responses**: All AI responses in Dutch
- **Financial Terminology**: Specialized for Dutch financial terms
- **Compliance Language**: Dutch regulatory and compliance terminology
- **PII Detection**: Automatically removes personal information
- **Concise Summaries**: Bullet-point Dutch summaries

## 🔒 Security

- **No Secrets in Code**: All configuration via environment variables
- **Input Validation**: Comprehensive input sanitization
- **File Security**: Secure file upload with validation
- **Database Security**: Parameterized queries, connection security
- **Network Security**: CORS configuration, secure headers

## 📈 Monitoring

### Health Checks
```bash
# Overall health
curl http://localhost:8050/health/

# Component health
curl http://localhost:8050/health/database
curl http://localhost:8050/health/milvus
curl http://localhost:8050/health/ollama
```

### Logs
```bash
# Docker logs
docker-compose logs -f fatrag

# Application logs
docker-compose exec fatrag tail -f logs/app.log
```

## 🐛 Troubleshooting

### Common Issues

1. **Services won't start**: Check port conflicts and Docker resources
2. **Database connection**: Verify MySQL credentials and connectivity
3. **Ollama not responding**: Check if models are pulled and service is running
4. **Slow responses**: Monitor system resources and model performance

### Getting Help

1. Check [README_DOCKER.md](README_DOCKER.md) for Docker-specific issues
2. Review logs for error messages
3. Check API documentation for correct usage
4. Monitor system health endpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: For the excellent web framework
- **Milvus**: For vector database capabilities
- **Ollama**: For LLM model management
- **Tailwind CSS**: For the utility-first CSS framework
- **Font Awesome**: For the icon library

## 📞 Support

For support and questions:

1. Check the documentation first
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and system information when reporting issues

---

**FATRAG** - Empowering financial professionals with AI-powered document analysis and insights.
