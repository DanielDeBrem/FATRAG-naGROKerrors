# FATRAG Docker Setup Guide

This guide explains how to set up and run FATRAG using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- At least 20GB free disk space

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FATRAG
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.docker .env
   # Edit .env with your secure passwords and settings
   nano .env
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Wait for services to be ready (2-5 minutes):**
   ```bash
   docker-compose logs -f fatrag
   ```

5. **Access the application:**
   - Main application: http://localhost:8050
   - Admin interface: http://localhost:8050/admin
   - Health check: http://localhost:8050/health
   - API docs: http://localhost:8050/docs

## Services Overview

### FATRAG Application (Port 8050)
- Main FastAPI application
- Automatically connects to all required services
- Health checks ensure proper startup order

### MySQL Database (Port 3306)
- Metadata storage for documents, queries, jobs
- User: `fatrag`, Database: `fatrag`
- Data persisted in `mysql_data` volume

### Milvus Vector Database (Port 19530)
- Vector storage for document embeddings
- Includes etcd and MinIO dependencies
- Web UI available at http://localhost:9091

### Redis (Port 6379)
- Background task queue and caching
- Data persisted in `redis_data` volume

### Ollama (Port 11434)
- LLM service for embeddings and generation
- Models stored in `ollama_data` volume
- Default models: `nomic-embed-text`, `llama2`

### Nginx (Optional, Ports 80/443)
- Reverse proxy for production deployments
- Enabled with: `docker-compose --profile production up`

## Environment Variables

### Required Variables
- `MYSQL_ROOT_PASSWORD`: MySQL root password
- `MYSQL_PASSWORD`: MySQL user password
- `SECRET_KEY`: Application secret key (32+ characters)

### Optional Variables
- `OLLAMA_EMBEDDING_MODEL`: Default embedding model
- `OLLAMA_GENERATION_MODEL`: Default generation model
- `LOG_LEVEL`: Application logging level
- `MAX_FILE_SIZE`: Maximum upload file size

## Common Operations

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fatrag
docker-compose logs -f mysql
docker-compose logs -f milvus
```

### Stop Services
```bash
docker-compose down
```

### Stop and Remove Volumes
```bash
docker-compose down -v
# WARNING: This deletes all data!
```

### Update Services
```bash
docker-compose pull
docker-compose up -d
```

### Access Service Containers
```bash
# Application container
docker-compose exec fatrag bash

# MySQL container
docker-compose exec mysql mysql -u fatrag -p fatrag

# Ollama container
docker-compose exec ollama ollama list
```

## Managing Ollama Models

### Pull Additional Models
```bash
docker-compose exec ollama ollama pull mistral
docker-compose exec ollama ollama pull codellama
```

### List Available Models
```bash
docker-compose exec ollama ollama list
```

### Change Default Models
Edit `.env`:
```bash
OLLAMA_EMBEDDING_MODEL=mistral
OLLAMA_GENERATION_MODEL=codellama
```

Then restart:
```bash
docker-compose restart fatrag
```

## Development Setup

### Development Mode
```bash
# Enable debug mode
export APP_DEBUG=true
docker-compose up --build
```

### Hot Reload
For development with hot reload:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Production Deployment

### Production Mode
```bash
# Use production profile (includes Nginx)
docker-compose --profile production up -d
```

### SSL Configuration
1. Place SSL certificates in `docker/nginx/ssl/`
2. Update `docker/nginx/nginx.conf` with your domain
3. Restart with production profile

### Backup Data
```bash
# MySQL backup
docker-compose exec mysql mysqldump -u root -p fatrag > backup.sql

# Volume backup
docker run --rm -v fatrag_mysql_data:/data -v $(pwd):/backup alpine tar czf /backup/mysql_backup.tar.gz -C /data .
```

## Troubleshooting

### Service Won't Start
1. Check port conflicts: `netstat -tulpn | grep -E ':(3306|8050|19530|11434)'`
2. Check Docker resources: `docker system df`
3. Check logs: `docker-compose logs`

### Database Connection Issues
```bash
# Test MySQL connection
docker-compose exec mysql mysql -u fatrag -p fatrag -e "SELECT 1"

# Reset database
docker-compose down -v
docker-compose up mysql
```

### Milvus Issues
```bash
# Check Milvus status
curl http://localhost:9091/healthz

# Reset Milvus data
docker-compose down -v
docker-compose up etcd minio milvus
```

### Ollama Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull models manually
docker-compose exec ollama ollama pull llama2
docker-compose exec ollama ollama pull nomic-embed-text
```

### Performance Issues
1. Increase Docker memory limits
2. Check system resources: `docker stats`
3. Monitor logs for errors
4. Consider scaling services

## Monitoring

### Health Checks
All services include health checks:
```bash
docker-compose ps
# Look for "healthy" status
```

### Application Health
- http://localhost:8050/health - Comprehensive health
- http://localhost:8050/health/live - Liveness probe
- http://localhost:8050/health/ready - Readiness probe

### Logs Monitoring
```bash
# Follow application logs
docker-compose logs -f fatrag

# Check error logs
docker-compose logs fatrag | grep ERROR
```

## Security Considerations

1. **Change default passwords** in `.env`
2. **Use strong secrets** for `SECRET_KEY`
3. **Enable firewalls** to restrict port access
4. **Use HTTPS** in production with Nginx
5. **Regular updates** of Docker images
6. **Monitor logs** for suspicious activity

## Support

For issues:
1. Check this guide first
2. Review Docker logs
3. Check GitHub Issues
4. Provide logs with bug reports

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FATRAG    │    │   MySQL     │    │   Milvus    │
│   (8050)    │◄──►│   (3306)    │◄──►│  (19530)    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Redis     │    │   Ollama    │    │   Nginx     │
│   (6379)    │    │  (11434)    │    │   (80/443)   │
└─────────────┘    └─────────────┘    └─────────────┘
