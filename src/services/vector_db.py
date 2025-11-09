"""
Milvus vector database integration for FATRAG.
"""
from typing import List, Dict, Optional, Any
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
import numpy as np
import logging
from datetime import datetime

from src.core.config import settings

logger = logging.getLogger(__name__)


class MilvusService:
    """Milvus vector database service."""
    
    def __init__(self):
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.user = settings.milvus_user
        self.password = settings.milvus_password
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to Milvus server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            connect_params = {
                "host": self.host,
                "port": self.port
            }
            
            if self.user and self.password:
                connect_params["user"] = self.user
                connect_params["password"] = self.password
            
            connections.connect(**connect_params)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def create_collection(self, collection_name: str, dimension: int = 768) -> Collection:
        """
        Create a collection for document chunks.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            
        Returns:
            Collection: Created collection
        """
        try:
            # Check if collection already exists
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return Collection(collection_name)
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="document_category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="financial_year", dtype=DataType.INT64),
                FieldSchema(name="contains_financial_data", dtype=DataType.BOOL),
                FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Document chunks collection for {collection_name}"
            )
            
            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created collection {collection_name} with dimension {dimension}")
            return collection
            
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def insert_chunks(self, collection_name: str, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Insert document chunks into collection.
        
        Args:
            collection_name: Name of the collection
            chunks_data: List of chunk data
            
        Returns:
            List[str]: List of inserted chunk IDs
        """
        try:
            collection = Collection(collection_name)
            
            # Prepare data for insertion
            ids = []
            document_ids = []
            chunk_indices = []
            contents = []
            page_numbers = []
            section_titles = []
            chunk_types = []
            document_categories = []
            company_names = []
            financial_years = []
            contains_financial_data = []
            embedding_models = []
            created_ats = []
            embeddings = []
            
            for chunk in chunks_data:
                ids.append(chunk["id"])
                document_ids.append(chunk["document_id"])
                chunk_indices.append(chunk["chunk_index"])
                contents.append(chunk["content"])
                page_numbers.append(chunk.get("page_number", -1))
                section_titles.append(chunk.get("section_title", ""))
                chunk_types.append(chunk.get("chunk_type", "text"))
                document_categories.append(chunk.get("document_category", ""))
                company_names.append(chunk.get("company_name", ""))
                financial_years.append(chunk.get("financial_year", -1))
                contains_financial_data.append(chunk.get("contains_financial_data", False))
                embedding_models.append(chunk.get("embedding_model", ""))
                created_ats.append(int(datetime.now().timestamp()))
                embeddings.append(chunk["embedding"])
            
            # Insert data
            data = [
                ids, document_ids, chunk_indices, contents, page_numbers,
                section_titles, chunk_types, document_categories, company_names,
                financial_years, contains_financial_data, embedding_models,
                created_ats, embeddings
            ]
            
            collection.insert(data)
            collection.flush()
            
            logger.info(f"Inserted {len(chunks_data)} chunks into {collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Error inserting chunks into {collection_name}: {e}")
            raise
    
    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        expr_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query vector
            limit: Number of results to return
            expr_filter: Optional filter expression
            
        Returns:
            List[Dict]: Search results
        """
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr_filter,
                output_fields=[
                    "id", "document_id", "chunk_index", "content", "page_number",
                    "section_title", "chunk_type", "document_category", "company_name",
                    "financial_year", "contains_financial_data", "embedding_model", "created_at"
                ]
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": hit.entity.get("id"),
                    "document_id": hit.entity.get("document_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "content": hit.entity.get("content"),
                    "page_number": hit.entity.get("page_number"),
                    "section_title": hit.entity.get("section_title"),
                    "chunk_type": hit.entity.get("chunk_type"),
                    "document_category": hit.entity.get("document_category"),
                    "company_name": hit.entity.get("company_name"),
                    "financial_year": hit.entity.get("financial_year"),
                    "contains_financial_data": hit.entity.get("contains_financial_data"),
                    "embedding_model": hit.entity.get("embedding_model"),
                    "created_at": hit.entity.get("created_at"),
                    "score": hit.score
                })
            
            logger.info(f"Found {len(formatted_results)} similar chunks in {collection_name}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {e}")
            raise
    
    def delete_by_document_id(self, collection_name: str, document_id: int) -> bool:
        """
        Delete chunks by document ID.
        
        Args:
            collection_name: Name of the collection
            document_id: Document ID
            
        Returns:
            bool: True if deletion successful
        """
        try:
            collection = Collection(collection_name)
            
            # Delete entities by document_id
            expr = f"document_id == {document_id}"
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"Deleted chunks for document {document_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunks from {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict: Collection statistics
        """
        try:
            if not utility.has_collection(collection_name):
                return {"exists": False}
            
            collection = Collection(collection_name)
            stats = {
                "exists": True,
                "name": collection_name,
                "num_entities": collection.num_entities,
                "schema": collection.schema.to_dict()
            }
            
            # Get index information
            indexes = collection.indexes
            if indexes:
                stats["index"] = indexes[0].to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return {"exists": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Milvus.
        
        Returns:
            Dict: Health check results
        """
        try:
            if not self.connected:
                return {"status": "disconnected", "error": "Not connected to Milvus"}
            
            # Check server version
            server_version = utility.get_server_version()
            
            # List collections
            collections = utility.list_collections()
            
            return {
                "status": "healthy",
                "connected": True,
                "server_version": server_version,
                "host": self.host,
                "port": self.port,
                "collections": collections,
                "collection_count": len(collections)
            }
            
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "host": self.host,
                "port": self.port
            }


# Global Milvus service instance
milvus_service = MilvusService()
