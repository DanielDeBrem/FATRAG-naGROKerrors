"""
Ollama integration service for FATRAG.
"""
from typing import List, Dict, Optional, Any
import httpx
import logging
import asyncio
from datetime import datetime

from src.core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama service for embeddings and text generation."""
    
    def __init__(self):
        self.base_url = settings.ollama_host
        self.embedding_model = settings.ollama_embedding_model
        self.generation_model = settings.ollama_generation_model
        self.timeout = settings.ollama_timeout
        
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make async request to Ollama API.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Dict: Response data
        """
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=data)
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            logger.error(f"Timeout connecting to Ollama at {self.base_url}")
            raise Exception(f"Ollama request timeout after {self.timeout}s")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error from Ollama: {e}")
            raise Exception(f"Ollama HTTP error: {e}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise
    
    async def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            bool: True if connection successful
        """
        try:
            url = f"{self.base_url}/api/tags"
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List[Dict]: List of available models
        """
        try:
            response = await self._make_request("tags", {})
            return response.get("models", [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in Ollama.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model exists
        """
        try:
            models = await self.list_models()
            return any(model.get("name") == model_name for model in models)
        except Exception as e:
            logger.error(f"Error checking model {model_name}: {e}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            bool: True if pull successful
        """
        try:
            logger.info(f"Pulling model {model_name}...")
            await self._make_request("pull", {"name": model_name})
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Model name (uses default if not provided)
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            model_name = model or self.embedding_model
            
            # Check if model exists
            if not await self.check_model_exists(model_name):
                logger.warning(f"Model {model_name} not found, attempting to pull...")
                if not await self.pull_model(model_name):
                    raise Exception(f"Failed to pull model {model_name}")
            
            data = {
                "model": model_name,
                "prompt": text
            }
            
            response = await self._make_request("embeddings", data)
            embedding = response.get("embedding", [])
            
            if not embedding:
                raise Exception("Empty embedding returned")
            
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Model name (uses default if not provided)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        model_name = model or self.embedding_model
        
        logger.info(f"Generating embeddings for {len(texts)} texts using model {model_name}")
        
        for i, text in enumerate(texts):
            try:
                embedding = await self.generate_embedding(text, model_name)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} embeddings")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for text {i}: {e}")
                # Add empty embedding to maintain order
                embeddings.append([])
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text response.
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if not provided)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        try:
            model_name = model or self.generation_model
            
            # Check if model exists
            if not await self.check_model_exists(model_name):
                logger.warning(f"Model {model_name} not found, attempting to pull...")
                if not await self.pull_model(model_name):
                    raise Exception(f"Failed to pull model {model_name}")
            
            # Build full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt += f"System: {system_prompt}\n\n"
            full_prompt += f"User: {prompt}\n\nAssistant: "
            
            data = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                data["options"]["num_predict"] = max_tokens
            
            response = await self._make_request("generate", data)
            generated_text = response.get("response", "")
            
            # Clean up the response
            if generated_text.startswith("Assistant: "):
                generated_text = generated_text[len("Assistant: "):]
            
            logger.debug(f"Generated response of length {len(generated_text)}")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_dutch_summary(
        self,
        text: str,
        max_length: int = 500,
        focus_financial: bool = True
    ) -> str:
        """
        Generate Dutch summary of text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            focus_financial: Whether to focus on financial aspects
            
        Returns:
            str: Dutch summary
        """
        try:
            system_prompt = """Je bent een financiële assistent die samenvattingen maakt in het Nederlands.
            Maak beknopte, bullet-point samenvattingen.
            Vermijd persoonlijke informatie (PII).
            Focus op financiële en fiscale aspecten indien aanwezig.
            Gebruik "onvoldoende data" als er niet genoeg informatie is."""
            
            if focus_financial:
                user_prompt = f"""Maak een Nederlandse samenvatting van de volgende tekst, focus op financiële en fiscale informatie:
                
                {text}
                
                Houd de samenvatting binnen {max_length} karakters."""
            else:
                user_prompt = f"""Maak een Nederlandse samenvatting van de volgende tekst:
                
                {text}
                
                Houd de samenvatting binnen {max_length} karakters."""
            
            summary = await self.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=max_length // 4  # Rough estimate of tokens needed
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating Dutch summary: {e}")
            return "onvoldoende data"
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Ollama service.
        
        Returns:
            Dict: Health check results
        """
        try:
            # Check basic connection
            is_connected = await self.check_connection()
            if not is_connected:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "error": "Cannot connect to Ollama",
                    "base_url": self.base_url
                }
            
            # List available models
            models = await self.list_models()
            model_names = [model.get("name") for model in models]
            
            # Check if default models are available
            embedding_available = self.embedding_model in model_names
            generation_available = self.generation_model in model_names
            
            return {
                "status": "healthy",
                "connected": True,
                "base_url": self.base_url,
                "models": model_names,
                "model_count": len(models),
                "default_embedding_model": {
                    "name": self.embedding_model,
                    "available": embedding_available
                },
                "default_generation_model": {
                    "name": self.generation_model,
                    "available": generation_available
                }
            }
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "base_url": self.base_url
            }


# Global Ollama service instance
ollama_service = OllamaService()
