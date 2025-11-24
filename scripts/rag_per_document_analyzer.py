#!/usr/bin/env python3
"""
RAG-Based Per-Document Analyzer for De Brem
Uses existing ingestion.py and rag_diagnostics.py patterns
"""
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import traceback

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import existing FATRAG modules
from ingestion import read_pdf_file, chunk_texts, ingest_texts
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# Targeted questions for notariele akten
DOCUMENT_QUESTIONS = [
    "Welke bedrijven of entiteiten worden genoemd in dit document?",
    "Welke transacties of rechtshandelingen vinden plaats?",
    "Welke bedragen, percentages of waarderingen worden genoemd?",
    "Welke belangrijke datums worden genoemd?",
    "Wie zijn de betrokken partijen (aandeelhouders, bestuurders, etc)?",
    "Wat is de aard van dit document (oprichting, inbreng, uitgifte, etc)?",
]

class PerDocumentRAGAnalyzer:
    def __init__(self):
        self.config = self.load_config()
        self.llm = self.init_llm()
        self.embed_model = self.init_embeddings()
        # We'll create temporary collections per document
        
    def load_config(self) -> Dict:
        config_path = ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            "vectordb": {
                "type": " chroma",
                "conn": str(ROOT / "fatrag_chroma_db"),
                "collection": "fatrag_docs"
            }
        }
    
    def init_llm(self):
        """Initialize LLM"""
        try:
            llm = ChatOllama(
                model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1:70b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.1,
                timeout=120,
            )
            print("✓ LLM initialized (llama3.1:70b)")
            return llm
        except Exception as e:
            print(f"ERROR initializing LLM: {e}")
            sys.exit(1)
    
    def init_embeddings(self):
        """Initialize embedding model"""
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    
    def create_document_vectorstore(self, doc_id: str) -> Chroma:
        """Create a temporary vectorstore for this document"""
        # Use hash to create short, valid collection name
        import hashlib
        doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
        collection_name = f"temp_{doc_hash}"
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed_model,
            persist_directory=str(ROOT / "fatrag_chroma_db")
        )
        return vectorstore
    
    def ingest_document(self, pdf_path: str, vectorstore: Chroma) -> int:
        """Ingest a single PDF into vectorstore"""
        print(f"  [1/3] Extracting and chunking PDF...")
        
        # Extract PDF
        text = read_pdf_file(pdf_path)
        
        if not text or len(text) < 100:
            print(f"    WARNING: PDF extraction yielded very little text ({len(text)} chars)")
            print(f"    PDF might be scan-based or empty")
            return 0
        
        print(f"    Extracted {len(text)} characters")
        
        # Chunk
        chunks = chunk_texts([text], chunk_size=500, chunk_overlap=100)
        print(f"    Created {len(chunks)} chunks")
        
        if not chunks:
            print(f"    WARNING: No chunks created from text")
            return 0
        
        # Ingest
        print(f"  [2/3] Embedding and indexing...")
        metadata = {
            "source": os.path.basename(pdf_path),
            "doc_id": os.path.basename(pdf_path),
        }
        n = ingest_texts(vectorstore, chunks, metadata, persist=True)
        print(f"    Indexed {n} chunks")
        
        return n
    
    def query_document(self, vectorstore: Chroma, question: str, k: int = 5) -> List[Dict]:
        """Query the document vectorstore"""
        try:
            results = vectorstore.similarity_search_with_score(question, k=k)
            hits = []
            for doc, score in results:
                hits.append({
                    "text": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                })
            return hits
        except Exception as e:
            print(f"    ERROR in query: {e}")
            return []
    
    def extract_answer(self, question: str, hits: List[Dict], doc_name: str) -> str:
        """Extract answer from retrieved chunks using LLM"""
        if not hits:
            return "GEEN INFORMATIE GEVONDEN"
        
        # Combine top 3 chunks
        context = "\n\n---\n\n".join(h["text"] for h in hits[:3])
        
        prompt = f"""Je bent een juridisch assistent gespecialiseerd in Nederlandse notariële akten.

Document: {doc_name}

Vraag: {question}

Context:
{context[:3000]}

Geef een kort, feitelijk antwoord gebaseerd ALLEEN op bovenstaande context.
Als de informatie niet in de context staat, antwoord: "NIET GEVONDEN".
Wees specifiek en noem concrete namen, bedragen, data indien aanwezig.

Antwoord:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def analyze_document(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze a single document via RAG"""
        doc_name = os.path.basename(pdf_path)
        print(f"\n{'='*70}")
        print(f"📄 Analyzing: {doc_name}")
        print(f"{'='*70}")
        
        result = {
            "filename": doc_name,
            "path": pdf_path,
            "timestamp": datetime.now().isoformat(),
            "qa_pairs": [],
            "summary": {},
        }
        
        try:
            # Create vectorstore for this document
            vectorstore = self.create_document_vectorstore(doc_name)
            
            # Ingest document
            chunk_count = self.ingest_document(pdf_path, vectorstore)
            result["chunk_count"] = chunk_count
            
            # Query with each question
            print(f"  [3/3] Querying document...")
            for i, question in enumerate(DOCUMENT_QUESTIONS, 1):
                print(f"    Q{i}: {question[:50]}...")
                
                # Retrieve relevant chunks
                hits = self.query_document(vectorstore, question, k=5)
                
                # Extract answer
                answer = self.extract_answer(question, hits, doc_name)
                
                result["qa_pairs"].append({
                    "question": question,
                    "answer": answer,
                    "chunks_used": len(hits),
                })
                
                print(f"      → {answer[:80]}...")
            
            # Create summary from all answers
            result["summary"] = self.create_summary(result["qa_pairs"], doc_name)
            
            print(f"\n✅ Analysis complete for {doc_name}")
            
            # Cleanup: delete temporary collection
            try:
                vectorstore._client.delete_collection(vectorstore._collection.name)
                print(f"  Cleaned up temporary collection")
            except:
                pass
            
        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ Error analyzing {doc_name}: {e}")
            traceback.print_exc()
        
        return result
    
    def create_summary(self, qa_pairs: List[Dict], doc_name: str) -> Dict:
        """Create structured summary from Q&A pairs"""
        # Extract key info from answers
        entities = []
        transactions = []
        amounts = []
        dates = []
        parties = []
        doc_type = "ONBEKEND"
        
        for qa in qa_pairs:
            answer = qa["answer"]
            if "NIET GEVONDEN" in answer or "GEEN INFORMATIE" in answer:
                continue
            
            # Try to categorize based on question
            if "bedrijven" in qa["question"].lower() or "entiteiten" in qa["question"].lower():
                entities.append(answer)
            elif "transacties" in qa["question"].lower():
                transactions.append(answer)
            elif "bedragen" in qa["question"].lower():
                amounts.append(answer)
            elif "datums" in qa["question"].lower():
                dates.append(answer)
            elif "partijen" in qa["question"].lower():
                parties.append(answer)
            elif "aard" in qa["question"].lower():
                doc_type = answer
        
        return {
            "document_type": doc_type,
            "entities": entities,
            "transactions": transactions,
            "amounts": amounts,
            "dates": dates,
            "parties": parties,
        }
    
    def analyze_all_documents(self, upload_dir: str) -> List[Dict]:
        """Analyze all PDFs in uploads directory"""
        upload_path = Path(upload_dir)
        pdf_files = sorted(upload_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {upload_dir}")
            return []
        
        print(f"\n{'='*70}")
        print(f"  RAG PER-DOCUMENT ANALYZER")
        print(f"{'='*70}")
        print(f"Found {len(pdf_files)} PDF files")
        print(f"{'='*70}\n")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}]")
            result = self.analyze_document(str(pdf_path))
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """Save results as JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual document results
        for result in results:
            filename = result["filename"].replace(".pdf", ".json")
            filepath = output_path / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath}")
        
        # Save combined results
        combined_path = output_path / "all_documents.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ All results saved: {combined_path}")


def main():
    upload_dir = ROOT / "fatrag_data" / "uploads"
    output_dir = ROOT / "outputs" / f"rag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        analyzer = PerDocumentRAGAnalyzer()
        results = analyzer.analyze_all_documents(str(upload_dir))
        analyzer.save_results(results, str(output_dir))
        
        print(f"\n{'='*70}")
        print(f"  ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Documents analyzed: {len(results)}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
        print(f"Next step: Run combine_business_structure.py to synthesize results")
        print(f"{'='*70}\n")
        
        return 0
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
