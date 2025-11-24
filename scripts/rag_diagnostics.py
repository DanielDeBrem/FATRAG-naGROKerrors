#!/usr/bin/env python3
"""
Autonomous RAG Diagnostics Pipeline for FATRAG
Executes full Evidence-First retrieval evaluation with audit report
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

# Disable ChromaDB telemetry to prevent PostHog errors
try:
    from scripts.chroma_telemetry_fix import *
except:
    pass

# Import existing FATRAG modules
try:
    import chromadb
    from langchain_ollama import ChatOllama
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Install: pip install chromadb langchain-ollama")
    sys.exit(1)

class RAGDiagnostics:
    def __init__(self):
        self.config = self.load_config()
        self.log_dir = ROOT / "logs"
        self.output_dir = ROOT / "outputs" / "diagnostics"
        self.log_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize vector store
        self.vdb = self.init_vectorstore()
        self.llm = self.init_llm()
        
        # Seed questions
        self.questions = self.load_questions()
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "vdb_info": {
                "type": self.config["vectordb"]["type"],
                "collection": self.config["vectordb"]["collection"],
            },
            "questions": [],
            "metrics": {},
        }
    
    def load_config(self) -> Dict:
        config_path = ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def load_questions(self) -> List[str]:
        q_file = ROOT / "scripts" / "seed_questions.txt"
        if not q_file.exists():
            return []
        questions = []
        with open(q_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    questions.append(line)
        return questions
    
    def init_vectorstore(self):
        """Initialize Chroma vectorstore (read-only)"""
        try:
            chroma_dir = self.config["vectordb"]["conn"]
            collection = self.config["vectordb"]["collection"]
            
            embed_model = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            
            vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embed_model,
                collection_name=collection
            )
            
            # Test query
            test_results = vectorstore.similarity_search("test", k=1)
            print(f"✓ VDB initialized: {len(test_results)} docs accessible")
            
            return vectorstore
        except Exception as e:
            print(f"ERROR initializing vectorstore: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def init_llm(self):
        """Initialize LLM (using existing Ollama)"""
        try:
            llm = ChatOllama(
                model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.1,
                timeout=60,
            )
            # Test
            llm.invoke("test")
            print("✓ LLM initialized")
            return llm
        except Exception as e:
            print(f"ERROR initializing LLM: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def retrieve(self, question: str, k: int = 10) -> List[Dict]:
        """Simple dense retrieval from Chroma"""
        try:
            results = self.vdb.similarity_search_with_score(question, k=k)
            hits = []
            for doc, score in results:
                hits.append({
                    "text": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "doc_id": doc.metadata.get("source", "unknown"),
                })
            return hits
        except Exception as e:
            print(f"ERROR in retrieval: {e}")
            return []
    
    def extract_evidence_l1(self, question: str, hits: List[Dict]) -> List[Dict]:
        """L1: Extract evidence from each hit using LLM"""
        evidence_list = []
        
        for i, hit in enumerate(hits[:5]):  # Top 5 for speed
            prompt = f"""Rol: register-paralegal NL. Taak: haal uitsluitend bewijs uit onderstaande context.

Vraag: {question}

Context (bron={hit['doc_id']}):
---
{hit['text'][:2000]}
---

Regels:
- Alleen feiten uit de context; geen extrapolaties.
- Noteer citaten (max 2 zinnen) met bron.
- Extraheer: facts, entities, amounts, dates.
- GEEN conclusies.

Geef je antwoord als JSON met keys: facts (list), entities (list), amounts (list), dates (list), citations (list), confidence (0-1)."""

            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Try to parse JSON
                try:
                    evidence = json.loads(content)
                except:
                    evidence = {
                        "facts": [],
                        "entities": [],
                        "amounts": [],
                        "dates": [],
                        "citations": [content[:200]],
                        "confidence": 0.5
                    }
                
                evidence["hit_index"] = i
                evidence["doc_id"] = hit["doc_id"]
                evidence_list.append(evidence)
                
            except Exception as e:
                print(f"  L1 extraction error for hit {i}: {e}")
                continue
        
        return evidence_list
    
    def synthesize_l2(self, question: str, l1_evidence: List[Dict]) -> Dict:
        """L2: Synthesize evidence across chunks"""
        # Combine all evidence
        all_facts = []
        all_entities = set()
        all_amounts = []
        all_dates = []
        all_citations = []
        
        for ev in l1_evidence:
            all_facts.extend(ev.get("facts", []))
            all_entities.update(ev.get("entities", []))
            all_amounts.extend(ev.get("amounts", []))
            all_dates.extend(ev.get("dates", []))
            all_citations.extend(ev.get("citations", []))
        
        return {
            "question": question,
            "total_evidence_chunks": len(l1_evidence),
            "summary": {
                "facts_count": len(all_facts),
                "entities_count": len(all_entities),
                "amounts_count": len(all_amounts),
                "dates_count": len(all_dates),
                "citations_count": len(all_citations),
            },
            "facts": all_facts[:20],  # Top 20
            "entities": list(all_entities)[:20],
            "amounts": all_amounts[:20],
            "dates": all_dates[:20],
            "citations": all_citations[:10],
        }
    
    def generate_final_answer(self, question: str, l2: Dict) -> str:
        """Final: Generate strict evidence-based answer"""
        
        prompt = f"""Rol: fiscalist NL 2025. Antwoord uitsluitend op basis van FEITEN en CITATEN uit L2.

Vraag: {question}

L2 Evidence:
- {l2['summary']['facts_count']} feiten
- {l2['summary']['entities_count']} entiteiten
- {l2['summary']['amounts_count']} bedragen

Feiten:
{chr(10).join('- ' + f for f in l2['facts'][:10])}

Citaten:
{chr(10).join('- ' + c for c in l2['citations'][:5])}

Regels:
- Elke bewering krijgt een voetnoot [doc].
- Bij onvoldoende bewijs: antwoord exact 'ONVOLDOENDE CONTEXT'.
- Geen nieuwe feiten buiten de citaten.

Output:
1) TL;DR (max 3 bullets)
2) Antwoord met voetnoten
3) Wat ontbreekt (indien van toepassing)"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer
        except Exception as e:
            return f"ERROR generating answer: {e}"
    
    def process_question(self, question: str) -> Dict:
        """Process single question through full pipeline"""
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        
        result = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # 1. Retrieve
            print("  [1/4] Retrieving...")
            hits = self.retrieve(question, k=10)
            result["retrieval"] = {
                "hits_count": len(hits),
                "top_scores": [h["score"] for h in hits[:5]],
                "top_docs": [h["doc_id"] for h in hits[:3]],
            }
            print(f"    Retrieved {len(hits)} chunks")
            
            # 2. L1 Evidence
            print("  [2/4] Extracting evidence (L1)...")
            l1_evidence = self.extract_evidence_l1(question, hits)
            result["l1_evidence_count"] = len(l1_evidence)
            print(f"    Extracted evidence from {len(l1_evidence)} chunks")
            
            # 3. L2 Synthesis
            print("  [3/4] Synthesizing (L2)...")
            l2 = self.synthesize_l2(question, l1_evidence)
            result["l2"] = l2
            print(f"    Synthesized: {l2['summary']}")
            
            # 4. Final Answer
            print("  [4/4] Generating final answer...")
            answer = self.generate_final_answer(question, l2)
            result["answer"] = answer
            print(f"    Answer length: {len(answer)} chars")
            
            # Check for "ONVOLDOENDE CONTEXT"
            result["has_answer"] = "ONVOLDOENDE CONTEXT" not in answer.upper()
            
        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()
        
        return result
    
    def run_pipeline(self):
        """Execute full pipeline on all questions"""
        print("\n" + "="*70)
        print(" FATRAG RAG DIAGNOSTICS PIPELINE")
        print("="*70)
        print(f"VDB: {self.config['vectordb']['type']} @ {self.config['vectordb']['conn']}")
        print(f"Collection: {self.config['vectordb']['collection']}")
        print(f"Questions: {len(self.questions)}")
        print("="*70)
        
        for question in self.questions:
            result = self.process_question(question)
            self.results["questions"].append(result)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Save results
        self.save_results()
        
        # Generate audit
        self.generate_audit()
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE")
        print("="*70)
        print(f"Results: {self.output_dir / 'results.json'}")
        print(f"Audit:   {ROOT / 'report' / 'audit.md'}")
        print("="*70 + "\n")
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        total = len(self.results["questions"])
        answered = sum(1 for q in self.results["questions"] if q.get("has_answer", False))
        no_answer = total - answered
        
        avg_hits = sum(q.get("retrieval", {}).get("hits_count", 0) for q in self.results["questions"]) / max(total, 1)
        avg_evidence = sum(q.get("l1_evidence_count", 0) for q in self.results["questions"]) / max(total, 1)
        
        self.results["metrics"] = {
            "total_questions": total,
            "answered": answered,
            "no_answer_rate": no_answer / max(total, 1),
            "avg_hits_retrieved": avg_hits,
            "avg_evidence_chunks": avg_evidence,
        }
    
    def save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved: {output_file}")
    
    def generate_audit(self):
        """Generate audit.md report"""
        report_dir = ROOT / "report"
        report_dir.mkdir(exist_ok=True)
        audit_file = report_dir / "audit.md"
        
        metrics = self.results["metrics"]
        
        audit = f"""# FATRAG RAG Diagnostics Audit

**Gegenereerd:** {self.results['timestamp']}  
**Vector DB:** {self.results['vdb_info']['type']}  
**Collection:** {self.results['vdb_info']['collection']}  

## Samenvatting

- **Totaal vragen:** {metrics['total_questions']}
- **Beantwoord:** {metrics['answered']} ({(1-metrics['no_answer_rate'])*100:.1f}%)
- **No-answer rate:** {metrics['no_answer_rate']*100:.1f}%
- **Gem. chunks opgehaald:** {metrics['avg_hits_retrieved']:.1f}
- **Gem. evidence chunks:** {metrics['avg_evidence_chunks']:.1f}

## Resultaten per Vraag

"""
        
        for i, q in enumerate(self.results["questions"], 1):
            status = "✅ Beantwoord" if q.get("has_answer") else "❌ Geen context"
            audit += f"""
### {i}. {q['question']}

**Status:** {status}  
**Chunks retrieved:** {q.get('retrieval', {}).get('hits_count', 0)}  
**Evidence chunks:** {q.get('l1_evidence_count', 0)}  

**Top docs:**
{chr(10).join('- ' + d for d in q.get('retrieval', {}).get('top_docs', []))}

**L2 Summary:**
- Facts: {q.get('l2', {}).get('summary', {}).get('facts_count', 0)}
- Entities: {q.get('l2', {}).get('summary', {}).get('entities_count', 0)}
- Amounts: {q.get('l2', {}).get('summary', {}).get('amounts_count', 0)}

**Answer preview:**
```
{q.get('answer', 'N/A')[:300]}...
```

---
"""
        
        audit += """
## Aanbevelingen

### Retrieval
- ✓ Dense retrieval werkt (Chroma)
- Consider: Hybride retrieval (BM25 + dense) voor betere recall
- Consider: Cross-encoder reranking voor precisie

### Chunking
- Check: Chunk sizes (huidige overlap/size optimaal?)
- Check: Metadata preservation (page numbers, sections)

### Prompting
- ✓ Evidence-first aanpak
- Consider: Strengere citatie-eisen in L1
- Consider: Conflict resolution in L2

### Evaluatie
- **Volgende stap:** Goldset (20-50 vragen + ground truth) voor Recall@k meting

## Checklist Verbeteringen

- [ ] Implementeer hybride retrieval (BM25 + dense)
- [ ] Add cross-encoder reranking
- [ ] Optimize chunk size/overlap
- [ ] Strengthen citation requirements in prompts
- [ ] Create goldset for quantitative eval
- [ ] Add hallucination detection
- [ ] Implement confidence thresholds

"""
        
        with open(audit_file, "w", encoding="utf-8") as f:
            f.write(audit)
        
        print(f"✓ Audit saved: {audit_file}")


def main():
    try:
        diagnostics = RAGDiagnostics()
        diagnostics.run_pipeline()
        return 0
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
