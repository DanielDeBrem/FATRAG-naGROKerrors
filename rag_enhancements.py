"""
RAG Quality Enhancements Module
Implements advanced retrieval and generation strategies to improve answer quality
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import json


# ============= QUERY ENHANCEMENT =============

def expand_query(query: str, synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Expand query with synonyms and related terms
    Returns list of query variations
    """
    if synonyms is None:
        # Dutch financial/fiscal synonyms
        synonyms = {
            "belasting": ["heffing", "fiscaal", "tax", "afdracht"],
            "waarde": ["valuta", "bedrag", "prijs", "taxatie"],
            "aandeel": ["participatie", "eigendom", "deelneming"],
            "schenking": ["gift", "donatie", "overdracht"],
            "erfenis": ["nalatenschap", "testament", "legaat"],
            "schuld": ["verplichting", "lening", "vordering"],
            "vermogen": ["kapitaal", "bezit", "activa"],
            "winst": ["resultaat", "rendement", "opbrengst"],
        }
    
    queries = [query]
    query_lower = query.lower()
    
    # Add synonym variations
    for term, syns in synonyms.items():
        if term in query_lower:
            for syn in syns[:2]:  # Max 2 synonyms per term
                queries.append(query_lower.replace(term, syn))
    
    return queries[:5]  # Max 5 query variations


def generate_hyde_document(query: str, llm_func) -> str:
    """
    HyDE (Hypothetical Document Embeddings)
    Generate a hypothetical answer, then use it for retrieval
    """
    hyde_prompt = f"""Genereer een kort hypothetisch antwoord op deze vraag, alsof je de relevante documenten al hebt gelezen.
Wees specifiek en gebruik concrete termen die waarschijnlijk in relevante documenten voorkomen.

Vraag: {query}

Hypothetisch antwoord (2-3 zinnen):"""
    
    try:
        hypothetical = llm_func(hyde_prompt)
        return hypothetical
    except Exception:
        return query  # Fallback to original query


def split_multi_query(query: str) -> List[str]:
    """
    Split complex queries into simpler sub-queries
    """
    # Check if query contains multiple questions
    if "en" in query.lower() and ("?" in query or len(query.split()) > 15):
        # Split on conjunctions
        parts = re.split(r'\s+en\s+|,\s+', query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if len(p.strip()) > 10]
    
    return [query]


# ============= HYBRID RETRIEVAL =============

def bm25_score(query: str, document: str) -> float:
    """
    Simple BM25-like scoring for keyword relevance
    Not full BM25 implementation but captures key concept
    """
    # Tokenize
    query_terms = set(query.lower().split())
    doc_terms = document.lower().split()
    doc_term_freq = Counter(doc_terms)
    
    score = 0.0
    k1 = 1.5  # Term frequency saturation parameter
    
    for term in query_terms:
        if term in doc_term_freq:
            tf = doc_term_freq[term]
            # Simplified BM25 formula
            score += (tf * (k1 + 1)) / (tf + k1)
    
    return score


def hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Combine vector similarity with keyword matching
    
    Args:
        query: Search query
        vector_results: Results from vector search with 'content' and 'score'
        vector_weight: Weight for vector similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
    
    Returns:
        Reranked results
    """
    # Add keyword scores
    for result in vector_results:
        content = result.get('content', '')
        keyword_score = bm25_score(query, content)
        result['keyword_score'] = keyword_score
        
        # Normalize scores to 0-1 range
        vector_score = result.get('score', 0.0)
        normalized_vector = min(1.0, max(0.0, vector_score))
        normalized_keyword = min(1.0, keyword_score / 10.0)  # Scale keyword score
        
        # Hybrid score
        result['hybrid_score'] = (
            vector_weight * normalized_vector +
            keyword_weight * normalized_keyword
        )
    
    # Sort by hybrid score
    return sorted(vector_results, key=lambda x: x.get('hybrid_score', 0), reverse=True)


def maximal_marginal_relevance(
    query: str,
    documents: List[Dict[str, Any]],
    lambda_param: float = 0.5,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    MMR (Maximal Marginal Relevance) for diversity in results
    Reduces redundancy by selecting diverse documents
    
    Args:
        query: Search query
        documents: List with 'content' and 'score'
        lambda_param: Trade-off between relevance and diversity (0-1)
        top_k: Number of documents to return
    """
    if not documents:
        return []
    
    selected = []
    remaining = documents.copy()
    
    # First document: highest relevance
    first = max(remaining, key=lambda x: x.get('score', 0))
    selected.append(first)
    remaining.remove(first)
    
    # Select remaining documents
    while len(selected) < top_k and remaining:
        max_mmr = -float('inf')
        best_doc = None
        
        for doc in remaining:
            relevance = doc.get('score', 0)
            
            # Calculate max similarity to already selected docs
            max_sim = 0.0
            for sel_doc in selected:
                # Simple similarity: overlap of words
                doc_words = set(doc.get('content', '').lower().split())
                sel_words = set(sel_doc.get('content', '').lower().split())
                if doc_words and sel_words:
                    similarity = len(doc_words & sel_words) / len(doc_words | sel_words)
                    max_sim = max(max_sim, similarity)
            
            # MMR formula
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr > max_mmr:
                max_mmr = mmr
                best_doc = doc
        
        if best_doc:
            selected.append(best_doc)
            remaining.remove(best_doc)
        else:
            break
    
    return selected


# ============= CHAIN-OF-THOUGHT PROMPTING =============

def build_cot_prompt(query: str, context: str, domain: str = "financial") -> str:
    """
    Build Chain-of-Thought prompt for better reasoning
    """
    if domain == "financial":
        return f"""Je bent een senior financieel adviseur. Analyseer deze vraag stap-voor-stap:

Vraag: {query}

Context uit documenten:
{context}

Denk stap-voor-stap:

1. KERNVRAAG IDENTIFICEREN
   - Wat wordt precies gevraagd?
   - Welke specifieke informatie is nodig?

2. RELEVANTE FEITEN ZOEKEN
   - Welke bedragen/percentages zijn relevant?
   - Welke data/termijnen zijn belangrijk?
   - Welke partijen zijn betrokken?

3. NEDERLANDSE FISCALE REGELS TOEPASSEN (2025)
   - Welke belastingregels zijn van toepassing?
   - Zijn er vrijstellingen of aftrekposten?
   - Wat zijn de aandachtspunten?

4. ADVIES FORMULEREN
   - Wat is het antwoord op de vraag?
   - Hoe onderbouw je dit met de feiten?
   - Wat zijn de vervolgstappen?

Geef je eindantwoord in puntvorm, professioneel Nederlands, zonder PII."""
    
    return f"""Analyseer deze vraag stap-voor-stap:

Vraag: {query}

Context: {context}

Redenering:
1. Identificeer kernvraag
2. Zoek relevante feiten in context
3. Pas domeinkennis toe
4. Formuleer antwoord met onderbouwing

Antwoord:"""


# ============= SELF-CONSISTENCY & VERIFICATION =============

def generate_multiple_answers(
    query: str,
    context: str,
    llm_func,
    n: int = 3,
    temperatures: List[float] = None
) -> List[str]:
    """
    Generate multiple answers with different temperatures
    For self-consistency checking
    """
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5]
    
    answers = []
    prompt = build_cot_prompt(query, context)
    
    for temp in temperatures[:n]:
        try:
            answer = llm_func(prompt, temperature=temp)
            answers.append(answer)
        except Exception:
            continue
    
    return answers


def verify_answer_against_context(answer: str, context: str) -> Dict[str, Any]:
    """
    Verify if answer is grounded in context
    Detect potential hallucinations
    """
    result = {
        "grounded": True,
        "confidence": 1.0,
        "issues": []
    }
    
    # Extract claims with specific values (bedragen, percentages, dates)
    amount_pattern = r'€\s?[\d.,]+'
    pct_pattern = r'\d+[,.]?\d*\s?%'
    date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{4}'
    
    answer_amounts = set(re.findall(amount_pattern, answer))
    answer_pcts = set(re.findall(pct_pattern, answer))
    answer_dates = set(re.findall(date_pattern, answer))
    
    context_amounts = set(re.findall(amount_pattern, context))
    context_pcts = set(re.findall(pct_pattern, context))
    context_dates = set(re.findall(date_pattern, context))
    
    # Check if answer values are in context
    ungrounded_amounts = answer_amounts - context_amounts
    ungrounded_pcts = answer_pcts - context_pcts
    ungrounded_dates = answer_dates - context_dates
    
    if ungrounded_amounts:
        result["issues"].append(f"Bedragen niet in context: {ungrounded_amounts}")
        result["grounded"] = False
        result["confidence"] *= 0.7
    
    if ungrounded_pcts:
        result["issues"].append(f"Percentages niet in context: {ungrounded_pcts}")
        result["grounded"] = False
        result["confidence"] *= 0.7
    
    if ungrounded_dates:
        result["issues"].append(f"Data niet in context: {ungrounded_dates}")
        result["grounded"] = False
        result["confidence"] *= 0.8
    
    # Check for hedging language
    hedging_phrases = [
        "onvoldoende data",
        "niet duidelijk",
        "mogelijk",
        "waarschijnlijk",
        "vermoedelijk"
    ]
    
    hedging_count = sum(1 for phrase in hedging_phrases if phrase in answer.lower())
    if hedging_count > 0:
        result["confidence"] *= (1 - hedging_count * 0.1)
    
    return result


# ============= EVIDENCE EXTRACTION WITH CITATIONS =============

def extract_evidence_with_citations(
    answer: str,
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract evidence and link to source chunks
    
    Args:
        answer: Generated answer
        context_chunks: List of dicts with 'content', 'source', 'chunk_id'
    
    Returns:
        Dict with claims and their citations
    """
    evidence = {
        "claims": [],
        "citations": [],
        "coverage": 0.0
    }
    
    # Extract statements from answer (sentences ending with .)
    statements = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    
    for stmt in statements:
        claim = {
            "statement": stmt,
            "cited_chunks": [],
            "confidence": 0.0
        }
        
        # Find which chunks support this claim
        for chunk in context_chunks:
            content = chunk.get('content', '')
            # Simple overlap check (in production, use semantic similarity)
            overlap_score = calculate_text_overlap(stmt, content)
            
            if overlap_score > 0.3:  # Threshold
                claim["cited_chunks"].append({
                    "chunk_id": chunk.get('chunk_id', 'unknown'),
                    "source": chunk.get('source', 'unknown'),
                    "overlap_score": overlap_score
                })
                claim["confidence"] = max(claim["confidence"], overlap_score)
        
        evidence["claims"].append(claim)
    
    # Calculate coverage: % of claims with at least one citation
    cited_claims = sum(1 for c in evidence["claims"] if c["cited_chunks"])
    evidence["coverage"] = cited_claims / len(evidence["claims"]) if evidence["claims"] else 0.0
    
    return evidence


def calculate_text_overlap(text1: str, text2: str) -> float:
    """Calculate simple word overlap between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


# ============= ADAPTIVE RAG ROUTING =============

def classify_query_type(query: str) -> Dict[str, Any]:
    """
    Classify query to determine optimal RAG strategy
    
    Returns:
        - type: simple_fact, multi_doc, complex_analysis
        - recommended_k: number of chunks to retrieve
        - recommended_strategy: retrieval strategy
    """
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Simple fact lookup patterns
    simple_patterns = [
        r'wat is (de|het)',
        r'hoeveel',
        r'welke datum',
        r'wanneer',
        r'wie is',
    ]
    
    # Multi-document patterns
    multi_doc_patterns = [
        r'vergelijk',
        r'verschil tussen',
        r'relatie tussen',
        r'overzicht van',
        r'samenvatting',
    ]
    
    # Complex analysis patterns
    complex_patterns = [
        r'analyseer',
        r'adviseer',
        r'wat zijn de gevolgen',
        r'strategie',
        r'scenario',
        r'optimaliseer',
    ]
    
    result = {
        "type": "simple_fact",
        "recommended_k": 3,
        "recommended_strategy": "vector_only",
        "use_cot": False,
        "verify_answer": False
    }
    
    # Check patterns
    if any(re.search(p, query_lower) for p in complex_patterns) or word_count > 20:
        result["type"] = "complex_analysis"
        result["recommended_k"] = 10
        result["recommended_strategy"] = "hybrid_mmr"
        result["use_cot"] = True
        result["verify_answer"] = True
    elif any(re.search(p, query_lower) for p in multi_doc_patterns):
        result["type"] = "multi_doc"
        result["recommended_k"] = 7
        result["recommended_strategy"] = "hybrid"
        result["use_cot"] = True
        result["verify_answer"] = False
    elif any(re.search(p, query_lower) for p in simple_patterns):
        result["type"] = "simple_fact"
        result["recommended_k"] = 3
        result["recommended_strategy"] = "vector_only"
        result["use_cot"] = False
        result["verify_answer"] = False
    
    return result


# ============= QUALITY METRICS =============

def calculate_enhanced_metrics(
    query: str,
    answer: str,
    context: str,
    evidence: Dict[str, Any],
    verification: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate comprehensive quality metrics
    """
    metrics = {}
    
    # Groundedness: % of claims with citations
    metrics["groundedness"] = evidence.get("coverage", 0.0)
    
    # Verification confidence
    metrics["verification_confidence"] = verification.get("confidence", 0.0)
    
    # Answer completeness: does it address the query?
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    metrics["query_coverage"] = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0.0
    
    # Specificity: presence of concrete info (amounts, dates, percentages)
    concrete_pattern = r'€|%|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d+[.,]\d+'
    concrete_matches = len(re.findall(concrete_pattern, answer))
    metrics["specificity"] = min(1.0, concrete_matches / 5.0)
    
    # Appropriate hedging (should say "onvoldoende data" when needed)
    has_uncertainty = any(phrase in answer.lower() for phrase in [
        "onvoldoende data", "niet duidelijk", "niet beschikbaar"
    ])
    context_is_limited = len(context) < 500 or context.count("€") < 2
    metrics["appropriate_hedging"] = 1.0 if (has_uncertainty and context_is_limited) or (not has_uncertainty and not context_is_limited) else 0.5
    
    # Overall quality score
    metrics["overall_quality"] = (
        0.3 * metrics["groundedness"] +
        0.25 * metrics["verification_confidence"] +
        0.2 * metrics["query_coverage"] +
        0.15 * metrics["specificity"] +
        0.1 * metrics["appropriate_hedging"]
    )
    
    return metrics


# ============= A/B TESTING FRAMEWORK =============

class ABTestManager:
    """Manage A/B tests for different RAG configurations"""
    
    def __init__(self, config_path: str = "config/ab_tests.json"):
        self.config_path = config_path
        self.tests = self._load_tests()
    
    def _load_tests(self) -> Dict[str, Any]:
        """Load A/B test configurations"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_tests(self):
        """Save A/B test configurations"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.tests, f, indent=2)
    
    def create_test(
        self,
        test_name: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        metric: str = "overall_quality"
    ):
        """Create new A/B test"""
        self.tests[test_name] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "results_a": [],
            "results_b": [],
            "created_at": str(pd.Timestamp.now())
        }
        self._save_tests()
    
    def get_variant(self, test_name: str, user_id: str) -> str:
        """Get variant for user (consistent assignment)"""
        if test_name not in self.tests:
            return "a"
        
        # Simple hash-based assignment for consistency
        hash_val = hash(f"{test_name}_{user_id}") % 2
        return "a" if hash_val == 0 else "b"
    
    def record_result(self, test_name: str, variant: str, metrics: Dict[str, float]):
        """Record test result"""
        if test_name not in self.tests:
            return
        
        key = f"results_{variant}"
        self.tests[test_name][key].append({
            "metrics": metrics,
            "timestamp": str(pd.Timestamp.now())
        })
        self._save_tests()
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get test results summary"""
        if test_name not in self.tests:
            return {}
        
        test = self.tests[test_name]
        metric = test["metric"]
        
        results_a = [r["metrics"].get(metric, 0) for r in test["results_a"]]
        results_b = [r["metrics"].get(metric, 0) for r in test["results_b"]]
        
        import numpy as np
        
        return {
            "variant_a": {
                "mean": np.mean(results_a) if results_a else 0,
                "std": np.std(results_a) if results_a else 0,
                "count": len(results_a)
            },
            "variant_b": {
                "mean": np.mean(results_b) if results_b else 0,
                "std": np.std(results_b) if results_b else 0,
                "count": len(results_b)
            },
            "difference": (np.mean(results_b) - np.mean(results_a)) if results_a and results_b else 0,
            "winner": "b" if results_b and results_a and np.mean(results_b) > np.mean(results_a) else "a"
        }


# Quick test
if __name__ == "__main__":
    # Test query enhancement
    query = "Wat is de belasting op schenking van aandelen?"
    expanded = expand_query(query)
    print("Expanded queries:", expanded)
    
    # Test query classification
    classification = classify_query_type(query)
    print("Query classification:", classification)
    
    print("✅ RAG enhancements module loaded!")
