"""
Document Classification Service for FATRAG
Automatically classifies uploaded documents using Ollama LLMs
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama


# Document classification categories (Dutch financial context)
DOCUMENT_CATEGORIES = {
    "financiële_rapporten": {
        "name": "Financiële Rapporten",
        "description": "Jaarrekeningen, kwartaalrapporten, jaarverslagen, balansspecificaties",
        "keywords": ["jaarrekening", "kwartaalrapport", "jaarverslag", "balans", "winst", "verlies", "EBITDA", "cashflow"]
    },
    "contracten": {
        "name": "Contracten",
        "description": "Arbeidsovereenkomsten, huurcontracten, koop/verkoop contracten, NDA's",
        "keywords": ["overeenkomst", "contract", "arbeidsovereenkomst", "huurcontract", "koopovereenkomst", "NDA", "confidentialiteit"]
    },
    "belasting_documenten": {
        "name": "Belasting Documenten",
        "description": "IB aangifte, VPB aangifte, belastingaanslagen, beschikkingen",
        "keywords": ["aangifte", "inkomstenbelasting", "vennootschapsbelasting", "BTW aangifte", "belastingdienst", "beschikking"]
    },
    "waarderingsrapporten": {
        "name": "Waarderingsrapporten",
        "description": "Bedrijfswaarderingen, taxaties, due diligence rapporten",
        "keywords": ["waardering", "taxatie", "bedrijfs waarde", "onderneming waarde", "due diligence", "DCF", "multiples"]
    },
    "juridische_akten": {
        "name": "Juridische Akten",
        "description": "Statuten BV/NV, maatschapscontracten, huwelijkse voorwaarden",
        "keywords": ["statuten", "BV", "NV", "maatschap", "stichting", "huwelijkse voorwaarden", "erven", "erven"]
    },
    "correspondentie": {
        "name": "Correspondentie",
        "description": "Brieven, e-mails, memo's, correspondentie",
        "keywords": ["brief", "email", "memo", "correspondentie", "geachte", "beste"]
    },
    "presentaties": {
        "name": "Presentaties",
        "description": "PowerPoint presentaties, rapportages, vergaderstukken",
        "keywords": ["presentatie", "diapresentatie", "vergadering", "agenda", "punten", "actie"]
    },
    "overig": {
        "name": "Overige Documenten",
        "description": "Overige documenten die niet in andere categorieën passen",
        "keywords": []
    }
}


def sample_text_for_classification(text: str, max_chars: int = 5000) -> str:
    """
    Intelligente tekst sampling voor document classificatie.
    Gebruikt gelaagd sampling: begin + midden + einde voor optimale context.
    """
    if not text or len(text) <= max_chars:
        return text

    # Gelaagde sampling strategie
    begin_len = max_chars // 3
    middle_len = max_chars // 3
    end_len = max_chars - begin_len - middle_len

    # Begin sectie (titel, headers, eerste inhoud)
    begin = text[:begin_len]

    # Midden sectie (kern van document)
    middle_start = len(text) // 2 - middle_len // 2
    middle = text[middle_start:middle_start + middle_len]
    if middle_start < begin_len:  # Overlap voorkomen
        middle = text[begin_len:begin_len + middle_len]

    # Eind sectie (samenvatting, conclusies, handtekeningen)
    end = text[-end_len:]

    return f"{begin}\n\n[MIDDEN SECTIE - KERN INHOUD]\n{middle}\n\n[EIND SECTIE - SAMENVATTING/CONCLUSIES]\n{end}"


def classify_document_with_ollama(text_sample: str, model_name: str = "qwen2.5:7b-instruct", base_url: Optional[str] = None) -> str:
    """
    Classificeert een document sample met Ollama LLM.
    Returns the category key (e.g., 'financiële_rapporten').
    """
    if not text_sample or not text_sample.strip():
        return "overig"

    try:
        # Build LLM instance
        llm = ChatOllama(
            model=model_name,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,  # Lage temperature voor consistente classificatie
            timeout=60
        )

        # Build classification prompt
        categories_text = "\n".join([
            f"- {cat['name']}: {cat['description']}"
            for cat_key, cat in DOCUMENT_CATEGORIES.items()
            if cat_key != "overig"
        ])
        categories_text += "\n- Overige Documenten: Documenten die niet in bovenstaande categorieën passen"

        prompt = f"""Je bent een expert in het classificeren van financiële en juridische documenten.

TAak: Analyseer de onderstaande documentfragmenten en bepaal welke categorie het beste past.

BESCHIKBARE CATEGORIEËN:
{categories_text}

DOCUMENT INHOUD:
```
{text_sample[:4000]}  # Beperk voor prompt lengte
```

GEVEN ALLEEN de naam van de meest geschikte categorie terug. Geef geen uitleg of extra tekst.

ANTWOORD:"""

        # Get classification from LLM
        response = llm.invoke(prompt)
        result = getattr(response, 'content', str(response)).strip()

        # Match result to categories (case-insensitive, partial matching)
        result_lower = result.lower()

        for cat_key, cat_info in DOCUMENT_CATEGORIES.items():
            cat_name = cat_info['name'].lower()
            if cat_name in result_lower:
                return cat_key

        # Fallback: keyword-based classification
        return classify_by_keywords(text_sample)

    except Exception as e:
        print(f"Ollama classificatie fout: {e}")
        return classify_by_keywords(text_sample)


def classify_by_keywords(text: str) -> str:
    """
    Fallback classificatie gebaseerd op keywords als Ollama niet beschikbaar is.
    """
    text_lower = text.lower()
    scores = {}

    for cat_key, cat_info in DOCUMENT_CATEGORIES.items():
        if cat_key == "overig":
            continue

        keywords = cat_info.get("keywords", [])
        score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        scores[cat_key] = score

    # Return highest scoring category, or 'overig' if no matches
    if scores:
        best_cat = max(scores.items(), key=lambda x: x[1])
        if best_cat[1] > 0:  # At least one keyword match
            return best_cat[0]

    return "overig"


def get_category_info(category_key: str) -> Dict[str, Any]:
    """
    Get full category information by key.
    """
    return DOCUMENT_CATEGORIES.get(category_key, DOCUMENT_CATEGORIES["overig"])


def list_categories() -> List[Dict[str, Any]]:
    """
    Get all available categories for API responses.
    """
    return [
        {"key": key, "name": cat["name"], "description": cat["description"]}
        for key, cat in DOCUMENT_CATEGORIES.items()
    ]
