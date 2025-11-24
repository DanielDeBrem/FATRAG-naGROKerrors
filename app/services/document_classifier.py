"""Document classification service.

Fase C van het plan:
- Bied één centrale plek om op basis van tekst + bestandsnaam een *document_type*
  en optionele metadata af te leiden.
- Voor nu puur heuristiek (rules op bestandsnaam/tekst); LLM-ondersteuning kan
  later worden toegevoegd.

Typen die we voorlopig onderscheiden:
- jaarrekening
- contract
- belastingaanslag
- financiele_memo
- arbeidsovereenkomst
- overeenkomst
- taxatie
- notariele_akte
- unknown
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple


# Eenvoudige type-strings (later evt. Enum/Pydantic)
JAARREKENING = "jaarrekening"
CONTRACT = "contract"
BELASTINGAANSLAG = "belastingaanslag"
FINANCIELE_MEMO = "financiele_memo"
ARBEIDSOVEREENKOMST = "arbeidsovereenkomst"
OVEREENKOMST = "overeenkomst"
TAXATIE = "taxatie"
NOTARIELE_AKTE = "notariele_akte"
UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
  document_type: str
  confidence: float
  hints: Dict[str, str]


class DocumentClassifierService:
  """Heuristische classifier op basis van bestandsnaam en tekstinhoud.

  Later kan hier een LLM-call bijkomen (met veilige prompts), maar de interface
  blijft gelijk: geef tekst + filename, krijg (type, confidence, hints).
  """

  def classify(self, text: str, filename: str = "") -> ClassificationResult:
    fname = (filename or "").lower()
    sample = (text or "")[:4000].lower()  # beperkte sample voor heuristiek

    # Hard filename-based hints
    if "jaarrekening" in fname or "jaarverslag" in fname or "financial_statements" in fname:
      return ClassificationResult(JAARREKENING, 0.9, {"basis": "filename"})

    if "arbeidsovereenkomst" in fname or "arbeidscontract" in fname:
      return ClassificationResult(ARBEIDSOVEREENKOMST, 0.9, {"basis": "filename"})

    if "taxatie" in fname or "valuation" in fname:
      return ClassificationResult(TAXATIE, 0.9, {"basis": "filename"})

    if "aanslag" in fname or "belastingaanslag" in fname:
      return ClassificationResult(BELASTINGAANSLAG, 0.9, {"basis": "filename"})

    if "notari" in fname and ("akte" in fname or "deed" in fname):
      return ClassificationResult(NOTARIELE_AKTE, 0.9, {"basis": "filename"})

    # Content-based heuristics
    # Jaarrekening
    if any(k in sample for k in ["balans per", "winst- en verliesrekening", "jaarrekening", "jaarverslag"]):
      return ClassificationResult(JAARREKENING, 0.8, {"basis": "tekst", "key": "jaarrekening-terminologie"})

    # Belastingaanslag
    if any(k in sample for k in ["aanslag inkomstenbelasting", "aanslag vennootschapsbelasting", "definitieve aanslag", "voorlopige aanslag"]):
      return ClassificationResult(BELASTINGAANSLAG, 0.8, {"basis": "tekst", "key": "belastingaanslag-terminologie"})

    # Arbeidsovereenkomst / overeenkomst
    if "arbeidsovereenkomst" in sample or "arbeidsvoorwaarden" in sample:
      return ClassificationResult(ARBEIDSOVEREENKOMST, 0.75, {"basis": "tekst", "key": "arbeidsovereenkomst-terminologie"})
    if any(k in sample for k in ["partijen komen overeen", "de partij bij deze overeenkomst", "overeenkomst tussen", "contractspartij"]):
      return ClassificationResult(CONTRACT, 0.7, {"basis": "tekst", "key": "contract-terminologie"})

    # Taxatie / waarderingsrapport
    if any(k in sample for k in ["marktwaarde", "taxatierapport", "waardering per", "waarde in het economische verkeer"]):
      return ClassificationResult(TAXATIE, 0.7, {"basis": "tekst", "key": "taxatie-terminologie"})

    # Notariële akte (globaal)
    if any(k in sample for k in ["ten overstaan van notaris", "de comparanten", "notariële akte", "akte van levering"]):
      return ClassificationResult(NOTARIELE_AKTE, 0.7, {"basis": "tekst", "key": "notariële-akte-terminologie"})

    # Financiële memo / adviesnotitie
    if any(k in sample for k in ["memo", "advies", "notitie", "conclusie", "aanbevelingen"]) and any(
      k in sample for k in ["belasting", "fiscale", "structuur", "herstructurering", "financieel"]
    ):
      return ClassificationResult(FINANCIELE_MEMO, 0.6, {"basis": "tekst", "key": "financiele-memo-terminologie"})

    # Geen duidelijke match
    return ClassificationResult(UNKNOWN, 0.3, {"basis": "onzeker"})
