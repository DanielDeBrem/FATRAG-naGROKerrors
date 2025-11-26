from __future__ import annotations

from typing import Optional


def truncate_text_chars(text: str, max_chars: int) -> str:
    """Eenvoudige, veilige truncatie op basis van aantal characters.

    Dit is een fallback als er geen tokeniser beschikbaar is.
    We laten een kleine marge en voegen een marker toe zodat duidelijk is dat er is ingekort.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    # Houd wat ruimte over voor de truncatie-marker
    marker = "\n\n[TRUNCATED VOOR CONTEXT-LIMIT]\n"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker


def apply_context_limit(
    text: str,
    max_context_tokens: Optional[int] = None,
    approx_chars_per_token: int = 4,
    hard_max_chars: int = 200_000,
) -> str:
    """Beperk de hoeveelheid tekst die wordt meegestuurd naar de LLM.

    - Als max_context_tokens is opgegeven: benader dit grofweg met een factor chars/token.
    - Anders gebruiken we een harde bovengrens op aantal characters.

    Let op:
    - Dit is een ruwe schatting, maar voldoende om runaway-prompts te voorkomen.
    - Voor preciezere limieten kan later een echte tokeniser worden geïntegreerd.
    """
    if not text:
        return text

    if max_context_tokens and max_context_tokens > 0:
        target_chars = max_context_tokens * approx_chars_per_token
        target_chars = min(target_chars, hard_max_chars)
    else:
        target_chars = hard_max_chars

    return truncate_text_chars(text, target_chars)
