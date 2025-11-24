"""
Disable ChromaDB telemetry to prevent PostHog errors.
Import this at the top of any script that uses chromadb.
"""
import os

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
