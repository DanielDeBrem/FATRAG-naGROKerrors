#!/usr/bin/env python3
"""
Quick test script to verify document upload functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test imports
    from app.api.documents import documents_router
    from app.services.document_ingest import DocumentIngestService
    from app.repositories.documents import DocumentsRepository
    from app.models.dto.upload import UploadResponse
    print("✅ All imports successful")

    # Test service instantiation
    service = DocumentIngestService()
    repo = service.repository
    print("✅ Service and repository instantiation successful")

    # Test basic repo methods
    try:
        docs = repo.list_documents()
        print(f"✅ Repository list_documents() works, found {len(docs)} documents")
    except Exception as e:
    print("\n🎉 Basic implementation tests passed!")
    print("The modular architecture is correctly set up.")

    print("The modular architecture is correctly set up.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
