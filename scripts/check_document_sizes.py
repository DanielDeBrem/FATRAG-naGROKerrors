#!/usr/bin/env python3
"""
Quick document size checker
Analyzes PDFs to estimate if they fit in llama3.1:70b context window (128K tokens)
"""

import os
import pdfplumber

UPLOADS_DIR = "fatrag_data/uploads"
MAX_CONTEXT = 128000  # llama3.1:70b context window
SAFE_THRESHOLD = 0.7  # Use 70% of context for safety

def estimate_tokens(text):
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def analyze_pdf(filepath):
    """Extract and analyze a PDF"""
    try:
        with pdfplumber.open(filepath) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            char_count = len(full_text)
            token_estimate = estimate_tokens(full_text)
            page_count = len(pdf.pages)
            
            return {
                "pages": page_count,
                "chars": char_count,
                "tokens": token_estimate,
                "fits": token_estimate < (MAX_CONTEXT * SAFE_THRESHOLD),
                "percentage": (token_estimate / MAX_CONTEXT) * 100
            }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 80)
    print("📊 DOCUMENT SIZE ANALYSIS")
    print("=" * 80)
    print(f"Max Context Window: {MAX_CONTEXT:,} tokens")
    print(f"Safe Threshold: {int(MAX_CONTEXT * SAFE_THRESHOLD):,} tokens (70%)")
    print()
    
    files = sorted([f for f in os.listdir(UPLOADS_DIR) if f.endswith('.pdf')])
    
    if not files:
        print("❌ No PDF files found in uploads directory")
        return
    
    results = []
    
    for filename in files:
        filepath = os.path.join(UPLOADS_DIR, filename)
        print(f"📄 Analyzing: {filename}")
        print(f"   File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        
        result = analyze_pdf(filepath)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            results.append({"file": filename, "status": "ERROR", "error": result['error']})
        else:
            status = "✅ FITS" if result["fits"] else "❌ TOO BIG"
            print(f"   Pages: {result['pages']}")
            print(f"   Characters: {result['chars']:,}")
            print(f"   Estimated tokens: {result['tokens']:,}")
            print(f"   Context usage: {result['percentage']:.1f}%")
            print(f"   Status: {status}")
            
            results.append({
                "file": filename,
                "status": "FITS" if result["fits"] else "TOO_BIG",
                "pages": result["pages"],
                "tokens": result["tokens"],
                "percentage": result["percentage"]
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    fits_count = sum(1 for r in results if r.get("status") == "FITS")
    too_big_count = sum(1 for r in results if r.get("status") == "TOO_BIG")
    error_count = sum(1 for r in results if r.get("status") == "ERROR")
    
    print(f"Total files: {len(results)}")
    print(f"✅ Can analyze: {fits_count}")
    print(f"❌ Too large: {too_big_count}")
    print(f"⚠️  Errors: {error_count}")
    print()
    
    if fits_count > 0:
        print("✅ GOOD NEWS: You can analyze these files with oneshot_analysis.py:")
        for r in results:
            if r.get("status") == "FITS":
                print(f"   • {r['file']} ({r['tokens']:,} tokens, {r['percentage']:.1f}% context)")
    
    if too_big_count > 0:
        print()
        print("❌ TOO LARGE for one-shot analysis:")
        for r in results:
            if r.get("status") == "TOO_BIG":
                print(f"   • {r['file']} ({r['tokens']:,} tokens, {r['percentage']:.1f}% context)")
        print()
        print("💡 These would need map-reduce pipeline or manual filtering")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
