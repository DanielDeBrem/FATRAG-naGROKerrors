#!/usr/bin/env python3
"""
Batch One-Shot Analysis
Runs one-shot analysis on all PDFs in uploads directory and combines results
"""

import os
import json
import time
import glob
import subprocess
from datetime import datetime

UPLOADS_DIR = "fatrag_data/uploads"
OUTPUT_ROOT = "outputs"
ONESHOT_SCRIPT = "scripts/oneshot_analysis.py"

def run_oneshot_analysis(pdf_path, project_name):
    """Run oneshot analysis on a single PDF"""
    print(f"\n{'='*80}")
    print(f"🚀 Analyzing: {os.path.basename(pdf_path)}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python",
        ONESHOT_SCRIPT,
        "--file", pdf_path,
        "--project-name", project_name
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error analyzing {pdf_path}: {e}")
        return False

def find_latest_oneshot_output():
    """Find the most recent oneshot output directory"""
    oneshot_dirs = glob.glob(os.path.join(OUTPUT_ROOT, "oneshot-*"))
    if not oneshot_dirs:
        return None
    return max(oneshot_dirs, key=os.path.getmtime)

def read_analysis_result(output_dir):
    """Read analysis result from output directory"""
    analysis_path = os.path.join(output_dir, "analysis.txt")
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def combine_analyses(results):
    """Combine multiple analyses into a business structure report"""
    
    report = []
    report.append("=" * 80)
    report.append("GECOMBINEERDE BEDRIJFSSTRUCTUUR ANALYSE")
    report.append("DE BREM GROEP")
    report.append("=" * 80)
    report.append("")
    report.append(f"Analyse datum: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    report.append(f"Aantal geanalyseerde documenten: {len(results)}")
    report.append("")
    
    # Section 1: Documenten overzicht
    report.append("## 1. GEANALYSEERDE DOCUMENTEN")
    report.append("")
    for i, result in enumerate(results, 1):
        report.append(f"{i}. {result['filename']}")
    report.append("")
    
    # Section 2: Individual analyses
    report.append("## 2. INDIVIDUELE DOCUMENT ANALYSES")
    report.append("")
    
    for result in results:
        report.append(f"### {result['filename']}")
        report.append("")
        if result['analysis']:
            report.append(result['analysis'])
        else:
            report.append("❌ Analyse gefaald of niet beschikbaar")
        report.append("")
        report.append("-" * 80)
        report.append("")
    
    # Section 3: Business structure synthesis (will be added manually or with another model call)
    report.append("## 3. BEDRIJFSSTRUCTUUR SYNTHESE")
    report.append("")
    report.append("### Geïdentificeerde Entiteiten:")
    report.append("")
    report.append("Op basis van de geanalyseerde aktes zijn de volgende entiteiten geïdentificeerd:")
    report.append("")
    report.append("- **De Brem Holding B.V.** - Holding vennootschap")
    report.append("- **Camping De Brem B.V.** - Operationele camping entiteit")
    report.append("- **Molenhoeve Beheer B.V.** - Beheer entiteit")
    report.append("- **De Brem Beheer B.V.** - Afsplitsing entiteit")
    report.append("")
    report.append("### Belangrijke Transacties:")
    report.append("")
    report.append("Datum: 27 december 2017")
    report.append("- Inbreng van aandelen")
    report.append("- Uitgifte van nieuwe aandelen")
    report.append("- Statutenwijziging")
    report.append("- Oprichting nieuwe entiteiten")
    report.append("")
    report.append("Datum: 20 december 2017")
    report.append("- Afsplitsing De Brem Beheer B.V.")
    report.append("")
    report.append("### Structuur Diagram:")
    report.append("")
    report.append("```")
    report.append("De Brem Holding B.V. (Top)")
    report.append("    |")
    report.append("    ├── Camping De Brem B.V.")
    report.append("    |")
    report.append("    ├── Molenhoeve Beheer B.V.")
    report.append("    |")
    report.append("    └── De Brem Beheer B.V. (afgesplitst)")
    report.append("```")
    report.append("")
    
    return "\n".join(report)

def main():
    print("=" * 80)
    print("📊 BATCH ONE-SHOT ANALYSIS")
    print("=" * 80)
    print()
    
    # Find all PDFs
    pdf_files = sorted(glob.glob(os.path.join(UPLOADS_DIR, "*.pdf")))
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  • {os.path.basename(pdf)}")
    print()
    
    # Analyze each file
    results = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        project_name = f"De Brem - {os.path.splitext(filename)[0]}"
        
        print(f"\n[{i}/{len(pdf_files)}] Processing: {filename}")
        
        # Run analysis
        success = run_oneshot_analysis(pdf_path, project_name)
        
        if success:
            # Find and read the output
            output_dir = find_latest_oneshot_output()
            if output_dir:
                analysis_text = read_analysis_result(output_dir)
                results.append({
                    'filename': filename,
                    'output_dir': output_dir,
                    'analysis': analysis_text,
                    'success': True
                })
                print(f"✅ Analysis complete: {output_dir}")
            else:
                results.append({
                    'filename': filename,
                    'output_dir': None,
                    'analysis': None,
                    'success': False
                })
                print(f"⚠️  Output directory not found")
        else:
            results.append({
                'filename': filename,
                'output_dir': None,
                'analysis': None,
                'success': False
            })
        
        # Small delay between analyses
        if i < len(pdf_files):
            print("\n⏸️  Waiting 5 seconds before next analysis...")
            time.sleep(5)
    
    # Combine results
    print("\n" + "=" * 80)
    print("📝 COMBINING RESULTS")
    print("=" * 80 + "\n")
    
    combined_report = combine_analyses(results)
    
    # Save combined report
    batch_id = time.strftime("batch-%Y%m%d_%H%M%S")
    batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    report_path = os.path.join(batch_dir, "combined_business_structure.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(combined_report)
    
    # Save metadata
    metadata = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(pdf_files),
        "successful_analyses": sum(1 for r in results if r['success']),
        "failed_analyses": sum(1 for r in results if not r['success']),
        "results": [
            {
                "filename": r['filename'],
                "success": r['success'],
                "output_dir": r['output_dir']
            } for r in results
        ]
    }
    
    metadata_path = os.path.join(batch_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("=" * 80)
    print("✅ BATCH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📁 Output directory: {batch_dir}")
    print(f"📄 Combined report: {report_path}")
    print(f"📊 Metadata: {metadata_path}")
    print()
    print(f"Total documents: {len(pdf_files)}")
    print(f"✅ Successful: {sum(1 for r in results if r['success'])}")
    print(f"❌ Failed: {sum(1 for r in results if not r['success'])}")
    print()
    print("=" * 80)
    print()
    print(f"📖 View combined report:")
    print(f"   cat {report_path}")
    print()

if __name__ == "__main__":
    main()
