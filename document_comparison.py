"""
Document Comparison Service
Feature 5: Side-by-side document comparison with diff highlighting
"""

import difflib
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import json


def compute_file_hash(content: str) -> str:
    """Compute SHA256 hash of file content for version tracking"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compute_diff(text1: str, text2: str, context_lines: int = 3) -> List[Dict]:
    """
    Compute unified diff between two texts with highlighted changes
    
    Returns:
        List of diff blocks with:
        - type: 'unchanged', 'added', 'removed', 'changed'
        - content: text content
        - line_number_1: line number in first doc
        - line_number_2: line number in second doc
    """
    lines1 = text1.splitlines(keepends=False)
    lines2 = text2.splitlines(keepends=False)
    
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    
    result = []
    line1_idx = 0
    line2_idx = 0
    
    for line in diff:
        prefix = line[:2]
        content = line[2:] if len(line) > 2 else ""
        
        if prefix == '  ':  # Unchanged
            result.append({
                'type': 'unchanged',
                'content': content,
                'line_number_1': line1_idx + 1,
                'line_number_2': line2_idx + 1
            })
            line1_idx += 1
            line2_idx += 1
        elif prefix == '- ':  # Removed
            result.append({
                'type': 'removed',
                'content': content,
                'line_number_1': line1_idx + 1,
                'line_number_2': None
            })
            line1_idx += 1
        elif prefix == '+ ':  # Added
            result.append({
                'type': 'added',
                'content': content,
                'line_number_1': None,
                'line_number_2': line2_idx + 1
            })
            line2_idx += 1
        elif prefix == '? ':  # Changed (marker line)
            continue
    
    return result


def get_similarity_score(text1: str, text2: str) -> float:
    """
    Compute similarity ratio between two texts (0.0 to 1.0)
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def extract_key_differences(diff_blocks: List[Dict], max_differences: int = 10) -> List[str]:
    """
    Extract key differences as human-readable summaries
    """
    differences = []
    
    added_lines = [b for b in diff_blocks if b['type'] == 'added']
    removed_lines = [b for b in diff_blocks if b['type'] == 'removed']
    
    if added_lines:
        differences.append(f"{len(added_lines)} regels toegevoegd")
    if removed_lines:
        differences.append(f"{len(removed_lines)} regels verwijderd")
    
    # Find substantial changes (non-empty lines)
    substantial_added = [b for b in added_lines if len(b['content'].strip()) > 10]
    substantial_removed = [b for b in removed_lines if len(b['content'].strip()) > 10]
    
    for block in substantial_added[:max_differences]:
        differences.append(f"+ {block['content'][:60]}...")
    
    for block in substantial_removed[:max_differences]:
        differences.append(f"- {block['content'][:60]}...")
    
    return differences


class DocumentVersionTracker:
    """Track document versions and changes over time"""
    
    def __init__(self):
        self.versions: Dict[str, List[Dict]] = {}
    
    def add_version(self, filename: str, content: str, uploaded_at: str) -> Dict:
        """Add a new version of a document"""
        content_hash = compute_file_hash(content)
        
        version = {
            'version_id': len(self.versions.get(filename, [])) + 1,
            'filename': filename,
            'content_hash': content_hash,
            'uploaded_at': uploaded_at,
            'size': len(content)
        }
        
        if filename not in self.versions:
            self.versions[filename] = []
        
        self.versions[filename].append(version)
        return version
    
    def get_versions(self, filename: str) -> List[Dict]:
        """Get all versions of a document"""
        return self.versions.get(filename, [])
    
    def compare_versions(self, filename: str, version1_id: int, version2_id: int, 
                        content1: str, content2: str) -> Dict:
        """Compare two versions of a document"""
        diff_blocks = compute_diff(content1, content2)
        similarity = get_similarity_score(content1, content2)
        key_diffs = extract_key_differences(diff_blocks)
        
        return {
            'filename': filename,
            'version1_id': version1_id,
            'version2_id': version2_id,
            'similarity': similarity,
            'diff_blocks': diff_blocks,
            'key_differences': key_diffs,
            'total_changes': len([b for b in diff_blocks if b['type'] != 'unchanged'])
        }


# Global version tracker instance
version_tracker = DocumentVersionTracker()
