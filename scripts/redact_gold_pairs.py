"""
scripts/redact_gold_pairs.py
-----------------------------
Redact case names from destination_context fields in gold pair JSONL files.

This script processes the gold_pairs_test.jsonl and gold_pairs_val.jsonl files
created by baseline_prep.py and redacts case names to prevent data leakage
during retrieval evaluation.

Uses eyecite for robust legal citation detection plus regex fallback patterns
for case names that appear without full citations.

Usage:
    python scripts/redact_gold_pairs.py
    python scripts/redact_gold_pairs.py --input data/processed/baseline/gold_pairs_test.jsonl
    python scripts/redact_gold_pairs.py --dry-run  # preview changes without writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed" / "baseline"
DEFAULT_FILES = ["gold_pairs_test.jsonl", "gold_pairs_val.jsonl"]


def redact_case_names(text: str) -> str:
    """
    Redact case names from text to prevent data leakage in retrieval evaluation.
    
    Uses eyecite to detect legal citations and extract case names, then replaces
    them with [CASE_NAME_REDACTED] to ensure the model cannot use case names as
    shortcuts during retrieval.
    
    Args:
        text: Input text potentially containing case names
        
    Returns:
        Text with case names replaced by [CASE_NAME_REDACTED]
    """
    if not text:
        return text
    
    try:
        from eyecite import get_citations
    except ImportError:
        raise RuntimeError("eyecite not installed — run: uv add eyecite")
    
    # Extract all citations from the text using eyecite
    citations = get_citations(text)
    
    # Build a list of (start, end) tuples for all citations to redact
    replacements = []
    for cite in citations:
        # eyecite provides span() method for citation positions
        if hasattr(cite, 'span'):
            start, end = cite.span()
            replacements.append((start, end))
    
    # Apply replacements in reverse order to maintain string positions
    redacted_text = text
    for start, end in sorted(replacements, reverse=True):
        redacted_text = redacted_text[:start] + '[CASE_NAME_REDACTED]' + redacted_text[end:]
    
    # Fallback: also use regex for common case name patterns that eyecite might miss
    # (e.g., case names without full citations, or partial references)
    case_name_patterns = [
        # Standard adversarial cases: "Party v. Party" or "Party v Party"
        # Limit to 3 words per party to avoid over-matching
        r'\b([A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+){0,3})\s+v\.?\s+([A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+){0,3})\b',
        # In re cases: "In re Smith"
        r'\bIn\s+re\s+([A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+){0,2})\b',
        # Ex parte cases: "Ex parte Smith"
        r'\bEx\s+parte\s+([A-Z][A-Za-z\'\-\.]+(?:\s+[A-Z][A-Za-z\'\-\.]+){0,2})\b',
    ]
    
    for pattern in case_name_patterns:
        redacted_text = re.sub(pattern, '[CASE_NAME_REDACTED]', redacted_text)
    
    return redacted_text


def redact_file(
    input_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Redact case names from destination_context in a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output file (defaults to input_path with .redacted suffix)
        dry_run: If True, preview changes without writing output
        
    Returns:
        Summary dict with counts
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path.with_suffix('.redacted.jsonl')
    
    records_processed = 0
    records_modified = 0
    total_redactions = 0
    example = None
    
    records = []
    
    print(f"Reading {input_path.name} ...")
    with input_path.open('r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            records_processed += 1
            
            # Redact destination_context if present
            if 'destination_context' in record and record['destination_context']:
                original = record['destination_context']
                redacted = redact_case_names(original)
                
                if original != redacted:
                    records_modified += 1
                    # Count number of redactions in this record
                    redaction_count = redacted.count('[CASE_NAME_REDACTED]')
                    total_redactions += redaction_count
                    
                    # Collect first example for the summary
                    if example is None:
                        example = {
                            'source_id': record.get('source_id'),
                            'dest_id': record.get('dest_id'),
                            'before': original[:300] + ('...' if len(original) > 300 else ''),
                            'after': redacted[:300] + ('...' if len(redacted) > 300 else ''),
                            'redaction_count': redaction_count,
                        }
                    
                    record['destination_context'] = redacted
            
            records.append(record)
    
    print(f"  Processed: {records_processed:,} records")
    print(f"  Modified:  {records_modified:,} records ({100*records_modified/max(1,records_processed):.1f}%)")
    print(f"  Total redactions: {total_redactions:,}")
    
    if not dry_run:
        print(f"Writing {output_path.name} ...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as fout:
            for record in records:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Wrote {output_path}")
    else:
        print(f"  [DRY RUN] Would write to {output_path}")
    
    return {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'records_processed': records_processed,
        'records_modified': records_modified,
        'total_redactions': total_redactions,
        'modification_rate_pct': round(100 * records_modified / max(1, records_processed), 2),
        'example': example,
    }


def main():
    import time
    
    parser = argparse.ArgumentParser(
        description='Redact case names from gold pair destination_context fields'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Directory containing gold pair files (default: data/processed/baseline)',
    )
    parser.add_argument(
        '--input',
        type=Path,
        help='Single input file to redact (overrides --input-dir)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: input with .redacted suffix)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing output files',
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite input files instead of creating .redacted versions',
    )
    
    args = parser.parse_args()
    
    t_start = time.time()
    
    # Determine which files to process
    if args.input:
        files_to_process = [args.input]
    else:
        files_to_process = [args.input_dir / fname for fname in DEFAULT_FILES]
    
    summaries = []
    
    for input_path in files_to_process:
        if not input_path.exists():
            print(f"SKIP: {input_path} (not found)", file=sys.stderr)
            continue
        
        print("\n" + "=" * 60)
        print(f"  Processing: {input_path.name}")
        print("=" * 60)
        
        if args.in_place:
            output_path = input_path
        elif args.output:
            output_path = args.output
        else:
            output_path = None  # Will use default .redacted suffix
        
        summary = redact_file(input_path, output_path, dry_run=args.dry_run)
        summaries.append(summary)
    
    elapsed = time.time() - t_start
    
    # Build overall summary
    if summaries:
        print("\n" + "=" * 60)
        print("  Overall Summary")
        print("=" * 60)
        total_processed = sum(s['records_processed'] for s in summaries)
        total_modified = sum(s['records_modified'] for s in summaries)
        total_redactions = sum(s['total_redactions'] for s in summaries)
        print(f"  Files processed:    {len(summaries)}")
        print(f"  Records processed:  {total_processed:,}")
        print(f"  Records modified:   {total_modified:,} ({100*total_modified/max(1,total_processed):.1f}%)")
        print(f"  Total redactions:   {total_redactions:,}")
        print(f"  Time Elapsed:            {elapsed:.1f}s")
        
        if args.dry_run:
            print("\n  [DRY RUN] No files were modified")
        
        # Write summary JSON
        if not args.dry_run:
            summary_json = {
                'files_processed': len(summaries),
                'total_records_processed': total_processed,
                'total_records_modified': total_modified,
                'total_redactions': total_redactions,
                'modification_rate_pct': round(100 * total_modified / max(1, total_processed), 2),
                'elapsed_sec': round(elapsed, 1),
                'in_place': args.in_place,
                'file_summaries': summaries,
            }
            
            # Write summary to the same directory as the first output file
            if summaries:
                first_output = Path(summaries[0]['output_file'])
                summary_path = first_output.parent / 'redact_gold_pairs_summary.json'
                with summary_path.open('w') as f:
                    json.dump(summary_json, f, indent=2)
                print(f"\n  Summary → {summary_path}")
    else:
        print("\nNo files processed", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
