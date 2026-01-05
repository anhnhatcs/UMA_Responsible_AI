#!/usr/bin/env python3
"""
Check which models have been completed in a results file.
Useful for determining what to skip when resuming after timeout.
"""

import json
import sys
from collections import defaultdict

def analyze_results(json_path):
    """Analyze a results file to see what's been completed."""
    
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        return
    
    # Count evaluations per model and candidate
    model_stats = defaultdict(lambda: defaultdict(int))
    total_by_model = defaultdict(int)
    
    for result in results:
        model = result['model']
        candidate = result['candidate']
        mitigation = result.get('mitigation_applied', False)
        
        # Only count non-mitigation runs for progress
        if not mitigation:
            model_stats[model][candidate] += 1
            total_by_model[model] += 1
    
    print(f"\n{'='*70}")
    print(f"RESULTS ANALYSIS: {json_path}")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(results)}")
    print(f"Models found: {len(model_stats)}")
    print("")
    
    # Show per-model breakdown
    print(f"{'Model':<20} {'Total Evals':<12} {'Candidates':<40}")
    print(f"{'-'*70}")
    
    all_completed_models = []
    all_incomplete_models = []
    
    for model in sorted(model_stats.keys()):
        candidates = model_stats[model]
        total = total_by_model[model]
        candidate_counts = [f"{c}:{n}" for c, n in sorted(candidates.items())]
        candidate_str = ", ".join(candidate_counts)
        
        # Determine if model is complete (assuming 6 candidates, check if all have same count)
        counts = list(candidates.values())
        if len(candidates) == 6 and len(set(counts)) == 1:
            status = "✓ COMPLETE"
            all_completed_models.append(model)
        else:
            status = "⚠ INCOMPLETE"
            all_incomplete_models.append(model)
        
        print(f"{model:<20} {total:<12} {candidate_str[:40]:<40} {status}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    if all_completed_models:
        print(f"\n✓ Completed models ({len(all_completed_models)}):")
        for model in all_completed_models:
            print(f"  - {model}")
    
    if all_incomplete_models:
        print(f"\n⚠ Incomplete models ({len(all_incomplete_models)}):")
        for model in all_incomplete_models:
            print(f"  - {model}")
            # Show which candidates are incomplete
            for candidate, count in sorted(model_stats[model].items()):
                expected = max(model_stats[model].values())
                if count < expected:
                    print(f"    {candidate}: {count}/{expected} runs")
    
    # Generate resume command
    print(f"\n{'='*70}")
    print(f"RESUME COMMAND")
    print(f"{'='*70}")
    
    if all_completed_models:
        skip_list = " ".join(all_completed_models)
        print(f"\nTo skip completed models:")
        print(f"\n  --skip-models {skip_list}")
        print(f"\nOr to resume from this file:")
        print(f"\n  --resume-from {json_path}")
        print(f"\nFull example:")
        print(f"\n  python run_bias_evaluation.py --resume-from {json_path} \\")
        print(f"         --models <remaining_models> --runs 300")
    else:
        print("\nNo completed models found - may need to start fresh.")
    
    print("")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_completed_models.py <results.json>")
        print("\nExample:")
        print("  python check_completed_models.py results/results_2025-01-01T12-00-00.json")
        sys.exit(1)
    
    analyze_results(sys.argv[1])

