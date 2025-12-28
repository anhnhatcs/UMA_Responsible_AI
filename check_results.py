#!/usr/bin/env python3
"""
Check Results Completeness Script
Reads all result files and reports which models/candidates have been evaluated.
"""

import json
import os
import glob
from collections import defaultdict
from datetime import datetime

# Expected configuration
EXPECTED_CANDIDATES = ['anonymous', 'lukas', 'andrei', 'mehmet', 'minh', 'wei']
EXPECTED_MODELS = [
    'gemma2-9b', 'gemma2-27b',
    'llama31-8b', 'llama31-70b',
    'ministral-8b', 'ministral-14b-reasoning',
    'mistral-small',
    'qwen3-4b', 'qwen3-8b', 'qwen3-30b', 'qwen3-32b'
]
MIN_RUNS_FOR_ANALYSIS = 25  # Minimum runs per candidate for statistical significance

def load_all_results(results_dir: str):
    """Load all JSON result files from the results directory."""
    all_results = []
    json_files = glob.glob(os.path.join(results_dir, "results_*.json"))
    
    print(f"\n{'='*70}")
    print("RESULT FILES FOUND")
    print('='*70)
    
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data)
                all_results.extend(data)
                print(f"  ✓ {filename}: {count} evaluations")
        except Exception as e:
            print(f"  ✗ {filename}: ERROR - {e}")
    
    print(f"\nTotal evaluations loaded: {len(all_results)}")
    return all_results


def analyze_completeness(results: list):
    """Analyze which models and candidates have been evaluated."""
    
    # Group by model -> candidate -> list of run_ids
    model_data = defaultdict(lambda: defaultdict(list))
    
    # Also track with/without mitigation
    mitigation_data = defaultdict(lambda: defaultdict(lambda: {'baseline': 0, 'mitigation': 0}))
    
    for r in results:
        model = r.get('model', 'unknown')
        candidate = r.get('candidate', 'unknown')
        run_id = r.get('run_id', 0)
        mitigation = r.get('mitigation_applied', False)
        
        model_data[model][candidate].append(run_id)
        
        if mitigation:
            mitigation_data[model][candidate]['mitigation'] += 1
        else:
            mitigation_data[model][candidate]['baseline'] += 1
    
    return model_data, mitigation_data


def print_summary_table(model_data: dict, mitigation_data: dict):
    """Print a summary table of completeness."""
    
    print(f"\n{'='*70}")
    print("MODEL COMPLETENESS SUMMARY")
    print('='*70)
    print(f"{'Model':<25} {'Candidates':<12} {'Total Runs':<12} {'Status'}")
    print('-'*70)
    
    models_complete = []
    models_incomplete = []
    models_missing = []
    
    for model in EXPECTED_MODELS:
        if model in model_data:
            candidates_done = len(model_data[model])
            total_runs = sum(len(runs) for runs in model_data[model].values())
            min_runs = min(len(runs) for runs in model_data[model].values()) if model_data[model] else 0
            
            if candidates_done == len(EXPECTED_CANDIDATES) and min_runs >= MIN_RUNS_FOR_ANALYSIS:
                status = "✓ COMPLETE"
                models_complete.append(model)
            elif candidates_done > 0:
                status = f"⚠ PARTIAL (min {min_runs} runs)"
                models_incomplete.append(model)
            else:
                status = "✗ MISSING"
                models_missing.append(model)
            
            print(f"{model:<25} {candidates_done}/{len(EXPECTED_CANDIDATES):<10} {total_runs:<12} {status}")
        else:
            print(f"{model:<25} {'0/6':<12} {'0':<12} ✗ NOT RUN")
            models_missing.append(model)
    
    return models_complete, models_incomplete, models_missing


def print_detailed_breakdown(model_data: dict, mitigation_data: dict):
    """Print detailed breakdown per model."""
    
    print(f"\n{'='*70}")
    print("DETAILED BREAKDOWN BY MODEL")
    print('='*70)
    
    for model in sorted(model_data.keys()):
        print(f"\n>>> {model}")
        print(f"    {'Candidate':<12} {'Baseline':<10} {'Mitigation':<12} {'Total':<8} {'Status'}")
        print(f"    {'-'*55}")
        
        for candidate in EXPECTED_CANDIDATES:
            if candidate in model_data[model]:
                baseline = mitigation_data[model][candidate]['baseline']
                mitigation = mitigation_data[model][candidate]['mitigation']
                total = len(model_data[model][candidate])
                
                if baseline >= MIN_RUNS_FOR_ANALYSIS:
                    status = "✓"
                elif baseline > 0:
                    status = f"⚠ need {MIN_RUNS_FOR_ANALYSIS - baseline} more"
                else:
                    status = "✗"
                
                print(f"    {candidate:<12} {baseline:<10} {mitigation:<12} {total:<8} {status}")
            else:
                print(f"    {candidate:<12} {'0':<10} {'0':<12} {'0':<8} ✗ MISSING")


def print_recommendations(models_complete: list, models_incomplete: list, models_missing: list, model_data: dict):
    """Print recommendations for completing the experiment."""
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print('='*70)
    
    if models_complete:
        print(f"\n✓ Models ready for analysis ({len(models_complete)}):")
        for m in models_complete:
            print(f"    - {m}")
    
    if models_incomplete:
        print(f"\n⚠ Models needing more runs ({len(models_incomplete)}):")
        for m in models_incomplete:
            # Calculate what's missing
            for candidate in EXPECTED_CANDIDATES:
                if candidate not in model_data[m] or len(model_data[m][candidate]) < MIN_RUNS_FOR_ANALYSIS:
                    current = len(model_data[m].get(candidate, []))
                    needed = MIN_RUNS_FOR_ANALYSIS - current
                    if needed > 0:
                        print(f"    - {m}: {candidate} needs {needed} more runs")
    
    if models_missing:
        print(f"\n✗ Models not yet run ({len(models_missing)}):")
        for m in models_missing:
            print(f"    - {m}")
    
    # Generate commands to complete missing runs
    print(f"\n{'='*70}")
    print("SUGGESTED COMMANDS TO COMPLETE EXPERIMENT")
    print('='*70)
    
    if models_missing:
        print("\n# Run missing models:")
        for m in models_missing:
            if '70b' in m:
                print(f'MODELS="{m}" RUNS=30 sbatch --gres=gpu:4 --partition=gpu_a100_short --time=00:30:00 run_experiment.sh custom')
            elif '27b' in m or '30b' in m or '32b' in m or 'mistral-small' in m:
                print(f'MODELS="{m}" RUNS=30 sbatch --gres=gpu:2 --partition=gpu_a100_short --time=00:30:00 run_experiment.sh custom')
            else:
                print(f'MODELS="{m}" RUNS=30 sbatch --gres=gpu:1 --partition=gpu_a100_short --time=00:30:00 run_experiment.sh custom')


def check_data_quality(results: list):
    """Check for data quality issues."""
    
    print(f"\n{'='*70}")
    print("DATA QUALITY CHECK")
    print('='*70)
    
    issues = []
    
    # Check for missing scores
    missing_cf = sum(1 for r in results if r.get('cultural_fit_score') is None)
    missing_hp = sum(1 for r in results if r.get('hiring_probability') is None)
    
    if missing_cf > 0:
        issues.append(f"⚠ {missing_cf} evaluations missing cultural_fit_score")
    if missing_hp > 0:
        issues.append(f"⚠ {missing_hp} evaluations missing hiring_probability")
    
    # Check score ranges
    cf_scores = [r.get('cultural_fit_score') for r in results if r.get('cultural_fit_score') is not None]
    hp_scores = [r.get('hiring_probability') for r in results if r.get('hiring_probability') is not None]
    
    if cf_scores:
        cf_out_of_range = sum(1 for s in cf_scores if s < 0 or s > 100)
        if cf_out_of_range > 0:
            issues.append(f"⚠ {cf_out_of_range} cultural_fit_scores out of 0-100 range")
    
    if hp_scores:
        hp_out_of_range = sum(1 for s in hp_scores if s < 0 or s > 100)
        if hp_out_of_range > 0:
            issues.append(f"⚠ {hp_out_of_range} hiring_probabilities out of 0-100 range")
    
    # Check visa mentions
    visa_mentions = sum(1 for r in results if r.get('visa_mentioned', False))
    visa_rate = (visa_mentions / len(results) * 100) if results else 0
    
    print(f"\nScore Statistics:")
    if cf_scores:
        print(f"  Cultural Fit: min={min(cf_scores):.1f}, max={max(cf_scores):.1f}, avg={sum(cf_scores)/len(cf_scores):.1f}")
    if hp_scores:
        print(f"  Hiring Prob:  min={min(hp_scores):.1f}, max={max(hp_scores):.1f}, avg={sum(hp_scores)/len(hp_scores):.1f}")
    print(f"  Visa Mention Rate: {visa_rate:.1f}% ({visa_mentions}/{len(results)})")
    
    if issues:
        print("\nIssues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No data quality issues found")


def main():
    # Determine results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    print(f"\n{'#'*70}")
    print("#  VISA WALL EXPERIMENT - RESULTS COMPLETENESS CHECK")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load all results
    results = load_all_results(results_dir)
    
    if not results:
        print("\nERROR: No results found!")
        return
    
    # Analyze completeness
    model_data, mitigation_data = analyze_completeness(results)
    
    # Print summary
    models_complete, models_incomplete, models_missing = print_summary_table(model_data, mitigation_data)
    
    # Print detailed breakdown
    print_detailed_breakdown(model_data, mitigation_data)
    
    # Check data quality
    check_data_quality(results)
    
    # Print recommendations
    print_recommendations(models_complete, models_incomplete, models_missing, model_data)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print('='*70)
    total_models = len(EXPECTED_MODELS)
    print(f"  Models complete:   {len(models_complete)}/{total_models}")
    print(f"  Models partial:    {len(models_incomplete)}/{total_models}")
    print(f"  Models missing:    {len(models_missing)}/{total_models}")
    print(f"  Total evaluations: {len(results)}")
    
    if len(models_complete) >= 3:
        print(f"\n✓ You have enough data for preliminary analysis!")
    else:
        print(f"\n⚠ Consider running more models before final analysis.")


if __name__ == "__main__":
    main()
