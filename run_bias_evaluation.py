#!/usr/bin/env python3
"""
The Visa Wall: Benchmarking LLM Bias Against Non-EU Applicants
Automatic Evaluation Script for NVIDIA A100 GPU

Configuration is loaded from config.yaml for easy modification.
"""

import argparse
import json
import re
import os
import csv
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from vllm import LLM, SamplingParams

# ============================================================================
# HUGGINGFACE AUTHENTICATION (for gated models like Gemma)
# ============================================================================
# Replace with your HuggingFace token from https://huggingface.co/settings/tokens
HF_TOKEN = "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"  # <-- PUT YOUR TOKEN HERE

# Set environment variables for HuggingFace authentication
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Login to HuggingFace
try:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print(f"✓ Logged in to HuggingFace successfully")
except Exception as e:
    print(f"⚠ HuggingFace login warning: {e}")

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvaluationResult:
    model: str
    candidate: str
    category: str
    run_id: int
    cultural_fit_score: Optional[int]
    hiring_probability: Optional[int]
    adjectives: List[str]
    reasoning: str
    raw_response: str
    visa_mentioned: bool
    visa_keywords: List[str]
    mitigation_applied: bool
    timestamp: str


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def build_candidate_prompt(candidate_id: str, config: Dict) -> str:
    """Build the evaluation prompt for a candidate from config."""
    c = config['candidates'][candidate_id]
    template = config['prompts']['candidate_template']
    task = config['prompts']['task']
    
    prompt = template.format(
        name=c['name'],
        location=c['location'],
        nationality=c['nationality'],
        education=c['education'],
        experience=c['experience'],
        language=c.get('language', 'Englisch (C1/Verhandlungssicher)')
    )
    return prompt + "\n" + task


def format_chat_prompt(model_name: str, system: str, user: str) -> str:
    """Format prompt according to model's chat template."""
    if "llama" in model_name.lower():
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    elif "ministral" in model_name.lower():
        # Ministral-8B uses the same format as Mistral-Small/Large (v7 template)
        return f"<s>[INST] {system}\n\n{user} [/INST]"
    elif "mistral" in model_name.lower():
        # Mistral-Small/Large v7 template
        return f"<s>[INST] {system}\n\n{user} [/INST]"
    elif "qwen" in model_name.lower():
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    elif "gemma" in model_name.lower():
        # Gemma 2/3 uses specific format
        return f"<bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
    else:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def clean_response(response: str) -> str:
    """
    Clean LLM output by removing Chain-of-Thought blocks and extracting actual content.
    Handles models like Qwen3-8B that use <think>...</think> tags.
    """
    cleaned = response
    
    # 1. Remove Chain-of-Thought (<think>...</think>) blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    
    # 2. Remove other common thinking/reasoning tags
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<scratch>.*?</scratch>', '', cleaned, flags=re.DOTALL)
    
    # 3. Extract content from markdown code blocks if present
    code_block_pattern = r"```(?:json)?\s*(.*?)```"
    code_matches = re.findall(code_block_pattern, cleaned, re.DOTALL)
    if code_matches:
        # Use the last code block found (usually the final answer)
        cleaned = code_matches[-1]
    
    # 4. Clean up whitespace
    cleaned = cleaned.strip()
    
    # If cleaning removed everything, fall back to original (minus think tags)
    if not cleaned:
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    return cleaned


def extract_scores(response: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract Cultural Fit Score and Hiring Probability from response."""
    # Clean the response first to remove <think> blocks
    response = clean_response(response)
    
    cultural_fit = None
    hiring_prob = None
    
    # Pattern for Cultural Fit Score
    cf_patterns = [
        r"cultural\s*fit\s*score[:\s]*(\d{1,3})",
        r"cultural\s*fit[:\s]*(\d{1,3})",
        r"fit\s*score[:\s]*(\d{1,3})",
    ]
    for pattern in cf_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                cultural_fit = score
                break
    
    # Pattern for Hiring Probability
    hp_patterns = [
        r"hiring\s*probability[:\s]*(\d{1,3})",
        r"einstellungswahrscheinlichkeit[:\s]*(\d{1,3})",
        r"hiring[:\s]*(\d{1,3})\s*%?",
    ]
    for pattern in hp_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                hiring_prob = score
                break
    
    # Fallback: look for any two numbers between 0-100
    if cultural_fit is None or hiring_prob is None:
        numbers = re.findall(r'\b(\d{1,3})\s*(?:%|/100|von 100)?', response)
        valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
        if len(valid_scores) >= 2:
            if cultural_fit is None:
                cultural_fit = valid_scores[0]
            if hiring_prob is None:
                hiring_prob = valid_scores[1] if len(valid_scores) > 1 else valid_scores[0]
    
    return cultural_fit, hiring_prob


def extract_adjectives(response: str) -> List[str]:
    """Extract adjectives from response."""
    # Clean the response first to remove <think> blocks
    response = clean_response(response)
    
    patterns = [
        r"adjektive?[:\s]*(.+?)(?:\n|begründ|$)",
        r"3\s*adjektive?[:\s]*(.+?)(?:\n|$)",
        r"beschreiben[:\s]*(.+?)(?:\n|begründ|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)
            words = re.split(r'[,;•\-\d\.\n]+', text)
            words = [w.strip().strip('"\'') for w in words if w.strip()]
            words = [w for w in words if 3 <= len(w) <= 30]
            if words:
                return words[:5]
    return []


def check_visa_mention(response: str, visa_keywords: List[str]) -> Tuple[bool, List[str]]:
    """Check if response mentions visa-related terms."""
    # Clean the response first to remove <think> blocks
    response = clean_response(response)
    
    found_keywords = []
    response_lower = response.lower()
    for keyword in visa_keywords:
        if keyword in response_lower:
            found_keywords.append(keyword)
    return len(found_keywords) > 0, found_keywords


def parse_response(response: str, visa_keywords: List[str]) -> Dict:
    """Parse model response to extract all metrics."""
    cf_score, hp_score = extract_scores(response)
    adjectives = extract_adjectives(response)
    visa_mentioned, found_keywords = check_visa_mention(response, visa_keywords)
    
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    reasoning = '. '.join(sentences[-2:]) if sentences else response[:500]
    
    return {
        "cultural_fit_score": cf_score,
        "hiring_probability": hp_score,
        "adjectives": adjectives,
        "reasoning": reasoning,
        "visa_mentioned": visa_mentioned,
        "visa_keywords": found_keywords,
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(
    model_keys: List[str],
    config: Dict,
    num_runs: int = 1,
    output_dir: str = "results",
    include_mitigation: bool = True,
    gpu_memory_utilization: float = 0.98,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = False,
) -> List[EvaluationResult]:
    """Run the full evaluation pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results: List[EvaluationResult] = []
    timestamp = datetime.now().isoformat()
    
    # Get settings from config
    settings = config.get('settings', {})
    sampling_params = SamplingParams(
        temperature=settings.get('temperature', 0.7),
        top_p=settings.get('top_p', 0.9),
        max_tokens=settings.get('max_tokens', 512),
        stop=["<|eot_id|>", "<|im_end|>", "</s>"]
    )
    
    system_prompt = config['prompts']['system']
    mitigation_prompt = config['prompts']['mitigation']
    visa_keywords = config.get('visa_keywords', [])
    candidates = config['candidates']
    models = config['models']
    
    for model_key in model_keys:
        if model_key not in models:
            print(f"WARNING: Model '{model_key}' not found in config, skipping.")
            continue
            
        model_path = models[model_key]['path']
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key} ({model_path})")
        print(f"{'='*60}")
        
        # Determine max_model_len based on model type to avoid OOM
        model_max_len = max_model_len
        if model_max_len is None:
            # Auto-detect sensible defaults for models with large context windows
            model_lower = model_path.lower()
            if "ministral" in model_lower or "mistral" in model_lower:
                model_max_len = 8192  # Ministral has 262k context but we don't need it
                print(f"  Auto-setting max_model_len=8192 for Mistral model")
            elif "gemma" in model_lower:
                model_max_len = 8192
                print(f"  Auto-setting max_model_len=8192 for Gemma model")
            elif "qwen" in model_lower:
                model_max_len = 32768
                print(f"  Auto-setting max_model_len=32768 for Qwen model")
        
        if model_max_len:
            print(f"  Using max_model_len: {model_max_len}")
        
        if enforce_eager:
            print(f"  Using enforce_eager=True (for Gemma 2 softcapping support)")
        
        # Load model
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
        }
        if model_max_len:
            llm_kwargs["max_model_len"] = model_max_len
        
        llm = LLM(**llm_kwargs)
        
        # Run evaluation for each candidate
        for candidate_id, candidate_info in candidates.items():
            for run_id in range(num_runs):
                print(f"\n  Evaluating: {candidate_id} (run {run_id + 1}/{num_runs})")
                
                # Standard evaluation
                user_prompt = build_candidate_prompt(candidate_id, config)
                full_prompt = format_chat_prompt(model_key, system_prompt, user_prompt)
                
                outputs = llm.generate([full_prompt], sampling_params)
                response = outputs[0].outputs[0].text
                
                parsed = parse_response(response, visa_keywords)
                
                result = EvaluationResult(
                    model=model_key,
                    candidate=candidate_id,
                    category=candidate_info["category"],
                    run_id=run_id,
                    cultural_fit_score=parsed["cultural_fit_score"],
                    hiring_probability=parsed["hiring_probability"],
                    adjectives=parsed["adjectives"],
                    reasoning=parsed["reasoning"],
                    raw_response=response,
                    visa_mentioned=parsed["visa_mentioned"],
                    visa_keywords=parsed["visa_keywords"],
                    mitigation_applied=False,
                    timestamp=timestamp,
                )
                all_results.append(result)
                
                print(f"    Cultural Fit: {parsed['cultural_fit_score']}")
                print(f"    Hiring Prob:  {parsed['hiring_probability']}")
                print(f"    Visa Mention: {parsed['visa_mentioned']} {parsed['visa_keywords']}")
                
                # Mitigation test for non-EU candidates
                if include_mitigation and not candidate_info.get('is_eu', True):
                    print("    Running mitigation test...")
                    
                    mitigated_system = mitigation_prompt + "\n" + system_prompt
                    full_prompt_mit = format_chat_prompt(model_key, mitigated_system, user_prompt)
                    
                    outputs_mit = llm.generate([full_prompt_mit], sampling_params)
                    response_mit = outputs_mit[0].outputs[0].text
                    
                    parsed_mit = parse_response(response_mit, visa_keywords)
                    
                    result_mit = EvaluationResult(
                        model=model_key,
                        candidate=candidate_id,
                        category=candidate_info["category"],
                        run_id=run_id,
                        cultural_fit_score=parsed_mit["cultural_fit_score"],
                        hiring_probability=parsed_mit["hiring_probability"],
                        adjectives=parsed_mit["adjectives"],
                        reasoning=parsed_mit["reasoning"],
                        raw_response=response_mit,
                        visa_mentioned=parsed_mit["visa_mentioned"],
                        visa_keywords=parsed_mit["visa_keywords"],
                        mitigation_applied=True,
                        timestamp=timestamp,
                    )
                    all_results.append(result_mit)
                    
                    print(f"    [MIT] Cultural Fit: {parsed_mit['cultural_fit_score']}")
                    print(f"    [MIT] Hiring Prob:  {parsed_mit['hiring_probability']}")
        
        # Clear GPU memory before loading next model
        del llm
        import torch
        torch.cuda.empty_cache()
    
    # Save results
    save_results(all_results, output_dir, timestamp)
    
    return all_results


def save_results(results: List[EvaluationResult], output_dir: str, timestamp: str):
    """Save results to JSON and CSV files."""
    
    # JSON output
    json_path = os.path.join(output_dir, f"results_{timestamp.replace(':', '-')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    # CSV output
    csv_path = os.path.join(output_dir, f"results_{timestamp.replace(':', '-')}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'model', 'candidate', 'category', 'run_id', 'mitigation_applied',
            'cultural_fit_score', 'hiring_probability', 'visa_mentioned',
            'visa_keywords', 'adjectives', 'reasoning'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'model': r.model,
                'candidate': r.candidate,
                'category': r.category,
                'run_id': r.run_id,
                'mitigation_applied': r.mitigation_applied,
                'cultural_fit_score': r.cultural_fit_score,
                'hiring_probability': r.hiring_probability,
                'visa_mentioned': r.visa_mentioned,
                'visa_keywords': '|'.join(r.visa_keywords),
                'adjectives': '|'.join(r.adjectives),
                'reasoning': r.reasoning[:200],
            }
            writer.writerow(row)
    print(f"Saved CSV: {csv_path}")
    
    # Summary statistics
    print_summary(results)


def print_summary(results: List[EvaluationResult]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if not r.mitigation_applied:
            key = (r.model, r.candidate)
            if r.cultural_fit_score:
                stats[key]['cf'].append(r.cultural_fit_score)
            if r.hiring_probability:
                stats[key]['hp'].append(r.hiring_probability)
            stats[key]['visa'].append(r.visa_mentioned)
    
    print(f"\n{'Model':<15} {'Candidate':<10} {'CF Score':<10} {'Hire Prob':<10} {'Visa%':<8}")
    print("-" * 55)
    
    for (model, candidate), values in sorted(stats.items()):
        cf_avg = sum(values['cf']) / len(values['cf']) if values['cf'] else 'N/A'
        hp_avg = sum(values['hp']) / len(values['hp']) if values['hp'] else 'N/A'
        visa_pct = sum(values['visa']) / len(values['visa']) * 100 if values['visa'] else 0
        
        cf_str = f"{cf_avg:.1f}" if isinstance(cf_avg, float) else cf_avg
        hp_str = f"{hp_avg:.1f}" if isinstance(hp_avg, float) else hp_avg
        
        print(f"{model:<15} {candidate:<10} {cf_str:<10} {hp_str:<10} {visa_pct:.0f}%")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="The Visa Wall: LLM Bias Evaluation"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of runs per candidate (default: 1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--no-mitigation",
        action="store_true",
        help="Skip mitigation tests"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=None,
        help="GPU memory utilization (default: from config)"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for large models (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length (context size) to limit memory usage (default: auto-detect)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph and use eager mode (required for Gemma 2 softcapping)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine models to run
    if args.models:
        model_keys = args.models
    else:
        model_keys = list(config['models'].keys())
    
    # GPU memory from config or CLI
    gpu_memory = args.gpu_memory or config.get('settings', {}).get('gpu_memory_utilization', 0.90)
    
    print("="*60)
    print("THE VISA WALL: LLM Bias Evaluation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Models: {model_keys}")
    print(f"Candidates: {list(config['candidates'].keys())}")
    print(f"Runs per candidate: {args.runs}")
    print(f"Output directory: {args.output}")
    print(f"Mitigation tests: {not args.no_mitigation}")
    if args.max_model_len:
        print(f"Max model length: {args.max_model_len}")
    
    run_evaluation(
        model_keys=model_keys,
        config=config,
        num_runs=args.runs,
        output_dir=args.output,
        include_mitigation=not args.no_mitigation,
        gpu_memory_utilization=gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )


if __name__ == "__main__":
    main()
