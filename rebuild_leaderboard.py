#!/usr/bin/env python3
"""
Utility script to rebuild the leaderboard.json from individual result files.
This script scans all result_*.json files in the millionaire_results directory
and generates a new leaderboard.json file.
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Any


def load_individual_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all individual result files from the results directory."""
    results = []
    
    # Find all result files
    result_files = list(results_dir.glob("result_*.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return results
    
    print(f"Found {len(result_files)} result files:")
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
                print(f"  ✓ Loaded {result_file.name} - {data.get('model', 'Unknown model')}")
        except Exception as e:
            print(f"  ✗ Failed to load {result_file.name}: {e}")
    
    return results


def create_leaderboard_entry(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Create a leaderboard entry from an evaluation result."""
    programs = evaluation.get('programs', [])
    
    # Calculate max earnings from a single program
    max_earnings = 0
    if programs:
        max_earnings = max(program.get('final_earnings', 0) for program in programs)
    
    # Calculate success rate
    success_rate = 0.0
    if evaluation.get('total_programs', 0) > 0:
        success_rate = round(
            (evaluation.get('successful_programs', 0) / evaluation.get('total_programs', 1)) * 100,
            1
        )
    
    return {
        "model": evaluation.get('model', 'Unknown'),
        "total_earnings": evaluation.get('total_earnings', 0),
        "average_earnings": round(evaluation.get('average_earnings', 0), 2),
        "successful_programs": evaluation.get('successful_programs', 0),
        "total_programs": evaluation.get('total_programs', 0),
        "success_rate": success_rate,
        "max_earnings_single_program": max_earnings
    }


def rebuild_leaderboard(results_dir: Path) -> None:
    """Rebuild the leaderboard from individual result files."""
    print(f"Rebuilding leaderboard from results in: {results_dir}")
    print("=" * 60)
    
    # Load all individual results
    evaluations = load_individual_results(results_dir)
    
    if not evaluations:
        print("No valid evaluation results found. Cannot create leaderboard.")
        return
    
    print(f"\nProcessing {len(evaluations)} evaluations...")
    
    # Create leaderboard entries
    leaderboard = []
    for evaluation in evaluations:
        entry = create_leaderboard_entry(evaluation)
        leaderboard.append(entry)
    
    # Sort by average earnings (descending)
    leaderboard.sort(key=lambda x: x['average_earnings'], reverse=True)
    
    # Determine total programs per model (should be consistent)
    total_programs_per_model = 0
    if leaderboard:
        total_programs_per_model = leaderboard[0]['total_programs']
    
    # Create the leaderboard summary
    leaderboard_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_models_evaluated": len(leaderboard),
        "total_programs_per_model": total_programs_per_model,
        "leaderboard": leaderboard
    }
    
    # Save the leaderboard
    leaderboard_file = results_dir / "leaderboard.json"
    with open(leaderboard_file, 'w', encoding='utf-8') as f:
        json.dump(leaderboard_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Leaderboard saved to: {leaderboard_file}")
    
    # Display summary
    print(f"\n{'='*80}")
    print("LEADERBOARD SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} | {'Avg Earnings':>12} | {'Total Earnings':>15} | {'Success':>10} | {'Max Single':>12}")
    print("-" * 80)
    
    for entry in leaderboard:
        print(
            f"{entry['model']:<35} | "
            f"€{entry['average_earnings']:>11,.2f} | "
            f"€{entry['total_earnings']:>14,} | "
            f"{entry['successful_programs']:>2}/{entry['total_programs']:<2} ({entry['success_rate']:>4.1f}%) | "
            f"€{entry['max_earnings_single_program']:>11,}"
        )
    
    print("-" * 80)
    print(f"Total models evaluated: {len(leaderboard)}")
    print(f"Programs per model: {total_programs_per_model}")


def create_combined_results(results_dir: Path) -> None:
    """Create combined_results.json from individual results."""
    print(f"\nCreating combined results file...")
    
    # Load all individual results
    evaluations = load_individual_results(results_dir)
    
    if not evaluations:
        print("No valid evaluation results found. Cannot create combined results.")
        return
    
    # Determine total programs per model
    total_programs_per_model = 0
    if evaluations:
        total_programs_per_model = evaluations[0].get('total_programs', 0)
    
    # Create combined results
    combined_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_models": len(evaluations),
        "total_programs_per_model": total_programs_per_model,
        "evaluations": evaluations
    }
    
    # Save combined results
    combined_file = results_dir / "combined_results.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Combined results saved to: {combined_file}")


def main():
    """Main function to rebuild leaderboard and combined results."""
    # Set up paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "millionaire_results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("Millionaire Benchmark Leaderboard Rebuilder")
    print(f"Results directory: {results_dir}")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    
    try:
        # Rebuild leaderboard
        rebuild_leaderboard(results_dir)
        
        # Create combined results
        create_combined_results(results_dir)
        
        print(f"\n🎉 Successfully rebuilt leaderboard and combined results!")
        
    except Exception as e:
        print(f"\n❌ Error rebuilding leaderboard: {e}")
        raise


if __name__ == "__main__":
    main()
