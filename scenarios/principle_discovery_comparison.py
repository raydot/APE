import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add parent directory to path to import principle_discovery
sys.path.insert(0, str(Path(__file__).parent))
from principle_discovery import run_principle_discovery
import matplotlib.pyplot as plt

load_dotenv()


def run_comparison_experiment(
    num_balls=8,
    table_size=5.0,
    max_velocity=2.0,
    elasticity=0.95,
    num_trials=20,
    num_collisions=20,
    base_seed=42
):
    """
    Run Principle Discovery comparison: learning vs no-learning observer.
    
    Args:
        num_balls: Number of actor balls in converging ring
        table_size: Size of square table (meters)
        max_velocity: Maximum initial velocity (m/s)
        elasticity: Coefficient of restitution
        num_trials: Number of trials per condition
        num_collisions: Target collisions per trial
        base_seed: Base random seed for reproducibility
    
    Returns:
        dict: Comparison results with statistics
    """
    
    print("\n" + "="*70)
    print("PRINCIPLE DISCOVERY COMPARISON EXPERIMENT")
    print("="*70)
    print(f"\nComparing observer learning across {num_trials} trials")
    print(f"Configuration:")
    print(f"  Actor balls: {num_balls}")
    print(f"  Table size: {table_size}m x {table_size}m")
    print(f"  Max velocity: {max_velocity} m/s")
    print(f"  Elasticity: {elasticity}")
    print(f"  Target collisions per trial: {num_collisions}")
    print(f"  Trials per condition: {num_trials}")
    print("="*70 + "\n")
    
    # Storage for results
    baseline_results = []
    learning_results = []
    
    # Create shared experience stores for each condition
    # Baseline doesn't use experience store, but learning trials share one
    from ape.learning import ExperienceStore, FeedbackGenerator
    
    learning_experience_store = ExperienceStore()
    learning_feedback_gen = FeedbackGenerator()
    
    # Run baseline trials (no learning)
    print("\n" + "="*70)
    print("BASELINE: NO LEARNING")
    print("="*70)
    print("Observer makes predictions without learning from past observations\n")
    
    for trial in range(num_trials):
        print(f"\n--- Baseline Trial {trial+1}/{num_trials} ---\n")
        
        result = run_principle_discovery(
            num_balls=num_balls,
            table_size=table_size,
            max_velocity=max_velocity,
            elasticity=elasticity,
            learning_enabled=False,
            trial_num=trial + 1,
            num_collisions=num_collisions,
            use_mlflow=True,
            seed=base_seed + trial,
            experience_store=None,  # No learning
            feedback_generator=None
        )
        
        baseline_results.append(result)
        
        print(f"\nBaseline Trial {trial+1} complete:")
        print(f"  Observations: {result['total_observations']}")
        print(f"  Avg error: {result['avg_prediction_error']:.3f} m/s")
        print(f"  Improvement: {result['improvement_percent']:.1f}%")
    
    # Run learning trials
    print("\n" + "="*70)
    print("LEARNING ENABLED")
    print("="*70)
    print("Observer learns from past observations using experience store\n")
    
    for trial in range(num_trials):
        print(f"\n--- Learning Trial {trial+1}/{num_trials} ---\n")
        
        result = run_principle_discovery(
            num_balls=num_balls,
            table_size=table_size,
            max_velocity=max_velocity,
            elasticity=elasticity,
            learning_enabled=True,
            trial_num=trial + 1,
            num_collisions=num_collisions,
            use_mlflow=True,
            seed=base_seed + trial,
            experience_store=learning_experience_store,  # Shared across learning trials
            feedback_generator=learning_feedback_gen
        )
        
        learning_results.append(result)
        
        print(f"\nLearning Trial {trial+1} complete:")
        print(f"  Observations: {result['total_observations']}")
        print(f"  Avg error: {result['avg_prediction_error']:.3f} m/s")
        print(f"  Improvement: {result['improvement_percent']:.1f}%")
    
    # Calculate statistics
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70 + "\n")
    
    # Extract metrics
    baseline_avg_errors = [r['avg_prediction_error'] for r in baseline_results]
    learning_avg_errors = [r['avg_prediction_error'] for r in learning_results]
    
    baseline_improvements = [r['improvement_percent'] for r in baseline_results]
    learning_improvements = [r['improvement_percent'] for r in learning_results]
    
    baseline_early_errors = [r['early_error'] for r in baseline_results]
    learning_early_errors = [r['early_error'] for r in learning_results]
    
    baseline_recent_errors = [r['recent_error'] for r in baseline_results]
    learning_recent_errors = [r['recent_error'] for r in learning_results]
    
    # Summary statistics
    print(f"{'Metric':<30} {'Baseline':<15} {'Learning':<15} {'Difference':<15}")
    print("-" * 75)
    
    baseline_avg_mean = np.mean(baseline_avg_errors)
    learning_avg_mean = np.mean(learning_avg_errors)
    print(f"{'Average Prediction Error':<30} {baseline_avg_mean:<15.3f} {learning_avg_mean:<15.3f} {baseline_avg_mean - learning_avg_mean:<15.3f}")
    
    baseline_early_mean = np.mean(baseline_early_errors)
    learning_early_mean = np.mean(learning_early_errors)
    print(f"{'Early Error (first 1/3)':<30} {baseline_early_mean:<15.3f} {learning_early_mean:<15.3f} {baseline_early_mean - learning_early_mean:<15.3f}")
    
    baseline_recent_mean = np.mean(baseline_recent_errors)
    learning_recent_mean = np.mean(learning_recent_errors)
    print(f"{'Recent Error (last 1/3)':<30} {baseline_recent_mean:<15.3f} {learning_recent_mean:<15.3f} {baseline_recent_mean - learning_recent_mean:<15.3f}")
    
    baseline_improvement_mean = np.mean(baseline_improvements)
    learning_improvement_mean = np.mean(learning_improvements)
    print(f"{'Improvement %':<30} {baseline_improvement_mean:<15.1f} {learning_improvement_mean:<15.1f} {learning_improvement_mean - baseline_improvement_mean:<15.1f}")
    
    # Statistical significance (t-test on average prediction error)
    from scipy import stats
    t_stat_avg, p_value_avg = stats.ttest_ind(baseline_avg_errors, learning_avg_errors)
    
    # T-test on improvement percentage
    t_stat_improve, p_value_improve = stats.ttest_ind(baseline_improvements, learning_improvements)
    
    print(f"\n{'Statistical Test (t-test on average error)':<50}")
    print(f"  Null hypothesis: Learning has no effect on prediction accuracy")
    print(f"  t-statistic: {t_stat_avg:.3f}")
    print(f"  p-value: {p_value_avg:.4f}")
    if p_value_avg < 0.05:
        print(f"  Result: ✓ SIGNIFICANT (p<0.05) - Learning reduces prediction error")
    else:
        print(f"  Result: ✗ NOT SIGNIFICANT (p≥0.05) - No clear learning effect")
    
    print(f"\n{'Statistical Test (t-test on improvement %)':<50}")
    print(f"  Null hypothesis: Learning has no effect on improvement rate")
    print(f"  t-statistic: {t_stat_improve:.3f}")
    print(f"  p-value: {p_value_improve:.4f}")
    if p_value_improve < 0.05:
        print(f"  Result: ✓ SIGNIFICANT (p<0.05) - Learning improves over time")
    else:
        print(f"  Result: ✗ NOT SIGNIFICANT (p≥0.05) - No clear improvement effect")
    
    # Create visualization
    print("\n[VISUALIZATION] Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Principle Discovery: Learning vs No-Learning ({num_balls} balls, {num_trials} trials)', fontsize=14, fontweight='bold')
    
    # Plot 1: Average prediction error per trial
    ax1 = axes[0, 0]
    trials = np.arange(1, num_trials + 1)
    ax1.plot(trials, baseline_avg_errors, 'o-', color='red', label='No Learning', alpha=0.7, linewidth=2)
    ax1.plot(trials, learning_avg_errors, 'o-', color='green', label='With Learning', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Average Prediction Error (m/s)', fontsize=11)
    ax1.set_title('Prediction Error Per Trial', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning curve (smoothed)
    ax2 = axes[0, 1]
    window = 5
    if num_trials >= window:
        baseline_smooth = np.convolve(baseline_avg_errors, np.ones(window)/window, mode='valid')
        learning_smooth = np.convolve(learning_avg_errors, np.ones(window)/window, mode='valid')
        smooth_trials = np.arange(window, num_trials + 1)
        ax2.plot(smooth_trials, baseline_smooth, '-', color='red', label='No Learning', linewidth=2.5)
        ax2.plot(smooth_trials, learning_smooth, '-', color='green', label='With Learning', linewidth=2.5)
    else:
        ax2.plot(trials, baseline_avg_errors, '-', color='red', label='No Learning', linewidth=2.5)
        ax2.plot(trials, learning_avg_errors, '-', color='green', label='With Learning', linewidth=2.5)
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Prediction Error (m/s)', fontsize=11)
    ax2.set_title(f'Learning Curve ({window}-trial rolling average)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement percentage per trial
    ax3 = axes[1, 0]
    ax3.plot(trials, baseline_improvements, 'o-', color='red', label='No Learning', alpha=0.7, linewidth=2)
    ax3.plot(trials, learning_improvements, 'o-', color='green', label='With Learning', alpha=0.7, linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Trial Number', fontsize=11)
    ax3.set_ylabel('Improvement (%)', fontsize=11)
    ax3.set_title('Within-Trial Improvement (Early vs Late)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average error comparison (bar chart)
    ax4 = axes[1, 1]
    categories = ['No Learning', 'With Learning']
    avg_errors = [baseline_avg_mean, learning_avg_mean]
    colors = ['red', 'green']
    bars = ax4.bar(categories, avg_errors, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Average Prediction Error (m/s)', fontsize=11)
    ax4.set_title('Average Observer Prediction Error', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(avg_errors) * 1.2)
    
    # Add value labels on bars
    for bar, error in zip(bars, avg_errors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2f} m/s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add statistical significance annotation
    significance_text = f"Statistical Test: p={p_value_avg:.4f}"
    if p_value_avg < 0.05:
        significance_text += f" (SIGNIFICANT ✓)"
        sig_color = 'green'
    else:
        significance_text += f" (NOT SIGNIFICANT)"
        sig_color = 'red'
    
    ax4.text(0.5, 0.02, significance_text, 
            transform=ax4.transAxes,
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=sig_color, alpha=0.2))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f'{timestamp}_principle_discovery_comparison_{num_balls}balls_{num_trials}trials.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[VISUALIZATION] Saved comparison plot to {filename}")
    
    plt.close()
    
    # Return comparison results
    return {
        'num_trials': num_trials,
        'num_balls': num_balls,
        'baseline_results': baseline_results,
        'learning_results': learning_results,
        'baseline_avg_error': baseline_avg_mean,
        'learning_avg_error': learning_avg_mean,
        'baseline_improvement': baseline_improvement_mean,
        'learning_improvement': learning_improvement_mean,
        'p_value_error': p_value_avg,
        'p_value_improvement': p_value_improve,
        't_stat_error': t_stat_avg,
        't_stat_improvement': t_stat_improve
    }


def main():
    """Run comparison experiment with command-line arguments."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Principle Discovery Comparison Experiment')
    parser.add_argument('--balls', type=int, default=8, help='Number of actor balls (default: 8)')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per condition (default: 20)')
    parser.add_argument('--collisions', type=int, default=20, help='Target collisions per trial (default: 20)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_comparison_experiment(
        num_balls=args.balls,
        num_trials=args.trials,
        num_collisions=args.collisions,
        base_seed=args.seed
    )
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nBaseline average error: {results['baseline_avg_error']:.3f} m/s")
    print(f"Learning average error: {results['learning_avg_error']:.3f} m/s")
    print(f"Error reduction: {results['baseline_avg_error'] - results['learning_avg_error']:.3f} m/s")
    print(f"\nBaseline improvement: {results['baseline_improvement']:.1f}%")
    print(f"Learning improvement: {results['learning_improvement']:.1f}%")
    print(f"\np-value (error): {results['p_value_error']:.4f}")
    print(f"p-value (improvement): {results['p_value_improvement']:.4f}")
    print(f"\nView results: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print()


if __name__ == "__main__":
    main()
