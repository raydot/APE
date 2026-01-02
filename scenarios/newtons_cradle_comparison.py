import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add parent directory to path to import newtons_cradle
sys.path.insert(0, str(Path(__file__).parent))
from newtons_cradle import run_newtons_cradle
import matplotlib.pyplot as plt

load_dotenv()


def run_comparison_experiment(
    num_balls=5,
    initial_velocity=2.0,
    elasticity=1.0,
    num_trials=20,
    visualize_final=False
):
    """
    Compare learning vs no-learning performance over multiple trials.
    
    Tests whether agents improve with experience or if they perform
    the same without learning.
    
    Args:
        num_balls: Number of balls in cradle
        initial_velocity: Initial velocity (m/s)
        elasticity: Coefficient of restitution
        num_trials: Number of trials to run for each condition
        visualize_final: Whether to visualize the final trial
    """
    
    print("\n" + "="*70)
    print("NEWTON'S CRADLE: LEARNING vs NO-LEARNING COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Balls: {num_balls}")
    print(f"  Initial velocity: {initial_velocity} m/s")
    print(f"  Elasticity: {elasticity}")
    print(f"  Trials per condition: {num_trials}")
    print("="*70 + "\n")
    
    # Run baseline (no learning)
    print("\n" + "="*70)
    print("BASELINE: NO LEARNING")
    print("="*70)
    print("Agents make predictions without learning from past collisions\n")
    
    baseline_results = []
    for trial in range(1, num_trials + 1):
        print(f"\n--- Baseline Trial {trial}/{num_trials} ---")
        result = run_newtons_cradle(
            num_balls=num_balls,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
            learning_enabled=False,  # No learning
            trial_num=trial,
            use_mlflow=True,
            visualize=False
        )
        baseline_results.append(result)
        
        # Quick summary
        print(f"  Momentum error: {result['momentum_error_percent']:.2f}%")
        print(f"  Final velocity error: {result['final_velocity_error_percent']:.2f}%")
        print(f"  Resolver acceptance rate: {result['resolver_acceptance_rate']*100:.1f}%")
        print(f"  Success: {'✓' if result['success'] else '✗'}")
    
    # Run with learning
    print("\n" + "="*70)
    print("EXPERIMENTAL: WITH LEARNING")
    print("="*70)
    print("Agents learn from past collisions and improve over time\n")
    
    learning_results = []
    for trial in range(1, num_trials + 1):
        print(f"\n--- Learning Trial {trial}/{num_trials} ---")
        result = run_newtons_cradle(
            num_balls=num_balls,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
            learning_enabled=True,  # Learning enabled
            trial_num=trial,
            use_mlflow=True,
            visualize=(visualize_final and trial == num_trials)  # Visualize last trial
        )
        learning_results.append(result)
        
        # Quick summary
        print(f"  Momentum error: {result['momentum_error_percent']:.2f}%")
        print(f"  Final velocity error: {result['final_velocity_error_percent']:.2f}%")
        print(f"  Resolver acceptance rate: {result['resolver_acceptance_rate']*100:.1f}%")
        print(f"  Success: {'✓' if result['success'] else '✗'}")
    
    # Analyze results
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    # Extract metrics
    baseline_momentum_errors = [r['momentum_error_percent'] for r in baseline_results]
    learning_momentum_errors = [r['momentum_error_percent'] for r in learning_results]
    
    baseline_velocity_errors = [r['final_velocity_error_percent'] for r in baseline_results]
    learning_velocity_errors = [r['final_velocity_error_percent'] for r in learning_results]
    
    # Extract resolver acceptance rates (THE REAL LEARNING METRIC)
    baseline_acceptance_rates = [r['resolver_acceptance_rate'] * 100 for r in baseline_results]
    learning_acceptance_rates = [r['resolver_acceptance_rate'] * 100 for r in learning_results]
    
    baseline_success_rate = sum(r['success'] for r in baseline_results) / len(baseline_results)
    learning_success_rate = sum(r['success'] for r in learning_results) / len(learning_results)
    
    # Overall statistics
    print(f"\n{'Metric':<30} {'Baseline':<20} {'Learning':<20} {'Improvement':<15}")
    print("-" * 85)
    
    baseline_mom_mean = np.mean(baseline_momentum_errors)
    learning_mom_mean = np.mean(learning_momentum_errors)
    mom_improvement = ((baseline_mom_mean - learning_mom_mean) / baseline_mom_mean * 100) if baseline_mom_mean > 0 else 0
    print(f"{'Momentum Error (mean)':<30} {baseline_mom_mean:>6.2f}% ± {np.std(baseline_momentum_errors):>5.2f}%   {learning_mom_mean:>6.2f}% ± {np.std(learning_momentum_errors):>5.2f}%   {mom_improvement:>+6.1f}%")
    
    baseline_vel_mean = np.mean(baseline_velocity_errors)
    learning_vel_mean = np.mean(learning_velocity_errors)
    vel_improvement = ((baseline_vel_mean - learning_vel_mean) / baseline_vel_mean * 100) if baseline_vel_mean > 0 else 0
    print(f"{'Final Velocity Error (mean)':<30} {baseline_vel_mean:>6.2f}% ± {np.std(baseline_velocity_errors):>5.2f}%   {learning_vel_mean:>6.2f}% ± {np.std(learning_velocity_errors):>5.2f}%   {vel_improvement:>+6.1f}%")
    
    success_improvement = (learning_success_rate - baseline_success_rate) * 100
    print(f"{'Success Rate':<30} {baseline_success_rate:>6.1%}              {learning_success_rate:>6.1%}              {success_improvement:>+6.1f}pp")
    
    # Resolver acceptance rate (ACTUAL LEARNING METRIC)
    baseline_accept_mean = np.mean(baseline_acceptance_rates)
    learning_accept_mean = np.mean(learning_acceptance_rates)
    accept_improvement = learning_accept_mean - baseline_accept_mean
    print(f"\n{'RESOLVER ACCEPTANCE RATE':<30} {'(Agent predictions accepted)':<40}")
    print(f"{'  Baseline (no learning)':<30} {baseline_accept_mean:>6.1f}% ± {np.std(baseline_acceptance_rates):>5.1f}%")
    print(f"{'  With learning':<30} {learning_accept_mean:>6.1f}% ± {np.std(learning_acceptance_rates):>5.1f}%")
    print(f"{'  Improvement':<30} {accept_improvement:>+6.1f}pp")
    
    # Early vs Late performance (learning curve)
    print(f"\n{'Learning Curve Analysis':<30} {'Early (1-5)':<20} {'Late ({}-{})':<20} {'Improvement':<15}".format(num_trials-4, num_trials))
    print("-" * 85)
    
    if num_trials >= 10:
        early_learning = learning_momentum_errors[:5]
        late_learning = learning_momentum_errors[-5:]
        
        early_mean = np.mean(early_learning)
        late_mean = np.mean(late_learning)
        learning_improvement = ((early_mean - late_mean) / early_mean * 100) if early_mean > 0 else 0
        
        print(f"{'Momentum Error (learning)':<30} {early_mean:>6.2f}% ± {np.std(early_learning):>5.2f}%   {late_mean:>6.2f}% ± {np.std(late_learning):>5.2f}%   {learning_improvement:>+6.1f}%")
        
        early_baseline = baseline_momentum_errors[:5]
        late_baseline = baseline_momentum_errors[-5:]
        baseline_change = ((np.mean(early_baseline) - np.mean(late_baseline)) / np.mean(early_baseline) * 100) if np.mean(early_baseline) > 0 else 0
        
        print(f"{'Momentum Error (baseline)':<30} {np.mean(early_baseline):>6.2f}% ± {np.std(early_baseline):>5.2f}%   {np.mean(late_baseline):>6.2f}% ± {np.std(late_baseline):>5.2f}%   {baseline_change:>+6.1f}%")
    
    # Statistical significance (t-test on ACCEPTANCE RATE - the real metric)
    from scipy import stats
    t_stat_accept, p_value_accept = stats.ttest_ind(baseline_acceptance_rates, learning_acceptance_rates)
    
    print(f"\n{'Statistical Test (t-test on acceptance rate)':<50}")
    print(f"  Null hypothesis: Learning has no effect on agent accuracy")
    print(f"  t-statistic: {t_stat_accept:.3f}")
    print(f"  p-value: {p_value_accept:.4f}")
    if p_value_accept < 0.05:
        print(f"  Result: ✓ SIGNIFICANT (p<0.05) - Learning improves agent predictions")
    else:
        print(f"  Result: ✗ NOT SIGNIFICANT (p≥0.05) - No clear learning effect")
    
    # Create visualization
    print("\n[VISUALIZATION] Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Newton\'s Cradle: Learning vs No-Learning ({num_balls} balls, {num_trials} trials)', fontsize=14, fontweight='bold')
    
    # Plot 1: RESOLVER ACCEPTANCE RATE (THE REAL LEARNING METRIC)
    ax1 = axes[0, 0]
    trials = list(range(1, num_trials + 1))
    ax1.plot(trials, baseline_acceptance_rates, 'o-', color='red', alpha=0.6, label='No Learning', linewidth=2, markersize=6)
    ax1.plot(trials, learning_acceptance_rates, 'o-', color='green', alpha=0.6, label='With Learning', linewidth=2, markersize=6)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Acceptance Rate (%)', fontsize=11)
    ax1.set_title('Resolver Acceptance Rate (Agent Prediction Accuracy)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Rolling average of acceptance rate
    ax2 = axes[0, 1]
    if num_trials >= 5:
        window = 5
        baseline_rolling = np.convolve(baseline_acceptance_rates, np.ones(window)/window, mode='valid')
        learning_rolling = np.convolve(learning_acceptance_rates, np.ones(window)/window, mode='valid')
        rolling_trials = list(range(window, num_trials + 1))
        
        ax2.plot(rolling_trials, baseline_rolling, '-', color='red', alpha=0.8, label='No Learning (5-trial avg)', linewidth=3)
        ax2.plot(rolling_trials, learning_rolling, '-', color='green', alpha=0.8, label='With Learning (5-trial avg)', linewidth=3)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Trial Number', fontsize=11)
        ax2.set_ylabel('Acceptance Rate (%) - 5-trial avg', fontsize=11)
        ax2.set_title('Learning Curve (Smoothed)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
    
    # Plot 3: Ground truth imposed rate (inverse of acceptance)
    ax3 = axes[1, 0]
    baseline_imposed_rates = [100 - rate for rate in baseline_acceptance_rates]
    learning_imposed_rates = [100 - rate for rate in learning_acceptance_rates]
    ax3.plot(trials, baseline_imposed_rates, 'o-', color='red', alpha=0.6, label='No Learning', linewidth=2, markersize=6)
    ax3.plot(trials, learning_imposed_rates, 'o-', color='green', alpha=0.6, label='With Learning', linewidth=2, markersize=6)
    ax3.set_xlabel('Trial Number', fontsize=11)
    ax3.set_ylabel('Ground Truth Imposed (%)', fontsize=11)
    ax3.set_title('Resolver Corrections (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # Plot 4: Acceptance rate comparison (bar chart)
    ax4 = axes[1, 1]
    categories = ['No Learning', 'With Learning']
    accept_rates = [baseline_accept_mean, learning_accept_mean]
    colors = ['red', 'green']
    bars = ax4.bar(categories, accept_rates, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.set_ylabel('Acceptance Rate (%)', fontsize=11)
    ax4.set_title('Average Agent Prediction Accuracy', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.legend()
    
    # Add value labels on bars
    for bar, rate in zip(bars, accept_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add statistical significance annotation
    significance_text = f"Statistical Test: p={p_value_accept:.4f}"
    if p_value_accept < 0.05:
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
    filename = output_dir / f'{timestamp}_newtons_cradle_comparison_{num_balls}balls_{num_trials}trials.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[VISUALIZATION] Saved comparison plot to {filename}")
    
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Evaluate based on ACCEPTANCE RATE (the real metric)
    if accept_improvement > 10:
        print("✓ Learning system shows CLEAR IMPROVEMENT in agent predictions")
        print(f"  Acceptance rate increased by {accept_improvement:.1f} percentage points")
    elif accept_improvement > 5:
        print("~ Learning system shows MODERATE IMPROVEMENT in agent predictions")
        print(f"  Acceptance rate increased by {accept_improvement:.1f} percentage points")
    elif accept_improvement > 0:
        print("~ Learning system shows SLIGHT IMPROVEMENT in agent predictions")
        print(f"  Acceptance rate increased by {accept_improvement:.1f} percentage points")
    else:
        print("✗ Learning system shows NO IMPROVEMENT in agent predictions")
        print(f"  Acceptance rate changed by {accept_improvement:.1f} percentage points")
    
    if p_value_accept < 0.05:
        print(f"✓ Difference is STATISTICALLY SIGNIFICANT (p={p_value_accept:.4f})")
    else:
        print(f"✗ Difference is NOT statistically significant (p={p_value_accept:.4f})")
    
    # Note about final errors
    print(f"\nNote: Final momentum/energy errors are ~0% in both conditions because")
    print(f"      the resolver imposes ground truth when agents are wrong.")
    print(f"      The ACCEPTANCE RATE shows whether agents actually learned.")
    
    if num_trials >= 10 and learning_improvement > 10:
        print(f"✓ Clear LEARNING CURVE detected ({learning_improvement:.1f}% improvement from early to late trials)")
    elif num_trials >= 10:
        print(f"~ Weak learning curve ({learning_improvement:.1f}% improvement from early to late trials)")
    
    print("\n" + "="*70)
    print(f"View detailed metrics: mlflow ui --backend-store-uri ./mlruns")
    print("="*70 + "\n")
    
    return {
        'baseline_results': baseline_results,
        'learning_results': learning_results,
        'baseline_success_rate': baseline_success_rate,
        'learning_success_rate': learning_success_rate,
        'baseline_acceptance_rate': baseline_accept_mean,
        'learning_acceptance_rate': learning_accept_mean,
        'acceptance_improvement': accept_improvement,
        'p_value': p_value_accept,
        'significant': p_value_accept < 0.05
    }


def main():
    """Run comparison experiments with different trial counts."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Compare learning vs no-learning in Newton\'s Cradle')
    parser.add_argument('--balls', type=int, default=5, help='Number of balls (default: 5)')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per condition (default: 20)')
    parser.add_argument('--velocity', type=float, default=2.0, help='Initial velocity (default: 2.0)')
    parser.add_argument('--visualize', action='store_true', help='Visualize final trial')
    
    args = parser.parse_args()
    
    run_comparison_experiment(
        num_balls=args.balls,
        initial_velocity=args.velocity,
        elasticity=1.0,
        num_trials=args.trials,
        visualize_final=args.visualize
    )


if __name__ == "__main__":
    main()
