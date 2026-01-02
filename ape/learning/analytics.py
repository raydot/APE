import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from .experience_store import ExperienceStore


class LearningAnalytics:
    """Analyze and visualize agent learning over time"""
    
    def __init__(self, experience_store: ExperienceStore):
        self.store = experience_store
    
    def plot_learning_curve(
        self,
        agent_id: str,
        window_size: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot learning curve showing accuracy over time
        
        Args:
            agent_id: Which agent to analyze
            window_size: Rolling window for smoothing
            save_path: Optional path to save figure
        """
        stats = self.store.get_agent_statistics(agent_id)
        
        if stats['total_experiences'] == 0:
            print(f"No experiences found for {agent_id}")
            return
        
        experiences = stats['experiences_over_time']
        
        timestamps = [exp['timestamp'] for exp in experiences]
        errors = [exp['error'] for exp in experiences]
        was_correct = [1 if exp['was_correct'] else 0 for exp in experiences]
        
        rolling_accuracy = []
        for i in range(len(was_correct)):
            start = max(0, i - window_size + 1)
            window = was_correct[start:i+1]
            rolling_accuracy.append(sum(window) / len(window))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(timestamps, rolling_accuracy, 'b-', linewidth=2, label=f'Rolling Accuracy (window={window_size})')
        ax1.axhline(y=stats['overall_accuracy'], color='r', linestyle='--', label=f'Overall Accuracy: {stats["overall_accuracy"]:.1%}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Learning Curve for {agent_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        ax2.scatter(timestamps, errors, alpha=0.5, s=20)
        ax2.plot(timestamps, errors, 'b-', alpha=0.3)
        ax2.axhline(y=stats['avg_prediction_error'], color='r', linestyle='--', label=f'Avg Error: {stats["avg_prediction_error"]:.3f}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Prediction Error (m/s)')
        ax2.set_title('Prediction Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning curve to {save_path}")
        
        plt.show()
    
    def compare_agents(
        self,
        agent_ids: List[str],
        save_path: Optional[str] = None
    ):
        """Compare learning performance across multiple agents"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
        
        for agent_id, color in zip(agent_ids, colors):
            stats = self.store.get_agent_statistics(agent_id)
            
            if stats['total_experiences'] == 0:
                continue
            
            experiences = stats['experiences_over_time']
            timestamps = [exp['timestamp'] for exp in experiences]
            errors = [exp['error'] for exp in experiences]
            was_correct = [1 if exp['was_correct'] else 0 for exp in experiences]
            
            window = 10
            rolling_acc = []
            for i in range(len(was_correct)):
                start = max(0, i - window + 1)
                w = was_correct[start:i+1]
                rolling_acc.append(sum(w) / len(w))
            
            ax1.plot(timestamps, rolling_acc, color=color, linewidth=2, label=agent_id)
            ax2.plot(timestamps, errors, color=color, alpha=0.6, label=agent_id)
            ax3.bar(agent_id, stats['overall_accuracy'], color=color)
            ax4.bar(agent_id, stats['avg_prediction_error'], color=color)
        
        ax1.set_title('Accuracy Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Rolling Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Prediction Error Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Overall Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4.set_title('Average Prediction Error')
        ax4.set_ylabel('Error (m/s)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        plt.show()
    
    def analyze_error_patterns(self, agent_id: str):
        """Identify common error patterns for an agent"""
        worst = self.store.get_worst_experiences(agent_id, limit=20)
        
        if not worst:
            print(f"No experiences found for {agent_id}")
            return
        
        print(f"\n=== Error Pattern Analysis for {agent_id} ===\n")
        
        error_types = {}
        for exp in worst:
            scenario_key = f"mass_ratio_{exp.other_mass/exp.my_mass:.1f}"
            
            if scenario_key not in error_types:
                error_types[scenario_key] = []
            error_types[scenario_key].append(exp.prediction_error)
        
        print("Error by scenario type:")
        for scenario_type, errors in error_types.items():
            avg_error = sum(errors) / len(errors)
            print(f"  {scenario_type}: {avg_error:.3f} avg error ({len(errors)} cases)")
        
        print(f"\nTop 5 worst predictions:")
        for i, exp in enumerate(worst[:5], 1):
            print(f"\n{i}. Error: {exp.prediction_error:.3f}")
            print(f"   Scenario: mass {exp.my_mass}kg @ {exp.my_velocity} vs mass {exp.other_mass}kg @ {exp.other_velocity}")
            print(f"   Predicted: {exp.predicted_my_velocity}")
            print(f"   Actual: {exp.actual_my_velocity}")
            print(f"   Reasoning: {exp.reasoning[:100]}...")
    
    def print_summary_report(self, agent_ids: List[str]):
        """Print comprehensive summary for all agents"""
        print("\n" + "="*60)
        print("LEARNING SYSTEM SUMMARY REPORT")
        print("="*60)
        
        for agent_id in agent_ids:
            stats = self.store.get_agent_statistics(agent_id)
            
            if stats['total_experiences'] == 0:
                print(f"\n{agent_id}: No experiences recorded")
                continue
            
            print(f"\n{agent_id}:")
            print(f"  Total experiences: {stats['total_experiences']}")
            print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
            print(f"  Recent accuracy: {stats['recent_accuracy']:.1%}")
            print(f"  Improvement: {stats['improvement']:+.1%}")
            print(f"  Avg prediction error: {stats['avg_prediction_error']:.3f} m/s")
            
            if stats['improvement'] > 0.05:
                print(f"  ✓ Agent is learning! ({stats['improvement']:.1%} improvement)")
            elif stats['improvement'] < -0.05:
                print(f"  ⚠ Agent is regressing ({stats['improvement']:.1%})")
            else:
                print(f"  → Agent performance stable")
