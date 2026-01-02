import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ExperimentTracker:
    """
    MLflow integration for tracking APE experiments
    
    Tracks:
    - Learning agent performance over time
    - Physics accuracy metrics
    - Model parameters and configurations
    - Visualizations and artifacts
    """
    
    def __init__(
        self,
        experiment_name: str = "APE-Learning",
        tracking_uri: str = "sqlite:///mlflow.db"
    ):
        """
        Initialize MLflow experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: Path to store MLflow data (local) or remote URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.active_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run"""
        self.active_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        return self.active_run
    
    def end_run(self):
        """End the current MLflow run"""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters"""
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_collision_metrics(
        self,
        collision_num: int,
        prediction_error: float,
        momentum_error: float,
        energy_error: float,
        was_correct: bool,
        agent_id: str
    ):
        """
        Log metrics for a single collision
        
        Args:
            collision_num: Collision number (used as step)
            prediction_error: Velocity prediction error (m/s)
            momentum_error: Momentum conservation error
            energy_error: Energy conservation error
            was_correct: Whether prediction was within tolerance
            agent_id: ID of the agent
        """
        metrics = {
            f"{agent_id}_prediction_error": prediction_error,
            f"{agent_id}_momentum_error": momentum_error,
            f"{agent_id}_energy_error": energy_error,
            f"{agent_id}_accuracy": 1.0 if was_correct else 0.0,
        }
        self.log_metrics(metrics, step=collision_num)
    
    def log_learning_stats(
        self,
        step: int,
        agent_id: str,
        stats: Dict[str, Any]
    ):
        """
        Log learning statistics from ExperienceStore
        
        Args:
            step: Current step/collision number
            agent_id: Agent ID
            stats: Statistics dict from get_agent_statistics()
        """
        metrics = {
            f"{agent_id}_total_experiences": stats['total_experiences'],
            f"{agent_id}_overall_accuracy": stats['overall_accuracy'],
            f"{agent_id}_recent_accuracy": stats['recent_accuracy'],
            f"{agent_id}_avg_error": stats['avg_prediction_error'],
            f"{agent_id}_improvement": stats.get('improvement', 0.0),
        }
        self.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_figure(self, figure: plt.Figure, filename: str):
        """
        Log a matplotlib figure as an artifact
        
        Args:
            figure: Matplotlib figure object
            filename: Name to save the figure as
        """
        mlflow.log_figure(figure, filename)
    
    def log_learning_curve(
        self,
        collision_numbers: list,
        accuracies: list,
        errors: list,
        agent_id: str
    ):
        """
        Create and log a learning curve visualization
        
        Args:
            collision_numbers: List of collision numbers
            accuracies: List of accuracy values (0 or 1)
            errors: List of prediction errors
            agent_id: Agent ID
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Rolling accuracy
        window = 5
        rolling_acc = []
        for i in range(len(accuracies)):
            start = max(0, i - window + 1)
            rolling_acc.append(np.mean(accuracies[start:i+1]))
        
        ax1.plot(collision_numbers, rolling_acc, 'b-', linewidth=2)
        ax1.set_xlabel('Collision Number')
        ax1.set_ylabel('Rolling Accuracy (window=5)')
        ax1.set_title(f'Learning Curve - {agent_id}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Prediction error
        ax2.plot(collision_numbers, errors, 'r-', alpha=0.6)
        ax2.scatter(collision_numbers, errors, alpha=0.3, s=20)
        ax2.set_xlabel('Collision Number')
        ax2.set_ylabel('Prediction Error (m/s)')
        ax2.set_title('Prediction Error Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self.log_figure(fig, f"learning_curve_{agent_id}.png")
        plt.close(fig)
    
    def log_model_config(self, config: Dict[str, Any]):
        """Log model configuration as JSON artifact"""
        import json
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            temp_path = f.name
        
        self.log_artifact(temp_path, "config")
        Path(temp_path).unlink()  # Clean up temp file
    
    def log_text(self, text: str, filename: str):
        """Log text content as an artifact"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        self.log_artifact(temp_path)
        Path(temp_path).unlink()


def track_learning_experiment(
    experiment_name: str,
    agent_id: str,
    model_name: str,
    learning_enabled: bool,
    max_examples: int,
    tolerance: float,
    num_collisions: int
) -> ExperimentTracker:
    """
    Convenience function to start a tracked learning experiment
    
    Args:
        experiment_name: Name of the experiment
        agent_id: Agent ID
        model_name: LLM model name
        learning_enabled: Whether learning is enabled
        max_examples: Max examples to retrieve
        tolerance: Prediction tolerance
        num_collisions: Number of collisions to run
    
    Returns:
        ExperimentTracker instance with active run
    """
    tracker = ExperimentTracker(experiment_name)
    
    tracker.start_run(
        run_name=f"{agent_id}_{model_name}",
        tags={
            "agent_id": agent_id,
            "model": model_name,
            "learning_enabled": str(learning_enabled),
        }
    )
    
    tracker.log_params({
        "agent_id": agent_id,
        "model_name": model_name,
        "learning_enabled": learning_enabled,
        "max_examples": max_examples,
        "tolerance": tolerance,
        "num_collisions": num_collisions,
    })
    
    return tracker
