import numpy as np
from typing import Dict, Any
from .experience_store import PhysicsExperience


class FeedbackGenerator:
    """
    Generates structured feedback from prediction errors
    
    Analyzes what went wrong and provides corrective guidance
    """
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
    
    def generate_feedback(
        self,
        predicted_velocity: np.ndarray,
        actual_velocity: np.ndarray,
        scenario: Dict[str, Any],
        reasoning: str
    ) -> Dict[str, Any]:
        """
        Analyze prediction error and generate feedback
        
        Returns detailed feedback about what went wrong
        """
        error_vector = actual_velocity - predicted_velocity
        error_magnitude = np.linalg.norm(error_vector)
        
        velocity_magnitude = np.linalg.norm(actual_velocity)
        relative_error = error_magnitude / (velocity_magnitude + 1e-6)
        
        was_correct = relative_error < self.tolerance
        
        error_analysis = self._analyze_error_type(
            predicted_velocity,
            actual_velocity,
            scenario
        )
        
        guidance = self._generate_guidance(
            error_analysis,
            scenario,
            reasoning
        )
        
        return {
            'error_magnitude': float(error_magnitude),
            'relative_error': float(relative_error),
            'was_correct': was_correct,
            'error_vector': error_vector.tolist(),
            'error_analysis': error_analysis,
            'guidance': guidance
        }
    
    def _analyze_error_type(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify what type of error occurred
        
        Common error patterns:
        - Magnitude error (speed wrong but direction ok)
        - Direction error (angle wrong)
        - Sign error (reversed direction)
        - Elasticity error (didn't apply coefficient correctly)
        - Component error (normal vs tangential)
        """
        pred_dir = predicted / (np.linalg.norm(predicted) + 1e-6)
        actual_dir = actual / (np.linalg.norm(actual) + 1e-6)
        
        direction_similarity = np.dot(pred_dir, actual_dir)
        
        pred_speed = np.linalg.norm(predicted)
        actual_speed = np.linalg.norm(actual)
        speed_ratio = pred_speed / (actual_speed + 1e-6)
        
        normal = np.array(scenario.get('collision_normal', [0, 1]))
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        pred_normal = np.dot(predicted, normal)
        actual_normal = np.dot(actual, normal)
        
        analysis = {
            'direction_similarity': float(direction_similarity),
            'speed_ratio': float(speed_ratio),
            'predicted_speed': float(pred_speed),
            'actual_speed': float(actual_speed),
        }
        
        if direction_similarity < 0:
            analysis['error_type'] = 'sign_error'
            analysis['description'] = 'Predicted velocity in completely wrong direction'
        elif abs(speed_ratio - 1.0) > 0.2 and direction_similarity > 0.9:
            analysis['error_type'] = 'magnitude_error'
            analysis['description'] = 'Direction correct but speed wrong'
        elif direction_similarity < 0.9:
            analysis['error_type'] = 'direction_error'
            analysis['description'] = 'Angle of velocity incorrect'
        elif abs(actual_normal) > 1e-6 and abs(pred_normal / actual_normal - scenario.get('my_elasticity', 1.0)) > 0.1:
            analysis['error_type'] = 'elasticity_error'
            analysis['description'] = 'Elasticity coefficient not applied correctly'
        else:
            analysis['error_type'] = 'small_numerical_error'
            analysis['description'] = 'Minor calculation error'
        
        return analysis
    
    def _generate_guidance(
        self,
        error_analysis: Dict[str, Any],
        scenario: Dict[str, Any],
        reasoning: str
    ) -> str:
        """Generate specific guidance based on error type"""
        error_type = error_analysis['error_type']
        
        guidance_map = {
            'sign_error': f"""
Your velocity prediction was in the completely wrong direction.

Common cause: Forgetting to reverse the normal component during collision.
Remember: For elastic collision, the component perpendicular to the surface
reverses direction and scales by elasticity coefficient.

Your reasoning was: "{reasoning[:200]}..."

Fix: Double-check the sign of the normal component after collision.
""",
            'magnitude_error': f"""
Your direction was correct, but the speed was wrong.

You predicted speed: {error_analysis['predicted_speed']:.2f} m/s
Actual speed: {error_analysis['actual_speed']:.2f} m/s
Ratio: {error_analysis['speed_ratio']:.2f}

Common cause: Incorrectly applying elasticity coefficient.
The normal component should be multiplied by the coefficient of restitution.

Elasticity in this scenario: {scenario.get('my_elasticity', 1.0)}

Your reasoning was: "{reasoning[:200]}..."

Fix: v_normal_after = -elasticity * v_normal_before
""",
            'direction_error': f"""
The angle of your predicted velocity was incorrect.

Direction similarity: {error_analysis['direction_similarity']:.2f} (should be ~1.0)

Common cause: Incorrect vector decomposition into normal/tangential components.
Remember:
- Normal component (perpendicular to surface): reverses and scales
- Tangential component (parallel to surface): unchanged

Collision normal in this scenario: {scenario.get('collision_normal')}

Your reasoning was: "{reasoning[:200]}..."

Fix: Carefully decompose velocity, transform normal component, recombine.
""",
            'elasticity_error': f"""
You didn't apply the elasticity coefficient correctly.

Elasticity coefficient: {scenario.get('my_elasticity', 1.0)}

Remember: The normal component of velocity should be:
v_normal_after = -elasticity * v_normal_before

Your reasoning was: "{reasoning[:200]}..."

Fix: Make sure to multiply by the coefficient of restitution.
""",
            'small_numerical_error': f"""
Your calculation was very close! Just a small numerical error.

This is likely due to rounding or minor calculation differences.
Your physics understanding appears correct.

Keep up the good work!
"""
        }
        
        return guidance_map.get(error_type, "Unknown error type. Review your calculation carefully.")
    
    def create_experience_from_collision(
        self,
        agent_id: str,
        timestamp: float,
        scenario: Dict[str, Any],
        predicted_my_velocity: np.ndarray,
        predicted_other_velocity: np.ndarray,
        reasoning: str,
        actual_my_velocity: np.ndarray,
        actual_other_velocity: np.ndarray,
        model_used: str,
        similar_experiences_used: int = 0
    ) -> PhysicsExperience:
        """
        Create a PhysicsExperience object from collision data
        
        Convenience method to package everything together
        """
        feedback = self.generate_feedback(
            predicted_my_velocity,
            actual_my_velocity,
            scenario,
            reasoning
        )
        
        my_mass = scenario['my_mass']
        other_mass = scenario['other_mass']
        
        momentum_before = (
            my_mass * np.array(scenario['my_velocity']) +
            other_mass * np.array(scenario['other_velocity'])
        )
        
        momentum_predicted = (
            my_mass * predicted_my_velocity +
            other_mass * predicted_other_velocity
        )
        
        momentum_actual = (
            my_mass * actual_my_velocity +
            other_mass * actual_other_velocity
        )
        
        momentum_error = np.linalg.norm(momentum_predicted - momentum_actual)
        
        energy_before = (
            0.5 * my_mass * np.dot(scenario['my_velocity'], scenario['my_velocity']) +
            0.5 * other_mass * np.dot(scenario['other_velocity'], scenario['other_velocity'])
        )
        
        energy_predicted = (
            0.5 * my_mass * np.dot(predicted_my_velocity, predicted_my_velocity) +
            0.5 * other_mass * np.dot(predicted_other_velocity, predicted_other_velocity)
        )
        
        energy_actual = (
            0.5 * my_mass * np.dot(actual_my_velocity, actual_my_velocity) +
            0.5 * other_mass * np.dot(actual_other_velocity, actual_other_velocity)
        )
        
        energy_error = abs(energy_predicted - energy_actual) / (energy_actual + 1e-6)
        
        return PhysicsExperience(
            scenario_id=f"{agent_id}_{int(timestamp*1000)}",
            timestamp=timestamp,
            agent_id=agent_id,
            my_velocity=scenario['my_velocity'],
            my_mass=my_mass,
            my_elasticity=scenario['my_elasticity'],
            other_velocity=scenario['other_velocity'],
            other_mass=other_mass,
            other_elasticity=scenario['other_elasticity'],
            collision_normal=scenario['collision_normal'],
            predicted_my_velocity=predicted_my_velocity.tolist(),
            predicted_other_velocity=predicted_other_velocity.tolist(),
            reasoning=reasoning,
            actual_my_velocity=actual_my_velocity.tolist(),
            actual_other_velocity=actual_other_velocity.tolist(),
            prediction_error=float(feedback['error_magnitude']),
            momentum_error=float(momentum_error),
            energy_error=float(energy_error),
            was_correct=feedback['was_correct'],
            similar_experiences_used=similar_experiences_used,
            model_used=model_used
        )
