import numpy as np
from typing import Dict, Any, Optional, List
import json

from .learning.experience_store import ExperienceStore, PhysicsExperience
from .learning.feedback import FeedbackGenerator


class ObserverAgent:
    """
    Observer agent that learns physics principles by watching collisions.
    
    Does not participate in collisions - only observes and predicts outcomes.
    Tests whether agents can discover conservation laws from observation alone.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_client,
        model_name: str,
        experience_store: Optional[ExperienceStore] = None,
        feedback_generator: Optional[FeedbackGenerator] = None,
        learning_enabled: bool = True,
        max_examples: int = 5
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.model_name = model_name
        self.experience_store = experience_store
        self.feedback_generator = feedback_generator
        self.learning_enabled = learning_enabled
        self.max_examples = max_examples
        
        self.observations = []
        self.predictions = {}
        self.prediction_errors = []
    
    def predict_collision(
        self,
        ball1_id: str,
        ball2_id: str,
        v1_before: np.ndarray,
        v2_before: np.ndarray,
        mass1: float,
        mass2: float,
        elasticity: float,
        normal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict collision outcome based on observed patterns.
        
        Args:
            ball1_id: ID of first ball
            ball2_id: ID of second ball
            v1_before: Velocity of ball 1 before collision
            v2_before: Velocity of ball 2 before collision
            mass1: Mass of ball 1
            mass2: Mass of ball 2
            elasticity: Coefficient of restitution
            normal: Collision normal vector
        
        Returns:
            dict with 'v1_after' and 'v2_after' predictions
        """
        
        # Retrieve similar past observations if learning enabled
        similar_examples = []
        if self.learning_enabled and self.experience_store and len(self.observations) > 0:
            # Create query scenario for similarity search
            query_scenario = {
                'my_mass': mass1,
                'my_velocity': v1_before.tolist(),
                'my_elasticity': elasticity,
                'other_mass': mass2,
                'other_velocity': v2_before.tolist(),
                'other_elasticity': elasticity,
                'collision_normal': normal.tolist()
            }
            
            # Search for similar past collisions
            similar_experiences = self.experience_store.retrieve_similar_experiences(
                query_scenario,
                limit=self.max_examples,
                agent_id=self.agent_id
            )
            
            if similar_experiences:
                similar_examples = [
                    {
                        'v1_before': exp.my_velocity,
                        'v2_before': exp.other_velocity,
                        'v1_after': exp.actual_my_velocity,
                        'v2_after': exp.actual_other_velocity,
                        'mass1': exp.my_mass,
                        'mass2': exp.other_mass,
                        'elasticity': exp.my_elasticity
                    }
                    for exp in similar_experiences
                ]
        
        # Build prompt for LLM
        prompt = self._build_prediction_prompt(
            v1_before, v2_before, mass1, mass2, elasticity, normal, similar_examples
        )
        
        # Get prediction from LLM
        try:
            if hasattr(self.llm_client, 'chat'):  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a physics observer learning collision dynamics."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
            else:  # Anthropic
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
            
            # Parse prediction
            v1_after, v2_after = self._parse_prediction(response_text)
            
            print(f"[{self.agent_id}] Prediction:")
            if similar_examples:
                print(f"  Used {len(similar_examples)} similar past observations")
            else:
                print(f"  No past observations (first-principles reasoning)")
            print(f"  v1_after: {v1_after}")
            print(f"  v2_after: {v2_after}")
            
        except Exception as e:
            print(f"[{self.agent_id}] Error in prediction: {e}")
            # Fallback: assume velocities don't change
            v1_after = v1_before.copy()
            v2_after = v2_before.copy()
        
        # Store prediction for later evaluation
        collision_id = f"{ball1_id}_{ball2_id}_{len(self.observations)}"
        self.predictions[collision_id] = {
            'v1_after': v1_after,
            'v2_after': v2_after
        }
        
        return {
            'v1_after': v1_after,
            'v2_after': v2_after
        }
    
    def record_observation(
        self,
        ball1_id: str,
        ball2_id: str,
        v1_before: np.ndarray,
        v2_before: np.ndarray,
        v1_after: np.ndarray,
        v2_after: np.ndarray,
        mass1: float,
        mass2: float,
        elasticity: float,
        normal: np.ndarray
    ):
        """
        Record observed collision outcome for learning.
        
        Args:
            ball1_id: ID of first ball
            ball2_id: ID of second ball
            v1_before: Actual velocity of ball 1 before collision
            v2_before: Actual velocity of ball 2 before collision
            v1_after: Actual velocity of ball 1 after collision
            v2_after: Actual velocity of ball 2 after collision
            mass1: Mass of ball 1
            mass2: Mass of ball 2
            elasticity: Coefficient of restitution
            normal: Collision normal vector
        """
        
        collision_id = f"{ball1_id}_{ball2_id}_{len(self.observations)}"
        
        # Calculate prediction error if we made a prediction
        if collision_id in self.predictions:
            pred = self.predictions[collision_id]
            error1 = np.linalg.norm(pred['v1_after'] - v1_after)
            error2 = np.linalg.norm(pred['v2_after'] - v2_after)
            avg_error = (error1 + error2) / 2
            
            self.prediction_errors.append(avg_error)
            
            print(f"[{self.agent_id}] Observation recorded:")
            print(f"  Prediction error: {avg_error:.3f} m/s")
            print(f"  Total observations: {len(self.observations) + 1}")
        
        # Store observation
        observation = {
            'ball1_id': ball1_id,
            'ball2_id': ball2_id,
            'v1_before': v1_before.tolist(),
            'v2_before': v2_before.tolist(),
            'v1_after': v1_after.tolist(),
            'v2_after': v2_after.tolist(),
            'mass1': mass1,
            'mass2': mass2,
            'elasticity': elasticity,
            'normal': normal.tolist()
        }
        self.observations.append(observation)
        
        # Store in experience store if learning enabled
        if self.learning_enabled and self.experience_store:
            import time
            
            # Get prediction if it exists
            pred = self.predictions.get(collision_id, {})
            pred_v1 = pred.get('v1_after', v1_after)
            pred_v2 = pred.get('v2_after', v2_after)
            
            # Create experience using the actual PhysicsExperience dataclass structure
            experience = PhysicsExperience(
                scenario_id=collision_id,
                timestamp=time.time(),
                agent_id=self.agent_id,
                my_velocity=v1_before.tolist(),
                my_mass=mass1,
                my_elasticity=elasticity,
                other_velocity=v2_before.tolist(),
                other_mass=mass2,
                other_elasticity=elasticity,
                collision_normal=normal.tolist(),
                predicted_my_velocity=pred_v1.tolist() if isinstance(pred_v1, np.ndarray) else pred_v1,
                predicted_other_velocity=pred_v2.tolist() if isinstance(pred_v2, np.ndarray) else pred_v2,
                reasoning="Observer prediction",
                actual_my_velocity=v1_after.tolist(),
                actual_other_velocity=v2_after.tolist(),
                prediction_error=avg_error,
                momentum_error=0.0,  # Not calculated for observer
                energy_error=0.0,    # Not calculated for observer
                was_correct=avg_error < 0.1,
                similar_experiences_used=0,  # Track this separately if needed
                model_used=self.model_name
            )
            
            self.experience_store.store_experience(experience)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about observer's learning progress."""
        
        if not self.prediction_errors:
            return {
                'total_observations': len(self.observations),
                'total_predictions': 0,
                'avg_error': 0.0,
                'recent_error': 0.0,
                'improvement': 0.0
            }
        
        # Calculate early vs late performance
        n = len(self.prediction_errors)
        early_errors = self.prediction_errors[:max(1, n//3)]
        late_errors = self.prediction_errors[-max(1, n//3):]
        
        early_avg = np.mean(early_errors) if early_errors else 0.0
        late_avg = np.mean(late_errors) if late_errors else 0.0
        improvement = early_avg - late_avg
        
        return {
            'total_observations': len(self.observations),
            'total_predictions': len(self.prediction_errors),
            'avg_error': np.mean(self.prediction_errors),
            'recent_error': late_avg,
            'early_error': early_avg,
            'improvement': improvement,
            'improvement_percent': (improvement / early_avg * 100) if early_avg > 0 else 0.0
        }
    
    def _create_collision_query(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        mass1: float,
        mass2: float,
        elasticity: float,
        normal: np.ndarray
    ) -> str:
        """Create text query for similarity search."""
        
        return (
            f"Collision: mass1={mass1:.2f}kg mass2={mass2:.2f}kg elasticity={elasticity:.2f} "
            f"v1=[{v1[0]:.2f},{v1[1]:.2f}] v2=[{v2[0]:.2f},{v2[1]:.2f}] "
            f"normal=[{normal[0]:.2f},{normal[1]:.2f}]"
        )
    
    def _build_prediction_prompt(
        self,
        v1_before: np.ndarray,
        v2_before: np.ndarray,
        mass1: float,
        mass2: float,
        elasticity: float,
        normal: np.ndarray,
        similar_examples: List[Dict]
    ) -> str:
        """Build prompt for LLM to predict collision outcome."""
        
        prompt = f"""You are observing a collision between two balls. Predict the velocities after collision.

COLLISION PARAMETERS:
- Ball 1: mass={mass1:.2f}kg, velocity_before=[{v1_before[0]:.3f}, {v1_before[1]:.3f}] m/s
- Ball 2: mass={mass2:.2f}kg, velocity_before=[{v2_before[0]:.3f}, {v2_before[1]:.3f}] m/s
- Elasticity: {elasticity:.2f}
- Collision normal: [{normal[0]:.3f}, {normal[1]:.3f}]
"""
        
        if similar_examples:
            prompt += f"\nPAST OBSERVATIONS ({len(similar_examples)} similar collisions):\n"
            for i, ex in enumerate(similar_examples, 1):
                prompt += f"\n{i}. Before: v1={ex['v1_before']}, v2={ex['v2_before']}"
                prompt += f"\n   After:  v1={ex['v1_after']}, v2={ex['v2_after']}"
                prompt += f"\n   (mass1={ex['mass1']}, mass2={ex['mass2']}, e={ex['elasticity']})"
        else:
            prompt += "\nNo past observations yet. Use physics principles to predict."
        
        prompt += """

PHYSICS PRINCIPLES:
1. Momentum conservation: m1*v1_before + m2*v2_before = m1*v1_after + m2*v2_after
2. Energy conservation (elastic): 0.5*m1*|v1|² + 0.5*m2*|v2|² = constant
3. Decompose velocities into normal (along collision) and tangential (perpendicular)
4. Normal components exchange based on masses, tangential components unchanged

EXAMPLE CALCULATION (1D head-on):
- Ball 1: m=1kg at v=2 m/s hits Ball 2: m=1kg at v=0 m/s
- Equal masses → velocities swap → v1_after=0 m/s, v2_after=2 m/s
- Check momentum: (1)(2) + (1)(0) = (1)(0) + (1)(2) = 2 kg⋅m/s ✓

TASK: Predict the velocities after this collision.

Respond with ONLY the predicted velocities in this format:
v1_after: [x, y]
v2_after: [x, y]
"""
        
        return prompt
    
    def _parse_prediction(self, response_text: str) -> tuple:
        """Parse LLM response to extract velocity predictions."""
        
        try:
            # Look for v1_after and v2_after in response
            lines = response_text.strip().split('\n')
            v1_after = None
            v2_after = None
            
            for line in lines:
                if 'v1_after' in line.lower():
                    # Extract [x, y] pattern
                    import re
                    match = re.search(r'\[([-\d.]+),\s*([-\d.]+)\]', line)
                    if match:
                        v1_after = np.array([float(match.group(1)), float(match.group(2))])
                
                if 'v2_after' in line.lower():
                    import re
                    match = re.search(r'\[([-\d.]+),\s*([-\d.]+)\]', line)
                    if match:
                        v2_after = np.array([float(match.group(1)), float(match.group(2))])
            
            if v1_after is not None and v2_after is not None:
                return v1_after, v2_after
            
            # Fallback: return zeros
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
            
        except Exception as e:
            print(f"[{self.agent_id}] Parse error: {e}")
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    def discover_principles(self) -> Dict[str, Any]:
        """
        After observing many collisions, ask agent what physics principles it learned.
        This tests meta-learning: can the agent articulate discovered patterns?
        
        Returns:
            dict with raw response and parsed findings
        """
        
        if len(self.observations) < 5:
            return {
                'raw_response': "Insufficient observations to discover principles (need at least 5)",
                'mentions_momentum': False,
                'mentions_energy': False,
                'mentions_conservation': False,
                'provides_formula': False,
                'prediction_stats': {}
            }
        
        # Sample diverse observations for analysis
        sample_size = min(10, len(self.observations))
        step = max(1, len(self.observations) // sample_size)
        sampled_obs = [self.observations[i] for i in range(0, len(self.observations), step)][:sample_size]
        
        # Calculate prediction statistics
        avg_error = np.mean(self.prediction_errors) if self.prediction_errors else 0
        early_errors = self.prediction_errors[:len(self.prediction_errors)//3] if self.prediction_errors else []
        late_errors = self.prediction_errors[-len(self.prediction_errors)//3:] if self.prediction_errors else []
        early_avg = np.mean(early_errors) if early_errors else 0
        late_avg = np.mean(late_errors) if late_errors else 0
        
        # Build prompt
        prompt = f"""You have observed {len(self.observations)} collisions between balls.

    YOUR PREDICTION PERFORMANCE:
    - Total predictions made: {len(self.prediction_errors)}
    - Average prediction error: {avg_error:.3f} m/s
    - Early predictions (first third): {early_avg:.3f} m/s
    - Recent predictions (last third): {late_avg:.3f} m/s
    - Improvement: {early_avg - late_avg:.3f} m/s

    Here are {len(sampled_obs)} representative examples:
    """
        
        for i, obs in enumerate(sampled_obs, 1):
            prompt += f"""
    {i}. Collision:
    Before: Ball 1 (m={obs['mass1']}kg) at v={obs['v1_before']} m/s
            Ball 2 (m={obs['mass2']}kg) at v={obs['v2_before']} m/s
    After:  Ball 1 at v={obs['v1_after']} m/s
            Ball 2 at v={obs['v2_after']} m/s
    Elasticity: {obs['elasticity']}
    """
        
        prompt += """

    Based on these observations, answer these questions:

    1. What determines the final velocities after a collision?
    2. Is there a relationship between mass and velocity change?
    3. What quantities are conserved in collisions?
    - Is momentum (mass × velocity) conserved? Verify with examples.
    - Is kinetic energy (½m*v²) conserved? Verify with examples.
    4. Can you state a general formula for predicting collision outcomes?
    5. How do collision angle and initial velocities affect the result?

    PHYSICS REFERENCE (for comparison):
    Classical physics predicts:
    - Momentum conservation: m1*v1_before + m2*v2_before = m1*v1_after + m2*v2_after
    - Energy conservation (elastic): ½m1*v1²_before + ½m2*v2²_before = ½m1*v1²_after + ½m2*v2²_after

    For each principle:
    1. State it clearly
    2. Show example calculation from data above
    3. Explain when/why it applies

    Did you discover these classical principles from observation alone?
    Explain the physics principles you learned and how confident you are in them.
    """
        
        try:
            if hasattr(self.llm_client, 'chat'):  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a physics observer analyzing collision patterns to discover fundamental principles."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                principles = response.choices[0].message.content
            else:  # Anthropic
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                principles = response.content[0].text
            
            # Parse key findings
            raw_lower = principles.lower()
            
            return {
                'raw_response': principles,
                'mentions_momentum': 'momentum' in raw_lower,
                'mentions_energy': 'energy' in raw_lower or 'kinetic' in raw_lower,
                'mentions_conservation': 'conserve' in raw_lower or 'constant' in raw_lower,
                'provides_formula': ('m1*v1' in principles or 'mv' in principles or 
                                'm₁v₁' in principles or 'Σmv' in principles),
                'discovers_classical_physics': ('momentum' in raw_lower and 'conserve' in raw_lower),
                'prediction_stats': {
                    'total_observations': len(self.observations),
                    'total_predictions': len(self.prediction_errors),
                    'avg_error': avg_error,
                    'early_error': early_avg,
                    'late_error': late_avg,
                    'improvement': early_avg - late_avg
                }
            }
            
        except Exception as e:
            return {
                'raw_response': f"Error discovering principles: {e}",
                'mentions_momentum': False,
                'mentions_energy': False,
                'mentions_conservation': False,
                'provides_formula': False,
                'prediction_stats': {}
            }
