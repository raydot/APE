import numpy as np
from typing import Dict, Any, List
import json

from ..negotiating_agent import NegotiatingBallAgent
from .experience_store import ExperienceStore, PhysicsExperience
from .feedback import FeedbackGenerator
from ..negotiation import VelocityProposal


class LearningBallAgent(NegotiatingBallAgent):
    """
    Ball agent that learns from experience
    
    Uses vector DB to retrieve similar past scenarios and incorporates
    learnings into reasoning
    """
    
    def __init__(
        self,
        agent_id: str,
        event_bus,
        world_state,
        llm_client,
        model_name: str,
        experience_store: ExperienceStore,
        feedback_generator: FeedbackGenerator,
        learning_enabled: bool = True,
        max_examples: int = 3
    ):
        super().__init__(agent_id, event_bus, world_state, llm_client, model_name)
        
        self.experience_store = experience_store
        self.feedback_generator = feedback_generator
        self.learning_enabled = learning_enabled
        self.max_examples = max_examples
        
        self.pending_predictions = {}
    
    def propose_velocity(
        self,
        collision_data: Dict,
        other_agent_id: str
    ) -> VelocityProposal:
        """
        Propose own velocity after ball-to-ball collision with learning
        """
        my_state = self.get_my_state()
        other_state = self.world_state.get_object(other_agent_id)
        
        if not my_state or not other_state:
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=my_state.velocity if my_state else np.array([0.0, 0.0]),
                reasoning="Error: couldn't get states",
                confidence=0.0
            )
        
        if self.agent_id == collision_data['ball1_id']:
            normal = collision_data['collision_normal']
        else:
            normal = -collision_data['collision_normal']
        
        scenario = {
            'my_velocity': my_state.velocity.tolist(),
            'my_mass': my_state.mass,
            'my_elasticity': my_state.elasticity,
            'other_velocity': other_state.velocity.tolist(),
            'other_mass': other_state.mass,
            'other_elasticity': other_state.elasticity,
            'collision_normal': normal.tolist()
        }
        
        similar_experiences = []
        if self.learning_enabled:
            similar_experiences = self.experience_store.retrieve_similar_experiences(
                query_scenario=scenario,
                limit=self.max_examples,
                min_score=0.7,
                agent_id=self.agent_id
            )
        
        prompt = self._build_prompt_with_learning(
            scenario,
            similar_experiences,
            other_agent_id
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            
            collision_id = f"{self.agent_id}_{other_agent_id}_{self.world_state.time}"
            self.pending_predictions[collision_id] = {
                'scenario': scenario,
                'predicted_my_velocity': np.array(result['your_velocity_after']),
                'reasoning': result['reasoning'],
                'similar_experiences_used': len(similar_experiences)
            }
            
            print(f"[{self.agent_id}] Learning-enhanced proposal:")
            print(f"  Used {len(similar_experiences)} similar past experiences")
            print(f"  Reasoning: {result['reasoning'][:150]}...")
            
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=np.array(result['your_velocity_after']),
                reasoning=result['reasoning'],
                confidence=result.get('confidence', 1.0)
            )
            
        except Exception as e:
            print(f"[{self.agent_id}] Error proposing velocity: {e}")
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=my_state.velocity,
                reasoning=f"Error: {e}",
                confidence=0.0
            )
    
    def _build_prompt_with_learning(
        self,
        scenario: Dict[str, Any],
        similar_experiences: List[PhysicsExperience],
        other_agent_id: str
    ) -> str:
        """Build LLM prompt that includes past learning examples"""
        base_prompt = f"""You are {self.agent_id}, a ball in a physics simulation.

You just collided with {other_agent_id}.

Your state BEFORE collision:
- Mass: {scenario['my_mass']} kg
- Velocity: {scenario['my_velocity']} m/s
- Elasticity: {scenario['my_elasticity']}

Other ball's state BEFORE collision:
- Mass: {scenario['other_mass']} kg
- Velocity: {scenario['other_velocity']} m/s
- Elasticity: {scenario['other_elasticity']}

Collision information:
- Normal vector: {scenario['collision_normal']} (points away from you toward other ball)
"""
        
        if similar_experiences:
            base_prompt += f"""
--- LEARNING FROM PAST EXPERIENCES ---

You have encountered {len(similar_experiences)} similar scenarios before.
Learn from these past outcomes to improve your prediction:

"""
            for i, exp in enumerate(similar_experiences, 1):
                base_prompt += f"""
Example {i}:
Scenario:
- Your velocity: {exp.my_velocity}, mass: {exp.my_mass}
- Other velocity: {exp.other_velocity}, mass: {exp.other_mass}
- Normal: {exp.collision_normal}, elasticity: {exp.my_elasticity}

Your previous prediction:
- Your velocity after: {exp.predicted_my_velocity}
- Reasoning: "{exp.reasoning[:100]}..."

Actual outcome:
- Your velocity after: {exp.actual_my_velocity}
- Prediction error: {exp.prediction_error:.3f} m/s
- Was correct: {exp.was_correct}

"""
                if not exp.was_correct:
                    feedback_data = self.feedback_generator.generate_feedback(
                        np.array(exp.predicted_my_velocity),
                        np.array(exp.actual_my_velocity),
                        {
                            'my_velocity': exp.my_velocity,
                            'my_mass': exp.my_mass,
                            'my_elasticity': exp.my_elasticity,
                            'other_velocity': exp.other_velocity,
                            'other_mass': exp.other_mass,
                            'other_elasticity': exp.other_elasticity,
                            'collision_normal': exp.collision_normal
                        },
                        exp.reasoning
                    )
                    base_prompt += f"""
Lesson learned:
{feedback_data['guidance']}
---
"""
        
        base_prompt += """
Physics principles for 2-ball elastic collision:
1. Decompose velocities into components parallel and perpendicular to collision normal
2. Perpendicular components: exchange momentum based on masses and elasticity
3. Parallel components: unchanged (no friction)
4. Use elasticity = min(your_e, their_e)

For 1D elastic collision along normal:
v1' = ((m1 - e*m2)*v1 + (1+e)*m2*v2) / (m1+m2)

Calculate YOUR velocity after the collision.
Use the lessons from past experiences to avoid previous mistakes.

Respond ONLY with JSON:
{
    "reasoning": "step-by-step calculation of YOUR velocity, referencing lessons learned if applicable",
    "your_velocity_after": [vx, vy],
    "confidence": 0.0-1.0
}"""
        
        return base_prompt
    
    def record_outcome(
        self,
        collision_id: str,
        actual_my_velocity: np.ndarray,
        actual_other_velocity: np.ndarray
    ):
        """
        Record the actual outcome and store as experience
        
        Call this after collision is resolved to provide feedback
        """
        if collision_id not in self.pending_predictions:
            print(f"[{self.agent_id}] Warning: No pending prediction for {collision_id}")
            return
        
        prediction = self.pending_predictions[collision_id]
        
        experience = self.feedback_generator.create_experience_from_collision(
            agent_id=self.agent_id,
            timestamp=self.world_state.time,
            scenario=prediction['scenario'],
            predicted_my_velocity=prediction['predicted_my_velocity'],
            predicted_other_velocity=actual_other_velocity,
            reasoning=prediction['reasoning'],
            actual_my_velocity=actual_my_velocity,
            actual_other_velocity=actual_other_velocity,
            model_used=self.model_name,
            similar_experiences_used=prediction['similar_experiences_used']
        )
        
        exp_id = self.experience_store.store_experience(experience)
        
        print(f"[{self.agent_id}] Stored experience {exp_id[:8]}...")
        print(f"  Prediction error: {experience.prediction_error:.3f} m/s")
        print(f"  Was correct: {experience.was_correct}")
        
        del self.pending_predictions[collision_id]
        
        return experience
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's learning progress"""
        return self.experience_store.get_agent_statistics(self.agent_id)
