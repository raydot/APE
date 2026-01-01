import numpy as np
import json
from typing import Optional

from .agents import Agent
from .events import PhysicsEvent, EventType, SimpleEventBus
from .physics import WorldState, PhysicsState
from .tools import ToolRegistry, calculate_elastic_collision
from .validator import PhysicsValidator, ValidationResult


class HybridBallAgent(Agent):
    """
    Hybrid agent that combines LLM reasoning with tool-based calculations.
    
    Strategy:
    1. LLM reasons about the physics situation
    2. Tool performs accurate calculation
    3. Validator checks if LLM reasoning was correct
    4. Use tool result (always accurate) but log LLM performance
    """
    
    def __init__(
        self,
        agent_id: str,
        event_bus: SimpleEventBus,
        world_state: WorldState,
        llm_client,
        model_name: str,
        tool_registry: ToolRegistry,
        validator: Optional[PhysicsValidator] = None,
        trust_llm_threshold: float = 0.05
    ):
        super().__init__(agent_id, event_bus, world_state)
        self.llm_client = llm_client
        self.model_name = model_name
        self.tool_registry = tool_registry
        self.validator = validator or PhysicsValidator(tolerance=trust_llm_threshold)
        self.trust_llm_threshold = trust_llm_threshold
        
        # Statistics
        self.llm_correct_count = 0
        self.llm_incorrect_count = 0
        self.tool_used_count = 0
    
    def handle_event(self, event: PhysicsEvent):
        if event.event_type == EventType.COLLISION:
            print(f"[{self.agent_id}] Handling collision with hybrid approach...")
            self._handle_collision_hybrid(event)
    
    def _handle_collision_hybrid(self, event: PhysicsEvent):
        state = self.get_my_state()
        if not state:
            return
        
        normal = np.array(event.data.get('normal', [0, 1]))
        velocity_before = state.velocity.copy()
        
        # Step 1: Get LLM reasoning
        print(f"[{self.agent_id}] Step 1: LLM reasoning...")
        llm_result = self._get_llm_reasoning(state, normal)
        
        # Step 2: Get tool calculation (ground truth)
        print(f"[{self.agent_id}] Step 2: Tool calculation...")
        tool_result = self._get_tool_calculation(state.velocity, normal, state.elasticity)
        
        # Step 3: Compare and validate
        print(f"[{self.agent_id}] Step 3: Validation...")
        llm_velocity = np.array(llm_result['new_velocity'])
        tool_velocity = np.array(tool_result['new_velocity'])
        
        # Calculate difference
        velocity_diff = np.linalg.norm(llm_velocity - tool_velocity)
        relative_error = velocity_diff / np.linalg.norm(tool_velocity) if np.linalg.norm(tool_velocity) > 0 else 0
        
        # Validate LLM result
        validation_results = self.validator.validate_collision(
            mass=state.mass,
            velocity_before=velocity_before,
            velocity_after=llm_velocity,
            position_before=state.position,
            position_after=state.position,
            elasticity=state.elasticity,
            surface_normal=normal
        )
        
        llm_passed = all(r.passed for r in validation_results)
        
        if llm_passed and relative_error < self.trust_llm_threshold:
            self.llm_correct_count += 1
            print(f"[{self.agent_id}] ✓ LLM reasoning CORRECT (error: {relative_error*100:.2f}%)")
            print(f"[{self.agent_id}]   LLM: {llm_result['reasoning']}")
            chosen_velocity = llm_velocity
            source = "LLM"
        else:
            self.llm_incorrect_count += 1
            self.tool_used_count += 1
            print(f"[{self.agent_id}] ✗ LLM reasoning INCORRECT (error: {relative_error*100:.2f}%)")
            print(f"[{self.agent_id}]   LLM said: {llm_velocity}")
            print(f"[{self.agent_id}]   Tool says: {tool_velocity}")
            print(f"[{self.agent_id}]   Using tool result (accurate)")
            
            # Show which validations failed
            for result in validation_results:
                if not result.passed:
                    print(f"[{self.agent_id}]   Failed: {result.law} - {result.message}")
            
            chosen_velocity = tool_velocity
            source = "Tool"
        
        # Step 4: Apply the chosen velocity
        self.update_velocity(chosen_velocity)
        print(f"[{self.agent_id}] Applied velocity from {source}: {chosen_velocity}")
    
    def _get_llm_reasoning(self, state: PhysicsState, normal: np.ndarray) -> dict:
        """Get LLM's reasoning about the collision"""
        prompt = f"""You are a ball in a physics simulation. You just collided with a surface.

Your current state:
- Position: {state.position.tolist()} m
- Velocity: {state.velocity.tolist()} m/s
- Mass: {state.mass} kg
- Elasticity: {state.elasticity}

Collision information:
- Surface normal: {normal.tolist()}

Calculate your new velocity after elastic collision.
The perpendicular component reverses and scales by elasticity.
The parallel component is unchanged (no friction).

Respond ONLY with JSON:
{{
    "reasoning": "brief physics explanation",
    "new_velocity": [vx, vy]
}}"""
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            return result
        except Exception as e:
            print(f"[{self.agent_id}] LLM error: {e}")
            # Fallback to tool if LLM fails
            return self._get_tool_calculation(state.velocity, normal, state.elasticity)
    
    def _get_tool_calculation(self, velocity: np.ndarray, normal: np.ndarray, elasticity: float) -> dict:
        """Get accurate calculation from tool"""
        return calculate_elastic_collision(
            velocity=velocity.tolist(),
            surface_normal=normal.tolist(),
            elasticity=elasticity
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (same as BallAgent)"""
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        else:
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.content[0].text
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response"""
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        return json.loads(response)
    
    def get_stats(self) -> dict:
        """Get statistics about LLM vs Tool usage"""
        total = self.llm_correct_count + self.llm_incorrect_count
        accuracy = self.llm_correct_count / total if total > 0 else 0
        
        return {
            'llm_correct': self.llm_correct_count,
            'llm_incorrect': self.llm_incorrect_count,
            'tool_used': self.tool_used_count,
            'total_collisions': total,
            'llm_accuracy': accuracy,
            'tool_usage_rate': self.tool_used_count / total if total > 0 else 0
        }
    
    def print_stats(self):
        """Print human-readable statistics"""
        stats = self.get_stats()
        
        print(f"\n[{self.agent_id}] Hybrid Agent Statistics:")
        print(f"  Total collisions: {stats['total_collisions']}")
        print(f"  LLM correct: {stats['llm_correct']} ({stats['llm_accuracy']*100:.1f}%)")
        print(f"  LLM incorrect: {stats['llm_incorrect']}")
        print(f"  Tool fallback used: {stats['tool_used']} ({stats['tool_usage_rate']*100:.1f}%)")
        
        if stats['llm_accuracy'] > 0.9:
            print(f"  → LLM is highly accurate, could reduce tool usage")
        elif stats['llm_accuracy'] < 0.5:
            print(f"  → LLM struggling, consider better prompts or model")
        else:
            print(f"  → Hybrid approach providing good balance")
