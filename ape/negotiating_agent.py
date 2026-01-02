import numpy as np
import json
from typing import Dict

from .agents import Agent
from .events import PhysicsEvent, EventType, SimpleEventBus
from .physics import WorldState
from .negotiation import VelocityProposal


class NegotiatingBallAgent(Agent):
    """
    Ball agent that proposes its own velocity after ball-to-ball collision.
    
    Simplified approach: Each agent only reasons about and proposes their own
    velocity. The resolver checks if both proposals are physically consistent.
    """
    
    def __init__(
        self,
        agent_id: str,
        event_bus: SimpleEventBus,
        world_state: WorldState,
        llm_client,
        model_name: str = "gpt-4o-mini"
    ):
        super().__init__(agent_id, event_bus, world_state)
        self.llm_client = llm_client
        self.model_name = model_name
    
    def handle_event(self, event: PhysicsEvent):
        """Handle events (ball-floor collisions still work as before)"""
        if event.event_type == EventType.COLLISION:
            # Ball-floor collision (existing behavior)
            self._handle_floor_collision(event)
    
    def _handle_floor_collision(self, event: PhysicsEvent):
        """Handle collision with floor (existing logic)"""
        state = self.get_my_state()
        if not state:
            return
        
        normal = np.array(event.data.get('normal', [0, 1]))
        
        print(f"[{self.agent_id}] Handling floor collision...")
        
        prompt = f"""You are a ball in a physics simulation. You just collided with the floor.

Your current state:
- Position: {state.position.tolist()} m
- Velocity: {state.velocity.tolist()} m/s
- Mass: {state.mass} kg
- Elasticity: {state.elasticity}

Surface information:
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
            
            print(f"[{self.agent_id}] LLM response:")
            print(f"    Reasoning: {result['reasoning']}")
            print(f"    New velocity: {result['new_velocity']}")
            
            new_velocity = np.array(result['new_velocity'])
            self.update_velocity(new_velocity)
            
        except Exception as e:
            print(f"[{self.agent_id}] Error: {e}")
    
    def propose_velocity(
        self,
        collision_data: Dict,
        other_agent_id: str
    ) -> VelocityProposal:
        """
        Propose own velocity after ball-to-ball collision.
        
        Args:
            collision_data: Collision info (normal, distance, etc.)
            other_agent_id: ID of the other ball
        
        Returns:
            VelocityProposal with own velocity only
        """
        my_state = self.get_my_state()
        other_state = self.world_state.get_object(other_agent_id)
        
        if not my_state or not other_state:
            # Fallback: don't change velocity
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=my_state.velocity if my_state else np.array([0.0, 0.0]),
                reasoning="Error: couldn't get states",
                confidence=0.0
            )
        
        # Determine collision normal from my perspective
        # Normal points from ball1 to ball2
        if self.agent_id == collision_data['ball1_id']:
            normal = collision_data['collision_normal']
        else:
            normal = -collision_data['collision_normal']
        
        # Build prompt (agent only calculates own velocity)
        prompt = f"""You are {self.agent_id}, a ball in a physics simulation.

You just collided with {other_agent_id}.

Your state BEFORE collision:
- Mass: {my_state.mass} kg
- Velocity: {my_state.velocity.tolist()} m/s
- Elasticity: {my_state.elasticity}

Other ball's state BEFORE collision:
- Mass: {other_state.mass} kg
- Velocity: {other_state.velocity.tolist()} m/s
- Elasticity: {other_state.elasticity}

Collision information:
- Normal vector: {normal.tolist()} (points away from you toward other ball)

Physics principles for 2-ball elastic collision:
1. Decompose velocities into components parallel and perpendicular to collision normal
2. Perpendicular components: exchange momentum based on masses and elasticity
3. Parallel components: unchanged (no friction)
4. Use elasticity = min(your_e, their_e) = {min(my_state.elasticity, other_state.elasticity)}

For 1D elastic collision along normal:
v1' = ((m1 - e*m2)*v1 + (1+e)*m2*v2) / (m1+m2)

Calculate YOUR velocity after the collision.

Respond ONLY with JSON:
{{
    "reasoning": "step-by-step calculation of YOUR velocity",
    "your_velocity_after": [vx, vy],
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            
            print(f"[{self.agent_id}] Proposal reasoning:")
            print(f"  {result['reasoning']}")
            
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=np.array(result['your_velocity_after']),
                reasoning=result['reasoning'],
                confidence=result.get('confidence', 1.0)
            )
            
        except Exception as e:
            print(f"[{self.agent_id}] Error proposing velocity: {e}")
            # Fallback: keep current velocity
            return VelocityProposal(
                agent_id=self.agent_id,
                proposed_velocity=my_state.velocity,
                reasoning=f"Error: {e}",
                confidence=0.0
            )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (supports both OpenAI and Anthropic)"""
        if hasattr(self.llm_client, 'chat'):
            # OpenAI
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        else:
            # Anthropic
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
