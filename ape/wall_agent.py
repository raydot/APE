import numpy as np
import json
from typing import Optional

from .agents import Agent
from .events import PhysicsEvent, EventType, SimpleEventBus
from .physics import WorldState


class WallAgent(Agent):
    """
    Wall agent that can be positioned at any angle.
    Unlike floor (always horizontal), walls have configurable surface normals.
    """
    
    def __init__(
        self,
        agent_id: str,
        event_bus: SimpleEventBus,
        world_state: WorldState,
        surface_normal: np.ndarray,
        llm_client=None,
        model_name: str = "gpt-4o-mini"
    ):
        super().__init__(agent_id, event_bus, world_state)
        self.surface_normal = surface_normal / np.linalg.norm(surface_normal)  # Normalize
        self.llm_client = llm_client
        self.model_name = model_name
        self.use_llm = llm_client is not None
    
    def handle_event(self, event: PhysicsEvent):
        if event.event_type == EventType.COLLISION:
            if self.use_llm:
                self._handle_collision_with_reasoning(event)
            else:
                print(f"[{self.agent_id}] I'm a wall. I don't move.")
    
    def _handle_collision_with_reasoning(self, event: PhysicsEvent):
        """
        Wall can optionally reason about collisions.
        This is interesting for angled walls where the physics is less obvious.
        """
        # Get the other agent involved
        other_agent_id = [aid for aid in event.involved_agents if aid != self.agent_id][0]
        other_state = self.world_state.get_object(other_agent_id)
        
        if not other_state:
            return
        
        # Wall reasons about what should happen
        prompt = f"""You are a wall in a physics simulation. A ball just collided with you.

Your properties:
- Surface normal: {self.surface_normal.tolist()} (direction perpendicular to your surface)
- You are immovable (infinite mass)

Ball properties:
- Velocity before collision: {other_state.velocity.tolist()} m/s
- Mass: {other_state.mass} kg
- Elasticity: {other_state.elasticity}

In an elastic collision with a wall:
- The component of velocity perpendicular to the wall reverses and scales by elasticity
- The component parallel to the wall is unchanged (no friction)
- The wall doesn't move (infinite mass)

What should happen to the ball? Provide brief reasoning.

Respond with JSON:
{{
    "reasoning": "explanation of what happens",
    "expected_outcome": "brief description of result"
}}"""
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            print(f"[{self.agent_id}] Wall's perspective:")
            print(f"  {result['reasoning']}")
            print(f"  Expected: {result['expected_outcome']}")
        except Exception as e:
            print(f"[{self.agent_id}] I'm a wall. I don't move. (LLM error: {e})")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM"""
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
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.content[0].text
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response"""
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        return json.loads(response)


class AngledWallBallAgent(Agent):
    """
    Ball agent that handles collisions with angled walls.
    Extends basic collision handling to work with arbitrary surface normals.
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
        if event.event_type == EventType.COLLISION:
            self._handle_collision(event)
    
    def _handle_collision(self, event: PhysicsEvent):
        state = self.get_my_state()
        if not state:
            return
        
        # Get surface normal from event data
        normal = np.array(event.data.get('normal', [0, 1]))
        
        # Get other agent to determine surface type
        other_agent_id = [aid for aid in event.involved_agents if aid != self.agent_id][0]
        
        print(f"[{self.agent_id}] Collision with {other_agent_id}")
        print(f"  Surface normal: {normal}")
        print(f"  Velocity before: {state.velocity}")
        
        prompt = f"""You are a ball in a physics simulation. You just collided with a surface.

Your current state:
- Position: {state.position.tolist()} m
- Velocity: {state.velocity.tolist()} m/s
- Mass: {state.mass} kg
- Elasticity: {state.elasticity}

Surface information:
- Surface normal: {normal.tolist()} (perpendicular to surface)
- Other agent: {other_agent_id}

For an angled surface collision:
1. Decompose velocity into components parallel and perpendicular to surface normal
2. Perpendicular component: reverses direction and scales by elasticity
3. Parallel component: unchanged (no friction)

Calculate your new velocity after collision.

Respond ONLY with JSON:
{{
    "reasoning": "step-by-step physics calculation",
    "new_velocity": [vx, vy]
}}"""
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            
            print(f"  Reasoning: {result['reasoning']}")
            print(f"  New velocity: {result['new_velocity']}")
            
            new_velocity = np.array(result['new_velocity'])
            self.update_velocity(new_velocity)
            
        except Exception as e:
            print(f"[{self.agent_id}] Error: {e}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM"""
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
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response"""
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        return json.loads(response)
