import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .events import PhysicsEvent, EventType, SimpleEventBus
from .physics import WorldState, PhysicsState

class Agent(ABC):
    def __init__(self, agent_id: str, event_bus: SimpleEventBus, world_state: WorldState):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.world_state = world_state

    def get_my_state(self) -> Optional[PhysicsState]:
        return self.world_state.get_object(self.agent_id)

    def update_velocity(self, new_velocity: np.ndarray):
        state = self.get_my_state()
        if state:
            state.velocity = new_velocity
            self.world_state.update_object(self.agent_id, state)
            print(f"[{self.agent_id}] Updated velocity to: {new_velocity}")

    @abstractmethod
    def handle_event(self, event: PhysicsEvent):
        pass

class FloorAgent(Agent):
    def handle_event(self, event: PhysicsEvent):
        if event.event_type == EventType.COLLISION:
            print(f"[{self.agent_id}] I'm a floor, I don't move (much).")

class BallAgent(Agent):
    def __init__(self, agent_id: str, event_bus: SimpleEventBus, world_state: WorldState, llm_client, model_name: str="gpt-4o-mini"):
        super().__init__(agent_id, event_bus, world_state)
        self.llm_client = llm_client
        self.model_name = model_name

    def handle_event(self, event: PhysicsEvent):
        if event.event_type == EventType.COLLISION:
            print(f"[{self.agent_id}] Handling collision event...")
            self._handle_collision(event)

    def _handle_collision(self, event: PhysicsEvent):
        state = self.get_my_state()
        if not state:
            return

        normal = np.array(event.data.get('normal', [0, 1]))

        prompt = self._build_collision_prompt(state, normal)

        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)

            print(f"[{self.agent_id}] LLM response:")
            print(f"    Reasoning:{result['reasoning']}")
            print(f"    New velocity: {result['new_velocity']}")

            new_velocity = np.array(result['new_velocity'])
            self.update_velocity(new_velocity)
            
        except Exception as e:
            print(f"[{self.agent_id}] Error processing LLM response:: {e}")

    def _build_collision_prompt(self, state: PhysicsState, normal: np.ndarray) -> str:
        return f"""You are a ball in a physics simulation. You just collided with a floor.
Your current state:
- Position {state.position.tolist()} m
- Velocity {state.velocity.tolist()} m/s
- Mass {state.mass} kg
- Elasticity {state.elasticity}

Collision information:
- Surface normal: {normal.tolist()}
Calculate your new velocity after elastic collision.
The perpendicular component reverses and scales by elasticity.
The parallel component remains unchanged (no friction).
Respond ONLY with JSON:
{{
    "reasoning": "bried physics explanation",
    "new_velocity": [vx, vy]
}}"""
        

    def _call_llm(self, prompt: str) -> str:
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
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])

        return json.loads(response)
