import numpy as np  
from dataclasses import dataclass
from typing import Dict, Optional
@dataclass
class PhysicsState:
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    elasticity: float

    def __str__(self):
        return f"pos: {self.position}, vel: {self.velocity}, m: {self.mass}kg, e: {self.elasticity}"

class WorldState:
    def __init__(self, gravity: np.ndarray = np.array([0, -9.81])):
        self._objects: Dict[str, PhysicsState] = {}
        self.gravity = gravity
        self.time = 0.0

    def add_object(self, agent_id: str, state: PhysicsState):
        self._objects[agent_id] = state
        print(f"[WORLD] Added {agent_id}: {state}")

    def get_object(self, agent_id: str) -> Optional[PhysicsState]:
        return self._objects.get(agent_id)

    def update_object(self, agent_id: str, state: PhysicsState):
        if agent_id in self._objects:
            self._objects[agent_id] = state
        else:
            raise KeyError(F"Agent {agent_id} not found in world state")

    def step(self, dt: float):
        for agent_id, state in self._objects.items():
            if state.mass < float('inf'):
                state.velocity += self.gravity * dt
                state.position += state.velocity * dt

        self.time += dt

    def get_all_objects(self) -> Dict[str, PhysicsState]:
        return self._objects.copy()

    def __str__(self):
        obj_summary = ", ".join(self._objects.keys())
        return f"WorldState(t={self.time}, objects=[{obj_summary}])"


    