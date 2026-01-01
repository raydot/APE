import numpy as np
from typing import List, Dict
from .events import SimpleEventBus, PhysicsEvent, EventType 
from .physics import WorldState
from .agents import Agent

class SimpleRuntime:
    def __init__(self, event_bus: SimpleEventBus, world_state: WorldState):
        self.event_bus = event_bus
        self.world_state = world_state
        self.agents: Dict[str, Agent] = {}
        self.iteration = 0

    def register_agent(self, agent: Agent):
        self.agents[agent.agent_id] = agent
        print(f"[RUNTIME] Registered agent: {agent.agent_id}")
    
    def run(self, max_iterations: int = 1000, dt: float = 0.1, print_every: int = 50):
        print(f"[RUNTIME] Starting simulation: max_iter={max_iterations}, dt={dt}")
        print(f"[RUNTIME] Initial state: {self.world_state}")
        for i in range(max_iterations):
            self.iteration = i

            self.world_state.step(dt)
            
            self._detect_collisions()
            
            self._process_events()

            if i % print_every == 0:
                self._print_state()

            if self._should_stop():
                print(f"[RUNTIME] Simulation stopped at iteration {i}")
                break

        print(f"[RUNTIME] Final state: {self.world_state}")
        self._print_summary()


    def _detect_collisions(self):
        for agent_id, state in self.world_state.get_all_objects().items():
            if state.position[1] <= 0 and state.velocity[1] < 0:
                state.position[1] = 0
                
                event = PhysicsEvent(
                    event_type=EventType.COLLISION,
                    timestamp=self.world_state.time,
                    involved_agents=[agent_id, 'floor-001'],
                    data={'normal': [0, 1]}
                )
                self.event_bus.emit(event)
    
    def _process_events(self):
        while self.event_bus.has_events():
            event = self.event_bus.get_next()
            if event:
                for agent_id in event.involved_agents:
                    if agent_id in self.agents:
                        self.agents[agent_id].handle_event(event)
    
    def _should_stop(self) -> bool:
        for agent_id, state in self.world_state.get_all_objects().items():
            if state.mass < float('inf'):
                if abs(state.velocity[1]) > 0.1 or state.position[1] > 0.01:
                    return False
        return True
    
    def _print_state(self):
        print(f"\n--- Iteration {self.iteration}, t={self.world_state.time:.3f}s ---")
        for agent_id, state in self.world_state.get_all_objects().items():
            if state.mass < float('inf'):
                print(f"  {agent_id}: {state}")
    
    def _print_summary(self):
        print("\n=== Event Log Summary ===")
        collision_count = sum(1 for e in self.event_bus.event_log 
                            if e.event_type == EventType.COLLISION)
        print(f"Total collisions: {collision_count}")
        print(f"Total events: {len(self.event_bus.event_log)}")
            
        