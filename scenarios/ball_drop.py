import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.agents import BallAgent, FloorAgent
from ape.runtime import SimpleRuntime

load_dotenv()


def main():
    print("=== APE: Ball Drop Scenario ===\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    runtime = SimpleRuntime(event_bus, world_state)
    
    ball_state = PhysicsState(
        position=np.array([0.0, 5.0]),
        velocity=np.array([0.0, 0.0]),
        mass=1.0,
        elasticity=0.8
    )
    world_state.add_object('ball-001', ball_state)
    
    floor_state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=1.0
    )
    world_state.add_object('floor-001', floor_state)
    
    llm_client = None
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        from openai import OpenAI
        llm_client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"
        print(f"Using OpenAI: {model}\n")
    else:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=api_key)
            model = "claude-haiku-4-5-20251001"
            print(f"Using Anthropic: {model}\n")
        else:
            raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
    
    ball_agent = BallAgent('ball-001', event_bus, world_state, llm_client, model)
    floor_agent = FloorAgent('floor-001', event_bus, world_state)
    
    runtime.register_agent(ball_agent)
    runtime.register_agent(floor_agent)
    
    runtime.run(max_iterations=1000, dt=0.01, print_every=50)
    
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    main()