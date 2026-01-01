import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus, EventType
from ape.physics import WorldState, PhysicsState
from ape.agents import BallAgent, FloorAgent
from ape.runtime import SimpleRuntime
from ape.validator import PhysicsValidator

load_dotenv()


def main():
    print("=== APE: Physics Validation Demo ===\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    runtime = SimpleRuntime(event_bus, world_state)
    validator = PhysicsValidator(tolerance=0.05)
    
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
    
    print("[VALIDATION] Running simulation with physics validation...\n")
    
    collision_count = 0
    max_iterations = 1000
    dt = 0.01
    
    for i in range(max_iterations):
        runtime.iteration = i
        
        state_before = world_state.get_object('ball-001')
        pos_before = state_before.position.copy()
        vel_before = state_before.velocity.copy()
        
        world_state.step(dt)
        runtime._detect_collisions()
        
        if event_bus.has_events():
            event = event_bus.get_next()
            if event and event.event_type == EventType.COLLISION:
                collision_count += 1
                print(f"\n=== Collision #{collision_count} at t={world_state.time:.3f}s ===")
                
                for agent_id in event.involved_agents:
                    if agent_id in runtime.agents:
                        runtime.agents[agent_id].handle_event(event)
                
                state_after = world_state.get_object('ball-001')
                pos_after = state_after.position.copy()
                vel_after = state_after.velocity.copy()
                
                print(f"\nValidating collision physics:")
                results = validator.validate_collision(
                    mass=state_after.mass,
                    velocity_before=vel_before,
                    velocity_after=vel_after,
                    position_before=pos_before,
                    position_after=pos_after,
                    elasticity=state_after.elasticity,
                    surface_normal=np.array([0.0, 1.0]),
                    gravity=9.8
                )
                
                for result in results:
                    status = "✓" if result.passed else "✗"
                    print(f"  {status} {result.law}: {result.message}")
        
        if runtime._should_stop():
            print(f"\n[VALIDATION] Simulation complete at iteration {i}")
            break
    
    print("\n" + "="*60)
    validator.print_summary()
    print("="*60)
    
    summary = validator.get_summary()
    if summary['total_violations'] == 0:
        print("\n🎉 LLM physics calculations are accurate!")
    else:
        print(f"\n⚠️  LLM made {summary['total_violations']} physics errors")
        print("Consider adjusting prompts or using a more capable model.")


if __name__ == "__main__":
    main()
