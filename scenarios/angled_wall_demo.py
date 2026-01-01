import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus, PhysicsEvent, EventType
from ape.physics import WorldState, PhysicsState
from ape.wall_agent import WallAgent, AngledWallBallAgent
from ape.runtime import SimpleRuntime

load_dotenv()


def main():
    print("=== APE: Angled Wall Demo ===\n")
    print("Testing ball collisions with angled surfaces\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    
    # Get LLM client
    llm_client = None
    model = None
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
            model = "claude-3-5-haiku-20241022"
            print(f"Using Anthropic: {model}\n")
    
    print("--- Creating Scenario ---\n")
    
    # Create ball with horizontal velocity
    ball_state = PhysicsState(
        position=np.array([2.0, 3.0]),
        velocity=np.array([-5.0, 0.0]),  # Moving left
        mass=1.0,
        elasticity=0.8
    )
    world_state.add_object('ball-001', ball_state)
    print(f"✓ Ball at (2.0, 3.0) moving left at -5.0 m/s")
    
    # Create angled wall (45 degrees)
    # Normal pointing up-right (away from ball)
    wall_normal = np.array([1.0, 1.0])  # 45 degree angle
    wall_state = PhysicsState(
        position=np.array([0.0, 2.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=0.9
    )
    world_state.add_object('wall-001', wall_state)
    print(f"✓ Angled wall at 45° (normal: {wall_normal / np.linalg.norm(wall_normal)})")
    
    # Create floor
    floor_state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=1.0
    )
    world_state.add_object('floor-001', floor_state)
    print(f"✓ Floor at y=0")
    
    # Create agents
    ball_agent = AngledWallBallAgent('ball-001', event_bus, world_state, llm_client, model)
    wall_agent = WallAgent('wall-001', event_bus, world_state, wall_normal, llm_client, model)
    
    # Simple floor agent
    from ape.agents import FloorAgent
    floor_agent = FloorAgent('floor-001', event_bus, world_state)
    
    print("\n" + "="*80)
    print("SIMULATION")
    print("="*80)
    print("Ball will hit angled wall, then bounce at an angle")
    print("="*80 + "\n")
    
    # Manual simulation (simplified for demo)
    max_iterations = 500
    dt = 0.01
    collision_count = 0
    
    for i in range(max_iterations):
        # Physics step
        world_state.step(dt)
        
        # Check for wall collision (simplified - check x position)
        ball_state = world_state.get_object('ball-001')
        
        # Wall collision when ball reaches x=0 and is moving left
        if ball_state.position[0] <= 0.5 and ball_state.velocity[0] < 0 and collision_count == 0:
            collision_count += 1
            print(f"\n[{world_state.time:.3f}s] WALL COLLISION DETECTED")
            print(f"  Ball position: {ball_state.position}")
            print(f"  Ball velocity: {ball_state.velocity}")
            
            # Emit collision event with angled normal
            event = PhysicsEvent(
                event_type=EventType.COLLISION,
                timestamp=world_state.time,
                involved_agents=['ball-001', 'wall-001'],
                data={'normal': wall_normal.tolist()}
            )
            event_bus.emit(event)
            
            # Process event
            ball_agent.handle_event(event)
            wall_agent.handle_event(event)
            
            print(f"\n  After collision:")
            ball_state = world_state.get_object('ball-001')
            print(f"  Ball velocity: {ball_state.velocity}")
            print(f"  (Should bounce up and right)")
        
        # Floor collision
        if ball_state.position[1] <= 0 and ball_state.velocity[1] < 0:
            ball_state.position[1] = 0
            
            event = PhysicsEvent(
                event_type=EventType.COLLISION,
                timestamp=world_state.time,
                involved_agents=['ball-001', 'floor-001'],
                data={'normal': [0, 1]}
            )
            event_bus.emit(event)
            
            ball_agent.handle_event(event)
            floor_agent.handle_event(event)
        
        # Stop if ball settled
        if abs(ball_state.velocity[0]) < 0.1 and abs(ball_state.velocity[1]) < 0.1 and ball_state.position[1] < 0.1:
            print(f"\n[{world_state.time:.3f}s] Ball settled")
            break
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    final_state = world_state.get_object('ball-001')
    print(f"\nFinal ball state:")
    print(f"  Position: {final_state.position}")
    print(f"  Velocity: {final_state.velocity}")
    
    print("\n💡 Angled Surface Insights:")
    print("  1. Surface normal determines bounce direction")
    print("  2. LLM must understand vector decomposition")
    print("  3. More complex than horizontal floor collisions")
    print("  4. Wall agent can reason about collisions from its perspective")
    print("  5. Opens up richer physics scenarios (ramps, corners, etc.)")


if __name__ == "__main__":
    main()
