import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus, PhysicsEvent, EventType
from ape.physics import WorldState, PhysicsState
from ape.negotiating_agent import NegotiatingBallAgent
from ape.resolver_agent import CollisionResolverAgent
from ape.collision_detection import BallBallCollisionDetector
from ape.tools import create_physics_tool_registry
from ape.ball_collision_viz import BallCollisionRecorder, visualize_ball_collision

load_dotenv()


def main():
    """
    Two balls collide head-on.
    
    Ball 1: Moving right at +5 m/s
    Ball 2: Moving left at -3 m/s
    
    Expected outcome (equal mass, elastic):
    - Velocities should swap
    - Ball 1: +5 → -3 m/s
    - Ball 2: -3 → +5 m/s
    """
    
    print("=== APE: Two Ball Head-On Collision ===\n")
    
    # Setup
    event_bus = SimpleEventBus()
    world = WorldState(gravity=np.array([0.0, 0.0]))  # No gravity for this test
    
    # LLM setup
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
        else:
            raise ValueError("No API key found")
    
    # Tool registry
    tool_registry = create_physics_tool_registry()
    
    # Collision detector
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    # Recorder for visualization
    recorder = BallCollisionRecorder()
    
    # Create resolver
    resolver = CollisionResolverAgent(event_bus, world, tool_registry)
    
    # Create two balls
    ball1_state = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),  # Moving right
        mass=1.0,
        elasticity=1.0  # Perfectly elastic
    )
    world.add_object("ball-001", ball1_state)
    
    ball2_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),  # Moving left
        mass=1.0,  # Equal mass
        elasticity=1.0
    )
    world.add_object("ball-002", ball2_state)
    
    # Create agents
    ball1_agent = NegotiatingBallAgent("ball-001", event_bus, world, llm_client, model)
    ball2_agent = NegotiatingBallAgent("ball-002", event_bus, world, llm_client, model)
    
    print("Initial Setup:")
    print(f"Ball 1: position={ball1_state.position}, velocity={ball1_state.velocity}")
    print(f"Ball 2: position={ball2_state.position}, velocity={ball2_state.velocity}")
    print("\nExpected outcome (equal mass elastic collision):")
    print("Ball 1: velocity should become -3.0 m/s")
    print("Ball 2: velocity should become +5.0 m/s")
    print("\n" + "="*60 + "\n")
    
    # Run simulation
    max_iterations = 200
    dt = 0.01
    ball_ids = ["ball-001", "ball-002"]
    
    for i in range(max_iterations):
        # Physics step
        world.step(dt)
        
        # Record frame
        recorder.record_frame(world, ball_ids)
        
        # Detect ball-ball collisions
        collisions = detector.detect_all_collisions(world.get_all_objects(), world.time)
        
        for collision in collisions:
            print(f"\n[{world.time:.3f}s] COLLISION DETECTED: {collision['ball1_id']} <-> {collision['ball2_id']}")
            print(f"  Distance: {collision['distance']:.3f}m")
            print(f"  Overlap: {collision['overlap']:.3f}m")
            print(f"  Normal: {collision['collision_normal']}")
            
            # Record collision
            recorder.record_collision(
                world.time,
                collision['ball1_id'],
                collision['ball2_id'],
                collision['collision_point']
            )
            
            # Separate overlapping balls
            state1 = world.get_object(collision['ball1_id'])
            state2 = world.get_object(collision['ball2_id'])
            
            new_pos1, new_pos2 = detector.separate_overlapping_balls(
                state1.position,
                state2.position,
                collision['overlap']
            )
            
            state1.position = new_pos1
            state2.position = new_pos2
            world.update_object(collision['ball1_id'], state1)
            world.update_object(collision['ball2_id'], state2)
            
            # Resolve collision through negotiation
            outcome = resolver.handle_collision(
                ball1_agent,
                ball2_agent,
                collision
            )
            
            # Record outcome
            recorder.record_negotiation_outcome(world.time, outcome)
        
        # Stop if balls are far apart and moving away
        state1 = world.get_object("ball-001")
        state2 = world.get_object("ball-002")
        distance = np.linalg.norm(state2.position - state1.position)
        
        if distance > 5.0 and i > 50:
            print(f"\n[{world.time:.3f}s] Balls separated, stopping simulation")
            break
    
    # Print results
    final_ball1 = world.get_object("ball-001")
    final_ball2 = world.get_object("ball-002")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Ball 1: velocity={final_ball1.velocity}")
    print(f"Ball 2: velocity={final_ball2.velocity}")
    
    # Check conservation
    momentum_before = 1.0 * np.array([5.0, 0.0]) + 1.0 * np.array([-3.0, 0.0])
    momentum_after = 1.0 * final_ball1.velocity + 1.0 * final_ball2.velocity
    
    energy_before = 0.5 * 1.0 * 25.0 + 0.5 * 1.0 * 9.0  # 17 J
    energy_after = 0.5 * 1.0 * np.dot(final_ball1.velocity, final_ball1.velocity) + \
                   0.5 * 1.0 * np.dot(final_ball2.velocity, final_ball2.velocity)
    
    print(f"\nConservation Check:")
    print(f"Momentum before: {momentum_before}")
    print(f"Momentum after: {momentum_after}")
    print(f"Momentum conserved: {np.allclose(momentum_before, momentum_after, atol=0.1)}")
    
    print(f"\nEnergy before: {energy_before:.3f} J")
    print(f"Energy after: {energy_after:.3f} J")
    print(f"Energy conserved: {np.allclose(energy_before, energy_after, atol=0.5)}")
    
    # Resolver stats
    resolver.print_stats()
    
    # Save recording
    recorder.save('ball_collision_data.json')
    
    # Visualize
    print("\n[VISUALIZATION] Creating animation...")
    visualize_ball_collision(recorder)


if __name__ == "__main__":
    main()
