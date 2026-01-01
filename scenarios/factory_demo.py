import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState
from ape.factory import AgentFactory
from ape.runtime import SimpleRuntime

load_dotenv()


def main():
    print("=== APE: Agent Factory Demo ===\n")
    
    # Setup
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
    else:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=api_key)
            model = "claude-3-5-haiku-20241022"
    
    # Create factory
    factory = AgentFactory(
        event_bus=event_bus,
        world_state=world_state,
        llm_client=llm_client,
        model_name=model
    )
    
    print("--- Available Templates ---")
    templates = factory.list_templates()
    for name, description in templates.items():
        print(f"  {name}: {description}")
    
    print("\n--- Creating Agents from Templates ---\n")
    
    # Create standard ball
    ball1 = factory.create_ball(
        position=np.array([0.0, 5.0]),
        template="ball"
    )
    print(f"✓ Created {ball1.agent_id} (standard ball)")
    
    # Create heavy ball
    ball2 = factory.create_ball(
        position=np.array([1.0, 3.0]),
        template="heavy_ball"
    )
    print(f"✓ Created {ball2.agent_id} (heavy ball)")
    
    # Create bouncy ball
    ball3 = factory.create_ball(
        position=np.array([-1.0, 4.0]),
        template="bouncy_ball"
    )
    print(f"✓ Created {ball3.agent_id} (bouncy ball)")
    
    # Create custom ball with overrides
    ball4 = factory.create_ball(
        position=np.array([0.5, 6.0]),
        template="ball",
        mass=2.0,  # Override mass
        elasticity=0.5,  # Override elasticity
        agent_id="custom-ball-001"
    )
    print(f"✓ Created {ball4.agent_id} (custom parameters)")
    
    # Create floor
    floor = factory.create_floor(position=np.array([0.0, 0.0]))
    print(f"✓ Created {floor.agent_id}")
    
    print("\n--- Agent Parameters ---")
    for agent_id in ['ball-001', 'heavy_ball-001', 'bouncy_ball-001', 'custom-ball-001']:
        state = world_state.get_object(agent_id)
        print(f"\n{agent_id}:")
        print(f"  Mass: {state.mass}kg")
        print(f"  Elasticity: {state.elasticity}")
        print(f"  Position: {state.position}")
    
    print("\n--- Running Multi-Ball Simulation ---\n")
    
    runtime = SimpleRuntime(event_bus, world_state)
    runtime.register_agent(ball1)
    runtime.register_agent(ball2)
    runtime.register_agent(ball3)
    runtime.register_agent(ball4)
    runtime.register_agent(floor)
    
    # Run for limited iterations (demo only)
    print("Running simulation with 4 balls...")
    print("(Limited to 200 iterations for demo)\n")
    
    collision_counts = {
        'ball-001': 0,
        'heavy_ball-001': 0,
        'bouncy_ball-001': 0,
        'custom-ball-001': 0
    }
    
    for i in range(200):
        runtime.iteration = i
        world_state.step(0.01)
        runtime._detect_collisions()
        
        # Count collisions
        while event_bus.has_events():
            event = event_bus.get_next()
            if event:
                for agent_id in event.involved_agents:
                    if agent_id in collision_counts:
                        collision_counts[agent_id] += 1
                
                for agent_id in event.involved_agents:
                    if agent_id in runtime.agents:
                        runtime.agents[agent_id].handle_event(event)
    
    print("\n--- Simulation Results ---")
    print(f"Total iterations: 200")
    print(f"\nCollisions per ball:")
    for agent_id, count in collision_counts.items():
        print(f"  {agent_id}: {count}")
    
    print("\n--- Final Positions ---")
    for agent_id in collision_counts.keys():
        state = world_state.get_object(agent_id)
        print(f"{agent_id}: y={state.position[1]:.3f}m, vy={state.velocity[1]:.3f}m/s")
    
    print("\n💡 Factory Benefits:")
    print("  1. Easy creation of multiple agents")
    print("  2. Consistent configuration via templates")
    print("  3. Parameter overrides for customization")
    print("  4. Automatic ID generation")
    print("  5. Centralized agent management")
    print("  6. Scales to complex multi-agent scenarios")


if __name__ == "__main__":
    main()
