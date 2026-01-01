import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState
from ape.factory import AgentFactory
from ape.runtime import SimpleRuntime

load_dotenv()


def main():
    print("=== APE: Two Balls Scenario ===\n")
    print("Testing multi-agent interactions with different ball types\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    runtime = SimpleRuntime(event_bus, world_state)
    
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
    
    # Create factory
    factory = AgentFactory(
        event_bus=event_bus,
        world_state=world_state,
        llm_client=llm_client,
        model_name=model
    )
    
    print("--- Creating Agents ---\n")
    
    # Create two different balls
    ball1 = factory.create_ball(
        position=np.array([0.0, 5.0]),
        template="ball",
        agent_id="standard-ball"
    )
    print(f"✓ Created {ball1.agent_id} at height 5.0m (standard, e=0.8)")
    
    ball2 = factory.create_ball(
        position=np.array([0.5, 3.0]),
        template="bouncy_ball",
        agent_id="bouncy-ball"
    )
    print(f"✓ Created {ball2.agent_id} at height 3.0m (bouncy, e=0.95)")
    
    floor = factory.create_floor(position=np.array([0.0, 0.0]))
    print(f"✓ Created {floor.agent_id}")
    
    # Show initial states
    print("\n--- Initial States ---")
    for agent_id in ['standard-ball', 'bouncy-ball']:
        state = world_state.get_object(agent_id)
        print(f"{agent_id}:")
        print(f"  Position: {state.position}")
        print(f"  Mass: {state.mass}kg")
        print(f"  Elasticity: {state.elasticity}")
    
    # Register agents
    runtime.register_agent(ball1)
    runtime.register_agent(ball2)
    runtime.register_agent(floor)
    
    print("\n" + "="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    print("Standard ball (e=0.8) vs Bouncy ball (e=0.95)")
    print("Watch how they bounce differently!")
    print("="*80 + "\n")
    
    # Track collisions per agent
    collision_counts = {
        'standard-ball': 0,
        'bouncy-ball': 0
    }
    
    max_iterations = 1000
    dt = 0.01
    
    for i in range(max_iterations):
        runtime.iteration = i
        world_state.step(dt)
        runtime._detect_collisions()
        
        # Count and process collisions
        while event_bus.has_events():
            event = event_bus.get_next()
            if event:
                for agent_id in event.involved_agents:
                    if agent_id in collision_counts:
                        collision_counts[agent_id] += 1
                        print(f"\n[{world_state.time:.3f}s] {agent_id} collision #{collision_counts[agent_id]}")
                
                for agent_id in event.involved_agents:
                    if agent_id in runtime.agents:
                        runtime.agents[agent_id].handle_event(event)
        
        # Check if both balls have settled
        ball1_state = world_state.get_object('standard-ball')
        ball2_state = world_state.get_object('bouncy-ball')
        
        ball1_settled = abs(ball1_state.velocity[1]) < 0.1 and ball1_state.position[1] < 0.01
        ball2_settled = abs(ball2_state.velocity[1]) < 0.1 and ball2_state.position[1] < 0.01
        
        if ball1_settled and ball2_settled:
            print(f"\n[RUNTIME] Both balls settled at iteration {i}")
            break
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nCollisions:")
    print(f"  Standard ball (e=0.8): {collision_counts['standard-ball']}")
    print(f"  Bouncy ball (e=0.95): {collision_counts['bouncy-ball']}")
    
    print(f"\nFinal positions:")
    for agent_id in ['standard-ball', 'bouncy-ball']:
        state = world_state.get_object(agent_id)
        print(f"  {agent_id}: y={state.position[1]:.4f}m, vy={state.velocity[1]:.4f}m/s")
    
    print("\n📊 Analysis:")
    if collision_counts['bouncy-ball'] > collision_counts['standard-ball']:
        print(f"  ✓ Bouncy ball bounced {collision_counts['bouncy-ball'] - collision_counts['standard-ball']} more times")
        print(f"    (Higher elasticity = more bounces)")
    
    print("\n💡 Multi-Agent Insights:")
    print("  1. Different agents can have different physics properties")
    print("  2. Agents operate independently but share the world")
    print("  3. Each agent reasons about its own collisions")
    print("  4. Factory makes it easy to create varied agents")
    print("  5. System scales to many agents naturally")


if __name__ == "__main__":
    main()
