import numpy as np
import os
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.agents import FloorAgent
from ape.hybrid_agent import HybridBallAgent
from ape.runtime import SimpleRuntime
from ape.tools import create_physics_tool_registry
from ape.validator import PhysicsValidator

load_dotenv()


def main():
    print("=== APE: Hybrid Agent Demo ===\n")
    print("Hybrid approach: LLM reasons + Tool calculates + Validator checks\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    runtime = SimpleRuntime(event_bus, world_state)
    
    # Setup tools and validator
    tool_registry = create_physics_tool_registry()
    validator = PhysicsValidator(tolerance=0.05)
    
    # Create ball and floor
    ball_state = PhysicsState(
        position=np.array([0.0, 5.0]),
        velocity=np.array([0.0, 0.0]),
        mass=1.0,
        elasticity=0.8
    )
    world_state.add_object('hybrid-ball-001', ball_state)
    
    floor_state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=1.0
    )
    world_state.add_object('floor-001', floor_state)
    
    # Get LLM client
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
            model = "claude-3-5-haiku-20241022"
            print(f"Using Anthropic: {model}\n")
        else:
            raise ValueError("No API key found")
    
    # Create hybrid agent
    ball_agent = HybridBallAgent(
        agent_id='hybrid-ball-001',
        event_bus=event_bus,
        world_state=world_state,
        llm_client=llm_client,
        model_name=model,
        tool_registry=tool_registry,
        validator=validator,
        trust_llm_threshold=0.05
    )
    
    floor_agent = FloorAgent('floor-001', event_bus, world_state)
    
    runtime.register_agent(ball_agent)
    runtime.register_agent(floor_agent)
    
    print("="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    print("Watch how the hybrid agent:")
    print("  1. Gets LLM reasoning")
    print("  2. Gets tool calculation")
    print("  3. Validates LLM result")
    print("  4. Uses tool if LLM is wrong")
    print("="*80 + "\n")
    
    # Run simulation
    max_iterations = 1000
    dt = 0.01
    
    for i in range(max_iterations):
        runtime.iteration = i
        world_state.step(dt)
        runtime._detect_collisions()
        runtime._process_events()
        
        if runtime._should_stop():
            print(f"\n[RUNTIME] Simulation complete at iteration {i}")
            break
    
    # Show statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    ball_agent.print_stats()
    
    stats = ball_agent.get_stats()
    
    print("\n💡 Hybrid Agent Benefits:")
    print("  1. Always uses accurate tool calculation")
    print("  2. Validates LLM reasoning quality")
    print("  3. Provides interpretable LLM explanations")
    print("  4. Tracks LLM performance over time")
    print("  5. Can adapt strategy based on LLM accuracy")
    
    print("\n📊 Cost Analysis:")
    print(f"  LLM calls made: {stats['total_collisions']}")
    print(f"  Tool calls made: {stats['total_collisions']} (always)")
    print(f"  LLM accuracy: {stats['llm_accuracy']*100:.1f}%")
    
    if stats['llm_accuracy'] > 0.95:
        print("\n✨ Optimization opportunity:")
        print("  LLM is very accurate - could skip tool validation")
        print("  for simple scenarios to save computation")
    elif stats['llm_accuracy'] < 0.7:
        print("\n⚠️  Recommendation:")
        print("  LLM accuracy is low - consider:")
        print("    - Better prompts")
        print("    - More capable model")
        print("    - Skip LLM entirely for this task")


if __name__ == "__main__":
    main()
