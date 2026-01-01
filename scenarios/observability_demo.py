import numpy as np
import os
import time
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.agents import BallAgent, FloorAgent
from ape.runtime import SimpleRuntime
from ape.observability import TraceCollector, TraceLevel

load_dotenv()


def main():
    print("=== APE: Observability Demo ===\n")
    
    # Create trace collector
    trace_collector = TraceCollector()
    
    # Log simulation start
    trace_collector.log(
        TraceLevel.INFO,
        "system",
        "simulation_start",
        "Starting ball drop simulation with observability",
        {"initial_height": 5.0, "elasticity": 0.8}
    )
    
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
    
    trace_collector.log(
        TraceLevel.INFO,
        "ball-001",
        "agent_created",
        "Ball agent initialized",
        {"position": ball_state.position.tolist(), "mass": ball_state.mass}
    )
    
    floor_state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=1.0
    )
    world_state.add_object('floor-001', floor_state)
    
    trace_collector.log(
        TraceLevel.INFO,
        "floor-001",
        "agent_created",
        "Floor agent initialized",
        {"position": floor_state.position.tolist()}
    )
    
    llm_client = None
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        from openai import OpenAI
        llm_client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"
        trace_collector.log(TraceLevel.INFO, "system", "llm_config", f"Using OpenAI: {model}")
    else:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=api_key)
            model = "claude-3-5-haiku-20241022"
            trace_collector.log(TraceLevel.INFO, "system", "llm_config", f"Using Anthropic: {model}")
        else:
            raise ValueError("No API key found")
    
    ball_agent = BallAgent('ball-001', event_bus, world_state, llm_client, model)
    floor_agent = FloorAgent('floor-001', event_bus, world_state)
    
    runtime.register_agent(ball_agent)
    runtime.register_agent(floor_agent)
    
    # Run simulation with tracing
    print("[OBSERVABILITY] Running simulation with full tracing...\n")
    
    max_iterations = 300  # Shorter for demo
    dt = 0.01
    collision_count = 0
    
    for i in range(max_iterations):
        runtime.iteration = i
        
        # Log physics step
        if i % 50 == 0:
            state = world_state.get_object('ball-001')
            trace_collector.log(
                TraceLevel.DEBUG,
                "ball-001",
                "physics_step",
                f"Physics update at iteration {i}",
                {
                    "position": state.position.tolist(),
                    "velocity": state.velocity.tolist(),
                    "time": world_state.time
                }
            )
        
        world_state.step(dt)
        runtime._detect_collisions()
        
        # Process events with tracing
        if event_bus.has_events():
            event = event_bus.get_next()
            if event:
                collision_count += 1
                
                trace_collector.log(
                    TraceLevel.INFO,
                    "ball-001",
                    "collision_detected",
                    f"Collision #{collision_count} detected",
                    {
                        "time": world_state.time,
                        "velocity_before": world_state.get_object('ball-001').velocity.tolist()
                    }
                )
                
                # Time the LLM call
                start_time = time.time()
                
                for agent_id in event.involved_agents:
                    if agent_id in runtime.agents:
                        runtime.agents[agent_id].handle_event(event)
                
                duration = time.time() - start_time
                
                # Log LLM interaction (simplified - in real scenario, extract from agent)
                state_after = world_state.get_object('ball-001')
                trace_collector.log_llm_call(
                    agent_id="ball-001",
                    model=model,
                    prompt=f"Calculate collision for velocity {state_after.velocity.tolist()}",
                    response=f"New velocity: {state_after.velocity.tolist()}",
                    reasoning="Elastic collision calculation",
                    duration=duration,
                    tokens_used=150,  # Estimated
                    cost=0.002  # Estimated
                )
                
                trace_collector.log(
                    TraceLevel.INFO,
                    "ball-001",
                    "collision_resolved",
                    f"Collision #{collision_count} resolved",
                    {
                        "velocity_after": state_after.velocity.tolist(),
                        "llm_duration": duration
                    }
                )
        
        if runtime._should_stop():
            trace_collector.log(
                TraceLevel.INFO,
                "system",
                "simulation_complete",
                f"Simulation stopped at iteration {i}",
                {"total_collisions": collision_count}
            )
            break
    
    # Save traces
    trace_collector.save('simulation_trace.json')
    
    # Display various trace views
    trace_collector.print_summary()
    
    print("\n" + "="*80)
    print("Would you like to see:")
    print("  1. Timeline view")
    print("  2. LLM conversations")
    print("  3. Agent-specific traces")
    print("="*80)
    
    # Show timeline
    trace_collector.print_timeline(max_entries=20)
    
    # Show LLM conversations
    trace_collector.print_llm_conversations(max_entries=3)
    
    print("\n💡 Observability Benefits:")
    print("  1. Debug agent behavior step-by-step")
    print("  2. Analyze LLM reasoning patterns")
    print("  3. Identify performance bottlenecks")
    print("  4. Track cost and token usage")
    print("  5. Replay and analyze past runs")
    print("  6. Compare different agent strategies")


if __name__ == "__main__":
    main()
