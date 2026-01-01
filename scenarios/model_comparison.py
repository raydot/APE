import numpy as np
import os
import time
from dotenv import load_dotenv
from typing import Dict, List

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.agents import BallAgent, FloorAgent
from ape.runtime import SimpleRuntime
from ape.validator import PhysicsValidator

load_dotenv()


class ModelBenchmark:
    def __init__(self):
        self.results = []
    
    def run_model(self, model_name: str, llm_client, max_iterations: int = 1000) -> Dict:
        """Run simulation with a specific model and collect metrics"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
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
        
        ball_agent = BallAgent('ball-001', event_bus, world_state, llm_client, model_name)
        floor_agent = FloorAgent('floor-001', event_bus, world_state)
        
        runtime.register_agent(ball_agent)
        runtime.register_agent(floor_agent)
        
        start_time = time.time()
        collision_count = 0
        dt = 0.01
        
        for i in range(max_iterations):
            runtime.iteration = i
            
            state_before = world_state.get_object('ball-001')
            vel_before = state_before.velocity.copy()
            pos_before = state_before.position.copy()
            
            world_state.step(dt)
            runtime._detect_collisions()
            runtime._process_events()
            
            # Validate if collision occurred
            if event_bus.event_log and len(event_bus.event_log) > collision_count:
                collision_count = len(event_bus.event_log)
                state_after = world_state.get_object('ball-001')
                
                validator.validate_collision(
                    mass=state_after.mass,
                    velocity_before=vel_before,
                    velocity_after=state_after.velocity,
                    position_before=pos_before,
                    position_after=state_after.position,
                    elasticity=state_after.elasticity,
                    surface_normal=np.array([0.0, 1.0]),
                    gravity=9.8
                )
            
            if runtime._should_stop():
                break
        
        elapsed_time = time.time() - start_time
        
        summary = validator.get_summary()
        
        result = {
            'model': model_name,
            'collisions': collision_count,
            'violations': summary['total_violations'],
            'violations_by_law': summary['violations_by_law'],
            'elapsed_time': elapsed_time,
            'avg_time_per_collision': elapsed_time / collision_count if collision_count > 0 else 0,
            'accuracy': 1.0 - (summary['total_violations'] / (collision_count * 3)) if collision_count > 0 else 0
        }
        
        self.results.append(result)
        
        print(f"\nResults:")
        print(f"  Collisions: {collision_count}")
        print(f"  Physics violations: {summary['total_violations']}")
        print(f"  Accuracy: {result['accuracy']*100:.1f}%")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"  Avg time/collision: {result['avg_time_per_collision']:.2f}s")
        
        return result
    
    def print_comparison(self):
        """Print comparison table of all models"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'Violations':<12} {'Avg Time':<12}")
        print(f"{'-'*80}")
        
        for result in sorted(self.results, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['model']:<30} "
                  f"{result['accuracy']*100:>10.1f}% "
                  f"{result['violations']:>10} "
                  f"{result['avg_time_per_collision']:>10.2f}s")
        
        print(f"\n{'='*80}")
        
        # Best model
        best = max(self.results, key=lambda x: x['accuracy'])
        fastest = min(self.results, key=lambda x: x['avg_time_per_collision'])
        
        print(f"\n🏆 Most Accurate: {best['model']} ({best['accuracy']*100:.1f}%)")
        print(f"⚡ Fastest: {fastest['model']} ({fastest['avg_time_per_collision']:.2f}s/collision)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if best['model'] == fastest['model']:
            print(f"   Use {best['model']} - best accuracy and speed!")
        else:
            print(f"   For accuracy: {best['model']}")
            print(f"   For speed: {fastest['model']}")
            print(f"   For balance: Consider cost vs accuracy tradeoff")


def main():
    print("=== APE: Model Comparison Benchmark ===\n")
    
    benchmark = ModelBenchmark()
    
    # Test OpenAI models
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        models_to_test = [
            "gpt-4o-mini",
            # "gpt-4o",  # Uncomment if you want to test (more expensive)
        ]
        
        for model in models_to_test:
            try:
                benchmark.run_model(model, client)
            except Exception as e:
                print(f"Error testing {model}: {e}")
    
    # Test Anthropic models
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=anthropic_key)
        
        models_to_test = [
            "claude-3-5-haiku-20241022",
            # "claude-3-5-sonnet-20241022",  # Uncomment if you want to test (more expensive)
        ]
        
        for model in models_to_test:
            try:
                benchmark.run_model(model, client)
            except Exception as e:
                print(f"Error testing {model}: {e}")
    
    # Print comparison
    if benchmark.results:
        benchmark.print_comparison()
    else:
        print("No models tested. Check your API keys in .env")


if __name__ == "__main__":
    main()
