import numpy as np
from ape.llm_router import LLMRouter, TaskComplexity


def main():
    print("=== APE: LLM Router Demo ===\n")
    print("Intelligent model selection based on task complexity\n")
    
    router = LLMRouter()
    
    print("--- Registered Models ---")
    for model_name, config in router.models.items():
        print(f"\n{model_name} ({config.provider}):")
        print(f"  Cost: ${config.cost_per_1k_tokens:.5f} per 1K tokens")
        print(f"  Capability: {config.capability_score:.2f}")
        print(f"  Speed: {config.speed_score:.2f}")
    
    print("\n" + "="*80)
    print("TASK ROUTING EXAMPLES")
    print("="*80)
    
    # Example 1: Simple task
    print("\n--- Example 1: Simple Horizontal Floor Collision ---")
    velocity = np.array([0.0, -5.0])
    normal = np.array([0.0, 1.0])
    elasticity = 0.8
    
    model, complexity = router.route_task(
        velocity=velocity,
        surface_normal=normal,
        elasticity=elasticity,
        context={'first_collision': False},
        optimize_for="balanced"
    )
    
    print(f"Velocity: {velocity}")
    print(f"Surface: Horizontal floor")
    print(f"Assessed complexity: {complexity.value}")
    print(f"Selected model: {model}")
    print(f"Reasoning: Simple vertical bounce, use cheap model")
    
    # Example 2: Moderate task
    print("\n--- Example 2: Moderate - Angled Surface ---")
    velocity = np.array([3.0, -7.0])
    normal = np.array([1.0, 1.0])  # 45 degree angle
    elasticity = 0.7
    
    model, complexity = router.route_task(
        velocity=velocity,
        surface_normal=normal,
        elasticity=elasticity,
        context={'first_collision': True},
        optimize_for="balanced"
    )
    
    print(f"Velocity: {velocity}")
    print(f"Surface: 45° angled wall")
    print(f"Assessed complexity: {complexity.value}")
    print(f"Selected model: {model}")
    print(f"Reasoning: Angled collision requires vector decomposition")
    
    # Example 3: Complex task
    print("\n--- Example 3: Complex - High Speed Angled Multi-Agent ---")
    velocity = np.array([12.0, -15.0])
    normal = np.array([0.7, 0.7])  # Arbitrary angle
    elasticity = 0.65
    
    model, complexity = router.route_task(
        velocity=velocity,
        surface_normal=normal,
        elasticity=elasticity,
        context={'first_collision': True, 'multiple_agents': True},
        optimize_for="balanced"
    )
    
    print(f"Velocity: {velocity}")
    print(f"Surface: Complex angled surface")
    print(f"Assessed complexity: {complexity.value}")
    print(f"Selected model: {model}")
    print(f"Reasoning: High speed + angle + multi-agent = use capable model")
    
    # Example 4: Cost optimization
    print("\n--- Example 4: Same Task, Different Optimization ---")
    velocity = np.array([3.0, -7.0])
    normal = np.array([1.0, 1.0])
    elasticity = 0.7
    
    print("\nOptimize for COST:")
    model_cost, _ = router.route_task(velocity, normal, elasticity, optimize_for="cost")
    print(f"  Selected: {model_cost}")
    
    print("\nOptimize for CAPABILITY:")
    model_cap, _ = router.route_task(velocity, normal, elasticity, optimize_for="capability")
    print(f"  Selected: {model_cap}")
    
    print("\nOptimize for SPEED:")
    model_speed, _ = router.route_task(velocity, normal, elasticity, optimize_for="speed")
    print(f"  Selected: {model_speed}")
    
    # Simulate usage and show stats
    print("\n" + "="*80)
    print("SIMULATED USAGE SCENARIO")
    print("="*80)
    
    # Simulate 100 collisions with varying complexity
    scenarios = [
        # 60% simple (horizontal floor bounces)
        *[(np.array([0.0, -5.0]), np.array([0.0, 1.0]), 0.8, {'first_collision': False})] * 60,
        # 30% moderate (angled surfaces)
        *[(np.array([3.0, -7.0]), np.array([1.0, 1.0]), 0.7, {'first_collision': True})] * 30,
        # 10% complex (high speed, angled, multi-agent)
        *[(np.array([12.0, -15.0]), np.array([0.7, 0.7]), 0.65, {'first_collision': True, 'multiple_agents': True})] * 10,
    ]
    
    print(f"\nSimulating {len(scenarios)} collision tasks...")
    
    for velocity, normal, elasticity, context in scenarios:
        model, complexity = router.route_task(velocity, normal, elasticity, context, optimize_for="balanced")
        # Simulate token usage (estimate)
        tokens = 150 if complexity == TaskComplexity.SIMPLE else 200 if complexity == TaskComplexity.MODERATE else 300
        router.record_cost(model, tokens)
    
    router.print_stats()
    
    print("\n💡 Router Benefits:")
    print("  1. Automatic model selection based on task difficulty")
    print("  2. Cost optimization for simple tasks")
    print("  3. Capability when needed for complex tasks")
    print("  4. Flexible optimization strategies")
    print("  5. Usage tracking and cost analysis")
    print("  6. Easy to add new models")
    
    print("\n📊 Cost Comparison:")
    stats = router.get_stats()
    print(f"  With routing: ${stats['total_cost']:.4f}")
    
    # Calculate if we used only expensive model
    expensive_cost = len(scenarios) * 300 * 0.003 / 1000
    print(f"  All premium model: ${expensive_cost:.4f}")
    
    savings = expensive_cost - stats['total_cost']
    savings_pct = (savings / expensive_cost * 100) if expensive_cost > 0 else 0
    print(f"  Savings: ${savings:.4f} ({savings_pct:.1f}%)")


if __name__ == "__main__":
    main()
