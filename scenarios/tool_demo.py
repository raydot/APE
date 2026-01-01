import numpy as np
from ape.tools import create_physics_tool_registry, calculate_elastic_collision


def main():
    print("=== APE: Tool Registry Demo ===\n")
    
    # Create tool registry
    registry = create_physics_tool_registry()
    
    print(f"Registered tools: {registry.list_tools()}\n")
    
    # Demo 1: Direct tool usage
    print("--- Demo 1: Direct Tool Usage ---")
    collision_tool = registry.get("calculate_elastic_collision")
    
    result = collision_tool.execute(
        velocity=[0.0, -9.898],
        surface_normal=[0.0, 1.0],
        elasticity=0.8
    )
    
    print(f"Input velocity: [0.0, -9.898] m/s")
    print(f"Surface normal: [0.0, 1.0]")
    print(f"Elasticity: 0.8")
    print(f"\nResult:")
    print(f"  New velocity: {result['new_velocity']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Demo 2: Tool schemas for LLM function calling
    print("\n--- Demo 2: Tool Schemas for LLM ---")
    schemas = registry.get_all_schemas()
    
    print(f"Available {len(schemas)} tools for LLM function calling:")
    for schema in schemas:
        print(f"\n  Tool: {schema['name']}")
        print(f"  Description: {schema['description']}")
        print(f"  Parameters: {list(schema['parameters']['properties'].keys())}")
    
    # Demo 3: Comparison with LLM reasoning
    print("\n--- Demo 3: Ground Truth vs LLM ---")
    print("This tool provides the 'correct' physics calculation.")
    print("In production, agents could:")
    print("  1. Use LLM to reason about physics (interpretable)")
    print("  2. Use tool for calculation (accurate)")
    print("  3. Compare results to validate LLM reasoning")
    print("  4. Fall back to tool if LLM makes errors")
    
    # Demo 4: Multiple collision scenarios
    print("\n--- Demo 4: Multiple Scenarios ---")
    scenarios = [
        {"velocity": [0.0, -10.0], "elasticity": 1.0, "name": "Perfect elastic"},
        {"velocity": [0.0, -10.0], "elasticity": 0.5, "name": "50% elastic"},
        {"velocity": [0.0, -10.0], "elasticity": 0.0, "name": "Perfectly inelastic"},
        {"velocity": [5.0, -10.0], "elasticity": 0.8, "name": "Angled collision"},
    ]
    
    for scenario in scenarios:
        result = calculate_elastic_collision(
            velocity=scenario["velocity"],
            surface_normal=[0.0, 1.0],
            elasticity=scenario["elasticity"]
        )
        print(f"\n{scenario['name']}:")
        print(f"  Before: {scenario['velocity']}")
        print(f"  After:  {result['new_velocity']}")


if __name__ == "__main__":
    main()
