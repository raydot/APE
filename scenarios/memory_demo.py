import numpy as np
from ape.memory import CollisionMemoryStore, MemoryAugmentedAgent


def main():
    print("=== APE: Memory System Demo ===\n")
    
    # Create memory store
    memory = CollisionMemoryStore(similarity_threshold=0.1)
    
    # Create a mock agent (in real scenario, this would be BallAgent)
    class MockAgent:
        pass
    
    agent = MemoryAugmentedAgent(MockAgent(), memory)
    
    print("--- Scenario 1: First Collision ---")
    print("Ball hits floor at -10 m/s with elasticity 0.8")
    
    velocity = np.array([0.0, -10.0])
    normal = np.array([0.0, 1.0])
    elasticity = 0.8
    
    new_vel, reasoning = agent.handle_collision(velocity, normal, elasticity)
    print(f"Result: {new_vel}")
    print(f"Reasoning: {reasoning}\n")
    
    print("--- Scenario 2: Similar Collision (Cache Hit) ---")
    print("Ball hits floor at -10.1 m/s with elasticity 0.8 (very similar)")
    
    velocity2 = np.array([0.0, -10.1])
    new_vel2, reasoning2 = agent.handle_collision(velocity2, normal, elasticity)
    print(f"Result: {new_vel2}")
    print(f"Reasoning: {reasoning2}\n")
    
    print("--- Scenario 3: Exact Same Collision (Cache Hit) ---")
    print("Ball hits floor at -10 m/s with elasticity 0.8 (exact match)")
    
    new_vel3, reasoning3 = agent.handle_collision(velocity, normal, elasticity)
    print(f"Result: {new_vel3}")
    print(f"Reasoning: {reasoning3}\n")
    
    print("--- Scenario 4: Different Elasticity (Cache Miss) ---")
    print("Ball hits floor at -10 m/s with elasticity 0.5 (different)")
    
    elasticity2 = 0.5
    new_vel4, reasoning4 = agent.handle_collision(velocity, normal, elasticity2)
    print(f"Result: {new_vel4}")
    print(f"Reasoning: {reasoning4}\n")
    
    print("--- Scenario 5: Angled Collision (Cache Miss) ---")
    print("Ball hits floor at angle [5, -10] m/s with elasticity 0.8")
    
    velocity3 = np.array([5.0, -10.0])
    new_vel5, reasoning5 = agent.handle_collision(velocity3, normal, elasticity)
    print(f"Result: {new_vel5}")
    print(f"Reasoning: {reasoning5}\n")
    
    # Show cache statistics
    print("="*60)
    print("CACHE PERFORMANCE")
    print("="*60)
    stats = agent.get_cache_stats()
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"Estimated cost savings: {stats['cost_savings']}")
    
    memory.print_stats()
    
    # Save memory for future use
    print("\n--- Saving Memory ---")
    memory.save('collision_memory.json')
    
    # Demo: Load memory
    print("\n--- Loading Memory ---")
    loaded_memory = CollisionMemoryStore.load('collision_memory.json')
    
    # Test with loaded memory
    print("\n--- Testing Loaded Memory ---")
    agent2 = MemoryAugmentedAgent(MockAgent(), loaded_memory)
    new_vel6, reasoning6 = agent2.handle_collision(velocity, normal, elasticity)
    print(f"Result from loaded memory: {new_vel6}")
    print(f"Cache hit: {reasoning6.startswith('[CACHED]')}")
    
    print("\n💡 Benefits of Memory System:")
    print("  1. Reduces LLM API calls (saves cost)")
    print("  2. Faster response time (no network latency)")
    print("  3. Consistent results for similar scenarios")
    print("  4. Can persist across simulation runs")
    print("  5. Enables learning from experience")


if __name__ == "__main__":
    main()
