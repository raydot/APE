import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class CollisionMemory:
    """Stores a collision scenario and its outcome"""
    velocity_before: List[float]
    velocity_after: List[float]
    surface_normal: List[float]
    elasticity: float
    mass: float
    reasoning: str
    hit_count: int = 1
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class CollisionMemoryStore:
    """
    Memory system for caching collision scenarios.
    Uses similarity matching to retrieve relevant past collisions.
    """
    
    def __init__(self, similarity_threshold: float = 0.1):
        self.memories: Dict[str, CollisionMemory] = {}
        self.similarity_threshold = similarity_threshold
    
    def _create_key(self, velocity: np.ndarray, normal: np.ndarray, elasticity: float) -> str:
        """Create a hash key for a collision scenario"""
        # Round to reduce key space
        v_rounded = np.round(velocity, 1)
        n_rounded = np.round(normal, 2)
        e_rounded = round(elasticity, 2)
        
        key_str = f"{v_rounded.tolist()}_{n_rounded.tolist()}_{e_rounded}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _calculate_similarity(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        e1: float,
        e2: float
    ) -> float:
        """
        Calculate similarity between two collision scenarios.
        Returns 0-1, where 1 is identical.
        """
        # Velocity similarity (normalized by magnitude)
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        v_similarity = np.dot(v1_norm, v2_norm)
        
        # Normal similarity
        n_similarity = np.dot(n1, n2)
        
        # Elasticity similarity
        e_similarity = 1.0 - abs(e1 - e2)
        
        # Combined similarity (weighted average)
        similarity = (v_similarity * 0.5 + n_similarity * 0.3 + e_similarity * 0.2)
        
        return max(0, similarity)
    
    def store(
        self,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
        surface_normal: np.ndarray,
        elasticity: float,
        mass: float,
        reasoning: str
    ):
        """Store a collision outcome in memory"""
        key = self._create_key(velocity_before, surface_normal, elasticity)
        
        if key in self.memories:
            # Update existing memory
            self.memories[key].hit_count += 1
        else:
            # Create new memory
            self.memories[key] = CollisionMemory(
                velocity_before=velocity_before.tolist(),
                velocity_after=velocity_after.tolist(),
                surface_normal=surface_normal.tolist(),
                elasticity=elasticity,
                mass=mass,
                reasoning=reasoning
            )
    
    def retrieve(
        self,
        velocity: np.ndarray,
        surface_normal: np.ndarray,
        elasticity: float
    ) -> Optional[CollisionMemory]:
        """
        Retrieve a similar collision from memory.
        Returns None if no similar collision found.
        """
        key = self._create_key(velocity, surface_normal, elasticity)
        
        # Exact match
        if key in self.memories:
            return self.memories[key]
        
        # Similarity search
        best_match = None
        best_similarity = 0.0
        
        for memory in self.memories.values():
            similarity = self._calculate_similarity(
                velocity,
                np.array(memory.velocity_before),
                surface_normal,
                np.array(memory.surface_normal),
                elasticity,
                memory.elasticity
            )
            
            if similarity > best_similarity and similarity >= (1.0 - self.similarity_threshold):
                best_similarity = similarity
                best_match = memory
        
        return best_match
    
    def save(self, filename: str):
        """Save memory store to JSON file"""
        data = {
            'memories': {k: v.to_dict() for k, v in self.memories.items()},
            'similarity_threshold': self.similarity_threshold
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[MEMORY] Saved {len(self.memories)} memories to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """Load memory store from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        store = cls(similarity_threshold=data.get('similarity_threshold', 0.1))
        store.memories = {
            k: CollisionMemory.from_dict(v)
            for k, v in data['memories'].items()
        }
        print(f"[MEMORY] Loaded {len(store.memories)} memories from {filename}")
        return store
    
    def get_stats(self) -> Dict:
        """Get statistics about the memory store"""
        if not self.memories:
            return {
                'total_memories': 0,
                'total_hits': 0,
                'avg_hits_per_memory': 0
            }
        
        total_hits = sum(m.hit_count for m in self.memories.values())
        
        return {
            'total_memories': len(self.memories),
            'total_hits': total_hits,
            'avg_hits_per_memory': total_hits / len(self.memories),
            'most_common': max(self.memories.values(), key=lambda m: m.hit_count).to_dict()
        }
    
    def print_stats(self):
        """Print human-readable statistics"""
        stats = self.get_stats()
        
        if stats['total_memories'] == 0:
            print("\n[MEMORY] No memories stored yet")
            return
        
        print(f"\n[MEMORY] Statistics:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total cache hits: {stats['total_hits']}")
        print(f"  Avg hits per memory: {stats['avg_hits_per_memory']:.1f}")
        print(f"  Cache hit rate: {(stats['total_hits'] / (stats['total_hits'] + stats['total_memories']) * 100):.1f}%")


class MemoryAugmentedAgent:
    """
    Wrapper that adds memory capabilities to any agent.
    Checks memory before calling LLM, stores results after.
    """
    
    def __init__(self, agent, memory_store: CollisionMemoryStore):
        self.agent = agent
        self.memory = memory_store
        self.cache_hits = 0
        self.cache_misses = 0
    
    def handle_collision(self, velocity: np.ndarray, surface_normal: np.ndarray, elasticity: float) -> Tuple[np.ndarray, str]:
        """
        Handle collision with memory caching.
        Returns (new_velocity, reasoning)
        """
        # Try to retrieve from memory
        memory = self.memory.retrieve(velocity, surface_normal, elasticity)
        
        if memory:
            self.cache_hits += 1
            print(f"[MEMORY] Cache HIT! Using stored result (hit count: {memory.hit_count})")
            return np.array(memory.velocity_after), f"[CACHED] {memory.reasoning}"
        
        # Cache miss - need to calculate (would call LLM in real scenario)
        self.cache_misses += 1
        print(f"[MEMORY] Cache MISS - would call LLM here")
        
        # For demo, use simple calculation
        from ape.tools import calculate_elastic_collision
        result = calculate_elastic_collision(
            velocity=velocity.tolist(),
            surface_normal=surface_normal.tolist(),
            elasticity=elasticity
        )
        
        new_velocity = np.array(result['new_velocity'])
        reasoning = result['reasoning']
        
        # Store in memory
        self.memory.store(
            velocity_before=velocity,
            velocity_after=new_velocity,
            surface_normal=surface_normal,
            elasticity=elasticity,
            mass=1.0,
            reasoning=reasoning
        )
        
        return new_velocity, reasoning
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cost_savings': f"${self.cache_hits * 0.002:.4f}"  # Assuming $0.002 per LLM call
        }
