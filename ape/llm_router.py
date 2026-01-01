import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    name: str
    provider: str  # "openai" or "anthropic"
    cost_per_1k_tokens: float
    capability_score: float  # 0-1, higher = more capable
    speed_score: float  # 0-1, higher = faster
    
    def get_cost_score(self) -> float:
        """Lower cost = higher score"""
        return 1.0 / (1.0 + self.cost_per_1k_tokens * 100)


class LLMRouter:
    """
    Intelligent router that selects the best LLM model based on task complexity.
    
    Strategy:
    - Simple tasks: Use cheapest, fastest model
    - Moderate tasks: Balance cost and capability
    - Complex tasks: Use most capable model
    """
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.usage_stats: Dict[str, int] = {}
        self.cost_stats: Dict[str, float] = {}
        
        self._register_default_models()
    
    def _register_default_models(self):
        """Register common models with their characteristics"""
        
        # OpenAI models
        self.register_model(ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            cost_per_1k_tokens=0.00015,  # $0.15 per 1M input tokens
            capability_score=0.7,
            speed_score=0.9
        ))
        
        self.register_model(ModelConfig(
            name="gpt-4o",
            provider="openai",
            cost_per_1k_tokens=0.0025,  # $2.50 per 1M input tokens
            capability_score=0.95,
            speed_score=0.7
        ))
        
        # Anthropic models
        self.register_model(ModelConfig(
            name="claude-3-5-haiku-20241022",
            provider="anthropic",
            cost_per_1k_tokens=0.0008,  # $0.80 per 1M input tokens
            capability_score=0.75,
            speed_score=0.85
        ))
        
        self.register_model(ModelConfig(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            cost_per_1k_tokens=0.003,  # $3.00 per 1M input tokens
            capability_score=0.98,
            speed_score=0.6
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a model configuration"""
        self.models[config.name] = config
        self.usage_stats[config.name] = 0
        self.cost_stats[config.name] = 0.0
    
    def assess_task_complexity(
        self,
        velocity: np.ndarray,
        surface_normal: np.ndarray,
        elasticity: float,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskComplexity:
        """
        Assess the complexity of a physics task.
        
        Factors:
        - Velocity magnitude (higher = more complex)
        - Surface angle (angled = more complex than horizontal)
        - Elasticity (edge cases like 0 or 1 are simpler)
        - Context (multiple agents, special conditions)
        """
        complexity_score = 0.0
        
        # Velocity complexity
        v_magnitude = np.linalg.norm(velocity)
        if v_magnitude > 10:
            complexity_score += 0.3
        elif v_magnitude > 5:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        # Surface angle complexity
        # Horizontal (normal = [0, 1]) is simplest
        normal_normalized = surface_normal / np.linalg.norm(surface_normal)
        angle_from_horizontal = abs(np.dot(normal_normalized, np.array([0, 1])))
        
        if angle_from_horizontal < 0.9:  # Significantly angled
            complexity_score += 0.3
        elif angle_from_horizontal < 0.99:  # Slightly angled
            complexity_score += 0.2
        else:  # Horizontal
            complexity_score += 0.1
        
        # Elasticity complexity
        if elasticity in [0.0, 1.0]:  # Edge cases are actually simpler
            complexity_score += 0.1
        else:
            complexity_score += 0.2
        
        # Context complexity
        if context:
            if context.get('multiple_agents', False):
                complexity_score += 0.2
            if context.get('first_collision', True):
                complexity_score += 0.1  # First collision is harder
            else:
                complexity_score += 0.05  # Subsequent collisions easier
        
        # Classify
        if complexity_score < 0.4:
            return TaskComplexity.SIMPLE
        elif complexity_score < 0.7:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.COMPLEX
    
    def select_model(
        self,
        complexity: TaskComplexity,
        provider_preference: Optional[str] = None,
        optimize_for: str = "balanced"  # "cost", "speed", "capability", "balanced"
    ) -> str:
        """
        Select the best model for a given task complexity.
        
        Args:
            complexity: Task complexity level
            provider_preference: Preferred provider ("openai" or "anthropic")
            optimize_for: Optimization strategy
        
        Returns:
            Model name to use
        """
        # Filter by provider if specified
        candidates = self.models.values()
        if provider_preference:
            candidates = [m for m in candidates if m.provider == provider_preference]
        
        if not candidates:
            raise ValueError(f"No models available for provider: {provider_preference}")
        
        # Score models based on complexity and optimization strategy
        scored_models = []
        
        for model in candidates:
            score = 0.0
            
            # Base score on complexity requirements
            if complexity == TaskComplexity.SIMPLE:
                # Prefer cheap and fast
                score += model.get_cost_score() * 0.5
                score += model.speed_score * 0.3
                score += model.capability_score * 0.2
            elif complexity == TaskComplexity.MODERATE:
                # Balance all factors
                score += model.get_cost_score() * 0.3
                score += model.speed_score * 0.3
                score += model.capability_score * 0.4
            else:  # COMPLEX
                # Prefer capability
                score += model.capability_score * 0.6
                score += model.speed_score * 0.2
                score += model.get_cost_score() * 0.2
            
            # Adjust based on optimization preference
            if optimize_for == "cost":
                score = model.get_cost_score()
            elif optimize_for == "speed":
                score = model.speed_score
            elif optimize_for == "capability":
                score = model.capability_score
            
            scored_models.append((score, model))
        
        # Select best model
        scored_models.sort(reverse=True, key=lambda x: x[0])
        selected_model = scored_models[0][1]
        
        # Track usage
        self.usage_stats[selected_model.name] += 1
        
        return selected_model.name
    
    def route_task(
        self,
        velocity: np.ndarray,
        surface_normal: np.ndarray,
        elasticity: float,
        context: Optional[Dict[str, Any]] = None,
        provider_preference: Optional[str] = None,
        optimize_for: str = "balanced"
    ) -> tuple[str, TaskComplexity]:
        """
        Complete routing: assess complexity and select model.
        
        Returns:
            (model_name, complexity)
        """
        complexity = self.assess_task_complexity(velocity, surface_normal, elasticity, context)
        model = self.select_model(complexity, provider_preference, optimize_for)
        
        return model, complexity
    
    def record_cost(self, model_name: str, tokens: int):
        """Record cost for a model usage"""
        if model_name in self.models:
            cost = (tokens / 1000) * self.models[model_name].cost_per_1k_tokens
            self.cost_stats[model_name] += cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage and cost statistics"""
        total_calls = sum(self.usage_stats.values())
        total_cost = sum(self.cost_stats.values())
        
        return {
            'total_calls': total_calls,
            'total_cost': total_cost,
            'usage_by_model': self.usage_stats.copy(),
            'cost_by_model': self.cost_stats.copy(),
            'avg_cost_per_call': total_cost / total_calls if total_calls > 0 else 0
        }
    
    def print_stats(self):
        """Print human-readable statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("LLM ROUTER STATISTICS")
        print("="*80)
        
        print(f"\nTotal calls: {stats['total_calls']}")
        print(f"Total cost: ${stats['total_cost']:.4f}")
        print(f"Avg cost per call: ${stats['avg_cost_per_call']:.4f}")
        
        print("\nUsage by model:")
        for model_name, count in stats['usage_by_model'].items():
            if count > 0:
                cost = stats['cost_by_model'][model_name]
                pct = (count / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
                print(f"  {model_name}: {count} calls ({pct:.1f}%), ${cost:.4f}")
        
        print("\n💡 Routing Strategy:")
        if stats['total_calls'] > 0:
            # Analyze which models were used most
            most_used = max(stats['usage_by_model'].items(), key=lambda x: x[1])
            print(f"  Most used: {most_used[0]} ({most_used[1]} calls)")
            
            # Cost efficiency
            if stats['avg_cost_per_call'] < 0.001:
                print(f"  ✓ Very cost-efficient routing")
            elif stats['avg_cost_per_call'] < 0.002:
                print(f"  ✓ Good cost efficiency")
            else:
                print(f"  ⚠️  Higher cost - using premium models frequently")
