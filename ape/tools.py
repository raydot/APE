from typing import Callable, Any, Dict, Optional
import numpy as np
import json
class Tool:
    def __init__(self, name: str, description: str, func: Callable, schema: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema or {}
    
    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)
    
    def to_llm_schema(self) -> Dict:
        """Convert tool to LLM function calling schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema
        }
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> list[Dict]:
        """Get all tool schemas for LLM function calling"""
        return [tool.to_llm_schema() for tool in self._tools.values()]


# Physics calculation tools
def calculate_elastic_collision(
    velocity: list[float],
    surface_normal: list[float],
    elasticity: float
) -> Dict[str, Any]:
    """
    Calculate post-collision velocity for elastic collision.
    
    Args:
        velocity: Current velocity [vx, vy]
        surface_normal: Surface normal vector [nx, ny]
        elasticity: Coefficient of restitution (0-1)
    
    Returns:
        Dict with new_velocity and reasoning
    """
    v = np.array(velocity)
    n = np.array(surface_normal)
    n = n / np.linalg.norm(n)  # Normalize
    
    # Decompose velocity into parallel and perpendicular components
    v_perp = np.dot(v, n) * n
    v_parallel = v - v_perp
    
    # Perpendicular component reverses and scales by elasticity
    v_perp_new = -v_perp * elasticity
    
    # Parallel component unchanged (no friction)
    v_new = v_parallel + v_perp_new
    
    return {
        "new_velocity": v_new.tolist(),
        "reasoning": f"Elastic collision: perpendicular component reversed and scaled by {elasticity}, parallel component unchanged"
    }


def calculate_gravitational_force(
    mass: float,
    gravity: float = 9.8
) -> Dict[str, Any]:
    """
    Calculate gravitational force on an object.
    
    Args:
        mass: Object mass in kg
        gravity: Gravitational acceleration in m/s^2
    
    Returns:
        Dict with force vector and reasoning
    """
    force = np.array([0.0, -mass * gravity])
    
    return {
        "force": force.tolist(),
        "reasoning": f"Gravitational force: F = mg = {mass} * {gravity} = {mass * gravity}N downward"
    }


def calculate_two_body_collision(
    mass1: float,
    velocity1: list[float],
    mass2: float,
    velocity2: list[float],
    collision_normal: list[float],
    elasticity: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate post-collision velocities for two moving balls.
    
    Uses 1D elastic collision formula along collision normal,
    preserves tangential components (no friction).
    
    Args:
        mass1: Mass of ball 1 (kg)
        velocity1: Velocity of ball 1 before collision [vx, vy]
        mass2: Mass of ball 2 (kg)
        velocity2: Velocity of ball 2 before collision [vx, vy]
        collision_normal: Unit vector from ball1 to ball2
        elasticity: Coefficient of restitution (0-1)
    
    Returns:
        Dict with velocity1_after, velocity2_after, and conservation data
    """
    v1 = np.array(velocity1)
    v2 = np.array(velocity2)
    normal = np.array(collision_normal)
    
    # Ensure normal is unit vector
    normal = normal / np.linalg.norm(normal)
    
    # Decompose velocities into normal and tangential components
    v1_normal_scalar = np.dot(v1, normal)
    v2_normal_scalar = np.dot(v2, normal)
    
    v1_normal = v1_normal_scalar * normal
    v2_normal = v2_normal_scalar * normal
    
    v1_tangent = v1 - v1_normal
    v2_tangent = v2 - v2_normal
    
    # 1D elastic collision formula for normal components
    # v1' = ((m1-e*m2)*v1 + (1+e)*m2*v2) / (m1+m2)
    # v2' = ((m2-e*m1)*v2 + (1+e)*m1*v1) / (m1+m2)
    total_mass = mass1 + mass2
    
    v1n_after = ((mass1 - elasticity * mass2) * v1_normal_scalar +
                 (1 + elasticity) * mass2 * v2_normal_scalar) / total_mass
    
    v2n_after = ((mass2 - elasticity * mass1) * v2_normal_scalar +
                 (1 + elasticity) * mass1 * v1_normal_scalar) / total_mass
    
    # Convert back to vectors
    v1_normal_after = v1n_after * normal
    v2_normal_after = v2n_after * normal
    
    # Tangential components unchanged (no friction)
    velocity1_after = v1_normal_after + v1_tangent
    velocity2_after = v2_normal_after + v2_tangent
    
    # Calculate conservation quantities
    momentum_before = mass1 * v1 + mass2 * v2
    momentum_after = mass1 * velocity1_after + mass2 * velocity2_after
    
    energy_before = 0.5 * mass1 * np.dot(v1, v1) + 0.5 * mass2 * np.dot(v2, v2)
    energy_after = 0.5 * mass1 * np.dot(velocity1_after, velocity1_after) + \
                   0.5 * mass2 * np.dot(velocity2_after, velocity2_after)
    
    energy_loss_pct = 100 * (energy_before - energy_after) / energy_before if energy_before > 0 else 0
    
    reasoning = f"""Two-body elastic collision (e={elasticity}):
Ball 1: m={mass1}kg, v={velocity1} → v'={velocity1_after.tolist()}
Ball 2: m={mass2}kg, v={velocity2} → v'={velocity2_after.tolist()}
Collision normal: {normal.tolist()}

Conservation:
- Momentum before: {momentum_before.tolist()}
- Momentum after: {momentum_after.tolist()}
- Energy before: {energy_before:.3f} J
- Energy after: {energy_after:.3f} J
- Energy loss: {energy_loss_pct:.1f}%"""
    
    return {
        'velocity1_after': velocity1_after.tolist(),
        'velocity2_after': velocity2_after.tolist(),
        'momentum_before': momentum_before.tolist(),
        'momentum_after': momentum_after.tolist(),
        'energy_before': float(energy_before),
        'energy_after': float(energy_after),
        'reasoning': reasoning.strip()
    }


def create_physics_tool_registry() -> ToolRegistry:
    """Create a tool registry with standard physics calculation tools"""
    registry = ToolRegistry()
    
    # Elastic collision tool
    collision_tool = Tool(
        name="calculate_elastic_collision",
        description="Calculate the new velocity after an elastic collision with a surface",
        func=calculate_elastic_collision,
        schema={
            "type": "object",
            "properties": {
                "velocity": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Current velocity vector [vx, vy] in m/s"
                },
                "surface_normal": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Surface normal vector [nx, ny]"
                },
                "elasticity": {
                    "type": "number",
                    "description": "Coefficient of restitution (0-1)"
                }
            },
            "required": ["velocity", "surface_normal", "elasticity"]
        }
    )
    registry.register(collision_tool)
    
    # Gravity tool
    gravity_tool = Tool(
        name="calculate_gravitational_force",
        description="Calculate gravitational force on an object",
        func=calculate_gravitational_force,
        schema={
            "type": "object",
            "properties": {
                "mass": {
                    "type": "number",
                    "description": "Object mass in kg"
                },
                "gravity": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: 9.8)"
                }
            },
            "required": ["mass"]
        }
    )
    registry.register(gravity_tool)
    
    # Two-body collision tool
    two_body_tool = Tool(
        name="calculate_two_body_collision",
        description="Calculate velocities after elastic collision between two moving balls",
        func=calculate_two_body_collision,
        schema={
            "type": "object",
            "properties": {
                "mass1": {
                    "type": "number",
                    "description": "Mass of ball 1 in kg"
                },
                "velocity1": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Velocity of ball 1 before collision [vx, vy]"
                },
                "mass2": {
                    "type": "number",
                    "description": "Mass of ball 2 in kg"
                },
                "velocity2": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Velocity of ball 2 before collision [vx, vy]"
                },
                "collision_normal": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Unit vector from ball1 to ball2"
                },
                "elasticity": {
                    "type": "number",
                    "description": "Coefficient of restitution (0-1, default: 1.0)"
                }
            },
            "required": ["mass1", "velocity1", "mass2", "velocity2", "collision_normal"]
        }
    )
    registry.register(two_body_tool)
    
    return registry