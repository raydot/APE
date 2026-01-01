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
    
    return registry