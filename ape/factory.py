import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .agents import Agent, BallAgent, FloorAgent
from .events import SimpleEventBus
from .physics import WorldState, PhysicsState
from .tools import ToolRegistry, create_physics_tool_registry
from .observability import TraceCollector


@dataclass
class AgentTemplate:
    """Template for creating agents with specific configurations"""
    agent_type: str
    default_params: Dict[str, Any]
    physics_params: Dict[str, Any]
    description: str


class AgentFactory:
    """
    Factory for creating agents with standardized configurations.
    Supports templates, custom parameters, and automatic ID generation.
    """
    
    def __init__(
        self,
        event_bus: SimpleEventBus,
        world_state: WorldState,
        llm_client=None,
        model_name: str = "gpt-4o-mini",
        tool_registry: Optional[ToolRegistry] = None,
        trace_collector: Optional[TraceCollector] = None
    ):
        self.event_bus = event_bus
        self.world_state = world_state
        self.llm_client = llm_client
        self.model_name = model_name
        self.tool_registry = tool_registry or create_physics_tool_registry()
        self.trace_collector = trace_collector
        
        self.templates: Dict[str, AgentTemplate] = {}
        self.agent_counters: Dict[str, int] = {}
        
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register built-in agent templates"""
        
        # Ball templates
        self.register_template(AgentTemplate(
            agent_type="ball",
            default_params={},
            physics_params={
                "mass": 1.0,
                "elasticity": 0.8,
                "velocity": [0.0, 0.0]
            },
            description="Standard ball with 80% elasticity"
        ))
        
        self.register_template(AgentTemplate(
            agent_type="heavy_ball",
            default_params={},
            physics_params={
                "mass": 5.0,
                "elasticity": 0.6,
                "velocity": [0.0, 0.0]
            },
            description="Heavy ball with lower elasticity"
        ))
        
        self.register_template(AgentTemplate(
            agent_type="bouncy_ball",
            default_params={},
            physics_params={
                "mass": 0.5,
                "elasticity": 0.95,
                "velocity": [0.0, 0.0]
            },
            description="Light, highly elastic ball"
        ))
        
        # Floor template
        self.register_template(AgentTemplate(
            agent_type="floor",
            default_params={},
            physics_params={
                "mass": float('inf'),
                "elasticity": 1.0,
                "velocity": [0.0, 0.0]
            },
            description="Static floor surface"
        ))
        
        # Wall template
        self.register_template(AgentTemplate(
            agent_type="wall",
            default_params={},
            physics_params={
                "mass": float('inf'),
                "elasticity": 0.9,
                "velocity": [0.0, 0.0]
            },
            description="Static wall surface"
        ))
    
    def register_template(self, template: AgentTemplate):
        """Register a new agent template"""
        self.templates[template.agent_type] = template
        if template.agent_type not in self.agent_counters:
            self.agent_counters[template.agent_type] = 0
    
    def _generate_agent_id(self, agent_type: str) -> str:
        """Generate unique agent ID"""
        self.agent_counters[agent_type] += 1
        return f"{agent_type}-{self.agent_counters[agent_type]:03d}"
    
    def create_ball(
        self,
        position: np.ndarray,
        template: str = "ball",
        velocity: Optional[np.ndarray] = None,
        mass: Optional[float] = None,
        elasticity: Optional[float] = None,
        agent_id: Optional[str] = None
    ) -> BallAgent:
        """
        Create a ball agent with specified parameters.
        
        Args:
            position: Initial position [x, y]
            template: Template name to use
            velocity: Override template velocity
            mass: Override template mass
            elasticity: Override template elasticity
            agent_id: Custom agent ID (auto-generated if None)
        """
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        
        tmpl = self.templates[template]
        
        # Merge template params with overrides
        physics_params = tmpl.physics_params.copy()
        if velocity is not None:
            physics_params['velocity'] = velocity.tolist()
        if mass is not None:
            physics_params['mass'] = mass
        if elasticity is not None:
            physics_params['elasticity'] = elasticity
        
        # Create physics state
        state = PhysicsState(
            position=position,
            velocity=np.array(physics_params['velocity']),
            mass=physics_params['mass'],
            elasticity=physics_params['elasticity']
        )
        
        # Generate ID
        if agent_id is None:
            agent_id = self._generate_agent_id(template)
        
        # Add to world
        self.world_state.add_object(agent_id, state)
        
        # Create agent
        agent = BallAgent(
            agent_id,
            self.event_bus,
            self.world_state,
            self.llm_client,
            self.model_name
        )
        
        if self.trace_collector:
            self.trace_collector.log(
                level="info",
                agent_id=agent_id,
                event_type="agent_created",
                message=f"Created {template} agent",
                data={
                    "position": position.tolist(),
                    "mass": physics_params['mass'],
                    "elasticity": physics_params['elasticity']
                }
            )
        
        return agent
    
    def create_floor(
        self,
        position: np.ndarray,
        agent_id: Optional[str] = None
    ) -> FloorAgent:
        """Create a floor agent"""
        template = self.templates["floor"]
        
        state = PhysicsState(
            position=position,
            velocity=np.array(template.physics_params['velocity']),
            mass=template.physics_params['mass'],
            elasticity=template.physics_params['elasticity']
        )
        
        if agent_id is None:
            agent_id = self._generate_agent_id("floor")
        
        self.world_state.add_object(agent_id, state)
        
        agent = FloorAgent(agent_id, self.event_bus, self.world_state)
        
        return agent
    
    def create_wall(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        agent_id: Optional[str] = None
    ) -> FloorAgent:
        """
        Create a wall agent (uses FloorAgent with different normal).
        In future, this could be a specialized WallAgent class.
        """
        template = self.templates["wall"]
        
        state = PhysicsState(
            position=position,
            velocity=np.array(template.physics_params['velocity']),
            mass=template.physics_params['mass'],
            elasticity=template.physics_params['elasticity']
        )
        
        if agent_id is None:
            agent_id = self._generate_agent_id("wall")
        
        self.world_state.add_object(agent_id, state)
        
        # For now, use FloorAgent (in future, create WallAgent with normal parameter)
        agent = FloorAgent(agent_id, self.event_bus, self.world_state)
        
        return agent
    
    def create_from_template(
        self,
        template_name: str,
        position: np.ndarray,
        **overrides
    ) -> Agent:
        """
        Create an agent from a template with parameter overrides.
        
        Args:
            template_name: Name of template to use
            position: Initial position
            **overrides: Any physics parameters to override
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Route to appropriate creation method
        if "ball" in template_name:
            return self.create_ball(
                position=position,
                template=template_name,
                velocity=overrides.get('velocity'),
                mass=overrides.get('mass'),
                elasticity=overrides.get('elasticity'),
                agent_id=overrides.get('agent_id')
            )
        elif template_name == "floor":
            return self.create_floor(
                position=position,
                agent_id=overrides.get('agent_id')
            )
        elif template_name == "wall":
            return self.create_wall(
                position=position,
                normal=overrides.get('normal', np.array([1.0, 0.0])),
                agent_id=overrides.get('agent_id')
            )
        else:
            raise ValueError(f"No creation method for template: {template_name}")
    
    def list_templates(self) -> Dict[str, str]:
        """List all available templates with descriptions"""
        return {
            name: template.description
            for name, template in self.templates.items()
        }
    
    def get_template_params(self, template_name: str) -> Dict[str, Any]:
        """Get default parameters for a template"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        return self.templates[template_name].physics_params.copy()
