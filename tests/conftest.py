import pytest
import numpy as np
from ape.physics import WorldState, PhysicsState
from ape.events import SimpleEventBus


@pytest.fixture
def world():
    """Create a WorldState with standard gravity"""
    return WorldState(gravity=np.array([0.0, -9.8]))


@pytest.fixture
def world_no_gravity():
    """Create a WorldState with no gravity"""
    return WorldState(gravity=np.array([0.0, 0.0]))


@pytest.fixture
def event_bus():
    """Create a SimpleEventBus"""
    return SimpleEventBus()


@pytest.fixture
def ball_state():
    """Create a standard PhysicsState for a ball"""
    return PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )


@pytest.fixture
def mock_llm_client():
    """Factory for creating mock LLM clients with custom responses"""
    class MockLLM:
        def __init__(self, response):
            self.response = response
        
        class Chat:
            def __init__(self, response):
                self.completions = self
                self.response = response
            
            def create(self, **kwargs):
                class Response:
                    def __init__(self, content):
                        self.choices = [type('obj', (), {
                            'message': type('obj', (), {'content': content})()
                        })()]
                return Response(self.response)
        
        @property
        def chat(self):
            return self.Chat(self.response)
    
    return MockLLM


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns valid JSON responses"""
    import json
    
    class MockOpenAI:
        def __init__(self):
            self.chat = self.Chat()
        
        class Chat:
            def __init__(self):
                self.completions = self
            
            def create(self, **kwargs):
                response_data = {
                    "reasoning": "Test reasoning",
                    "your_velocity_after": [1.0, 2.0],
                    "confidence": 0.9
                }
                
                class Response:
                    def __init__(self, content):
                        self.choices = [type('obj', (), {
                            'message': type('obj', (), {'content': content})()
                        })()]
                
                return Response(json.dumps(response_data))
    
    return MockOpenAI()
