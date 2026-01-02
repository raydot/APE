import pytest
import numpy as np
import tempfile
import shutil
from ape.learning import LearningBallAgent, ExperienceStore, FeedbackGenerator
from ape.physics import WorldState, PhysicsState
from ape.events import SimpleEventBus
from ape.negotiation import VelocityProposal


@pytest.fixture
def temp_qdrant_path():
    """Create temporary directory for Qdrant storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def learning_components(temp_qdrant_path):
    """Create learning system components"""
    experience_store = ExperienceStore(
        collection_name="test_learning",
        qdrant_path=temp_qdrant_path
    )
    feedback_generator = FeedbackGenerator(tolerance=0.05)
    return experience_store, feedback_generator


@pytest.fixture
def learning_agent(learning_components, mock_openai_client):
    """Create a LearningBallAgent for testing"""
    experience_store, feedback_generator = learning_components
    event_bus = SimpleEventBus()
    world = WorldState(gravity=np.array([0.0, 0.0]))
    
    ball_state = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-001", ball_state)
    
    agent = LearningBallAgent(
        agent_id="ball-001",
        event_bus=event_bus,
        world_state=world,
        llm_client=mock_openai_client,
        model_name="test-model",
        experience_store=experience_store,
        feedback_generator=feedback_generator,
        learning_enabled=True,
        max_examples=3
    )
    
    return agent


@pytest.mark.integration
def test_learning_agent_initialization(learning_agent):
    """Test LearningBallAgent initializes correctly"""
    assert learning_agent.agent_id == "ball-001"
    assert learning_agent.learning_enabled is True
    assert learning_agent.max_examples == 3
    assert learning_agent.experience_store is not None
    assert learning_agent.feedback_generator is not None


@pytest.mark.integration
def test_learning_agent_propose_velocity_no_experiences(learning_agent):
    """Test proposing velocity with no past experiences"""
    collision_data = {
        'ball1_id': 'ball-001',
        'ball2_id': 'ball-002',
        'collision_normal': np.array([1.0, 0.0]),
        'overlap': 0.1
    }
    
    # Add other ball to world
    other_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    learning_agent.world_state.add_object("ball-002", other_state)
    
    proposal = learning_agent.propose_velocity(collision_data, "ball-002")
    
    assert isinstance(proposal, VelocityProposal)
    assert proposal.agent_id == "ball-001"
    assert len(proposal.proposed_velocity) == 2


@pytest.mark.integration
def test_learning_agent_record_outcome(learning_agent):
    """Test recording collision outcome"""
    collision_data = {
        'ball1_id': 'ball-001',
        'ball2_id': 'ball-002',
        'collision_normal': np.array([1.0, 0.0]),
        'overlap': 0.1
    }
    
    # Add other ball to world
    other_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    learning_agent.world_state.add_object("ball-002", other_state)
    
    # Make a proposal (creates pending prediction)
    proposal = learning_agent.propose_velocity(collision_data, "ball-002")
    
    # Record outcome
    collision_id = f"ball-001_ball-002_{learning_agent.world_state.time}"
    actual_my_velocity = np.array([-3.0, 0.0])
    actual_other_velocity = np.array([5.0, 0.0])
    
    experience = learning_agent.record_outcome(
        collision_id,
        actual_my_velocity,
        actual_other_velocity
    )
    
    assert experience is not None
    assert experience.agent_id == "ball-001"
    assert experience.prediction_error >= 0


@pytest.mark.integration
def test_learning_agent_retrieves_experiences(learning_agent):
    """Test that agent retrieves past experiences"""
    # Store some experiences first
    from ape.learning import PhysicsExperience
    
    for i in range(3):
        exp = PhysicsExperience(
            scenario_id=f"test_{i:03d}",
            timestamp=float(i),
            agent_id="ball-001",
            my_velocity=[5.0, 0.0],
            my_mass=1.0,
            my_elasticity=1.0,
            other_velocity=[-3.0, 0.0],
            other_mass=1.0,
            other_elasticity=1.0,
            collision_normal=[1.0, 0.0],
            predicted_my_velocity=[-2.5, 0.0],
            predicted_other_velocity=[4.5, 0.0],
            reasoning="Test",
            actual_my_velocity=[-3.0, 0.0],
            actual_other_velocity=[5.0, 0.0],
            prediction_error=0.5,
            momentum_error=0.1,
            energy_error=0.05,
            was_correct=False,
            similar_experiences_used=0,
            model_used="test-model"
        )
        learning_agent.experience_store.store_experience(exp)
    
    collision_data = {
        'ball1_id': 'ball-001',
        'ball2_id': 'ball-002',
        'collision_normal': np.array([1.0, 0.0]),
        'overlap': 0.1
    }
    
    # Add other ball to world
    other_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    learning_agent.world_state.add_object("ball-002", other_state)
    
    # Propose velocity (should retrieve experiences)
    proposal = learning_agent.propose_velocity(collision_data, "ball-002")
    
    # Check that proposal was made
    assert isinstance(proposal, VelocityProposal)


@pytest.mark.integration
def test_learning_agent_get_stats(learning_agent):
    """Test getting learning statistics"""
    # Store some experiences
    from ape.learning import PhysicsExperience
    
    for i in range(5):
        exp = PhysicsExperience(
            scenario_id=f"test_{i:03d}",
            timestamp=float(i),
            agent_id="ball-001",
            my_velocity=[5.0, 0.0],
            my_mass=1.0,
            my_elasticity=1.0,
            other_velocity=[-3.0, 0.0],
            other_mass=1.0,
            other_elasticity=1.0,
            collision_normal=[1.0, 0.0],
            predicted_my_velocity=[-2.5, 0.0],
            predicted_other_velocity=[4.5, 0.0],
            reasoning="Test",
            actual_my_velocity=[-3.0, 0.0],
            actual_other_velocity=[5.0, 0.0],
            prediction_error=0.5,
            momentum_error=0.1,
            energy_error=0.05,
            was_correct=(i % 2 == 0),
            similar_experiences_used=0,
            model_used="test-model"
        )
        learning_agent.experience_store.store_experience(exp)
    
    stats = learning_agent.get_learning_stats()
    
    assert stats['total_experiences'] == 5
    assert 'overall_accuracy' in stats
    assert 'recent_accuracy' in stats
    assert 'avg_prediction_error' in stats


@pytest.mark.integration
def test_learning_agent_disabled_learning(learning_components, mock_openai_client):
    """Test agent with learning disabled"""
    experience_store, feedback_generator = learning_components
    event_bus = SimpleEventBus()
    world = WorldState(gravity=np.array([0.0, 0.0]))
    
    ball_state = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-001", ball_state)
    
    agent = LearningBallAgent(
        agent_id="ball-001",
        event_bus=event_bus,
        world_state=world,
        llm_client=mock_openai_client,
        model_name="test-model",
        experience_store=experience_store,
        feedback_generator=feedback_generator,
        learning_enabled=False,
        max_examples=3
    )
    
    collision_data = {
        'ball1_id': 'ball-001',
        'ball2_id': 'ball-002',
        'collision_normal': np.array([1.0, 0.0]),
        'overlap': 0.1
    }
    
    # Add other ball to world
    other_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    agent.world_state.add_object("ball-002", other_state)
    
    # Should still work but not retrieve experiences
    proposal = agent.propose_velocity(collision_data, "ball-002")
    
    assert isinstance(proposal, VelocityProposal)
