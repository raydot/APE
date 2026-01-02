import pytest
import numpy as np
import tempfile
import shutil
from ape.learning import ExperienceStore, PhysicsExperience


@pytest.fixture
def temp_qdrant_path():
    """Create temporary directory for Qdrant storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def experience_store(temp_qdrant_path):
    """Create ExperienceStore with temporary storage"""
    return ExperienceStore(
        collection_name="test_collection",
        qdrant_path=temp_qdrant_path
    )


@pytest.fixture
def sample_experience():
    """Create a sample PhysicsExperience"""
    return PhysicsExperience(
        scenario_id="test_001",
        timestamp=1.0,
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
        reasoning="Test reasoning",
        actual_my_velocity=[-3.0, 0.0],
        actual_other_velocity=[5.0, 0.0],
        prediction_error=0.5,
        momentum_error=0.1,
        energy_error=0.05,
        was_correct=False,
        similar_experiences_used=0,
        model_used="test-model"
    )


@pytest.mark.unit
def test_experience_store_initialization(temp_qdrant_path):
    """Test ExperienceStore initializes correctly"""
    store = ExperienceStore(
        collection_name="test_init",
        qdrant_path=temp_qdrant_path
    )
    
    assert store.collection_name == "test_init"
    assert store.encoder is not None


@pytest.mark.unit
def test_store_experience(experience_store, sample_experience):
    """Test storing an experience"""
    exp_id = experience_store.store_experience(sample_experience)
    
    assert exp_id is not None
    assert isinstance(exp_id, str)
    assert len(exp_id) > 0


@pytest.mark.unit
def test_retrieve_similar_experiences(experience_store, sample_experience):
    """Test retrieving similar experiences"""
    # Store an experience
    experience_store.store_experience(sample_experience)
    
    # Query for similar scenarios
    query_scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    
    similar = experience_store.retrieve_similar_experiences(
        query_scenario=query_scenario,
        limit=5,
        min_score=0.5
    )
    
    assert len(similar) > 0
    assert isinstance(similar[0], PhysicsExperience)


@pytest.mark.unit
def test_retrieve_by_agent_id(experience_store):
    """Test filtering experiences by agent_id"""
    # Store experiences for different agents
    exp1 = PhysicsExperience(
        scenario_id="test_001",
        timestamp=1.0,
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
    
    exp2 = PhysicsExperience(
        scenario_id="test_002",
        timestamp=2.0,
        agent_id="ball-002",
        my_velocity=[3.0, 0.0],
        my_mass=1.0,
        my_elasticity=1.0,
        other_velocity=[-2.0, 0.0],
        other_mass=1.0,
        other_elasticity=1.0,
        collision_normal=[1.0, 0.0],
        predicted_my_velocity=[-1.5, 0.0],
        predicted_other_velocity=[2.5, 0.0],
        reasoning="Test",
        actual_my_velocity=[-2.0, 0.0],
        actual_other_velocity=[3.0, 0.0],
        prediction_error=0.3,
        momentum_error=0.05,
        energy_error=0.02,
        was_correct=True,
        similar_experiences_used=0,
        model_used="test-model"
    )
    
    experience_store.store_experience(exp1)
    experience_store.store_experience(exp2)
    
    query_scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    
    # Retrieve only ball-001's experiences
    similar = experience_store.retrieve_similar_experiences(
        query_scenario=query_scenario,
        limit=5,
        min_score=0.0,
        agent_id="ball-001"
    )
    
    assert len(similar) == 1
    assert similar[0].agent_id == "ball-001"


@pytest.mark.unit
def test_get_agent_statistics(experience_store):
    """Test getting statistics for an agent"""
    # Store multiple experiences
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
            prediction_error=0.5 * (i + 1),
            momentum_error=0.1,
            energy_error=0.05,
            was_correct=(i % 2 == 0),
            similar_experiences_used=0,
            model_used="test-model"
        )
        experience_store.store_experience(exp)
    
    stats = experience_store.get_agent_statistics("ball-001")
    
    assert stats['total_experiences'] == 5
    assert stats['overall_accuracy'] == 0.6  # 3 out of 5 correct
    assert 'avg_prediction_error' in stats
    assert 'recent_accuracy' in stats


@pytest.mark.unit
def test_get_best_experiences(experience_store):
    """Test getting best predictions"""
    # Store experiences with different errors
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
            prediction_error=float(i),
            momentum_error=0.1,
            energy_error=0.05,
            was_correct=False,
            similar_experiences_used=0,
            model_used="test-model"
        )
        experience_store.store_experience(exp)
    
    best = experience_store.get_best_experiences(agent_id="ball-001", limit=3)
    
    assert len(best) == 3
    # Should be sorted by error (lowest first)
    assert best[0].prediction_error <= best[1].prediction_error
    assert best[1].prediction_error <= best[2].prediction_error


@pytest.mark.unit
def test_get_worst_experiences(experience_store):
    """Test getting worst predictions"""
    # Store experiences with different errors
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
            prediction_error=float(i),
            momentum_error=0.1,
            energy_error=0.05,
            was_correct=False,
            similar_experiences_used=0,
            model_used="test-model"
        )
        experience_store.store_experience(exp)
    
    worst = experience_store.get_worst_experiences(agent_id="ball-001", limit=3)
    
    assert len(worst) == 3
    # Should be sorted by error (highest first)
    assert worst[0].prediction_error >= worst[1].prediction_error
    assert worst[1].prediction_error >= worst[2].prediction_error


@pytest.mark.unit
def test_clear_collection(experience_store, sample_experience):
    """Test clearing all experiences"""
    # Store an experience
    experience_store.store_experience(sample_experience)
    
    # Clear collection
    experience_store.clear_collection()
    
    # Should have no experiences
    stats = experience_store.get_agent_statistics("ball-001")
    assert stats['total_experiences'] == 0
