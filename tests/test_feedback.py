import pytest
import numpy as np
from ape.learning import FeedbackGenerator, PhysicsExperience


@pytest.fixture
def feedback_generator():
    """Create a FeedbackGenerator with default tolerance"""
    return FeedbackGenerator(tolerance=0.05)


@pytest.mark.unit
def test_feedback_generator_initialization():
    """Test FeedbackGenerator initializes with tolerance"""
    generator = FeedbackGenerator(tolerance=0.1)
    assert generator.tolerance == 0.1


@pytest.mark.unit
def test_generate_feedback_correct_prediction(feedback_generator):
    """Test feedback for correct prediction"""
    predicted = np.array([3.0, 4.0])
    actual = np.array([3.01, 4.01])
    scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    reasoning = "Test reasoning"
    
    feedback = feedback_generator.generate_feedback(
        predicted, actual, scenario, reasoning
    )
    
    assert feedback['was_correct'] == True
    assert feedback['error_magnitude'] < 0.05


@pytest.mark.unit
def test_generate_feedback_incorrect_prediction(feedback_generator):
    """Test feedback for incorrect prediction"""
    predicted = np.array([3.0, 4.0])
    actual = np.array([1.0, 2.0])
    scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    reasoning = "Test reasoning"
    
    feedback = feedback_generator.generate_feedback(
        predicted, actual, scenario, reasoning
    )
    
    assert feedback['was_correct'] == False
    assert feedback['error_magnitude'] > 0.05
    assert 'error_analysis' in feedback
    assert 'guidance' in feedback


@pytest.mark.unit
def test_error_analysis_sign_error(feedback_generator):
    """Test detecting sign error (completely wrong direction)"""
    predicted = np.array([5.0, 0.0])
    actual = np.array([-5.0, 0.0])
    scenario = {
        'collision_normal': [1.0, 0.0],
        'my_elasticity': 1.0
    }
    
    analysis = feedback_generator._analyze_error_type(predicted, actual, scenario)
    
    assert analysis['error_type'] == 'sign_error'


@pytest.mark.unit
def test_error_analysis_magnitude_error(feedback_generator):
    """Test detecting magnitude error (direction ok, speed wrong)"""
    predicted = np.array([3.0, 0.0])
    actual = np.array([6.0, 0.0])
    scenario = {
        'collision_normal': [1.0, 0.0],
        'my_elasticity': 1.0
    }
    
    analysis = feedback_generator._analyze_error_type(predicted, actual, scenario)
    
    assert analysis['error_type'] == 'magnitude_error'


@pytest.mark.unit
def test_error_analysis_direction_error(feedback_generator):
    """Test detecting direction error (angle wrong)"""
    predicted = np.array([5.0, 0.0])
    actual = np.array([3.0, 4.0])
    scenario = {
        'collision_normal': [1.0, 0.0],
        'my_elasticity': 1.0
    }
    
    analysis = feedback_generator._analyze_error_type(predicted, actual, scenario)
    
    assert analysis['error_type'] == 'direction_error'


@pytest.mark.unit
def test_create_experience_from_collision(feedback_generator):
    """Test creating PhysicsExperience from collision data"""
    scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    
    experience = feedback_generator.create_experience_from_collision(
        agent_id="ball-001",
        timestamp=1.0,
        scenario=scenario,
        predicted_my_velocity=np.array([-2.5, 0.0]),
        predicted_other_velocity=np.array([4.5, 0.0]),
        reasoning="Test reasoning",
        actual_my_velocity=np.array([-3.0, 0.0]),
        actual_other_velocity=np.array([5.0, 0.0]),
        model_used="test-model",
        similar_experiences_used=0
    )
    
    assert isinstance(experience, PhysicsExperience)
    assert experience.agent_id == "ball-001"
    assert experience.timestamp == 1.0
    assert experience.prediction_error > 0
    assert experience.momentum_error >= 0
    assert experience.energy_error >= 0


@pytest.mark.unit
def test_momentum_error_calculation(feedback_generator):
    """Test momentum error is calculated correctly"""
    scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    
    # Perfect prediction should have zero momentum error
    experience = feedback_generator.create_experience_from_collision(
        agent_id="ball-001",
        timestamp=1.0,
        scenario=scenario,
        predicted_my_velocity=np.array([-3.0, 0.0]),
        predicted_other_velocity=np.array([5.0, 0.0]),
        reasoning="Perfect prediction",
        actual_my_velocity=np.array([-3.0, 0.0]),
        actual_other_velocity=np.array([5.0, 0.0]),
        model_used="test-model",
        similar_experiences_used=0
    )
    
    assert experience.momentum_error == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_energy_error_calculation(feedback_generator):
    """Test energy error is calculated correctly"""
    scenario = {
        'my_velocity': [5.0, 0.0],
        'my_mass': 1.0,
        'my_elasticity': 1.0,
        'other_velocity': [-3.0, 0.0],
        'other_mass': 1.0,
        'other_elasticity': 1.0,
        'collision_normal': [1.0, 0.0]
    }
    
    # Perfect prediction should have zero energy error
    experience = feedback_generator.create_experience_from_collision(
        agent_id="ball-001",
        timestamp=1.0,
        scenario=scenario,
        predicted_my_velocity=np.array([-3.0, 0.0]),
        predicted_other_velocity=np.array([5.0, 0.0]),
        reasoning="Perfect prediction",
        actual_my_velocity=np.array([-3.0, 0.0]),
        actual_other_velocity=np.array([5.0, 0.0]),
        model_used="test-model",
        similar_experiences_used=0
    )
    
    assert experience.energy_error == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_guidance_generation(feedback_generator):
    """Test guidance is generated for different error types"""
    predicted = np.array([5.0, 0.0])
    actual = np.array([-5.0, 0.0])
    scenario = {
        'collision_normal': [1.0, 0.0],
        'my_elasticity': 1.0
    }
    reasoning = "Test reasoning"
    
    feedback = feedback_generator.generate_feedback(
        predicted, actual, scenario, reasoning
    )
    
    assert 'guidance' in feedback
    assert len(feedback['guidance']) > 0
    assert isinstance(feedback['guidance'], str)
