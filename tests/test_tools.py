import pytest
import numpy as np
from ape.tools import calculate_elastic_collision, calculate_two_body_collision


@pytest.mark.unit
def test_elastic_collision_wall_bounce():
    """Test elastic collision with wall (surface reflection)"""
    velocity = [5.0, 3.0]
    surface_normal = [1.0, 0.0]
    elasticity = 1.0
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    # X component should reverse, Y unchanged
    assert result['new_velocity'][0] == pytest.approx(-5.0)
    assert result['new_velocity'][1] == pytest.approx(3.0)


@pytest.mark.unit
def test_two_body_collision_equal_mass():
    """Test two-body collision between equal mass objects"""
    m1 = 1.0
    v1 = [5.0, 0.0]
    m2 = 1.0
    v2 = [-3.0, 0.0]
    e = 1.0
    normal = [1.0, 0.0]
    
    result = calculate_two_body_collision(m1, v1, m2, v2, normal, e)
    
    # For equal masses and perfect elasticity, velocities should swap
    assert 'velocity1_after' in result
    assert 'velocity2_after' in result


@pytest.mark.unit
def test_two_body_momentum_conservation():
    """Test momentum is conserved in two-body collision"""
    m1 = 2.0
    v1 = [5.0, 2.0]
    m2 = 1.5
    v2 = [-3.0, 1.0]
    e = 1.0
    normal = [1.0, 0.0]
    
    momentum_before = np.array(v1) * m1 + np.array(v2) * m2
    
    result = calculate_two_body_collision(m1, v1, m2, v2, normal, e)
    
    v1_after = np.array(result['velocity1_after'])
    v2_after = np.array(result['velocity2_after'])
    momentum_after = v1_after * m1 + v2_after * m2
    
    np.testing.assert_array_almost_equal(momentum_before, momentum_after)


@pytest.mark.unit
def test_inelastic_wall_collision():
    """Test inelastic collision with wall loses energy"""
    velocity = [4.0, 0.0]
    surface_normal = [1.0, 0.0]
    elasticity = 0.5
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    # Speed should be reduced
    speed_before = abs(velocity[0])
    speed_after = abs(result['new_velocity'][0])
    assert speed_after < speed_before


@pytest.mark.unit
def test_elastic_collision_perpendicular():
    """Test ball bouncing perpendicular to surface"""
    velocity = [5.0, 0.0]
    surface_normal = [1.0, 0.0]
    elasticity = 1.0
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    assert result['new_velocity'][0] == pytest.approx(-5.0)
    assert result['new_velocity'][1] == pytest.approx(0.0)


@pytest.mark.unit
def test_elastic_collision_angled():
    """Test ball bouncing at angle to surface"""
    velocity = [3.0, 4.0]
    surface_normal = [1.0, 0.0]
    elasticity = 1.0
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    # Normal component should reverse, tangential unchanged
    assert result['new_velocity'][0] == pytest.approx(-3.0)
    assert result['new_velocity'][1] == pytest.approx(4.0)


@pytest.mark.unit
def test_elastic_collision_inelastic():
    """Test inelastic surface reflection"""
    velocity = [5.0, 0.0]
    surface_normal = [1.0, 0.0]
    elasticity = 0.8
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    # Should be -0.8 * 5.0 = -4.0
    assert result['new_velocity'][0] == pytest.approx(-4.0)
    assert result['new_velocity'][1] == pytest.approx(0.0)


@pytest.mark.unit
def test_two_body_stationary_target():
    """Test collision with stationary object"""
    m1 = 1.0
    v1 = [5.0, 0.0]
    m2 = 1.0
    v2 = [0.0, 0.0]
    e = 1.0
    normal = [1.0, 0.0]
    
    result = calculate_two_body_collision(m1, v1, m2, v2, normal, e)
    
    # For equal masses, velocities should swap
    assert result['velocity1_after'][0] == pytest.approx(0.0, abs=0.1)
    assert result['velocity2_after'][0] == pytest.approx(5.0, abs=0.1)


@pytest.mark.unit
def test_two_body_different_masses():
    """Test collision with different masses"""
    m1 = 2.0
    v1 = [4.0, 0.0]
    m2 = 1.0
    v2 = [0.0, 0.0]
    e = 1.0
    normal = [1.0, 0.0]
    
    result = calculate_two_body_collision(m1, v1, m2, v2, normal, e)
    
    v1_after = result['velocity1_after']
    v2_after = result['velocity2_after']
    
    # Heavier object should continue forward but slower
    assert v1_after[0] > 0
    assert v1_after[0] < v1[0]
    
    # Lighter object should move faster
    assert v2_after[0] > 0


@pytest.mark.unit
def test_elastic_collision_2d():
    """Test 2D collision preserves tangential components"""
    velocity = [5.0, 3.0]
    surface_normal = [1.0, 0.0]
    elasticity = 1.0
    
    result = calculate_elastic_collision(velocity, surface_normal, elasticity)
    
    # Y component (tangential) should be unchanged
    assert result['new_velocity'][1] == pytest.approx(3.0)
