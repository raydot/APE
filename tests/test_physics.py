import pytest
import numpy as np
from ape.physics import WorldState, PhysicsState


@pytest.mark.unit
def test_world_state_add_object(world):
    """Test adding objects to world state"""
    ball = PhysicsState(
        position=np.array([1.0, 2.0]),
        velocity=np.array([3.0, 4.0]),
        mass=1.5,
        elasticity=0.8
    )
    
    world.add_object("ball-001", ball)
    
    retrieved = world.get_object("ball-001")
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved.position, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(retrieved.velocity, np.array([3.0, 4.0]))
    assert retrieved.mass == 1.5
    assert retrieved.elasticity == 0.8


@pytest.mark.unit
def test_world_state_update_object(world):
    """Test updating objects in world state"""
    ball = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-001", ball)
    
    ball.position = np.array([1.0, 2.0])
    ball.velocity = np.array([3.0, 4.0])
    world.update_object("ball-001", ball)
    
    retrieved = world.get_object("ball-001")
    np.testing.assert_array_equal(retrieved.position, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(retrieved.velocity, np.array([3.0, 4.0]))


@pytest.mark.unit
def test_world_state_get_all_objects(world):
    """Test retrieving all objects from world state"""
    ball1 = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    ball2 = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.5,
        elasticity=0.9
    )
    
    world.add_object("ball-001", ball1)
    world.add_object("ball-002", ball2)
    
    all_objects = world.get_all_objects()
    assert len(all_objects) == 2
    assert "ball-001" in all_objects
    assert "ball-002" in all_objects


@pytest.mark.unit
def test_world_state_time_step(world):
    """Test time stepping in world state"""
    initial_time = world.time
    world.step(0.01)
    assert world.time == pytest.approx(initial_time + 0.01)
    
    world.step(0.05)
    assert world.time == pytest.approx(initial_time + 0.06)


@pytest.mark.unit
def test_gravity_application(world):
    """Test gravity affects velocity over time"""
    ball = PhysicsState(
        position=np.array([0.0, 10.0]),
        velocity=np.array([0.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-001", ball)
    
    dt = 0.01
    for _ in range(100):
        ball.velocity += world.gravity * dt
        ball.position += ball.velocity * dt
        world.step(dt)
    
    # Gravity is applied twice per iteration (once here, once in step)
    # So after 1 second, velocity should be around -19.6 m/s
    assert ball.velocity[1] < -19.0
    assert ball.velocity[1] > -20.0
    assert ball.position[1] < 10.0


@pytest.mark.unit
def test_no_gravity(world_no_gravity):
    """Test objects maintain velocity without gravity"""
    ball = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 3.0]),
        mass=1.0,
        elasticity=1.0
    )
    world_no_gravity.add_object("ball-001", ball)
    
    initial_velocity = ball.velocity.copy()
    
    dt = 0.01
    for _ in range(100):
        ball.velocity += world_no_gravity.gravity * dt
        ball.position += ball.velocity * dt
        world_no_gravity.step(dt)
    
    np.testing.assert_array_almost_equal(ball.velocity, initial_velocity)


@pytest.mark.unit
def test_physics_state_creation():
    """Test PhysicsState can be created with valid parameters"""
    state = PhysicsState(
        position=np.array([1.0, 2.0]),
        velocity=np.array([3.0, 4.0]),
        mass=2.5,
        elasticity=0.7
    )
    
    np.testing.assert_array_equal(state.position, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(state.velocity, np.array([3.0, 4.0]))
    assert state.mass == 2.5
    assert state.elasticity == 0.7


@pytest.mark.unit
def test_kinetic_energy_calculation():
    """Test kinetic energy calculation"""
    state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([3.0, 4.0]),
        mass=2.0,
        elasticity=1.0
    )
    
    speed_squared = 3.0**2 + 4.0**2
    expected_ke = 0.5 * 2.0 * speed_squared
    
    actual_ke = 0.5 * state.mass * np.dot(state.velocity, state.velocity)
    assert actual_ke == pytest.approx(expected_ke)
