import pytest
import numpy as np
from ape.collision_detection import BallBallCollisionDetector
from ape.physics import PhysicsState


@pytest.mark.unit
def test_ball_ball_collision_detection():
    """Test detecting collision between two balls"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    ball1 = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball2 = PhysicsState(
        position=np.array([0.25, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2
    }
    
    collisions = detector.detect_all_collisions(objects, 0.0)
    
    assert len(collisions) == 1
    collision = collisions[0]
    assert 'ball-001' in [collision['ball1_id'], collision['ball2_id']]
    assert 'ball-002' in [collision['ball1_id'], collision['ball2_id']]
    assert 'collision_normal' in collision
    assert 'overlap' in collision


@pytest.mark.unit
def test_ball_ball_no_collision():
    """Test no collision when balls are far apart"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    ball1 = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball2 = PhysicsState(
        position=np.array([5.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2
    }
    
    collisions = detector.detect_all_collisions(objects, 0.0)
    
    assert len(collisions) == 0


@pytest.mark.unit
def test_ball_ball_collision_normal():
    """Test collision normal points from ball1 to ball2"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    ball1 = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball2 = PhysicsState(
        position=np.array([0.2, 0.0]),
        velocity=np.array([-1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2
    }
    
    collisions = detector.detect_all_collisions(objects, 0.0)
    
    assert len(collisions) == 1
    collision = collisions[0]
    
    # Normal should point from ball1 to ball2 (positive x direction)
    normal = collision['collision_normal']
    assert normal[0] > 0
    assert abs(normal[1]) < 0.01
    
    # Normal should be unit length
    assert np.linalg.norm(normal) == pytest.approx(1.0)


@pytest.mark.unit
def test_ball_ball_overlap_calculation():
    """Test overlap distance calculation"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    ball1 = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    # Balls centers 0.2 apart, radii sum to 0.3, so overlap is 0.1
    ball2 = PhysicsState(
        position=np.array([0.2, 0.0]),
        velocity=np.array([-1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2
    }
    
    collisions = detector.detect_all_collisions(objects, 0.0)
    
    assert len(collisions) == 1
    collision = collisions[0]
    assert collision['overlap'] == pytest.approx(0.1, abs=0.01)


@pytest.mark.unit
def test_ball_ball_separation():
    """Test separating overlapping balls"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([0.2, 0.0])
    overlap = 0.1
    
    new_pos1, new_pos2 = detector.separate_overlapping_balls(pos1, pos2, overlap)
    
    # Distance between new positions should equal sum of radii
    distance = np.linalg.norm(new_pos2 - new_pos1)
    assert distance == pytest.approx(0.3, abs=0.01)
    
    # Both balls should move equally (same mass assumed)
    assert np.linalg.norm(new_pos1 - pos1) == pytest.approx(0.05, abs=0.01)
    assert np.linalg.norm(new_pos2 - pos2) == pytest.approx(0.05, abs=0.01)


@pytest.mark.unit
def test_collision_cooldown():
    """Test collision cooldown prevents duplicate detections"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    detector.cooldown_duration = 0.1
    
    ball1 = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball2 = PhysicsState(
        position=np.array([0.2, 0.0]),
        velocity=np.array([-1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2
    }
    
    # First detection should succeed
    collisions1 = detector.detect_all_collisions(objects, 0.0)
    assert len(collisions1) == 1
    
    # Second detection at same time should be blocked by cooldown
    collisions2 = detector.detect_all_collisions(objects, 0.0)
    assert len(collisions2) == 0
    
    # Detection after cooldown should succeed
    collisions3 = detector.detect_all_collisions(objects, 0.2)
    assert len(collisions3) == 1


@pytest.mark.unit
def test_multiple_ball_collisions():
    """Test detecting multiple simultaneous collisions"""
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    ball1 = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball2 = PhysicsState(
        position=np.array([0.2, 0.0]),
        velocity=np.array([-1.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball3 = PhysicsState(
        position=np.array([0.0, 0.5]),
        velocity=np.array([0.0, -1.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    ball4 = PhysicsState(
        position=np.array([0.0, 0.7]),
        velocity=np.array([0.0, 1.0]),
        mass=1.0,
        elasticity=1.0
    )
    
    objects = {
        'ball-001': ball1,
        'ball-002': ball2,
        'ball-003': ball3,
        'ball-004': ball4
    }
    
    collisions = detector.detect_all_collisions(objects, 0.0)
    
    # Should detect 2 separate collisions
    assert len(collisions) == 2
