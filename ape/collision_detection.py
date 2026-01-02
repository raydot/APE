import numpy as np
from typing import Optional, List, Dict, Tuple
from .physics import PhysicsState


class BallBallCollisionDetector:
    """
    Detects collisions between moving balls.
    
    Handles:
    - Overlap detection (distance < r1 + r2)
    - Collision normal calculation
    - Cooldown to prevent duplicate detection
    - Position correction for overlapping balls
    """
    
    def __init__(self, ball_radius: float = 0.15):
        self.ball_radius = ball_radius
        self.collision_cooldown: Dict[Tuple[str, str], float] = {}
        self.cooldown_duration = 0.1  # seconds
    
    def detect_collision(
        self,
        ball1_id: str,
        ball1_pos: np.ndarray,
        ball2_id: str,
        ball2_pos: np.ndarray,
        current_time: float
    ) -> Optional[Dict]:
        """
        Check if two balls are colliding.
        
        Args:
            ball1_id: ID of first ball
            ball1_pos: Position of first ball
            ball2_id: ID of second ball
            ball2_pos: Position of second ball
            current_time: Current simulation time
        
        Returns:
            Collision data dict if colliding, None otherwise
        """
        # Check cooldown (prevent duplicate detection)
        pair_key = tuple(sorted([ball1_id, ball2_id]))
        if pair_key in self.collision_cooldown:
            if current_time - self.collision_cooldown[pair_key] < self.cooldown_duration:
                return None
        
        # Calculate distance between centers
        displacement = ball2_pos - ball1_pos
        distance = np.linalg.norm(displacement)
        
        # Check for overlap (with small epsilon to avoid floating point issues)
        collision_threshold = 2 * self.ball_radius
        
        if distance < collision_threshold and distance > 1e-6:
            # Collision detected!
            collision_normal = displacement / distance  # Unit vector from ball1 to ball2
            overlap = collision_threshold - distance
            
            # Record cooldown
            self.collision_cooldown[pair_key] = current_time
            
            return {
                'ball1_id': ball1_id,
                'ball2_id': ball2_id,
                'collision_normal': collision_normal,
                'distance': distance,
                'overlap': overlap,
                'collision_point': ball1_pos + displacement * 0.5  # Midpoint
            }
        
        return None
    
    def detect_all_collisions(
        self,
        objects: Dict[str, PhysicsState],
        current_time: float
    ) -> List[Dict]:
        """
        Check all pairs of balls for collisions.
        
        Args:
            objects: Dict of agent_id -> PhysicsState
            current_time: Current simulation time
        
        Returns:
            List of collision data dicts
        """
        collisions = []
        
        # Get all ball IDs (exclude floor, walls, etc.)
        ball_ids = [
            aid for aid in objects.keys() 
            if 'ball' in aid.lower() and objects[aid].mass < float('inf')
        ]
        
        # Check all unique pairs
        for i, ball1_id in enumerate(ball_ids):
            for ball2_id in ball_ids[i+1:]:  # Avoid duplicate pairs and self-collision
                ball1_pos = objects[ball1_id].position
                ball2_pos = objects[ball2_id].position
                
                collision = self.detect_collision(
                    ball1_id, ball1_pos,
                    ball2_id, ball2_pos,
                    current_time
                )
                
                if collision:
                    collisions.append(collision)
        
        return collisions
    
    def separate_overlapping_balls(
        self,
        ball1_pos: np.ndarray,
        ball2_pos: np.ndarray,
        overlap: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate two overlapping balls to just touching.
        
        Moves each ball back by half the overlap distance.
        
        Args:
            ball1_pos: Position of first ball
            ball2_pos: Position of second ball
            overlap: How much they overlap
        
        Returns:
            (new_ball1_pos, new_ball2_pos)
        """
        displacement = ball2_pos - ball1_pos
        distance = np.linalg.norm(displacement)
        
        if distance < 1e-6:
            # Balls at same position - separate arbitrarily
            separation = np.array([overlap / 2, 0])
            return ball1_pos - separation, ball2_pos + separation
        
        direction = displacement / distance
        separation = direction * (overlap / 2)
        
        return ball1_pos - separation, ball2_pos + separation
    
    def clear_cooldowns(self):
        """Clear all collision cooldowns (useful for testing)"""
        self.collision_cooldown.clear()
