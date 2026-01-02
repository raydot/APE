import numpy as np
import os
from dotenv import load_dotenv
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.learning import ExperienceStore, FeedbackGenerator
from ape.observer_agent import ObserverAgent
from ape.collision_detection import BallBallCollisionDetector
from ape.tools import create_physics_tool_registry
from ape.tracking import ExperimentTracker

load_dotenv()


def calculate_elastic_collision_2d(
    v1: np.ndarray,
    v2: np.ndarray,
    m1: float,
    m2: float,
    normal: np.ndarray,
    elasticity: float = 1.0
) -> tuple:
    """
    Calculate ground truth collision outcome using physics equations.
    
    Args:
        v1: Velocity of ball 1 before collision
        v2: Velocity of ball 2 before collision
        m1: Mass of ball 1
        m2: Mass of ball 2
        normal: Collision normal vector (normalized)
        elasticity: Coefficient of restitution
    
    Returns:
        (v1_after, v2_after): Velocities after collision
    """
    
    # Decompose velocities into normal and tangential components
    v1_normal = np.dot(v1, normal) * normal
    v1_tangent = v1 - v1_normal
    
    v2_normal = np.dot(v2, normal) * normal
    v2_tangent = v2 - v2_normal
    
    # Calculate new normal velocities using 1D collision formula
    v1n_mag = np.dot(v1, normal)
    v2n_mag = np.dot(v2, normal)
    
    # Conservation of momentum and energy in normal direction
    v1n_mag_after = ((m1 - elasticity * m2) * v1n_mag + (1 + elasticity) * m2 * v2n_mag) / (m1 + m2)
    v2n_mag_after = ((m2 - elasticity * m1) * v2n_mag + (1 + elasticity) * m1 * v1n_mag) / (m1 + m2)
    
    v1_normal_after = v1n_mag_after * normal
    v2_normal_after = v2n_mag_after * normal
    
    # Tangential components unchanged
    v1_after = v1_normal_after + v1_tangent
    v2_after = v2_normal_after + v2_tangent
    
    return v1_after, v2_after


def run_principle_discovery(
    num_balls=3,
    table_size=5.0,
    max_velocity=2.0,
    elasticity=0.95,
    learning_enabled=True,
    trial_num=1,
    num_collisions=20,
    use_mlflow=True,
    seed=None
):
    """
    Principle Discovery experiment: Observer learns physics by watching collisions.
    
    Args:
        num_balls: Number of actor balls (default 3)
        table_size: Size of square table (meters)
        max_velocity: Maximum initial velocity (m/s)
        elasticity: Coefficient of restitution
        learning_enabled: Whether observer learns from past observations
        trial_num: Trial number for tracking
        num_collisions: Target number of collisions to observe
        use_mlflow: Whether to track with MLflow
        seed: Random seed for reproducibility
    
    Returns:
        dict: Results including observer stats, prediction errors
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"PRINCIPLE DISCOVERY EXPERIMENT - Trial {trial_num}")
    print(f"{'='*60}")
    print(f"Actor balls: {num_balls}, Table size: {table_size}m x {table_size}m")
    print(f"Max velocity: {max_velocity} m/s, Elasticity: {elasticity}")
    print(f"Observer learning: {learning_enabled}")
    print(f"Target collisions: {num_collisions}")
    print(f"{'='*60}\n")
    
    # Initialize MLflow tracking
    tracker = None
    if use_mlflow:
        tracker = ExperimentTracker(
            experiment_name="principle_discovery"
        )
        tracker.start_run(
            run_name=f"trial_{trial_num}_{'learning' if learning_enabled else 'baseline'}"
        )
        tracker.log_params({
            'num_balls': num_balls,
            'table_size': table_size,
            'max_velocity': max_velocity,
            'elasticity': elasticity,
            'learning_enabled': learning_enabled,
            'trial_num': trial_num,
            'num_collisions': num_collisions
        })
    
    # Initialize components
    event_bus = SimpleEventBus()
    world = WorldState()
    tools = create_physics_tool_registry()
    
    # Initialize LLM client
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        from openai import OpenAI
        llm_client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"
        print(f"Using OpenAI: {model}\n")
    else:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=api_key)
            model = "claude-3-5-haiku-20241022"
            print(f"Using Anthropic: {model}\n")
        else:
            raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    if tracker:
        tracker.log_params({'model': model})
    
    # Initialize observer's learning system
    experience_store = None
    feedback_gen = None
    if learning_enabled:
        experience_store = ExperienceStore()
        feedback_gen = FeedbackGenerator()
    
    # Create observer agent
    observer = ObserverAgent(
        agent_id="observer",
        llm_client=llm_client,
        model_name=model,
        experience_store=experience_store,
        feedback_generator=feedback_gen,
        learning_enabled=learning_enabled,
        max_examples=5
    )
    
    # Create actor balls in ring formation (like billiards)
    ball_radius = 0.15
    ball_mass = 1.0
    
    center = np.array([table_size / 2, table_size / 2])
    ring_radius = 1.5
    
    print("Initializing actor balls (converging ring):")
    for i in range(num_balls):
        ball_id = f"ball-{i:03d}"
        
        # Position evenly spaced in ring
        angle = (i / num_balls) * 2 * np.pi
        pos = center + ring_radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Velocity pointing toward center with variation
        direction = center - pos
        direction = direction / np.linalg.norm(direction)
        
        angle_variation = np.random.uniform(-np.pi/12, np.pi/12)
        cos_var = np.cos(angle_variation)
        sin_var = np.sin(angle_variation)
        direction = np.array([
            direction[0] * cos_var - direction[1] * sin_var,
            direction[0] * sin_var + direction[1] * cos_var
        ])
        
        speed = np.random.uniform(1.2, max_velocity)
        vel = direction * speed
        
        state = PhysicsState(
            position=pos,
            velocity=vel,
            mass=ball_mass,
            elasticity=elasticity
        )
        world.add_object(ball_id, state)
        
        print(f"  {ball_id}: pos={pos}, vel={vel}, speed={np.linalg.norm(vel):.2f} m/s")
    
    # Initialize collision detector
    detector = BallBallCollisionDetector(ball_radius)
    
    # Simulation parameters
    dt = 0.01
    max_time = 30.0
    
    collision_count = 0
    prediction_errors = []
    
    print(f"\nStarting simulation (max {max_time}s, target {num_collisions} collisions)...\n")
    
    # Main simulation loop
    while world.time < max_time and collision_count < num_collisions:
        # Update positions
        for i in range(num_balls):
            ball_id = f"ball-{i:03d}"
            state = world.get_object(ball_id)
            state.position += state.velocity * dt
            world.update_object(ball_id, state)
        
        world.time += dt
        
        # Detect collisions
        collisions = detector.detect_all_collisions(world.get_all_objects(), world.time)
        
        # Handle collisions
        for collision in collisions:
            collision_count += 1
            print(f"\n{'='*60}")
            print(f"[Collision {collision_count}] {collision['ball1_id']} ↔ {collision['ball2_id']}")
            print(f"{'='*60}")
            
            # Get pre-collision states
            state1 = world.get_object(collision['ball1_id'])
            state2 = world.get_object(collision['ball2_id'])
            v1_before = state1.velocity.copy()
            v2_before = state2.velocity.copy()
            
            print(f"Before: v1={v1_before}, v2={v2_before}")
            
            # Get collision normal
            collision_normal = collision['collision_normal']
            
            # Observer predicts outcome
            prediction = observer.predict_collision(
                collision['ball1_id'],
                collision['ball2_id'],
                v1_before,
                v2_before,
                state1.mass,
                state2.mass,
                elasticity,
                collision_normal
            )
            
            # Calculate ground truth outcome
            v1_after, v2_after = calculate_elastic_collision_2d(
                v1_before,
                v2_before,
                state1.mass,
                state2.mass,
                collision_normal,
                elasticity
            )
            
            print(f"Actual: v1={v1_after}, v2={v2_after}")
            
            # Calculate prediction error
            error1 = np.linalg.norm(prediction['v1_after'] - v1_after)
            error2 = np.linalg.norm(prediction['v2_after'] - v2_after)
            avg_error = (error1 + error2) / 2
            prediction_errors.append(avg_error)
            
            print(f"\nPrediction error: {avg_error:.3f} m/s")
            
            # Apply ground truth velocities
            state1.velocity = v1_after
            state2.velocity = v2_after
            
            # Separate overlapping balls
            new_pos1, new_pos2 = detector.separate_overlapping_balls(
                state1.position,
                state2.position,
                collision['overlap']
            )
            state1.position = new_pos1
            state2.position = new_pos2
            
            world.update_object(collision['ball1_id'], state1)
            world.update_object(collision['ball2_id'], state2)
            
            # Observer records observation
            observer.record_observation(
                collision['ball1_id'],
                collision['ball2_id'],
                v1_before,
                v2_before,
                v1_after,
                v2_after,
                state1.mass,
                state2.mass,
                elasticity,
                collision_normal
            )
            
            # Track with MLflow
            if tracker:
                tracker.log_metrics({
                    'prediction_error': avg_error
                }, step=collision_count)
            
            if collision_count >= num_collisions:
                break
    
    # Final analysis
    print(f"\n{'='*60}")
    print("OBSERVER LEARNING ANALYSIS")
    print(f"{'='*60}")
    
    stats = observer.get_learning_stats()
    
    print(f"\nTotal observations: {stats['total_observations']}")
    print(f"Total predictions: {stats['total_predictions']}")
    
    if stats['total_predictions'] > 0:
        print(f"\nPrediction accuracy:")
        print(f"  Average error: {stats['avg_error']:.3f} m/s")
        print(f"  Early error (first 1/3): {stats['early_error']:.3f} m/s")
        print(f"  Recent error (last 1/3): {stats['recent_error']:.3f} m/s")
        print(f"  Improvement: {stats['improvement']:.3f} m/s ({stats['improvement_percent']:.1f}%)")
        
        if stats['improvement'] > 0.1:
            print(f"\n✓ Observer LEARNED: Error decreased by {stats['improvement']:.3f} m/s")
        elif stats['improvement'] > 0:
            print(f"\n~ Observer showed SLIGHT improvement: {stats['improvement']:.3f} m/s")
        else:
            print(f"\n✗ Observer did NOT improve: Error changed by {stats['improvement']:.3f} m/s")
    
    # Compile results
    results = {
        'trial_num': trial_num,
        'num_balls': num_balls,
        'table_size': table_size,
        'max_velocity': max_velocity,
        'elasticity': elasticity,
        'learning_enabled': learning_enabled,
        'collision_count': collision_count,
        'total_observations': stats['total_observations'],
        'total_predictions': stats['total_predictions'],
        'avg_prediction_error': stats['avg_error'],
        'early_error': stats['early_error'],
        'recent_error': stats['recent_error'],
        'improvement': stats['improvement'],
        'improvement_percent': stats['improvement_percent'],
        'prediction_errors': prediction_errors
    }
    
    # Log final metrics to MLflow
    if tracker:
        tracker.log_metrics({
            'collision_count': collision_count,
            'total_observations': stats['total_observations'],
            'avg_prediction_error': stats['avg_error'],
            'early_error': stats['early_error'],
            'recent_error': stats['recent_error'],
            'improvement': stats['improvement'],
            'improvement_percent': stats['improvement_percent']
        })
        
        tracker.end_run()
    
    return results


def main():
    """Run Principle Discovery experiment with single trial."""
    
    print("\n" + "="*60)
    print("APE EMERGENCE EXPERIMENT: PRINCIPLE DISCOVERY")
    print("="*60)
    print("\nObjective: Can an observer learn physics by watching collisions?")
    print("Expected: Prediction error decreases as observer learns patterns")
    print("="*60 + "\n")
    
    # Configuration
    num_balls = 3
    table_size = 5.0
    max_velocity = 2.0
    elasticity = 0.95
    num_collisions = 20
    
    print(f"Configuration:")
    print(f"  Actor balls: {num_balls}")
    print(f"  Table size: {table_size}m x {table_size}m")
    print(f"  Max velocity: {max_velocity} m/s")
    print(f"  Elasticity: {elasticity}")
    print(f"  Target collisions: {num_collisions}")
    print()
    
    # Run single trial
    results = run_principle_discovery(
        num_balls=num_balls,
        table_size=table_size,
        max_velocity=max_velocity,
        elasticity=elasticity,
        learning_enabled=True,
        trial_num=1,
        num_collisions=num_collisions,
        use_mlflow=True,
        seed=42
    )
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nView results: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print()


if __name__ == "__main__":
    main()
