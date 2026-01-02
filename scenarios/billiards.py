import numpy as np
import os
from dotenv import load_dotenv
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.learning import ExperienceStore, FeedbackGenerator, LearningBallAgent
from ape.resolver_agent import CollisionResolverAgent
from ape.collision_detection import BallBallCollisionDetector
from ape.tools import create_physics_tool_registry
from ape.tracking import ExperimentTracker
from ape.ball_collision_viz import BallCollisionRecorder, visualize_ball_collision

load_dotenv()


def run_billiards(
    num_balls=6,
    table_size=5.0,
    max_velocity=2.0,
    elasticity=0.95,
    learning_enabled=True,
    trial_num=1,
    use_mlflow=True,
    visualize=False,
    seed=None
):
    """
    Billiards experiment: Test learning in complex 2D multi-body collisions.
    
    Args:
        num_balls: Number of balls on table (default 6)
        table_size: Size of square table (meters)
        max_velocity: Maximum initial velocity magnitude (m/s)
        elasticity: Coefficient of restitution (0.95 = slightly inelastic)
        learning_enabled: Whether agents learn from past collisions
        trial_num: Trial number for tracking
        use_mlflow: Whether to track with MLflow
        visualize: Whether to create visualization
        seed: Random seed for reproducibility
    
    Returns:
        dict: Results including resolver stats, errors, collision count
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"BILLIARDS EXPERIMENT - Trial {trial_num}")
    print(f"{'='*60}")
    print(f"Balls: {num_balls}, Table size: {table_size}m x {table_size}m")
    print(f"Max velocity: {max_velocity} m/s, Elasticity: {elasticity}")
    print(f"Learning: {learning_enabled}")
    print(f"{'='*60}\n")
    
    # Initialize MLflow tracking
    tracker = None
    if use_mlflow:
        tracker = ExperimentTracker(
            experiment_name="billiards"
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
            'trial_num': trial_num
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
    
    # Initialize learning system
    experience_store = None
    feedback_gen = None
    if learning_enabled:
        experience_store = ExperienceStore()
        feedback_gen = FeedbackGenerator()
    
    # Create balls in ring formation, all moving toward center
    ball_radius = 0.15
    ball_mass = 1.0
    agents = []
    
    center = np.array([table_size / 2, table_size / 2])
    ring_radius = 1.5  # Balls start in ring around center
    
    print("Initializing balls (converging ring):")
    for i in range(num_balls):
        ball_id = f"ball-{i:03d}"
        
        # Position evenly spaced in ring around center
        angle = (i / num_balls) * 2 * np.pi
        pos = center + ring_radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Velocity pointing toward center with slight variation
        direction = center - pos
        direction = direction / np.linalg.norm(direction)
        
        # Add small random variation (±15 degrees)
        angle_variation = np.random.uniform(-np.pi/12, np.pi/12)
        cos_var = np.cos(angle_variation)
        sin_var = np.sin(angle_variation)
        direction = np.array([
            direction[0] * cos_var - direction[1] * sin_var,
            direction[0] * sin_var + direction[1] * cos_var
        ])
        
        speed = np.random.uniform(1.2, max_velocity)
        vel = direction * speed
        
        # Create physics state
        state = PhysicsState(
            position=pos,
            velocity=vel,
            mass=ball_mass,
            elasticity=elasticity
        )
        world.add_object(ball_id, state)
        
        # Create agent
        if learning_enabled:
            agent = LearningBallAgent(
                agent_id=ball_id,
                event_bus=event_bus,
                world_state=world,
                llm_client=llm_client,
                model_name=model,
                experience_store=experience_store,
                feedback_generator=feedback_gen,
                learning_enabled=True,
                max_examples=3
            )
        else:
            from ape.negotiating_agent import NegotiatingBallAgent
            agent = NegotiatingBallAgent(
                agent_id=ball_id,
                event_bus=event_bus,
                world_state=world,
                llm_client=llm_client,
                model_name=model
            )
        agents.append(agent)
        
        print(f"  {ball_id}: pos={pos}, vel={vel}, speed={np.linalg.norm(vel):.2f} m/s")
    
    # Initialize resolver and collision detector
    resolver = CollisionResolverAgent(event_bus, world, tools)
    detector = BallBallCollisionDetector(ball_radius)
    
    # Initialize visualization recorder
    recorder = None
    if visualize:
        recorder = BallCollisionRecorder()
    
    # Calculate initial momentum and energy
    all_states = [world.get_object(f"ball-{i:03d}") for i in range(num_balls)]
    initial_momentum = sum(s.velocity for s in all_states)
    initial_energy = sum(0.5 * s.mass * np.dot(s.velocity, s.velocity) for s in all_states)
    
    print(f"\nInitial total momentum: {initial_momentum}")
    print(f"Initial total energy: {initial_energy:.4f} J")
    
    # Simulation parameters
    dt = 0.01
    max_time = 30.0
    max_collisions = 100
    settle_threshold = 0.05
    
    collision_count = 0
    collision_log = []
    momentum_errors = []
    energy_errors = []
    
    print(f"\nStarting simulation (max {max_time}s, max {max_collisions} collisions)...\n")
    
    # Main simulation loop
    while world.time < max_time and collision_count < max_collisions:
        # Update positions
        for i in range(num_balls):
            ball_id = f"ball-{i:03d}"
            state = world.get_object(ball_id)
            state.position += state.velocity * dt
            world.update_object(ball_id, state)
        
        world.time += dt
        
        # Record frame for visualization
        if recorder:
            recorder.record_frame(world, [f"ball-{i:03d}" for i in range(num_balls)])
        
        # Detect collisions
        collisions = detector.detect_all_collisions(world.get_all_objects(), world.time)
        
        # Handle collisions
        for collision in collisions:
            collision_count += 1
            print(f"[Collision {collision_count}] {collision['ball1_id']} ↔ {collision['ball2_id']}")
            
            # Get pre-collision states
            state1 = world.get_object(collision['ball1_id'])
            state2 = world.get_object(collision['ball2_id'])
            v1_before = state1.velocity.copy()
            v2_before = state2.velocity.copy()
            
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
            
            # Get agents
            agent1 = next(a for a in agents if a.agent_id == collision['ball1_id'])
            agent2 = next(a for a in agents if a.agent_id == collision['ball2_id'])
            
            # Resolve collision through negotiation
            outcome = resolver.handle_collision(agent1, agent2, collision)
            
            # Record collision for visualization
            if recorder:
                recorder.record_collision(
                    world.time,
                    collision['ball1_id'],
                    collision['ball2_id'],
                    collision['collision_point']
                )
                recorder.record_negotiation_outcome(world.time, outcome)
            
            # Get post-collision velocities
            state1 = world.get_object(collision['ball1_id'])
            state2 = world.get_object(collision['ball2_id'])
            v1_after = state1.velocity.copy()
            v2_after = state2.velocity.copy()
            
            # Calculate momentum and energy conservation
            momentum_before = v1_before + v2_before
            momentum_after = v1_after + v2_after
            momentum_error = np.linalg.norm(momentum_after - momentum_before)
            
            energy_before = 0.5 * (np.dot(v1_before, v1_before) + np.dot(v2_before, v2_before))
            energy_after = 0.5 * (np.dot(v1_after, v1_after) + np.dot(v2_after, v2_after))
            energy_error = abs(energy_after - energy_before)
            
            momentum_errors.append(momentum_error)
            energy_errors.append(energy_error)
            
            print(f"  Momentum error: {momentum_error:.4f} kg⋅m/s")
            print(f"  Energy error: {energy_error:.4f} J\n")
            
            # Log collision
            collision_log.append({
                'collision_num': collision_count,
                'time': world.time,
                'ball1': collision['ball1_id'],
                'ball2': collision['ball2_id'],
                'momentum_error': momentum_error,
                'energy_error': energy_error
            })
            
            # Record outcome for learning agents
            if learning_enabled:
                collision_id = f"{collision['ball1_id']}_{collision['ball2_id']}_{world.time}"
                if isinstance(agent1, LearningBallAgent):
                    agent1.record_outcome(collision_id, v1_after, v2_after)
                if isinstance(agent2, LearningBallAgent):
                    agent2.record_outcome(collision_id, v2_after, v1_after)
            
            # Track with MLflow
            if tracker:
                tracker.log_metrics({
                    'momentum_error': momentum_error,
                    'energy_error': energy_error
                }, step=collision_count)
        
        # Check if system has settled (all velocities below threshold)
        all_states = [world.get_object(f"ball-{i:03d}") for i in range(num_balls)]
        max_speed = max(np.linalg.norm(s.velocity) for s in all_states)
        
        if max_speed < settle_threshold:
            print(f"System settled at t={world.time:.2f}s (max speed: {max_speed:.4f} m/s)")
            break
    
    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_states = [world.get_object(f"ball-{i:03d}") for i in range(num_balls)]
    final_velocities = [s.velocity for s in final_states]
    
    print(f"\nFinal velocities:")
    for i, v in enumerate(final_velocities):
        speed = np.linalg.norm(v)
        print(f"  Ball {i}: {v} (speed: {speed:.4f} m/s)")
    
    # Calculate final total momentum and energy
    final_momentum = sum(final_velocities)
    final_energy = sum(0.5 * s.mass * np.dot(s.velocity, s.velocity) for s in final_states)
    
    total_momentum_error = np.linalg.norm(final_momentum - initial_momentum)
    total_energy_error = abs(final_energy - initial_energy)
    
    print(f"\nInitial momentum: {initial_momentum}")
    print(f"Final momentum:   {final_momentum}")
    print(f"Momentum error:   {total_momentum_error:.4f} kg⋅m/s ({total_momentum_error/np.linalg.norm(initial_momentum)*100:.2f}%)")
    
    print(f"\nInitial energy: {initial_energy:.4f} J")
    print(f"Final energy:   {final_energy:.4f} J")
    print(f"Energy error:   {total_energy_error:.4f} J ({total_energy_error/initial_energy*100:.2f}%)")
    
    print(f"\nTotal collisions: {collision_count}")
    if momentum_errors:
        print(f"Avg momentum error per collision: {np.mean(momentum_errors):.4f} kg⋅m/s")
        print(f"Avg energy error per collision: {np.mean(energy_errors):.4f} J")
    
    # Success criteria (more lenient for 2D)
    success = (
        total_momentum_error / np.linalg.norm(initial_momentum) < 0.10 and  # <10% momentum error
        total_energy_error / initial_energy < 0.15  # <15% energy error
    )
    
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Conservation laws {'satisfied' if success else 'violated'}")
    
    # Get resolver statistics
    resolver_stats = resolver.get_stats()
    
    print(f"\n{'='*60}")
    print("RESOLVER STATISTICS")
    print(f"{'='*60}")
    print(f"Total negotiations: {resolver_stats['total_negotiations']}")
    print(f"Proposals accepted: {resolver_stats['proposals_accepted']} ({resolver_stats['acceptance_rate']*100:.1f}%)")
    print(f"Proposals rejected: {resolver_stats['proposals_rejected']} ({resolver_stats['rejection_rate']*100:.1f}%)")
    print(f"Ground truth imposed: {resolver_stats['ground_truth_imposed']} ({resolver_stats['ground_truth_rate']*100:.1f}%)")
    
    # Compile results
    results = {
        'trial_num': trial_num,
        'num_balls': num_balls,
        'table_size': table_size,
        'max_velocity': max_velocity,
        'elasticity': elasticity,
        'learning_enabled': learning_enabled,
        'collision_count': collision_count,
        'simulation_time': world.time,
        'final_velocities': [v.tolist() for v in final_velocities],
        'initial_momentum': initial_momentum.tolist(),
        'final_momentum': final_momentum.tolist(),
        'total_momentum_error': total_momentum_error,
        'momentum_error_percent': total_momentum_error / np.linalg.norm(initial_momentum) * 100,
        'total_energy_error': total_energy_error,
        'energy_error_percent': total_energy_error / initial_energy * 100,
        'success': success,
        'collision_log': collision_log,
        'resolver_acceptance_rate': resolver_stats['acceptance_rate'],
        'resolver_rejection_rate': resolver_stats['rejection_rate'],
        'resolver_ground_truth_rate': resolver_stats['ground_truth_rate'],
        'resolver_proposals_accepted': resolver_stats['proposals_accepted'],
        'resolver_proposals_rejected': resolver_stats['proposals_rejected'],
        'resolver_ground_truth_imposed': resolver_stats['ground_truth_imposed']
    }
    
    # Log final metrics to MLflow
    if tracker:
        tracker.log_metrics({
            'collision_count': collision_count,
            'simulation_time': world.time,
            'total_momentum_error': total_momentum_error,
            'momentum_error_percent': results['momentum_error_percent'],
            'total_energy_error': total_energy_error,
            'energy_error_percent': results['energy_error_percent'],
            'success': 1.0 if success else 0.0,
            'resolver_acceptance_rate': resolver_stats['acceptance_rate'],
            'resolver_rejection_rate': resolver_stats['rejection_rate'],
            'resolver_ground_truth_rate': resolver_stats['ground_truth_rate']
        })
        
        # Log learning stats if enabled
        if learning_enabled and agents:
            for agent in agents:
                if isinstance(agent, LearningBallAgent):
                    stats = agent.get_learning_stats()
                    if stats['total_experiences'] > 0:
                        tracker.log_learning_stats(
                            step=collision_count,
                            agent_id=agent.agent_id,
                            stats=stats
                        )
        
        tracker.end_run()
    
    # Create visualization if requested
    if visualize and recorder:
        print("\n[VISUALIZATION] Creating animation...")
        recorder.save(f"billiards_trial{trial_num}.json")
        anim = visualize_ball_collision(recorder, ball_radius=ball_radius)
    
    return results


def main():
    """Run Billiards experiment with single trial."""
    
    print("\n" + "="*60)
    print("APE EMERGENCE EXPERIMENT: BILLIARDS")
    print("="*60)
    print("\nObjective: Test learning in complex 2D multi-body collisions")
    print("Expected: Agents learn to predict angled collisions")
    print("="*60 + "\n")
    
    # Configuration
    num_balls = 6
    table_size = 5.0
    max_velocity = 2.0
    elasticity = 0.95
    
    print(f"Configuration:")
    print(f"  Balls: {num_balls}")
    print(f"  Table size: {table_size}m x {table_size}m")
    print(f"  Max velocity: {max_velocity} m/s")
    print(f"  Elasticity: {elasticity}")
    print()
    
    # Run single trial
    results = run_billiards(
        num_balls=num_balls,
        table_size=table_size,
        max_velocity=max_velocity,
        elasticity=elasticity,
        learning_enabled=True,
        trial_num=1,
        use_mlflow=True,
        visualize=True,
        seed=42
    )
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nView results: mlflow ui --backend-store-uri ./mlruns")
    print()


if __name__ == "__main__":
    main()
