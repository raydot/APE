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


def run_newtons_cradle(
    num_balls=3,
    initial_velocity=2.0,
    elasticity=1.0,
    learning_enabled=True,
    trial_num=1,
    use_mlflow=True,
    visualize=False
):
    """
    Newton's Cradle experiment: Test momentum transfer through chain of collisions.
    
    Args:
        num_balls: Number of balls in the cradle (default 3)
        initial_velocity: Initial velocity of first ball (m/s)
        elasticity: Coefficient of restitution (1.0 = perfectly elastic)
        learning_enabled: Whether agents learn from past collisions
        trial_num: Trial number for tracking
        use_mlflow: Whether to track with MLflow
    
    Returns:
        dict: Results including final momentum, errors, collision count
    """
    
    print(f"\n{'='*60}")
    print(f"NEWTON'S CRADLE - Trial {trial_num}")
    print(f"{'='*60}")
    print(f"Balls: {num_balls}, Initial velocity: {initial_velocity} m/s")
    print(f"Elasticity: {elasticity}, Learning: {learning_enabled}")
    print(f"{'='*60}\n")
    
    # Initialize MLflow tracking
    tracker = None
    if use_mlflow:
        tracker = ExperimentTracker(experiment_name="Newtons-Cradle")
        tracker.start_run(run_name=f"{num_balls}balls_trial{trial_num}")
        tracker.log_params({
            'num_balls': num_balls,
            'initial_velocity': initial_velocity,
            'elasticity': elasticity,
            'learning_enabled': learning_enabled,
            'trial_num': trial_num
        })
    
    # Setup
    event_bus = SimpleEventBus()
    world = WorldState(gravity=np.array([0.0, 0.0]))  # No gravity for simplicity
    
    # LLM setup
    llm_client = None
    model = None
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
    
    # Learning system (shared across all balls if enabled)
    experience_store = None
    feedback_generator = None
    if learning_enabled:
        experience_store = ExperienceStore(
            collection_name=f"newtons_cradle_trial{trial_num}",
            qdrant_path="./qdrant_storage"
        )
        feedback_generator = FeedbackGenerator(tolerance=0.05)
    
    # Tool registry and detector
    tool_registry = create_physics_tool_registry()
    ball_radius = 0.15
    detector = BallBallCollisionDetector(ball_radius=ball_radius)
    
    # Create resolver
    resolver = CollisionResolverAgent(event_bus, world, tool_registry)
    
    # Create visualization recorder if requested
    recorder = None
    if visualize:
        recorder = BallCollisionRecorder()
    
    # Create balls in a line, touching
    spacing = ball_radius * 2  # Balls just touching
    agents = []
    
    for i in range(num_balls):
        ball_id = f"ball-{i:03d}"
        
        # Position balls in a line
        position = np.array([i * spacing, 1.0])
        
        # First ball gets initial velocity, others are stationary
        if i == 0:
            velocity = np.array([initial_velocity, 0.0])
        else:
            velocity = np.array([0.0, 0.0])
        
        # Create physics state
        state = PhysicsState(
            position=position,
            velocity=velocity,
            mass=1.0,
            elasticity=elasticity
        )
        world.add_object(ball_id, state)
        
        # Create agent
        if learning_enabled:
            agent = LearningBallAgent(
                ball_id,
                event_bus,
                world,
                llm_client,
                model,
                experience_store,
                feedback_generator,
                learning_enabled=True,
                max_examples=3
            )
        else:
            from ape.negotiating_agent import NegotiatingBallAgent
            agent = NegotiatingBallAgent(ball_id, event_bus, world, llm_client, model)
        
        agents.append(agent)
        print(f"Created {ball_id} at position {position}, velocity {velocity}")
    
    print(f"\nInitial momentum: {initial_velocity * 1.0:.3f} kg⋅m/s")
    print(f"Expected final: Ball 0 stops, Ball {num_balls-1} moves at {initial_velocity:.3f} m/s\n")
    
    # Track metrics
    collision_count = 0
    momentum_errors = []
    energy_errors = []
    collision_log = []
    
    # Calculate initial total momentum and energy
    initial_momentum = initial_velocity * 1.0  # mass = 1.0
    initial_energy = 0.5 * 1.0 * initial_velocity**2
    
    # Simulate
    dt = 0.01
    max_steps = 1000
    max_collisions = num_balls * 2  # Expect roughly num_balls-1 collisions
    
    print("Starting simulation...\n")
    
    for step in range(max_steps):
        world.step(dt)
        
        # Record frame for visualization
        if recorder:
            ball_ids = [f"ball-{i:03d}" for i in range(num_balls)]
            recorder.record_frame(world, ball_ids)
        
        # Detect collisions
        collisions = detector.detect_all_collisions(world.get_all_objects(), world.time)
        
        if collisions and collision_count < max_collisions:
            for collision in collisions:
                collision_count += 1
                
                print(f"[Collision {collision_count}] {collision['ball1_id']} ↔ {collision['ball2_id']}")
                
                # Separate overlapping balls
                state1 = world.get_object(collision['ball1_id'])
                state2 = world.get_object(collision['ball2_id'])
                
                # Store pre-collision velocities
                v1_before = state1.velocity.copy()
                v2_before = state2.velocity.copy()
                
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
                
                print(f"  Before: {collision['ball1_id']}={v1_before[0]:.3f}, {collision['ball2_id']}={v2_before[0]:.3f}")
                print(f"  After:  {collision['ball1_id']}={v1_after[0]:.3f}, {collision['ball2_id']}={v2_after[0]:.3f}")
                
                # Calculate momentum and energy conservation
                momentum_before = v1_before[0] + v2_before[0]
                momentum_after = v1_after[0] + v2_after[0]
                momentum_error = abs(momentum_after - momentum_before)
                
                energy_before = 0.5 * (v1_before[0]**2 + v2_before[0]**2)
                energy_after = 0.5 * (v1_after[0]**2 + v2_after[0]**2)
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
                    'v1_before': v1_before[0],
                    'v2_before': v2_before[0],
                    'v1_after': v1_after[0],
                    'v2_after': v2_after[0],
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
                        'energy_error': energy_error,
                        'ball1_velocity_change': abs(v1_after[0] - v1_before[0]),
                        'ball2_velocity_change': abs(v2_after[0] - v2_before[0])
                    }, step=collision_count)
        
        # Check if system has settled (all velocities near zero except possibly last ball)
        all_states = [world.get_object(f"ball-{i:03d}") for i in range(num_balls)]
        velocities = [s.velocity[0] for s in all_states]
        
        # System settled if only last ball is moving or all stopped
        if collision_count >= num_balls - 1:
            max_velocity = max(abs(v) for v in velocities)
            if max_velocity < 0.01:  # All stopped
                print("System settled (all stopped)")
                break
            elif all(abs(v) < 0.01 for i, v in enumerate(velocities) if i != num_balls - 1):
                # All stopped except possibly last ball
                if abs(velocities[-1]) > 0.01:
                    print("System settled (momentum transferred to last ball)")
                    break
    
    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_states = [world.get_object(f"ball-{i:03d}") for i in range(num_balls)]
    final_velocities = [s.velocity[0] for s in final_states]
    
    print(f"\nFinal velocities:")
    for i, v in enumerate(final_velocities):
        print(f"  Ball {i}: {v:.4f} m/s")
    
    # Calculate final total momentum and energy
    final_momentum = sum(final_velocities)
    final_energy = 0.5 * sum(v**2 for v in final_velocities)
    
    total_momentum_error = abs(final_momentum - initial_momentum)
    total_energy_error = abs(final_energy - initial_energy)
    
    print(f"\nInitial momentum: {initial_momentum:.4f} kg⋅m/s")
    print(f"Final momentum:   {final_momentum:.4f} kg⋅m/s")
    print(f"Momentum error:   {total_momentum_error:.4f} kg⋅m/s ({total_momentum_error/initial_momentum*100:.2f}%)")
    
    print(f"\nInitial energy: {initial_energy:.4f} J")
    print(f"Final energy:   {final_energy:.4f} J")
    print(f"Energy error:   {total_energy_error:.4f} J ({total_energy_error/initial_energy*100:.2f}%)")
    
    print(f"\nTotal collisions: {collision_count}")
    if momentum_errors:
        print(f"Avg momentum error per collision: {np.mean(momentum_errors):.4f} kg⋅m/s")
        print(f"Avg energy error per collision: {np.mean(energy_errors):.4f} J")
    
    # Check if final state matches expected (ball 0 stopped, last ball moving)
    expected_final_velocity = initial_velocity
    actual_final_velocity = final_velocities[-1]
    final_velocity_error = abs(actual_final_velocity - expected_final_velocity)
    
    print(f"\nExpected final ball velocity: {expected_final_velocity:.4f} m/s")
    print(f"Actual final ball velocity:   {actual_final_velocity:.4f} m/s")
    print(f"Error: {final_velocity_error:.4f} m/s ({final_velocity_error/expected_final_velocity*100:.2f}%)")
    
    # Success criteria
    success = (
        total_momentum_error / initial_momentum < 0.05 and  # <5% momentum error
        final_velocity_error / expected_final_velocity < 0.10  # <10% final velocity error
    )
    
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Momentum transfer {'accurate' if success else 'inaccurate'}")
    
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
        'initial_velocity': initial_velocity,
        'elasticity': elasticity,
        'learning_enabled': learning_enabled,
        'collision_count': collision_count,
        'final_velocities': final_velocities,
        'initial_momentum': initial_momentum,
        'final_momentum': final_momentum,
        'total_momentum_error': total_momentum_error,
        'momentum_error_percent': total_momentum_error / initial_momentum * 100,
        'total_energy_error': total_energy_error,
        'energy_error_percent': total_energy_error / initial_energy * 100,
        'expected_final_velocity': expected_final_velocity,
        'actual_final_velocity': actual_final_velocity,
        'final_velocity_error': final_velocity_error,
        'final_velocity_error_percent': final_velocity_error / expected_final_velocity * 100,
        'success': success,
        'collision_log': collision_log,
        # Add resolver statistics
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
            'total_momentum_error': total_momentum_error,
            'momentum_error_percent': results['momentum_error_percent'],
            'total_energy_error': total_energy_error,
            'energy_error_percent': results['energy_error_percent'],
            'final_velocity_error': final_velocity_error,
            'final_velocity_error_percent': results['final_velocity_error_percent'],
            'success': 1.0 if success else 0.0,
            # Log resolver statistics
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
        recorder.save(f"newtons_cradle_trial{trial_num}.json")
        anim = visualize_ball_collision(recorder, ball_radius=ball_radius)
    
    return results


def main():
    """Run Newton's Cradle experiment with multiple trials."""
    
    print("\n" + "="*60)
    print("APE EMERGENCE EXPERIMENT: NEWTON'S CRADLE")
    print("="*60)
    print("\nObjective: Test momentum transfer through chain of collisions")
    print("Expected: Ball 0 stops, Ball N moves with same velocity")
    print("="*60 + "\n")
    
    # Configuration
    num_balls = 5
    initial_velocity = 2.0
    elasticity = 1.0
    num_trials = 1  # Single trial with visualization
    
    print(f"Configuration:")
    print(f"  Balls: {num_balls}")
    print(f"  Initial velocity: {initial_velocity} m/s")
    print(f"  Elasticity: {elasticity}")
    print(f"  Trials: {num_trials}")
    print()
    
    # Run trials
    all_results = []
    
    for trial in range(1, num_trials + 1):
        results = run_newtons_cradle(
            num_balls=num_balls,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
            learning_enabled=True,
            trial_num=trial,
            use_mlflow=True,
            visualize=True  # Enable visualization
        )
        all_results.append(results)
    
    # Aggregate analysis
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)
    
    momentum_errors = [r['momentum_error_percent'] for r in all_results]
    final_velocity_errors = [r['final_velocity_error_percent'] for r in all_results]
    success_rate = sum(r['success'] for r in all_results) / len(all_results)
    
    print(f"\nMomentum error: {np.mean(momentum_errors):.2f}% ± {np.std(momentum_errors):.2f}%")
    print(f"Final velocity error: {np.mean(final_velocity_errors):.2f}% ± {np.std(final_velocity_errors):.2f}%")
    print(f"Success rate: {success_rate:.1%} ({sum(r['success'] for r in all_results)}/{len(all_results)} trials)")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nView results: mlflow ui --backend-store-uri ./mlruns")
    print()


if __name__ == "__main__":
    main()
