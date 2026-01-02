import numpy as np
import os
from dotenv import load_dotenv

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.learning import ExperienceStore, FeedbackGenerator, LearningBallAgent, LearningAnalytics
from ape.resolver_agent import CollisionResolverAgent
from ape.collision_detection import BallBallCollisionDetector
from ape.tools import create_physics_tool_registry
from ape.tracking import ExperimentTracker

load_dotenv()


def main():
    """
    Learning system demo with MLflow experiment tracking
    
    Tracks all metrics, parameters, and artifacts to MLflow for analysis
    """
    
    print("=== APE: Learning System Demo with MLflow Tracking ===\n")
    
    # Initialize MLflow tracker
    tracker = ExperimentTracker(
        experiment_name="APE-Learning-System",
        tracking_uri="./mlruns"
    )
    
    # Setup
    event_bus = SimpleEventBus()
    world = WorldState(gravity=np.array([0.0, 0.0]))
    
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
            raise ValueError("No API key found")
    
    # Learning system components
    experience_store = ExperienceStore(
        collection_name="learning_demo_tracked",
        qdrant_path="./qdrant_storage"
    )
    
    # Clear previous demo data
    try:
        experience_store.clear_collection()
        print("[LEARNING] Cleared previous experiences\n")
    except:
        pass
    
    feedback_generator = FeedbackGenerator(tolerance=0.05)
    
    # Tool registry and detector
    tool_registry = create_physics_tool_registry()
    detector = BallBallCollisionDetector(ball_radius=0.15)
    
    # Create resolver
    resolver = CollisionResolverAgent(event_bus, world, tool_registry)
    
    # Configuration
    num_collisions = 20
    learning_enabled = True
    max_examples = 3
    tolerance = 0.05
    
    # Start MLflow run
    tracker.start_run(
        run_name=f"learning_demo_{model}",
        tags={
            "model": model,
            "learning_enabled": str(learning_enabled),
            "scenario": "repeated_collisions"
        }
    )
    
    # Log parameters
    tracker.log_params({
        "model_name": model,
        "learning_enabled": learning_enabled,
        "max_examples": max_examples,
        "tolerance": tolerance,
        "num_collisions": num_collisions,
        "ball_radius": 0.15,
        "gravity": "none",
    })
    
    # Create learning agent (ball-001)
    ball1_state = PhysicsState(
        position=np.array([0.0, 1.0]),
        velocity=np.array([5.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-001", ball1_state)
    
    ball1_agent = LearningBallAgent(
        "ball-001",
        event_bus,
        world,
        llm_client,
        model,
        experience_store,
        feedback_generator,
        learning_enabled=learning_enabled,
        max_examples=max_examples
    )
    
    # Create non-learning agent (ball-002)
    ball2_state = PhysicsState(
        position=np.array([2.0, 1.0]),
        velocity=np.array([-3.0, 0.0]),
        mass=1.0,
        elasticity=1.0
    )
    world.add_object("ball-002", ball2_state)
    
    from ape.negotiating_agent import NegotiatingBallAgent
    ball2_agent = NegotiatingBallAgent("ball-002", event_bus, world, llm_client, model)
    
    print("="*60)
    print("TRAINING SESSION: 20 Collisions with MLflow Tracking")
    print("="*60)
    
    # Track metrics over time
    collision_numbers = []
    accuracies = []
    errors = []
    
    # Run collision scenarios
    for collision_num in range(1, num_collisions + 1):
        print(f"\n{'='*60}")
        print(f"COLLISION {collision_num}/{num_collisions}")
        print(f"{'='*60}")
        
        # Reset positions with variations
        variation = np.random.uniform(-0.5, 0.5, 2)
        ball1_state.position = np.array([0.0, 1.0])
        ball1_state.velocity = np.array([5.0, 0.0]) + variation
        
        ball2_state.position = np.array([2.0, 1.0])
        ball2_state.velocity = np.array([-3.0, 0.0]) - variation * 0.5
        
        world.update_object("ball-001", ball1_state)
        world.update_object("ball-002", ball2_state)
        
        print(f"\nInitial velocities:")
        print(f"  ball-001: {ball1_state.velocity}")
        print(f"  ball-002: {ball2_state.velocity}")
        
        # Simulate until collision
        dt = 0.01
        max_steps = 300
        
        for step in range(max_steps):
            world.step(dt)
            
            collisions = detector.detect_all_collisions(world.get_all_objects(), world.time)
            
            if collisions:
                collision = collisions[0]
                
                # Separate overlapping balls
                state1 = world.get_object(collision['ball1_id'])
                state2 = world.get_object(collision['ball2_id'])
                
                new_pos1, new_pos2 = detector.separate_overlapping_balls(
                    state1.position,
                    state2.position,
                    collision['overlap']
                )
                
                state1.position = new_pos1
                state2.position = new_pos2
                world.update_object(collision['ball1_id'], state1)
                world.update_object(collision['ball2_id'], state2)
                
                # Resolve collision
                outcome = resolver.handle_collision(
                    ball1_agent,
                    ball2_agent,
                    collision
                )
                
                # Record outcome for learning
                final_ball1 = world.get_object("ball-001")
                final_ball2 = world.get_object("ball-002")
                
                collision_id = f"ball-001_ball-002_{world.time}"
                experience = ball1_agent.record_outcome(
                    collision_id,
                    final_ball1.velocity,
                    final_ball2.velocity
                )
                
                # Track metrics
                if experience:
                    collision_numbers.append(collision_num)
                    accuracies.append(1.0 if experience.was_correct else 0.0)
                    errors.append(experience.prediction_error)
                    
                    # Log to MLflow
                    tracker.log_collision_metrics(
                        collision_num=collision_num,
                        prediction_error=experience.prediction_error,
                        momentum_error=experience.momentum_error,
                        energy_error=experience.energy_error,
                        was_correct=experience.was_correct,
                        agent_id="ball-001"
                    )
                
                break
        
        # Log learning progress every 5 collisions
        if collision_num % 5 == 0:
            print(f"\n--- Learning Progress After {collision_num} Collisions ---")
            stats = ball1_agent.get_learning_stats()
            print(f"Total experiences: {stats['total_experiences']}")
            print(f"Overall accuracy: {stats['overall_accuracy']:.1%}")
            print(f"Recent accuracy: {stats['recent_accuracy']:.1%}")
            print(f"Avg error: {stats['avg_prediction_error']:.3f} m/s")
            
            # Log stats to MLflow
            tracker.log_learning_stats(
                step=collision_num,
                agent_id="ball-001",
                stats=stats
            )
            
            if stats['total_experiences'] > 10:
                improvement = stats['recent_accuracy'] - stats['overall_accuracy']
                if improvement > 0:
                    print(f"✓ Improving! (+{improvement:.1%})")
                elif improvement < 0:
                    print(f"⚠ Declining ({improvement:.1%})")
    
    # Final analysis
    print("\n" + "="*60)
    print("FINAL LEARNING ANALYSIS")
    print("="*60)
    
    analytics = LearningAnalytics(experience_store)
    analytics.print_summary_report(["ball-001"])
    
    # Log final statistics
    final_stats = ball1_agent.get_learning_stats()
    tracker.log_metrics({
        "final_accuracy": final_stats['overall_accuracy'],
        "final_avg_error": final_stats['avg_prediction_error'],
        "total_experiences": final_stats['total_experiences'],
    })
    
    # Create and log learning curve
    if collision_numbers:
        tracker.log_learning_curve(
            collision_numbers=collision_numbers,
            accuracies=accuracies,
            errors=errors,
            agent_id="ball-001"
        )
    
    # Log best and worst predictions as text
    best = experience_store.get_best_experiences("ball-001", limit=3)
    worst = experience_store.get_worst_experiences("ball-001", limit=3)
    
    summary_text = f"""
Learning Experiment Summary
===========================

Model: {model}
Total Collisions: {num_collisions}
Learning Enabled: {learning_enabled}

Final Statistics:
- Overall Accuracy: {final_stats['overall_accuracy']:.1%}
- Recent Accuracy: {final_stats['recent_accuracy']:.1%}
- Average Error: {final_stats['avg_prediction_error']:.3f} m/s
- Improvement: {final_stats.get('improvement', 0.0):+.1%}

Best Predictions:
"""
    for i, exp in enumerate(best, 1):
        summary_text += f"\n{i}. Error: {exp.prediction_error:.4f} m/s"
        summary_text += f"\n   Predicted: {exp.predicted_my_velocity}"
        summary_text += f"\n   Actual: {exp.actual_my_velocity}\n"
    
    summary_text += "\nWorst Predictions:"
    for i, exp in enumerate(worst, 1):
        summary_text += f"\n{i}. Error: {exp.prediction_error:.4f} m/s"
        summary_text += f"\n   Predicted: {exp.predicted_my_velocity}"
        summary_text += f"\n   Actual: {exp.actual_my_velocity}\n"
    
    tracker.log_text(summary_text, "experiment_summary.txt")
    
    # End MLflow run
    tracker.end_run()
    
    print("\n✓ Experiment tracked to MLflow!")
    print(f"View results: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
