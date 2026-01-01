import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import json
from dotenv import load_dotenv

from ape.events import SimpleEventBus
from ape.physics import WorldState, PhysicsState
from ape.agents import BallAgent, FloorAgent
from ape.runtime import SimpleRuntime

load_dotenv()


class SimulationRecorder:
    def __init__(self):
        self.frames = []
    
    def record_frame(self, world_state: WorldState):
        frame = {}
        for agent_id, state in world_state.get_all_objects().items():
            if state.mass < float('inf'):
                frame[agent_id] = {
                    'position': state.position.copy(),
                    'velocity': state.velocity.copy(),
                    'time': world_state.time
                }
        self.frames.append(frame)
    
    def get_trajectory(self, agent_id: str):
        positions = []
        velocities = []
        times = []
        
        for frame in self.frames:
            if agent_id in frame:
                positions.append(frame[agent_id]['position'])
                velocities.append(frame[agent_id]['velocity'])
                times.append(frame[agent_id]['time'])
        
        return np.array(positions), np.array(velocities), np.array(times)
    
    def save(self, filename: str):
        """Save recorded frames to JSON file"""
        data = {
            'frames': [
                {
                    agent_id: {
                        'position': frame[agent_id]['position'].tolist(),
                        'velocity': frame[agent_id]['velocity'].tolist(),
                        'time': frame[agent_id]['time']
                    }
                    for agent_id in frame
                }
                for frame in self.frames
            ]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"[SAVED] Simulation data to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """Load recorded frames from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        recorder = cls()
        recorder.frames = [
            {
                agent_id: {
                    'position': np.array(frame_data[agent_id]['position']),
                    'velocity': np.array(frame_data[agent_id]['velocity']),
                    'time': frame_data[agent_id]['time']
                }
                for agent_id in frame_data
            }
            for frame_data in data['frames']
        ]
        print(f"[LOADED] Simulation data from {filename}")
        return recorder


def run_simulation_with_recording():
    print("=== APE: Ball Drop with Visualization ===\n")
    
    event_bus = SimpleEventBus()
    world_state = WorldState(gravity=np.array([0.0, -9.8]))
    runtime = SimpleRuntime(event_bus, world_state)
    recorder = SimulationRecorder()
    
    ball_state = PhysicsState(
        position=np.array([0.0, 5.0]),
        velocity=np.array([0.0, 0.0]),
        mass=1.0,
        elasticity=0.8
    )
    world_state.add_object('ball-001', ball_state)
    
    floor_state = PhysicsState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=float('inf'),
        elasticity=1.0
    )
    world_state.add_object('floor-001', floor_state)
    
    llm_client = None
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
            model = "claude-haiku-4-5-20251001"
            print(f"Using Anthropic: {model}\n")
        else:
            raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
    
    ball_agent = BallAgent('ball-001', event_bus, world_state, llm_client, model)
    floor_agent = FloorAgent('floor-001', event_bus, world_state)
    
    runtime.register_agent(ball_agent)
    runtime.register_agent(floor_agent)
    
    print("[RECORDING] Running simulation...")
    max_iterations = 1000
    dt = 0.01
    
    for i in range(max_iterations):
        runtime.iteration = i
        world_state.step(dt)
        runtime._detect_collisions()
        runtime._process_events()
        
        recorder.record_frame(world_state)
        
        if runtime._should_stop():
            print(f"[RECORDING] Simulation complete at iteration {i}")
            break
    
    print(f"[RECORDING] Recorded {len(recorder.frames)} frames\n")
    return recorder, event_bus


def visualize(recorder: SimulationRecorder, event_bus: SimpleEventBus):
    positions, velocities, times = recorder.get_trajectory('ball-001')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Trajectory
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.5, 5.5)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Ball Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='brown', linewidth=3, label='Floor')
    
    trajectory_line, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=1, label='Path')
    ball_point, = ax1.plot([], [], 'ro', markersize=15, label='Ball')
    ax1.legend()
    
    # Right plot: Velocity over time
    ax2.set_xlim(0, times[-1])
    ax2.set_ylim(min(velocities[:, 1]) - 1, max(velocities[:, 1]) + 1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Velocity (m/s)')
    ax2.set_title('Velocity Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    velocity_line, = ax2.plot([], [], 'g-', linewidth=2)
    
    # Collision markers
    collision_times = [e.timestamp for e in event_bus.event_log if e.event_type.value == 'collision']
    for ct in collision_times:
        ax2.axvline(x=ct, color='red', linestyle='--', alpha=0.3)
    
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        trajectory_line.set_data([], [])
        ball_point.set_data([], [])
        velocity_line.set_data([], [])
        time_text.set_text('')
        return trajectory_line, ball_point, velocity_line, time_text
    
    def animate(frame):
           # Update trajectory (show center path)
    trajectory_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
    
    # Update ball position (offset by radius so it sits on floor)
    ball_radius = 0.15  # Visual radius in meters
    ball_point.set_data([positions[frame, 0]], [positions[frame, 1] + ball_radius])
    
    # Update velocity plot
    velocity_line.set_data(times[:frame+1], velocities[:frame+1, 1])
    
    # Update time text
    time_text.set_text(f't = {times[frame]:.2f}s\nv = {velocities[frame, 1]:.2f} m/s')
    
    return trajectory_line, ball_point, velocity_line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(positions), interval=20, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()


def main():
    import sys
    
    # Check if we should load existing data
    if len(sys.argv) > 1 and sys.argv[1] == '--replay':
        print("=== Replaying saved simulation ===\n")
        recorder = SimulationRecorder.load('simulation_data.json')
        event_bus = SimpleEventBus()  # Empty bus for replay
    else:
        recorder, event_bus = run_simulation_with_recording()
        recorder.save('simulation_data.json')
        
        print("=== Event Summary ===")
        collision_count = sum(1 for e in event_bus.event_log if e.event_type.value == 'collision')
        print(f"Total collisions: {collision_count}")
        print(f"Total events: {len(event_bus.event_log)}\n")
    
    print("[VISUALIZATION] Creating animation...")
    visualize(recorder, event_bus)


if __name__ == "__main__":
    main()