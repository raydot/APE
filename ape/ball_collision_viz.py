import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from typing import Dict, List
import json


class BallCollisionRecorder:
    """
    Records ball-to-ball collision simulation data for visualization.
    Tracks multiple balls and their interactions.
    """
    
    def __init__(self):
        self.frames = []
        self.collision_events = []
        self.negotiation_outcomes = []
    
    def record_frame(self, world_state, ball_ids: List[str]):
        """
        Record positions and velocities of all balls at current timestep.
        
        Args:
            world_state: WorldState object
            ball_ids: List of ball agent IDs to track
        """
        frame = {
            'time': world_state.time,
            'balls': {}
        }
        
        for ball_id in ball_ids:
            state = world_state.get_object(ball_id)
            if state:
                frame['balls'][ball_id] = {
                    'position': state.position.copy(),
                    'velocity': state.velocity.copy(),
                    'mass': state.mass
                }
        
        self.frames.append(frame)
    
    def record_collision(self, time: float, ball1_id: str, ball2_id: str, collision_point: np.ndarray):
        """Record a collision event"""
        self.collision_events.append({
            'time': time,
            'ball1': ball1_id,
            'ball2': ball2_id,
            'point': collision_point.copy()
        })
    
    def record_negotiation_outcome(self, time: float, outcome: Dict):
        """Record negotiation outcome for analysis"""
        # Convert to JSON-serializable format
        agreement_score = outcome.get('agreement_score')
        if agreement_score and isinstance(agreement_score, dict):
            # Convert numpy values to Python types
            agreement_score = {
                k: float(v) if hasattr(v, 'item') else v
                for k, v in agreement_score.items()
            }
        
        self.negotiation_outcomes.append({
            'time': time,
            'source': outcome.get('source'),
            'valid': bool(outcome.get('valid')) if outcome.get('valid') is not None else None,
            'agreement_score': agreement_score
        })
    
    def save(self, filename: str):
        """Save recording to JSON file"""
        data = {
            'frames': [
                {
                    'time': f['time'],
                    'balls': {
                        ball_id: {
                            'position': ball_data['position'].tolist(),
                            'velocity': ball_data['velocity'].tolist(),
                            'mass': ball_data['mass']
                        }
                        for ball_id, ball_data in f['balls'].items()
                    }
                }
                for f in self.frames
            ],
            'collision_events': [
                {
                    'time': e['time'],
                    'ball1': e['ball1'],
                    'ball2': e['ball2'],
                    'point': e['point'].tolist()
                }
                for e in self.collision_events
            ],
            'negotiation_outcomes': self.negotiation_outcomes
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[RECORDER] Saved {len(self.frames)} frames to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """Load recording from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        recorder = cls()
        
        # Reconstruct frames
        recorder.frames = [
            {
                'time': f['time'],
                'balls': {
                    ball_id: {
                        'position': np.array(ball_data['position']),
                        'velocity': np.array(ball_data['velocity']),
                        'mass': ball_data['mass']
                    }
                    for ball_id, ball_data in f['balls'].items()
                }
            }
            for f in data['frames']
        ]
        
        # Reconstruct collision events
        recorder.collision_events = [
            {
                'time': e['time'],
                'ball1': e['ball1'],
                'ball2': e['ball2'],
                'point': np.array(e['point'])
            }
            for e in data['collision_events']
        ]
        
        recorder.negotiation_outcomes = data.get('negotiation_outcomes', [])
        
        print(f"[RECORDER] Loaded {len(recorder.frames)} frames from {filename}")
        return recorder


def visualize_ball_collision(recorder: BallCollisionRecorder, ball_radius: float = 0.15):
    """
    Create animated visualization of ball-to-ball collision.
    
    Shows:
    - Ball trajectories (different colors)
    - Collision points
    - Velocity vectors
    - Energy over time
    - Negotiation outcomes
    """
    
    if not recorder.frames:
        print("No frames to visualize")
        return
    
    # Extract data
    times = [f['time'] for f in recorder.frames]
    ball_ids = list(recorder.frames[0]['balls'].keys())
    
    # Get trajectories for each ball
    trajectories = {}
    for ball_id in ball_ids:
        positions = []
        velocities = []
        for frame in recorder.frames:
            if ball_id in frame['balls']:
                positions.append(frame['balls'][ball_id]['position'])
                velocities.append(frame['balls'][ball_id]['velocity'])
        trajectories[ball_id] = {
            'positions': np.array(positions),
            'velocities': np.array(velocities)
        }
    
    # Calculate energy over time
    total_energy = []
    for frame in recorder.frames:
        energy = 0.0
        for ball_id, ball_data in frame['balls'].items():
            v = ball_data['velocity']
            m = ball_data['mass']
            energy += 0.5 * m * np.dot(v, v)
        total_energy.append(energy)
    
    # Setup figure
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Trajectory view
    ax_traj = plt.subplot(2, 2, 1)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_title('Ball Trajectories')
    ax_traj.grid(True, alpha=0.3)
    
    # Set limits based on data
    all_positions = np.vstack([traj['positions'] for traj in trajectories.values()])
    x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
    y_min, y_max = all_positions[:, 1].min() - 0.5, all_positions[:, 1].max() + 0.5
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    
    # Colors for each ball
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    ball_colors = {ball_id: colors[i % len(colors)] for i, ball_id in enumerate(ball_ids)}
    
    # Trajectory lines and ball circles
    traj_lines = {}
    ball_circles = {}
    velocity_arrows = {}
    
    for ball_id in ball_ids:
        # Trajectory line
        line, = ax_traj.plot([], [], '-', color=ball_colors[ball_id], alpha=0.3, linewidth=1, label=ball_id)
        traj_lines[ball_id] = line
        
        # Ball circle
        circle = Circle((0, 0), ball_radius, color=ball_colors[ball_id], alpha=0.8)
        ax_traj.add_patch(circle)
        ball_circles[ball_id] = circle
        
        # Velocity arrow
        arrow = ax_traj.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, 
                             fc=ball_colors[ball_id], ec=ball_colors[ball_id], alpha=0.6)
        velocity_arrows[ball_id] = arrow
    
    # Collision markers
    collision_markers = ax_traj.scatter([], [], c='yellow', s=200, marker='*', 
                                       edgecolors='red', linewidths=2, zorder=10, label='Collision')
    
    ax_traj.legend(loc='upper right')
    
    # Top right: Energy over time
    ax_energy = plt.subplot(2, 2, 2)
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Total Energy (J)')
    ax_energy.set_title('System Energy')
    ax_energy.grid(True, alpha=0.3)
    
    energy_line, = ax_energy.plot([], [], 'g-', linewidth=2)
    
    # Mark collision times
    for event in recorder.collision_events:
        ax_energy.axvline(x=event['time'], color='red', linestyle='--', alpha=0.3)
    
    # Bottom left: Velocity magnitudes
    ax_vel = plt.subplot(2, 2, 3)
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Speed (m/s)')
    ax_vel.set_title('Ball Speeds')
    ax_vel.grid(True, alpha=0.3)
    
    vel_lines = {}
    for ball_id in ball_ids:
        line, = ax_vel.plot([], [], '-', color=ball_colors[ball_id], linewidth=2, label=ball_id)
        vel_lines[ball_id] = line
    ax_vel.legend()
    
    # Bottom right: Negotiation outcomes
    ax_info = plt.subplot(2, 2, 4)
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    def init():
        """Initialize animation"""
        for line in traj_lines.values():
            line.set_data([], [])
        energy_line.set_data([], [])
        for line in vel_lines.values():
            line.set_data([], [])
        collision_markers.set_offsets(np.empty((0, 2)))
        info_text.set_text('')
        return list(traj_lines.values()) + [energy_line] + list(vel_lines.values()) + [collision_markers, info_text]
    
    def animate(frame_idx):
        """Update animation frame"""
        frame = recorder.frames[frame_idx]
        current_time = frame['time']
        
        # Update trajectories and balls
        for ball_id in ball_ids:
            if ball_id in frame['balls']:
                # Update trajectory line
                positions = trajectories[ball_id]['positions'][:frame_idx+1]
                traj_lines[ball_id].set_data(positions[:, 0], positions[:, 1])
                
                # Update ball position
                pos = frame['balls'][ball_id]['position']
                ball_circles[ball_id].center = (pos[0], pos[1])
                
                # Update velocity arrow
                vel = frame['balls'][ball_id]['velocity']
                # Remove old arrow and create new one
                velocity_arrows[ball_id].remove()
                arrow = ax_traj.arrow(pos[0], pos[1], vel[0]*0.2, vel[1]*0.2,
                                     head_width=0.08, head_length=0.08,
                                     fc=ball_colors[ball_id], ec=ball_colors[ball_id], alpha=0.6)
                velocity_arrows[ball_id] = arrow
        
        # Update collision markers
        collisions_so_far = [e for e in recorder.collision_events if e['time'] <= current_time]
        if collisions_so_far:
            collision_points = np.array([e['point'] for e in collisions_so_far])
            collision_markers.set_offsets(collision_points)
        
        # Update energy plot
        energy_line.set_data(times[:frame_idx+1], total_energy[:frame_idx+1])
        ax_energy.set_xlim(0, times[-1])
        ax_energy.set_ylim(0, max(total_energy) * 1.1)
        
        # Update velocity plots
        for ball_id in ball_ids:
            velocities = trajectories[ball_id]['velocities'][:frame_idx+1]
            speeds = np.linalg.norm(velocities, axis=1)
            vel_lines[ball_id].set_data(times[:frame_idx+1], speeds)
        ax_vel.set_xlim(0, times[-1])
        all_speeds = [np.linalg.norm(trajectories[bid]['velocities'], axis=1).max() for bid in ball_ids]
        ax_vel.set_ylim(0, max(all_speeds) * 1.1)
        
        # Update info text
        info_lines = [f"Time: {current_time:.3f}s\n"]
        
        for ball_id in ball_ids:
            if ball_id in frame['balls']:
                pos = frame['balls'][ball_id]['position']
                vel = frame['balls'][ball_id]['velocity']
                speed = np.linalg.norm(vel)
                info_lines.append(f"\n{ball_id}:")
                info_lines.append(f"  pos: [{pos[0]:.2f}, {pos[1]:.2f}]")
                info_lines.append(f"  vel: [{vel[0]:.2f}, {vel[1]:.2f}]")
                info_lines.append(f"  speed: {speed:.2f} m/s")
        
        # Show recent collisions
        recent_collisions = [e for e in collisions_so_far if current_time - e['time'] < 0.5]
        if recent_collisions:
            info_lines.append(f"\n\nRecent Collisions:")
            for e in recent_collisions[-3:]:
                info_lines.append(f"  {e['ball1']} <-> {e['ball2']}")
        
        # Show negotiation stats
        if recorder.negotiation_outcomes:
            accepted = sum(1 for o in recorder.negotiation_outcomes if o.get('valid'))
            total = len(recorder.negotiation_outcomes)
            info_lines.append(f"\n\nNegotiation Stats:")
            info_lines.append(f"  Accepted: {accepted}/{total}")
            info_lines.append(f"  Rate: {100*accepted/total:.1f}%")
        
        info_text.set_text(''.join(info_lines))
        
        return list(traj_lines.values()) + [energy_line] + list(vel_lines.values()) + [collision_markers, info_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(recorder.frames), interval=20, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim
