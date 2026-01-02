import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EnergySnapshot:
    """Snapshot of system energy at a point in time"""
    time: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    ball_energies: Dict[str, float]


class EnergyTracker:
    """
    Tracks total system energy across collisions.
    
    Validates:
    - Energy conservation (elastic collisions)
    - Energy loss (inelastic collisions)
    - Detects energy "leaks" from bad physics
    """
    
    def __init__(self, gravity: float = 9.8):
        self.gravity = gravity
        self.snapshots: List[EnergySnapshot] = []
        self.violations: List[Dict] = []
    
    def record_snapshot(self, world_state, ball_ids: List[str]):
        """
        Record energy snapshot at current timestep.
        
        Args:
            world_state: WorldState object
            ball_ids: List of ball agent IDs to track
        """
        kinetic = 0.0
        potential = 0.0
        ball_energies = {}
        
        for ball_id in ball_ids:
            state = world_state.get_object(ball_id)
            if state and state.mass < float('inf'):
                # Kinetic energy
                v_squared = np.dot(state.velocity, state.velocity)
                ke = 0.5 * state.mass * v_squared
                
                # Potential energy (relative to y=0)
                pe = state.mass * self.gravity * state.position[1]
                
                kinetic += ke
                potential += pe
                ball_energies[ball_id] = ke + pe
        
        total = kinetic + potential
        
        snapshot = EnergySnapshot(
            time=world_state.time,
            kinetic_energy=kinetic,
            potential_energy=potential,
            total_energy=total,
            ball_energies=ball_energies
        )
        
        self.snapshots.append(snapshot)
        
        # Check for violations (energy increase)
        if len(self.snapshots) > 1:
            prev_snapshot = self.snapshots[-2]
            energy_change = total - prev_snapshot.total_energy
            
            # Allow small increase due to numerical errors (0.1%)
            if energy_change > prev_snapshot.total_energy * 0.001:
                self.violations.append({
                    'time': world_state.time,
                    'energy_before': prev_snapshot.total_energy,
                    'energy_after': total,
                    'increase': energy_change,
                    'increase_pct': 100 * energy_change / prev_snapshot.total_energy
                })
    
    def get_energy_loss(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Dict:
        """
        Calculate energy loss between two snapshots.
        
        Args:
            start_idx: Index of starting snapshot
            end_idx: Index of ending snapshot (None = last)
        
        Returns:
            Dict with energy loss statistics
        """
        if not self.snapshots:
            return {'error': 'No snapshots recorded'}
        
        if end_idx is None:
            end_idx = len(self.snapshots) - 1
        
        start = self.snapshots[start_idx]
        end = self.snapshots[end_idx]
        
        energy_loss = start.total_energy - end.total_energy
        loss_pct = 100 * energy_loss / start.total_energy if start.total_energy > 0 else 0
        
        return {
            'start_time': start.time,
            'end_time': end.time,
            'start_energy': start.total_energy,
            'end_energy': end.total_energy,
            'energy_loss': energy_loss,
            'loss_pct': loss_pct,
            'duration': end.time - start.time
        }
    
    def validate_collision(
        self,
        before_idx: int,
        after_idx: int,
        expected_elasticity: float,
        tolerance: float = 0.05
    ) -> Dict:
        """
        Validate energy change across a collision.
        
        For elastic collision (e=1.0): energy should be conserved
        For inelastic (e<1.0): energy should decrease by ~(1-e²)
        
        Args:
            before_idx: Snapshot index before collision
            after_idx: Snapshot index after collision
            expected_elasticity: Expected coefficient of restitution
            tolerance: Allowed error
        
        Returns:
            Dict with validation results
        """
        if before_idx >= len(self.snapshots) or after_idx >= len(self.snapshots):
            return {'error': 'Invalid snapshot indices'}
        
        before = self.snapshots[before_idx]
        after = self.snapshots[after_idx]
        
        energy_change = after.total_energy - before.total_energy
        energy_change_pct = energy_change / before.total_energy if before.total_energy > 0 else 0
        
        # Expected energy retention for inelastic collision
        expected_retention = expected_elasticity ** 2
        expected_energy = before.total_energy * expected_retention
        
        actual_retention = after.total_energy / before.total_energy if before.total_energy > 0 else 0
        
        error = abs(actual_retention - expected_retention)
        
        valid = error <= tolerance
        
        return {
            'valid': valid,
            'before_energy': before.total_energy,
            'after_energy': after.total_energy,
            'expected_energy': expected_energy,
            'energy_change': energy_change,
            'energy_change_pct': energy_change_pct * 100,
            'expected_retention': expected_retention * 100,
            'actual_retention': actual_retention * 100,
            'error': error * 100,
            'tolerance': tolerance * 100
        }
    
    def get_stats(self) -> Dict:
        """Get energy tracking statistics"""
        if not self.snapshots:
            return {'error': 'No snapshots recorded'}
        
        energies = [s.total_energy for s in self.snapshots]
        
        return {
            'total_snapshots': len(self.snapshots),
            'initial_energy': self.snapshots[0].total_energy,
            'final_energy': self.snapshots[-1].total_energy,
            'total_loss': self.snapshots[0].total_energy - self.snapshots[-1].total_energy,
            'loss_pct': 100 * (self.snapshots[0].total_energy - self.snapshots[-1].total_energy) / self.snapshots[0].total_energy if self.snapshots[0].total_energy > 0 else 0,
            'min_energy': min(energies),
            'max_energy': max(energies),
            'violations': len(self.violations),
            'duration': self.snapshots[-1].time - self.snapshots[0].time
        }
    
    def print_stats(self):
        """Print human-readable statistics"""
        stats = self.get_stats()
        
        if 'error' in stats:
            print(f"[ENERGY] {stats['error']}")
            return
        
        print(f"\n{'='*60}")
        print("ENERGY TRACKING STATISTICS")
        print(f"{'='*60}")
        print(f"Duration: {stats['duration']:.3f}s")
        print(f"Snapshots: {stats['total_snapshots']}")
        
        print(f"\nEnergy:")
        print(f"  Initial: {stats['initial_energy']:.3f} J")
        print(f"  Final: {stats['final_energy']:.3f} J")
        print(f"  Loss: {stats['total_loss']:.3f} J ({stats['loss_pct']:.1f}%)")
        
        print(f"\nRange:")
        print(f"  Min: {stats['min_energy']:.3f} J")
        print(f"  Max: {stats['max_energy']:.3f} J")
        
        if self.violations:
            print(f"\n⚠ Energy Violations: {len(self.violations)}")
            print("  (Energy increased - violates thermodynamics)")
            for v in self.violations[:3]:
                print(f"  t={v['time']:.3f}s: +{v['increase']:.3f}J (+{v['increase_pct']:.2f}%)")
            if len(self.violations) > 3:
                print(f"  ... and {len(self.violations) - 3} more")
        else:
            print(f"\n✓ No energy violations detected")
        
        # Interpretation
        if stats['loss_pct'] < 1:
            print(f"\n✓ Excellent energy conservation")
        elif stats['loss_pct'] < 10:
            print(f"\n✓ Good energy conservation (expected for inelastic collisions)")
        elif stats['loss_pct'] < 30:
            print(f"\n⚠ Moderate energy loss (check elasticity values)")
        else:
            print(f"\n✗ High energy loss (possible physics errors)")
    
    def plot_energy(self):
        """Plot energy over time (requires matplotlib)"""
        import matplotlib.pyplot as plt
        
        if not self.snapshots:
            print("No data to plot")
            return
        
        times = [s.time for s in self.snapshots]
        kinetic = [s.kinetic_energy for s in self.snapshots]
        potential = [s.potential_energy for s in self.snapshots]
        total = [s.total_energy for s in self.snapshots]
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(times, kinetic, 'b-', label='Kinetic Energy', linewidth=2)
        plt.plot(times, potential, 'r-', label='Potential Energy', linewidth=2)
        plt.plot(times, total, 'g-', label='Total Energy', linewidth=2, linestyle='--')
        
        # Mark violations
        for v in self.violations:
            plt.axvline(x=v['time'], color='orange', linestyle=':', alpha=0.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('System Energy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
