import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VelocityProposal:
    """
    Agent's proposal for their own velocity after collision.
    Simplified: each agent only proposes their own velocity.
    """
    agent_id: str
    proposed_velocity: np.ndarray
    reasoning: str
    confidence: float = 1.0  # 0-1, how confident agent is


class NegotiationProtocol:
    """
    Protocol for validating collision proposals from multiple agents.
    
    Checks:
    1. Momentum conservation
    2. Energy conservation (or acceptable loss)
    3. Physical plausibility
    """
    
    @staticmethod
    def validate_proposals(
        proposal1: VelocityProposal,
        proposal2: VelocityProposal,
        mass1: float,
        mass2: float,
        velocity1_before: np.ndarray,
        velocity2_before: np.ndarray,
        tolerance: float = 0.05
    ) -> Tuple[bool, str, dict]:
        """
        Validate if two velocity proposals satisfy conservation laws.
        
        Args:
            proposal1: First agent's velocity proposal
            proposal2: Second agent's velocity proposal
            mass1: Mass of first agent
            mass2: Mass of second agent
            velocity1_before: Velocity of first agent before collision
            velocity2_before: Velocity of second agent before collision
            tolerance: Allowed error (0-1, as fraction)
        
        Returns:
            (valid: bool, reason: str, details: dict)
        """
        v1_after = proposal1.proposed_velocity
        v2_after = proposal2.proposed_velocity
        
        details = {}
        
        # Check 1: Momentum conservation
        momentum_before = mass1 * velocity1_before + mass2 * velocity2_before
        momentum_after = mass1 * v1_after + mass2 * v2_after
        
        momentum_error = np.linalg.norm(momentum_after - momentum_before)
        momentum_magnitude = np.linalg.norm(momentum_before)
        momentum_error_pct = momentum_error / (momentum_magnitude + 1e-6)
        
        details['momentum_before'] = momentum_before
        details['momentum_after'] = momentum_after
        details['momentum_error_pct'] = momentum_error_pct
        
        if momentum_error_pct > tolerance:
            return False, f"Momentum not conserved: error={momentum_error_pct:.1%} (limit: {tolerance:.1%})", details
        
        # Check 2: Energy (should decrease or stay same, never increase)
        energy_before = 0.5 * mass1 * np.dot(velocity1_before, velocity1_before) + \
                        0.5 * mass2 * np.dot(velocity2_before, velocity2_before)
        
        energy_after = 0.5 * mass1 * np.dot(v1_after, v1_after) + \
                       0.5 * mass2 * np.dot(v2_after, v2_after)
        
        energy_change = energy_after - energy_before
        energy_change_pct = energy_change / (energy_before + 1e-6)
        
        details['energy_before'] = energy_before
        details['energy_after'] = energy_after
        details['energy_change_pct'] = energy_change_pct
        
        # Allow small energy increase due to numerical errors
        if energy_change_pct > tolerance:
            return False, f"Energy increased: {energy_change_pct:.1%} (violates thermodynamics)", details
        
        # Allow up to 50% energy loss (very inelastic collision)
        if energy_change_pct < -0.5:
            return False, f"Energy loss too large: {energy_change_pct:.1%} (suspicious)", details
        
        # Check 3: Velocity magnitude sanity check
        # Post-collision velocities shouldn't be much larger than pre-collision
        v1_before_mag = np.linalg.norm(velocity1_before)
        v2_before_mag = np.linalg.norm(velocity2_before)
        v1_after_mag = np.linalg.norm(v1_after)
        v2_after_mag = np.linalg.norm(v2_after)
        
        max_before = max(v1_before_mag, v2_before_mag)
        max_after = max(v1_after_mag, v2_after_mag)
        
        # Allow 2x increase (for light ball hit by heavy ball)
        if max_after > max_before * 2.5:
            return False, f"Velocity increase suspicious: {max_before:.2f} → {max_after:.2f} m/s", details
        
        # All checks passed
        return True, "Conservation laws satisfied", details
    
    @staticmethod
    def calculate_agreement_score(
        proposal1: VelocityProposal,
        proposal2: VelocityProposal,
        ground_truth_v1: np.ndarray,
        ground_truth_v2: np.ndarray
    ) -> dict:
        """
        Calculate how close proposals are to ground truth.
        
        Returns dict with error metrics for analysis.
        """
        error1 = np.linalg.norm(proposal1.proposed_velocity - ground_truth_v1)
        error2 = np.linalg.norm(proposal2.proposed_velocity - ground_truth_v2)
        
        # Relative errors
        gt1_mag = np.linalg.norm(ground_truth_v1)
        gt2_mag = np.linalg.norm(ground_truth_v2)
        
        rel_error1 = error1 / (gt1_mag + 1e-6)
        rel_error2 = error2 / (gt2_mag + 1e-6)
        
        return {
            'agent1_error': error1,
            'agent2_error': error2,
            'agent1_rel_error': rel_error1,
            'agent2_rel_error': rel_error2,
            'avg_rel_error': (rel_error1 + rel_error2) / 2,
            'both_accurate': rel_error1 < 0.1 and rel_error2 < 0.1
        }
