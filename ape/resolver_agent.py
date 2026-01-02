import numpy as np
from typing import Dict, Optional

from .negotiation import VelocityProposal, NegotiationProtocol
from .events import SimpleEventBus
from .physics import WorldState
from .tools import ToolRegistry


class CollisionResolverAgent:
    """
    Mediates ball-to-ball collisions.
    
    Responsibilities:
    1. Request velocity proposals from both balls
    2. Validate proposals against conservation laws
    3. If valid: accept proposals
    4. If invalid: impose ground truth solution
    5. Track negotiation statistics
    """
    
    def __init__(
        self,
        event_bus: SimpleEventBus,
        world_state: WorldState,
        tool_registry: ToolRegistry,
        trace_collector=None
    ):
        self.event_bus = event_bus
        self.world = world_state
        self.tools = tool_registry
        self.trace = trace_collector
        
        # Statistics
        self.stats = {
            'total_negotiations': 0,
            'proposals_accepted': 0,
            'proposals_rejected': 0,
            'ground_truth_imposed': 0,
            'momentum_violations': 0,
            'energy_violations': 0
        }
    
    def handle_collision(
        self,
        ball1_agent,
        ball2_agent,
        collision_data: Dict
    ) -> Dict:
        """
        Mediate collision between two ball agents.
        
        Args:
            ball1_agent: First ball agent (has propose_velocity method)
            ball2_agent: Second ball agent (has propose_velocity method)
            collision_data: Collision info from detector
        
        Returns:
            Outcome dict with both new velocities and metadata
        """
        self.stats['total_negotiations'] += 1
        
        ball1_id = collision_data['ball1_id']
        ball2_id = collision_data['ball2_id']
        
        print(f"\n[RESOLVER] Mediating collision: {ball1_id} <-> {ball2_id}")
        
        if self.trace:
            self.trace.log(
                level='info',
                agent_id='resolver',
                event_type='negotiation_start',
                message=f"Starting negotiation for {ball1_id} <-> {ball2_id}",
                data={
                    'ball1': ball1_id,
                    'ball2': ball2_id,
                    'collision_normal': collision_data['collision_normal'].tolist()
                }
            )
        
        # Get current states
        state1 = self.world.get_object(ball1_id)
        state2 = self.world.get_object(ball2_id)
        
        # Step 1: Get proposals from both agents
        print(f"[RESOLVER] Requesting proposals from both agents...")
        
        proposal1 = ball1_agent.propose_velocity(collision_data, ball2_id)
        proposal2 = ball2_agent.propose_velocity(collision_data, ball1_id)
        
        print(f"[RESOLVER] Proposals received:")
        print(f"  {ball1_id}: {proposal1.proposed_velocity} (confidence: {proposal1.confidence:.2f})")
        print(f"  {ball2_id}: {proposal2.proposed_velocity} (confidence: {proposal2.confidence:.2f})")
        
        # Step 2: Validate proposals
        valid, reason, details = NegotiationProtocol.validate_proposals(
            proposal1, proposal2,
            state1.mass, state2.mass,
            state1.velocity, state2.velocity,
            tolerance=0.05
        )
        
        if valid:
            print(f"[RESOLVER] ✓ Proposals VALID: {reason}")
            print(f"  Momentum error: {details['momentum_error_pct']:.2%}")
            print(f"  Energy change: {details['energy_change_pct']:.2%}")
            self.stats['proposals_accepted'] += 1
            
            # Accept agent proposals
            outcome = {
                'velocity1': proposal1.proposed_velocity,
                'velocity2': proposal2.proposed_velocity,
                'source': 'agent_negotiation',
                'valid': True,
                'details': details
            }
        else:
            print(f"[RESOLVER] ✗ Proposals INVALID: {reason}")
            self.stats['proposals_rejected'] += 1
            
            if 'Momentum' in reason:
                self.stats['momentum_violations'] += 1
            if 'Energy' in reason:
                self.stats['energy_violations'] += 1
            
            # Impose ground truth
            print(f"[RESOLVER] ⚠ Imposing ground truth solution")
            self.stats['ground_truth_imposed'] += 1
            
            tool = self.tools.get("calculate_two_body_collision")
            result = tool.execute(
                mass1=state1.mass,
                velocity1=state1.velocity.tolist(),
                mass2=state2.mass,
                velocity2=state2.velocity.tolist(),
                collision_normal=collision_data['collision_normal'].tolist(),
                elasticity=min(state1.elasticity, state2.elasticity)
            )
            
            outcome = {
                'velocity1': np.array(result['velocity1_after']),
                'velocity2': np.array(result['velocity2_after']),
                'source': 'ground_truth',
                'valid': False,
                'rejection_reason': reason,
                'tool_reasoning': result['reasoning']
            }
        
        # Step 3: Calculate agreement with ground truth (for analysis)
        tool = self.tools.get("calculate_two_body_collision")
        ground_truth = tool.execute(
            mass1=state1.mass,
            velocity1=state1.velocity.tolist(),
            mass2=state2.mass,
            velocity2=state2.velocity.tolist(),
            collision_normal=collision_data['collision_normal'].tolist(),
            elasticity=min(state1.elasticity, state2.elasticity)
        )
        
        agreement = NegotiationProtocol.calculate_agreement_score(
            proposal1, proposal2,
            np.array(ground_truth['velocity1_after']),
            np.array(ground_truth['velocity2_after'])
        )
        
        outcome['agreement_score'] = agreement
        
        if agreement['both_accurate']:
            print(f"[RESOLVER] ✓ Both agents were accurate (avg error: {agreement['avg_rel_error']:.1%})")
        else:
            print(f"[RESOLVER] ⚠ Agent accuracy issues (avg error: {agreement['avg_rel_error']:.1%})")
        
        # Step 4: Apply outcome
        state1.velocity = outcome['velocity1']
        state2.velocity = outcome['velocity2']
        self.world.update_object(ball1_id, state1)
        self.world.update_object(ball2_id, state2)
        
        print(f"[RESOLVER] Applied velocities:")
        print(f"  {ball1_id}: {outcome['velocity1']}")
        print(f"  {ball2_id}: {outcome['velocity2']}")
        
        if self.trace:
            self.trace.log(
                level='info',
                agent_id='resolver',
                event_type='negotiation_complete',
                message=f"Negotiation complete: {outcome['source']}",
                data={
                    'outcome': {
                        'velocity1': outcome['velocity1'].tolist(),
                        'velocity2': outcome['velocity2'].tolist(),
                        'source': outcome['source']
                    },
                    'valid': valid,
                    'agreement_score': agreement
                }
            )
        
        return outcome
    
    def get_stats(self) -> Dict:
        """Get negotiation statistics"""
        stats = self.stats.copy()
        
        if stats['total_negotiations'] > 0:
            stats['acceptance_rate'] = stats['proposals_accepted'] / stats['total_negotiations']
            stats['rejection_rate'] = stats['proposals_rejected'] / stats['total_negotiations']
            stats['ground_truth_rate'] = stats['ground_truth_imposed'] / stats['total_negotiations']
        else:
            stats['acceptance_rate'] = 0.0
            stats['rejection_rate'] = 0.0
            stats['ground_truth_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print human-readable statistics"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("RESOLVER STATISTICS")
        print(f"{'='*60}")
        print(f"Total negotiations: {stats['total_negotiations']}")
        
        if stats['total_negotiations'] > 0:
            print(f"\nProposal outcomes:")
            print(f"  Accepted: {stats['proposals_accepted']} ({stats['acceptance_rate']*100:.1f}%)")
            print(f"  Rejected: {stats['proposals_rejected']} ({stats['rejection_rate']*100:.1f}%)")
            print(f"  Ground truth imposed: {stats['ground_truth_imposed']} ({stats['ground_truth_rate']*100:.1f}%)")
            
            print(f"\nViolation types:")
            print(f"  Momentum violations: {stats['momentum_violations']}")
            print(f"  Energy violations: {stats['energy_violations']}")
            
            if stats['acceptance_rate'] > 0.8:
                print(f"\n✓ High acceptance rate - agents understand physics well")
            elif stats['acceptance_rate'] > 0.5:
                print(f"\n⚠ Moderate acceptance rate - agents need improvement")
            else:
                print(f"\n✗ Low acceptance rate - agents struggling with physics")
