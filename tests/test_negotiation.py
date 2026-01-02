import pytest
import numpy as np
from ape.negotiation import VelocityProposal


@pytest.mark.unit
def test_velocity_proposal_creation():
    """Test creating a VelocityProposal"""
    proposal = VelocityProposal(
        agent_id="ball-001",
        proposed_velocity=np.array([1.0, 2.0]),
        reasoning="Test reasoning",
        confidence=0.9
    )
    
    assert proposal.agent_id == "ball-001"
    np.testing.assert_array_equal(proposal.proposed_velocity, np.array([1.0, 2.0]))
    assert proposal.reasoning == "Test reasoning"
    assert proposal.confidence == 0.9




@pytest.mark.unit
def test_velocity_proposal_default_confidence():
    """Test VelocityProposal has default confidence"""
    proposal = VelocityProposal(
        agent_id="ball-001",
        proposed_velocity=np.array([1.0, 2.0]),
        reasoning="Test"
    )
    
    assert proposal.confidence == 1.0


