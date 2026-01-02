import pytest
from ape.events import SimpleEventBus, PhysicsEvent, EventType


@pytest.mark.unit
def test_event_bus_emit():
    """Test emitting events"""
    bus = SimpleEventBus()
    
    event = PhysicsEvent(
        event_type=EventType.COLLISION,
        timestamp=1.0,
        involved_agents=['ball-001', 'ball-002'],
        data={'test': 'data'}
    )
    
    bus.emit(event)
    
    assert bus.has_events()
    assert len(bus.event_log) == 1


@pytest.mark.unit
def test_event_bus_get_next():
    """Test retrieving events from queue"""
    bus = SimpleEventBus()
    
    event1 = PhysicsEvent(
        event_type=EventType.COLLISION,
        timestamp=1.0,
        involved_agents=['ball-001'],
        data={'order': 1}
    )
    
    event2 = PhysicsEvent(
        event_type=EventType.STATE_UPDATE,
        timestamp=2.0,
        involved_agents=['ball-002'],
        data={'order': 2}
    )
    
    bus.emit(event1)
    bus.emit(event2)
    
    retrieved1 = bus.get_next()
    retrieved2 = bus.get_next()
    
    assert retrieved1.data['order'] == 1
    assert retrieved2.data['order'] == 2


@pytest.mark.unit
def test_event_bus_has_events():
    """Test checking if events are in queue"""
    bus = SimpleEventBus()
    
    assert not bus.has_events()
    
    event = PhysicsEvent(
        event_type=EventType.FORCE_APPLIED,
        timestamp=1.0,
        involved_agents=['ball-001'],
        data={}
    )
    
    bus.emit(event)
    assert bus.has_events()
    
    bus.get_next()
    assert not bus.has_events()


@pytest.mark.unit
def test_event_bus_empty_queue():
    """Test getting from empty queue returns None"""
    bus = SimpleEventBus()
    
    result = bus.get_next()
    assert result is None


@pytest.mark.unit
def test_event_bus_event_log():
    """Test event log persists all events"""
    bus = SimpleEventBus()
    
    event1 = PhysicsEvent(
        event_type=EventType.COLLISION,
        timestamp=1.0,
        involved_agents=['ball-001'],
        data={}
    )
    
    event2 = PhysicsEvent(
        event_type=EventType.STATE_UPDATE,
        timestamp=2.0,
        involved_agents=['ball-002'],
        data={}
    )
    
    bus.emit(event1)
    bus.emit(event2)
    
    # Drain queue
    bus.get_next()
    bus.get_next()
    
    # Log should still have both events
    assert len(bus.event_log) == 2


@pytest.mark.unit
def test_event_bus_clear_log():
    """Test clearing event log"""
    bus = SimpleEventBus()
    
    event = PhysicsEvent(
        event_type=EventType.COLLISION,
        timestamp=1.0,
        involved_agents=['ball-001'],
        data={}
    )
    
    bus.emit(event)
    assert len(bus.event_log) == 1
    
    bus.clear_log()
    assert len(bus.event_log) == 0
