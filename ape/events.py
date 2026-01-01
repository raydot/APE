from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime
from queue import Queue

class EventType(Enum):
    COLLISION = "collision"
    FORCE_APPLIED = "force_applied"
    STATE_UPDATE = "state_update"

@dataclass
class PhysicsEvent:
    event_type: EventType
    timestamp: float
    involved_agents: list[str]
    data: dict[str, Any]

    def __str__(self):
        return f"[{self.event_type.value}] {self.involved_agents} @t={self.timestamp: .3f}"

class SimpleEventBus:
    def __init__(self):
        self._queue = Queue()
        self._event_log = []

    def emit(self, event: PhysicsEvent):
        self._queue.put(event)
        self._event_log.append(event)
        print(f"[EVENT] {event}")

    def get_next(self) -> Optional[PhysicsEvent]:
        if not self._queue.empty():
            return self._queue.get()
        return None

    def has_events(self) -> bool:
        return not self._queue.empty()
    
    @property
    def event_log(self) -> list[PhysicsEvent]:
        return self._event_log.copy()
    
    def clear_log(self):
        self._event_log.clear()