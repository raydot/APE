import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class TraceLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TraceEntry:
    """Single entry in the trace log"""
    timestamp: float
    level: TraceLevel
    agent_id: str
    event_type: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['level'] = self.level.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['level'] = TraceLevel(data['level'])
        return cls(**data)


@dataclass
class LLMTrace:
    """Detailed trace of an LLM interaction"""
    agent_id: str
    timestamp: float
    model: str
    prompt: str
    response: str
    reasoning: str
    duration: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class TraceCollector:
    """
    Collects and manages execution traces for debugging and analysis.
    Provides structured logging of agent actions, LLM calls, and events.
    """
    
    def __init__(self):
        self.traces: List[TraceEntry] = []
        self.llm_traces: List[LLMTrace] = []
        self.start_time = time.time()
    
    def log(
        self,
        level: TraceLevel,
        agent_id: str,
        event_type: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """Add a trace entry"""
        entry = TraceEntry(
            timestamp=time.time() - self.start_time,
            level=level,
            agent_id=agent_id,
            event_type=event_type,
            message=message,
            data=data or {}
        )
        self.traces.append(entry)
    
    def log_llm_call(
        self,
        agent_id: str,
        model: str,
        prompt: str,
        response: str,
        reasoning: str,
        duration: float,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """Log an LLM interaction"""
        trace = LLMTrace(
            agent_id=agent_id,
            timestamp=time.time() - self.start_time,
            model=model,
            prompt=prompt,
            response=response,
            reasoning=reasoning,
            duration=duration,
            tokens_used=tokens_used,
            cost=cost
        )
        self.llm_traces.append(trace)
    
    def get_agent_traces(self, agent_id: str) -> List[TraceEntry]:
        """Get all traces for a specific agent"""
        return [t for t in self.traces if t.agent_id == agent_id]
    
    def get_llm_traces(self, agent_id: Optional[str] = None) -> List[LLMTrace]:
        """Get LLM traces, optionally filtered by agent"""
        if agent_id:
            return [t for t in self.llm_traces if t.agent_id == agent_id]
        return self.llm_traces
    
    def get_traces_by_type(self, event_type: str) -> List[TraceEntry]:
        """Get all traces of a specific event type"""
        return [t for t in self.traces if t.event_type == event_type]
    
    def get_traces_by_level(self, level: TraceLevel) -> List[TraceEntry]:
        """Get all traces at a specific level"""
        return [t for t in self.traces if t.level == level]
    
    def save(self, filename: str):
        """Save traces to JSON file"""
        data = {
            'traces': [t.to_dict() for t in self.traces],
            'llm_traces': [t.to_dict() for t in self.llm_traces],
            'start_time': self.start_time,
            'duration': time.time() - self.start_time
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[TRACE] Saved {len(self.traces)} traces to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """Load traces from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        collector = cls()
        collector.traces = [TraceEntry.from_dict(t) for t in data['traces']]
        collector.llm_traces = [LLMTrace.from_dict(t) for t in data['llm_traces']]
        collector.start_time = data['start_time']
        print(f"[TRACE] Loaded {len(collector.traces)} traces from {filename}")
        return collector
    
    def print_summary(self):
        """Print human-readable summary of traces"""
        print("\n" + "="*80)
        print("TRACE SUMMARY")
        print("="*80)
        
        print(f"\nTotal traces: {len(self.traces)}")
        print(f"Total LLM calls: {len(self.llm_traces)}")
        print(f"Duration: {time.time() - self.start_time:.2f}s")
        
        # Traces by level
        print("\nTraces by level:")
        for level in TraceLevel:
            count = len(self.get_traces_by_level(level))
            if count > 0:
                print(f"  {level.value}: {count}")
        
        # Traces by agent
        agents = set(t.agent_id for t in self.traces)
        print(f"\nTraces by agent:")
        for agent_id in sorted(agents):
            count = len(self.get_agent_traces(agent_id))
            print(f"  {agent_id}: {count}")
        
        # LLM statistics
        if self.llm_traces:
            total_duration = sum(t.duration for t in self.llm_traces)
            avg_duration = total_duration / len(self.llm_traces)
            
            print(f"\nLLM Statistics:")
            print(f"  Total calls: {len(self.llm_traces)}")
            print(f"  Total time: {total_duration:.2f}s")
            print(f"  Avg time/call: {avg_duration:.2f}s")
            
            if any(t.cost for t in self.llm_traces):
                total_cost = sum(t.cost or 0 for t in self.llm_traces)
                print(f"  Total cost: ${total_cost:.4f}")
    
    def print_timeline(self, max_entries: int = 50):
        """Print chronological timeline of events"""
        print("\n" + "="*80)
        print("EVENT TIMELINE")
        print("="*80)
        
        for i, trace in enumerate(self.traces[:max_entries]):
            level_symbol = {
                TraceLevel.DEBUG: "🔍",
                TraceLevel.INFO: "ℹ️",
                TraceLevel.WARNING: "⚠️",
                TraceLevel.ERROR: "❌"
            }.get(trace.level, "•")
            
            print(f"\n[{trace.timestamp:6.3f}s] {level_symbol} {trace.agent_id}")
            print(f"  {trace.event_type}: {trace.message}")
            
            if trace.data:
                for key, value in trace.data.items():
                    if isinstance(value, (list, dict)):
                        print(f"    {key}: {json.dumps(value)[:60]}...")
                    else:
                        print(f"    {key}: {value}")
        
        if len(self.traces) > max_entries:
            print(f"\n... and {len(self.traces) - max_entries} more entries")
    
    def print_llm_conversations(self, max_entries: int = 5):
        """Print detailed LLM conversation logs"""
        print("\n" + "="*80)
        print("LLM CONVERSATIONS")
        print("="*80)
        
        for i, trace in enumerate(self.llm_traces[:max_entries]):
            print(f"\n{'='*80}")
            print(f"LLM Call #{i+1} - {trace.agent_id} @ {trace.timestamp:.3f}s")
            print(f"Model: {trace.model} | Duration: {trace.duration:.2f}s")
            if trace.cost:
                print(f"Cost: ${trace.cost:.4f}")
            print(f"{'='*80}")
            
            print("\n📤 PROMPT:")
            print("-" * 80)
            print(trace.prompt[:500])
            if len(trace.prompt) > 500:
                print(f"... ({len(trace.prompt) - 500} more characters)")
            
            print("\n📥 RESPONSE:")
            print("-" * 80)
            print(trace.response[:500])
            if len(trace.response) > 500:
                print(f"... ({len(trace.response) - 500} more characters)")
            
            print("\n💭 REASONING:")
            print("-" * 80)
            print(trace.reasoning)
        
        if len(self.llm_traces) > max_entries:
            print(f"\n... and {len(self.llm_traces) - max_entries} more LLM calls")


class ObservableAgent:
    """
    Mixin class that adds observability to agents.
    Agents can inherit from this to get automatic trace logging.
    """
    
    def __init__(self, trace_collector: Optional[TraceCollector] = None):
        self.trace_collector = trace_collector or TraceCollector()
    
    def _log(self, level: TraceLevel, event_type: str, message: str, data: Optional[Dict] = None):
        """Log a trace entry"""
        if hasattr(self, 'agent_id'):
            self.trace_collector.log(level, self.agent_id, event_type, message, data)
    
    def _log_llm(
        self,
        model: str,
        prompt: str,
        response: str,
        reasoning: str,
        duration: float,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """Log an LLM interaction"""
        if hasattr(self, 'agent_id'):
            self.trace_collector.log_llm_call(
                self.agent_id, model, prompt, response, reasoning,
                duration, tokens_used, cost
            )
