# APE (Agentic Physics Engine)

An experimental physics engine where physical objects are autonomous agents that reason about physics using LLMs, rather than traditional numerical solvers.

## Overview

APE explores a novel approach to physics simulation: instead of hardcoded equations, physical objects use large language models to reason about and negotiate their interactions. A ball doesn't follow `v = v0 + at`—it _thinks_ about what should happen when it hits the floor.

This is primarily an **agent infrastructure project** using physics as a concrete test domain. Physics provides:

- Objective success criteria (conservation laws)
- Visual debugging (you can see when it's wrong)
- Multi-agent coordination challenges
- Both fast reflexes (collision detection) and slow reasoning (outcome negotiation)

## Architecture

**Hybrid approach**: Start with hardcoded agents to validate the core idea, then extract infrastructure patterns from what actually works.

**Current Status**: Week 1 implementation

- Ball and floor agents negotiating collisions via LLM reasoning
- Simple event system for physics interactions
- 2D simulation with gravity and elastic collisions
- Matplotlib visualization

## Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ape

# Add API keys to .env
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY

# Run basic ball drop scenario
python scenarios/ball_drop.py

# Run with visualization
python scenarios/visualize.py
```

## Project Structure

```
ape/
├── ape/              # Core engine
│   ├── events.py     # Event system
│   ├── agents.py     # Agent base classes
│   ├── physics.py    # World state management
│   ├── runtime.py    # Orchestration loop
│   └── tools.py      # Agent tool definitions
├── scenarios/        # Test scenarios
│   ├── ball_drop.py  # Basic collision test
│   └── visualize.py  # Animated visualization
└── tests/            # Unit tests
```

## How It Works

1. **Physics objects are agents**: Each ball, floor, or obstacle is an autonomous agent
2. **Events trigger reasoning**: When a collision is detected, agents receive an event
3. **LLM calculates outcomes**: Agents use GPT-4 or Claude to reason about physics and calculate their new state
4. **Agents negotiate**: Multiple agents can coordinate on complex interactions

Example agent reasoning:

```
[ball-001] Collision detected with floor
[ball-001] LLM reasoning: "Elastic collision with e=0.8, perpendicular velocity
           component reverses and scales by elasticity coefficient"
[ball-001] Calculated new velocity: [0, 8.0] m/s
```

## Technical Stack

- **Python 3.12**
- **LLMs**: OpenAI GPT-4, Anthropic Claude
- **Physics**: NumPy for vector math
- **Visualization**: Matplotlib
- **Event System**: In-memory queue

## Roadmap

**Week 1** (Current): Basic ball-floor collision with LLM reasoning ✓

**Week 2+**: Extract infrastructure patterns

- Observability and trace viewer
- Tool registry with validation
- Physics validator (conservation laws)
- Agent factory and templates
- LLM router for model selection
- Memory system for agent learning

## Philosophy

- **Iteration speed over performance**: Accept 1-10 FPS to prove the concept
- **Build infrastructure from pain points**: Extract patterns after discovering what's needed
- **Physics as validation**: Conservation laws provide objective correctness metrics

## License

MIT License - see LICENSE file

## Author

Dave Kanter (2026)
