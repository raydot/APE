# APE (Agentic Physics Engine)

An experimental physics engine where physical objects are autonomous agents that reason about physics using LLMs, rather than traditional numerical solvers.

## Overview

APE explores a novel approach to physics simulation: physical objects are autonomous agents that use LLMs to reason about collisions and propose outcomes. When two balls collide, each agent predicts its post-collision velocity. A resolver validates these proposals against conservation laws—if valid, they're accepted; if not, ground truth physics is imposed.

**Key insight**: Agents can be wrong and learn from corrections. The system tracks whether agents improve over time by storing experiences and retrieving similar past collisions for few-shot learning.

This is primarily an **agent infrastructure project** using physics as a concrete test domain. Physics provides:

- Objective success criteria (conservation laws)
- Visual debugging (you can see when it's wrong)
- Multi-agent coordination challenges (proposal/validation cycles)
- Learning feedback (ground truth available when agents fail)

## Research Findings

After running three experiments (Newton's Cradle, Billiards, Principle Discovery) over 60 trials total, here's what we learned about LLM agents doing physics:

### TL;DR: LLMs Can't Do Physics Math (But We Knew That)

**Quantitative results:**

- **1D collisions (Newton's Cradle)**: 47% baseline accuracy, 77% with experience retrieval (+30pp, p<0.001)
- **2D collisions (Billiards)**: 5% baseline accuracy, 1% with experience retrieval (-4pp, p<0.05)
- **2D prediction (Observer)**: 1.44 m/s error baseline, 1.51 m/s with learning (+5%, p=0.38, not significant)

**Key findings:**

1. **Dimensionality matters**: LLM accuracy drops 9x going from 1D to 2D (47% → 5%)
2. **Experience retrieval helps only in simple cases**: +30pp improvement in 1D, -4pp in 2D
3. **Complex tasks break transfer learning**: Retrieved examples can mislead when task complexity increases
4. **Hybrid architecture necessary**: Even with learning, agents fail 23-95% of the time, requiring symbolic fallback

**What this means:**

- Don't use pure LLM agents for physics simulation (you knew this)
- Few-shot learning from experience has sharp complexity limits (you knew this too)
- Hybrid LLM + symbolic approaches work (Resolver achieves 95-100% accuracy by catching agent errors)
- 2D vector decomposition defeats current LLMs (not surprising)

**Conclusion**: This confirms that LLMs struggle with multi-step numerical reasoning, especially in higher dimensions. Experience-based few-shot learning helps in simple cases but fails or degrades performance in complex scenarios. The real value is the infrastructure for building multi-agent systems with validation, learning, and fallback mechanisms.

See `scenarios/` for experiment implementations and `mlruns/` for detailed data.

## Architecture

**Hybrid approach**: Agents propose outcomes using LLM reasoning, resolver validates against physics, ground truth applied when agents fail.

**Components:**

- **Agents**: LLM-powered reasoning (GPT-4o-mini, Claude Haiku)
- **Resolver**: Validates proposals against conservation laws
- **Experience Store**: Qdrant vector DB for storing/retrieving past collisions
- **Learning System**: Retrieves similar past experiences for few-shot learning
- **Tracking**: MLflow for experiment metrics and comparison

**Current Status**: Three experiments complete with statistical analysis

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

## Running Experiments

APE includes three emergence experiments testing multi-agent physics reasoning and learning:

### Experiment 1: Newton's Cradle (1D Collisions)

```bash
# Single trial with 5 balls
python scenarios/newtons_cradle.py

# Compare learning vs no-learning (20 trials)
python scenarios/newtons_cradle_comparison.py

# Custom configuration
python scenarios/newtons_cradle_comparison.py --balls 3 --trials 50
```

**Tests**: Momentum transfer through chain collisions in 1D
**Result**: Learning helps (+30pp improvement, 47% → 77% accuracy)

### Experiment 2: Billiards (2D Collisions)

```bash
# Single trial with 6 balls
python scenarios/billiards.py

# Compare learning vs no-learning (20 trials)
python scenarios/billiards_comparison.py

# Custom configuration
python scenarios/billiards_comparison.py --balls 8 --trials 20
```

**Tests**: Multi-body 2D collisions with vector decomposition
**Result**: Learning hurts (-4pp, 5% → 1% accuracy) - retrieved examples mislead

### Experiment 3: Principle Discovery (Observer Learning)

```bash
# Observer watches 8 balls colliding
python scenarios/principle_discovery.py

# Compare learning vs no-learning (20 trials)
python scenarios/principle_discovery_comparison.py
```

**Tests**: Can observer agent discover physics principles from observation?
**Result**: Learning has no effect (1.44 vs 1.51 m/s error, not significant)

### View Results

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000 to explore metrics, compare runs, and visualize learning curves.

## Project Structure

```
ape/
├── ape/                    # Core engine
│   ├── events.py          # Event system
│   ├── agents.py          # Agent base classes
│   ├── learning_agent.py  # LLM agents with experience retrieval
│   ├── observer_agent.py  # Non-participating observer
│   ├── resolver_agent.py  # Validates agent proposals
│   ├── physics.py         # World state management
│   ├── runtime.py         # Orchestration loop
│   ├── tools.py           # Physics calculation tools
│   └── learning/          # Experience store and feedback
│       ├── experience_store.py  # Qdrant vector DB
│       └── feedback.py          # Learning feedback generation
├── scenarios/             # Experiments
│   ├── newtons_cradle.py
│   ├── newtons_cradle_comparison.py
│   ├── billiards.py
│   ├── billiards_comparison.py
│   ├── principle_discovery.py
│   └── principle_discovery_comparison.py
└── tests/                 # Unit tests
```

## How It Works

1. **Physics objects are agents**: Each ball is an autonomous agent with LLM reasoning
2. **Events trigger reasoning**: When collision detected, agents propose outcomes
3. **Agents negotiate**: Both agents predict post-collision velocities
4. **Resolver validates**: Checks momentum/energy conservation (5% tolerance)
5. **Learning from experience**: Agents store outcomes, retrieve similar past collisions
6. **Fallback to ground truth**: If proposals invalid, resolver imposes correct physics

Example agent reasoning (from logs):

```
[ball-000] Learning-enhanced proposal:
  Used 3 similar past experiences
  Reasoning: "Decompose velocity into normal and tangential components.
             Exchange normal components based on masses and elasticity..."
[RESOLVER] ✗ Proposals INVALID: Momentum not conserved: error=450% (limit: 5%)
[RESOLVER] ⚠ Imposing ground truth solution
```

**The pattern**: Agents try → usually fail (especially in 2D) → resolver catches errors → system works

## Technical Stack

- **Python 3.12**
- **LLMs**: OpenAI GPT-4o-mini, Anthropic Claude Haiku
- **Vector DB**: Qdrant (for experience storage/retrieval)
- **Tracking**: MLflow (experiment metrics and comparison)
- **Physics**: NumPy for vector math
- **Visualization**: Matplotlib animations
- **Statistics**: SciPy (t-tests for significance)

## What Worked, What Didn't

**✓ Worked:**

- Multi-agent coordination with resolver validation
- Experience storage and retrieval (Qdrant + embeddings)
- Learning helps in simple 1D scenarios (+30pp)
- MLflow tracking for experiment comparison
- Statistical analysis showing clear trends

**✗ Didn't Work:**

- LLM agents can't do 2D vector math reliably (5% accuracy)
- Experience retrieval actively misleads in complex scenarios (-4pp)
- Agents don't improve within trials (no learning curve)
- 2D principle discovery shows no learning benefit

**→ Validated Need For:**

- Hybrid architecture (LLM + symbolic fallback)
- Clear complexity limits on when to use LLMs
- Validation layer to catch agent errors

## Key Takeaways for Building Agent Systems

1. **Always validate agent outputs** - LLMs are confident but often wrong
2. **Provide fallback mechanisms** - Don't trust agents with critical calculations
3. **Experience retrieval works for simple cases only** - Complex tasks need different approaches
4. **Track everything** - MLflow essential for understanding agent behavior
5. **Statistical rigor matters** - 20+ trials needed to see real patterns

## Philosophy

- **Iteration speed over performance**: Accept 1-10 FPS to prove concepts quickly
- **Build infrastructure from pain points**: Extract patterns after discovering what's needed
- **Physics as validation**: Conservation laws provide objective correctness metrics
- **Fail fast with data**: Run experiments, measure everything, move on when patterns are clear

## Development Speed

This entire project (3 experiments, 60 trials, statistical analysis, infrastructure) was built in ~24 hours as a rapid exploration. The goal was to answer: "Can LLM agents do physics?" Answer: No, but they can propose reasonable guesses that a validator can check.

## License

MIT License - see LICENSE file

## Author

Dave Kanter (2026)
