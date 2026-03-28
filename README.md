---
title: Cognitive Load Env
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# AI Cognitive Load Management Environment
**A dynamic evaluation framework for modeling human cognitive load, stress, and burnout under automated scheduling constraints.**

## Problem Statement
Current AI scheduling and task management systems focus exclusively on computational efficiency, ignoring the psychological realities of human workers. They lack intrinsic consequences for burning out employees, treating time linearly rather than dynamically. We need a way to benchmark how well AI systems balance high-throughput productivity with long-term human well-being. This OpenEnv-compatible project fills that gap.

## Research Concept & Uniqueness
Traditional job-shop scheduling environments treat tasks as static entities and time as a rigid resource. This falls short of reality when applied to human-AI collaboration. This project provides a **novel OpenEnv-compatible benchmark environment for evaluating AI systems in human cognitive load management**. 

It introduces:
- **Human Behavioral Dynamics**: Exponential stress scaling near deadlines, fatigue accumulation from consecutive effort `work_streaks`, and diminishing returns on sequential breaks.
- **Adaptive Difficulty Mechanics**: Evaluates how agents deal with environments that scale complexity relative to success—simulating "flow states" and increased responsibility via random interruptions.
- **Long-Term Accountability**: Focuses not simply on short-horizon greed, but punishes extended cognitive burnout.

### Simple Demo Flow
The simulation works sequentially on the OpenEnv spec:
1. **Agent Logic**: The LLM queries the `state()` reading active tasks, stress constraints, and fatigue.
2. **Action Choice**: The agent fires a generic JSON representation `schedule_task` or `suggest_break`.
3. **Environment Step**: Time cascades, deadlines approach unpredictably depending on scenario difficulty, and human-fatigue accumulates.
4. **Reward Layer**: Agent is penalized heavily for burnout loops, but massively rewarded for balancing sanity with throughput constraints.

### Example Benchmark Results
When testing environments sequentially using `benchmark.py` across seed bounds:
- **Easy Scenario**: Simple time management. The LLMBaseline leverages optimized contextual parsing to perfectly coordinate the human calendar. RuleBased gets median efficiency, Random Agent inevitably crashes human workload parameters and scores zero.
- **Hard Scenario**: Extremely dense. Spawns high-intensity "ad-hoc" assignments randomly modifying the schedule bounds. All agents struggle massively as predictable workflow heuristics collapse, effectively stress-testing dynamic AI Alignment algorithms to their absolute cognitive limit.

## Core Setup & Endpoints

### Local Testing
1. Install dependencies: `pip install -r requirements.txt`
2. Run server: `uvicorn app.main:app --reload`
3. API Endpoints:
   - `GET /tasks`: Returns JSON schemas for `Observation` and `Action`.
   - `GET /grader`: Generates deterministic grading evaluations for the current state.
   - `POST /baseline`: Runs an entire simulation episode directly server-side and outputs telemetry.

### Benchmarking
Run the multi-agent comparison demonstrating the rigor of the evaluation framework:
```bash
python benchmark.py
```
This runs a Random Agent, a Rule-Based Heuristic Agent, and an LLM Baseline against the environment's `hard` scenario. Expect the Random Agent to fail heavily (< 0.2) and the Rule-Based setting a median threshold.

### Deployment to Hugging Face Spaces

This project is fully Dockerized and pre-configured to run on port `7860`, meeting Hugging Face Spaces requirements natively.

1. **Create Space**: On Hugging Face, create a new Space and select **Docker** as the SDK.
2. **Add Secrets**: In the Space settings, add your API keys as Secrets:
   - `GROQ_API_KEY` (if using the lightning-fast Llama-3 baseline)
   - `OPENAI_API_KEY` (if using GPT-4o-mini)
3. **Upload Files**: Push this repository to your Space. The platform will automatically build the `Dockerfile` and expose the evaluation endpoints.

### Local Docker Testing
```bash
docker build -t cognitive-env .
docker run -p 7860:7860 -e GROQ_API_KEY="your_api_key_here" cognitive-env
```

## OpenEnv Compatibility
Fully supports standardized AI verification via compliant `step()`, `reset()`, and `state()` routines documented in the included `openenv.yaml` schema.
