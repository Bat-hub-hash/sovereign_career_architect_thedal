# Sovereign Career Architect

An autonomous AI agent for career navigation with cognitive architecture, built for AI-VERSE 2026 hackathon.

## Overview

The Sovereign Career Architect is not just a chatbot that gives advice; it's an autonomous, stateful, voice-enabled proxy that navigates your career lifecycle. It addresses the "scattered platforms" and "manual effort" problems in job hunting by automating job search workflows, managing applications, and conducting real-time interview simulations in Indic languages.

## Key Features

- **Cognitive Architecture**: Built on LangGraph with cyclic state machine for reasoning, planning, and self-correction
- **Persistent Memory**: Multi-scoped memory (User, Session, Agent) that evolves over months using Mem0
- **Autonomous Action**: Browser automation using computer vision to interact with job portals
- **Sovereign Interface**: Voice interaction in 10 Indic languages using Vapi.ai and Sarvam-1
- **Human-in-the-Loop**: Accountable agency with approval workflows for high-stakes actions

## Technical Architecture

```
Voice Interface (Vapi.ai + Sarvam-1)
           ↓
Cognitive Core (LangGraph State Machine)
    ↓        ↓        ↓
Memory Layer  Browser Agent  Safety Layer
(Mem0)       (Computer Vision) (HITL)
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o vision)
- Groq API key (for Llama-3-70B reasoning)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Keerthivasan-Venkitajalam/Thedal
cd sovereign-career-architect
```

2. Create virtual environment:
```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Install Playwright browsers:
```bash
python -m playwright install
```

### Running the Application

```bash
# Start the API server
uvicorn sovereign_career_architect.api.main:app --reload

# Or use the CLI
sca --help
```

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run property-based tests only
pytest -m property

# Run integration tests
pytest -m integration
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
flake8 src tests
mypy src
```

## Architecture Components

### Cognitive Core (LangGraph)
- **Profiler Node**: Retrieves user context from memory
- **Planner Node**: Generates step-by-step execution plans
- **Executor Node**: Performs actions via browser automation
- **Reviewer Node**: Validates outputs and provides feedback
- **Archivist Node**: Updates long-term memory

### Memory Layer (Mem0)
- **User Memory**: Static facts (education, skills, preferences)
- **Session Memory**: Immediate context (current job search)
- **Agent Memory**: Learned interaction patterns

### Browser Agent
- Computer vision-based element identification
- Stealth mode for anti-bot evasion
- Form filling and file upload automation
- Dynamic adaptation to DOM changes

### Voice Interface
- Ultra-low latency voice interaction (<500ms)
- Indic language processing with Sarvam-1
- Cultural context preservation
- Real-time streaming responses

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for AI-VERSE 2026 hackathon
- Powered by LangGraph, Mem0, Browser-use, Vapi.ai, and Sarvam-1
- Designed for the Indian AI ecosystem with sovereignty in mind