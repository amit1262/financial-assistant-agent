# Financial Assistant Agent

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange?logo=langchain)
![LangChain](https://img.shields.io/badge/LangChain-Framework-blue?logo=langchain)
![MCP](https://img.shields.io/badge/MCP-Extensible--Tools-purple)
![A2A](https://img.shields.io/badge/A2A-Planned-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![AlphaVantage](https://img.shields.io/badge/Data-AlphaVantage-yellow)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-black?logo=openai)
![MLflow](https://img.shields.io/badge/MLflow-Observability-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Status](https://img.shields.io/badge/status-active-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/amit1262/financial-assistant-agent/develop)

## Overview

The **Financial Assistant Agent** is a multi-agent AI system designed to assist users with comprehensive financial inquiries. From generic conceptual discussions and continued conversations to highly specific price-action analysis (e.g., _"What happened to NVDA in the last couple of days?"_), the platform provides data-grounded and concise insights.

Built on a modular architecture powered by **LangGraph** and **FastAPI**, the platform intelligently combines specialized autonomous agents to process technical, fundamental, and sentiment-based signals. The system is designed for multi-turn interactions, allowing users to follow up on complex findings with further questions.

The system is architected to support an **Agent-to-Agent (A2A)** communication pattern (v2.0), where a central Advisor Agent will orchestrate tasks between specialized units like the Technical Analysis Agent.

**Key Capabilities:**

- 🤖 **Conversational Intelligence**: Handles both conceptual finance queries and real-time market deep-dives with full context awareness.
- 📈 **Data-Grounded Analysis**: Multi-agent collaboration to synthesize Technical, Fundamental, and News/Sentiment indicators.
- 📉 **Future-Ready (A2A)**: Planned autonomous delegation between the central Advisor and specialized Market Intelligence units.
- 🔌 **MCP Integration**: Uses the Model Context Protocol (MCP) for extensible tool discovery and execution.
- 📊 **Agent Tracing**: Observability powered by **MLflow** for tracking agent trajectories, prompt versioning, and cost-performance trade-offs.

---

## 🚀 Key Features

- **Intelligent Routing**: Automatically determines whether a query requires conceptual explanation or data-driven analysis.
- **Multi-Turn Workflows**: Supports continued conversations, allowing users to drill down into specific data points or broader market trends.
- **LangGraph State Management**: Maintains conversational context across complex reasoning loops and conditional edge transitions.
- **Asynchronous Execution**: High-concurrency backend designed for parallel tool calls and multi-agent interaction.
- **Concise Outputs**: Specialized prompts ensure data is prioritized over fluff, providing clear and actionable financial summaries.

---

## 🏗️ Architecture

The project follows a **Multi-Agent Network** design:

1. **User Request** hits the Advisor Agent (e.g., "Why is MSFT trending up?").
2. **Advisor Agent** analyzes intent and delegates sub-tasks (Technical, News, Fundamentals) to the specialized units.
3. **MIU (Technical Agent)** uses LangGraph to pull live indicators (SMA, RSI, etc.) via MCP Tools.
4. **Knowledge Synthesis**: Sub-agent results are synthesized into a unified, concise response back to the Advisor.
5. **Final Delivery**: User receives a grounded answer and can initiate follow-up questions immediately.

---

## 🧰 Tech Stack

- **Frameworks:** FastAPI, LangGraph, LangChain
- **AI/ML:** OpenAI (via LangChain and OpenRouter), LangChain MCP Adapters
- **Observability:** MLflow
- **Package Management:** UV (Astral)
- **Deployment:** Docker & Docker Compose
- **Data Integration:** Model Context Protocol (MCP) for Market Data

---

## 🤖 Agents

| Agent                        | Technology          | Responsibility                                                                                                                                                          |
| ---------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Advisor Agent**            | FastAPI, LangChain  | The central orchestrator that intelligently breaks down user tasks, creates and assigns these tasks to sub-agents, and combines their outputs into a cohesive response. |
| **Market Intelligence Unit** | FastAPI, MCP Client | Hosts specialized agents for deep-dives into financial data.                                                                                                            |

### Specialized Sub-Agents under Market Intelligence Unit:

| Sub-Agent                | Technology     | Responsibility                                                            |
| ------------------------ | -------------- | ------------------------------------------------------------------------- |
| **Technical Analysis**   | LangGraph, MCP | Uses SMA, EMA, and RSI tools via MCP to analyze price trends.             |
| **Fundamental Analysis** | LangGraph, MCP | Analyzes Balance Sheets and Cash Flows for long-term financial health.    |
| **Sentiment Analysis**   | LangGraph, MCP | Scrapes news and social signals to score market sentiment and narratives. |

---

## 🚀 Upcoming Features & Roadmap

The project is under active development with the following milestones planned:

1. **Expanded Market Intelligence**:
   - **Fundamental Analysis Agent**: Retrieval and analysis of balance sheets, income statements, and cash flow data via specialized MCP tools.
   - **News & Sentiment Analysis Agent**: News scraping and sentiment scoring to correlate price action with market narratives.
2. **Advanced Agent Collaboration**:
   - **Production-grade A2A**: Full peer-to-peer communication between agents allowing the Advisor to trigger multi-step, cross-agent workflows.
3. **Enhanced Risk Modeling**: Multi-factor portfolio risk assessment integrating both technical and fundamental signals.

---

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- Docker & Docker Compose
- [UV](https://github.com/astral-sh/uv) (Recommended for package management)

### Setup

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd financial-assistant-agent
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root and in the respective agent directories:

   ```bash
   # Root / Market Intelligence Unit .env
   OPENAI_API_KEY="your-key"
   MLFLOW_TRACKING_URI="http://mlflow_container:5000"
   ```

3. **Install Dependencies (Local Development)**:
   ```bash
   uv sync
   ```

---

## 🏃 Running the Application

### Using Docker Compose (Recommended)

This will set up the Advisor Agent, Market Intelligence Unit, and MLflow Tracking Server.

```bash
docker-compose -f docker-compose-local.yml up -d
```

### Accessing the Services

- **Advisor Agent API**: `http://localhost:8000/docs`
- **Technical Analysis API**: `http://localhost:8001/docs` (depending on your `MIU_PORT`)
- **MLflow Dashboard**: `http://localhost:5000`

---

## 🙏 Acknowledgments

- **AlphaVantage** for providing the technical analysis data endpoints.
- **Anthropic/OpenAI** for the underlying LLM capabilities.
- **Astral (UV)** for making environment management lightning fast.
- **Model Context Protocol (MCP)** designers for the extensible tool strategy.

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**
