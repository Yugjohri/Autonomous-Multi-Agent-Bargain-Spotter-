#  Autonomous Multi-Agent Bargain Spotter

An end-to-end **autonomous multi-agent AI system** that identifies high-value product deals by collaboratively scanning listings, estimating fair prices, analyzing trends, and making independent purchase decisions.

This project simulates **real-world autonomous AI workflows**, where multiple specialized agents reason, communicate, and act without human intervention.

---

##  Key Features

-  **Multi-Agent Architecture** with role-specialized agents
-  **Autonomous Orchestration** & decision-making
-  **Pricing Intelligence** using historical trends & estimation agents
-  **Memory-Aware Agents** with persistent state
-  **Vector Database Retrieval (ChromaDB)** for semantic search
-  **Deployment-Ready Design** (modular, testable, extensible)

---

##  Agent Roles

| Agent | Responsibility |
|-----|----------------|
| Deal Scanner | Identifies relevant product listings |
| Pricing Agent | Estimates fair market price |
| Evaluation Agent | Determines deal quality |
| Memory Agent | Retains historical decisions |
| Orchestrator | Coordinates agent collaboration |

---

##  How It Works

1. Product data is embedded and stored in a **vector database**
2. Agents retrieve relevant context using semantic search
3. Pricing agents estimate fair value
4. Evaluation agents assess deal quality
5. The orchestrator agent makes autonomous decisions
6. Results are logged and persisted via agent memory

