# Customer Support Agent

A multi-agent customer support demo built with Strands Agents, Amazon Bedrock, and AgentCore. It routes requests to inventory, policy, and refund specialists, then produces a customer-facing response. Policy searches run across three Bedrock Knowledge Bases in parallel.

## Project layout

- `customer_support_agent/src/agent_orchestrator.py` — run, test, and deploy the agent system
- `customer_support_agent/src/` — agent, retrieval, and observability modules
- `customer_support_agent/infrastructure/` — AWS stack template and data utilities
- `customer_support_agent/tests/` — test and connectivity scripts

## Run locally

Requires Python 3.12, AWS credentials, and access to the required Amazon Bedrock models and resources. Install dependencies from the project directory:

```bash
cd customer_support_agent
pip install -r requirements.txt
```

Configure AWS resource and Knowledge Base settings in `.env` as needed. Then run:

```bash
python src/agent_orchestrator.py chat
python src/agent_orchestrator.py test
python src/agent_orchestrator.py invoke "Where is my order?"  # deployed runtime
```

Deploy with `python src/agent_orchestrator.py deploy`. AWS infrastructure is defined in `infrastructure/starter_stack.yaml`; deployment requires the AgentCore CLI and appropriate AWS permissions.
