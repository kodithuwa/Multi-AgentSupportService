import os
import sys
from strands import Agent, tool
from strands.models import BedrockModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from bedrock_kb_retrieval import (
    retrieve_from_knowledge_base,
    format_kb_results,
)


model = BedrockModel(
    model_id=config.WORKER_MODEL_ID,
    temperature=0.1,
)


# -----------------------------------------
# 1. Knowledge Base retrieval tools
# -----------------------------------------

@tool
def search_returns_policy(query: str) -> str:
    """Search the Returns Knowledge Base for return and refund policy."""
    results = retrieve_from_knowledge_base(
        query, config.RETURNS_KB_ID
    )
    return format_kb_results(results)


@tool
def search_shipping_policy(query: str) -> str:
    """Search the Shipping Knowledge Base for delivery and shipping policy."""
    results = retrieve_from_knowledge_base(
        query, config.SHIPPING_KB_ID
    )
    return format_kb_results(results)


@tool
def search_warranty_policy(query: str) -> str:
    """Search the Warranty Knowledge Base for warranty coverage and claims."""
    results = retrieve_from_knowledge_base(
        query, config.WARRANTY_KB_ID
    )
    return format_kb_results(results)


# -----------------------------------------
# 2. Three specialized agents
# -----------------------------------------

returns_agent = Agent(
    model=model,
    system_prompt=(
        "You are ReturnsAgent. Answer questions about returns and refunds. "
        "Use the Returns Knowledge Base tool for policy facts. "
        "Do not invent policy details."
    ),
    tools=[search_returns_policy],
)

shipping_agent = Agent(
    model=model,
    system_prompt=(
        "You are ShippingAgent. Answer questions about shipping and delivery. "
        "Use the Shipping Knowledge Base tool for policy facts. "
        "Do not invent policy details."
    ),
    tools=[search_shipping_policy],
)

warranty_agent = Agent(
    model=model,
    system_prompt=(
        "You are WarrantyAgent. Answer questions about warranty coverage "
        "and claims. Use the Warranty Knowledge Base tool for policy facts. "
        "Do not invent policy details."
    ),
    tools=[search_warranty_policy],
)


# -----------------------------------------
# 3. Expose specialists to the orchestrator
# -----------------------------------------

@tool
def ask_returns_agent(question: str) -> str:
    """Delegate a return or refund question to ReturnsAgent."""
    return str(returns_agent(question))


@tool
def ask_shipping_agent(question: str) -> str:
    """Delegate a shipping or delivery question to ShippingAgent."""
    return str(shipping_agent(question))


@tool
def ask_warranty_agent(question: str) -> str:
    """Delegate a warranty question to WarrantyAgent."""
    return str(warranty_agent(question))


# -----------------------------------------
# 4. Orchestrator agent
# -----------------------------------------

orchestrator_agent = Agent(
    model=model,
    system_prompt="""
You are NovaMart's customer-support orchestrator.

Delegate policy questions to the appropriate specialist:
- ReturnsAgent: returns and refunds.
- ShippingAgent: shipping and delivery.
- WarrantyAgent: warranty coverage and claims.

You may call multiple specialists when a question spans domains.
Combine their findings into one clear customer-facing answer.
Do not invent policy facts. If the evidence is insufficient,
tell the customer what needs to be verified.
""",
    tools=[
        ask_returns_agent,
        ask_shipping_agent,
        ask_warranty_agent,
    ],
)


if __name__ == "__main__":
    response = orchestrator_agent(
        "Can I return a damaged product, and is it covered by warranty?"
    )
    print(response)