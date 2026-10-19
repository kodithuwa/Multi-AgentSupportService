import boto3
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# One client shared by the three retrieval tasks
kb_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=config.AWS_REGION,
)

KNOWLEDGE_BASES = {
    "Returns": config.RETURNS_KB_ID,
    "Shipping": config.SHIPPING_KB_ID,
    "Warranty": config.WARRANTY_KB_ID,
}

def test_knowledge_base(name, kb_id):
    """Test retrieval from one Knowledge Base."""
    if not kb_id:
        return name, False, "Knowledge Base ID is empty"

    try:
        response = kb_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                "text": f"What is the {name.lower()} policy?"
            },
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            },
        )

        results = response.get("retrievalResults", [])
        passages = [
            item.get("content", {}).get("text", "")
            for item in results
        ]

        return name, True, {
            "kb_id": kb_id,
            "count": len(results),
            "passages": passages,
        }

    except Exception as exc:
        return name, False, (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    # Confirm which AWS identity is being used
    sts = boto3.client("sts", region_name=config.AWS_REGION)
    print("AWS identity:", sts.get_caller_identity()["Arn"])
    print(f"Testing {len(KNOWLEDGE_BASES)} Knowledge Bases in parallel...\n")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(test_knowledge_base, name, kb_id): name
            for name, kb_id in KNOWLEDGE_BASES.items()
        }

        for future in as_completed(futures):
            name, success, details = future.result()

            print(f"\n--- {name} ---")
            if success:
                print(f"KB ID: {details['kb_id']}")
                print(f"Retrieved results: {details['count']}")

                for i, passage in enumerate(details["passages"], 1):
                    print(f"\nResult {i}: {passage[:500]}")
            else:
                print("FAILED:")
                print(details)