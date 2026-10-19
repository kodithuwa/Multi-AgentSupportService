
import os
import boto3
import logging

logger = logging.getLogger(__name__)

def create_aws_session():
    """Create an AWS session using credentials from VS Code launch.json."""

    access_key = "<<ACCESS_KEY>>"
    secret_key = "<<SECRET_KEY>>"
    session_token = "<<SESSION_TOKEN>>"
    region = "us-east-1"

    if not access_key or not secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is missing "
            "from the environment."
        )

    credentials = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region,
    }

    if session_token:
        credentials["aws_session_token"] = session_token

    session = boto3.Session(**credentials)

    logger.info("AWS session created. Region: %s", region)
    return session


def query_knowledge_base(session, kb_id, query):
    """Retrieve documents from an Amazon Bedrock Knowledge Base."""

    client = session.client("bedrock-agent-runtime")

    try:
        response = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            },
        )

        results = response.get("retrievalResults", [])
        logger.info("Knowledge Base returned %d results", len(results))

        for index, result in enumerate(results, start=1):
            content = result.get("content", {})
            logger.info(
                "Result %d | Score: %s | Content: %s",
                index,
                result.get("score"),
                content.get("text", ""),
            )

        return results

    except Exception as exc:
        logger.exception("Knowledge Base retrieval failed")
        print({exc})
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    aws_session = create_aws_session()

    query_knowledge_base(
        aws_session,
        "DFSSN7CFZN",
        "What is the refund policy?",
    )