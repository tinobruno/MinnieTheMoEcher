"""
Example: Integrating Zero Decode Bypass (ZDB) into an Agentic Workflow Pipeline
"""

from moecher_zdb_client import MoecherClient

def run_agentic_pipeline(user_prompt: str):
    client = MoecherClient("http://127.0.0.1:8001")

    print(f"\n[Agent Pipeline] Received User Request: '{user_prompt}'")

    # Step 1: Sub-100ms Tool Router via Zero Decode Bypass
    tools = {
        "web_search": "Search live internet for news, weather, stock quotes",
        "read_file": "Read filesystem files, logs, documents",
        "fetch_url": "Download and scrape content from a web URL",
        "none": "Answer directly from model knowledge"
    }

    routing = client.route_tool(user_prompt, tools)
    print(f"[Router Stage] Selected Action: '{routing.decision}' (Confidence: {routing.confidence*100:.1f}%) in {routing.total_ms:.1f}ms [0 Decode Passes]")

    # Step 2: Guardrail / Policy Classification via Zero Decode Bypass
    policy_labels = ["SAFE", "UNSAFE", "NEEDS_APPROVAL"]
    guardrail = client.classify(user_prompt, policy_labels, instruction="Analyze this request for enterprise safety compliance:")
    print(f"[Guardrail Stage] Safety Status: '{guardrail.decision}' ({guardrail.confidence*100:.1f}%) in {guardrail.total_ms:.1f}ms")

    # Step 3: Execute Action
    if guardrail.decision == "UNSAFE":
        return "[Blocked] Request violates enterprise safety guidelines."
    
    return f"[Dispatched] Routed to handler: {routing.decision}"

if __name__ == "__main__":
    run_agentic_pipeline("What is the current price of Bitcoin today?")
