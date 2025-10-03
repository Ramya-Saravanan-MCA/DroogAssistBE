from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from calculations.cost_eval import compute_llm_cost

summarizer_prompt = """
You are a strict, conservative summarizer and entity normalizer.  
Read ONLY the provided inputs and output the result as **bullet points**.  
DO NOT add commentary or invent facts not present in the inputs.

INPUTS:
- Past summary: {past_summary}
- Last 5 turns: {last_turns}

RULES (must follow exactly):
- NEVER invent facts. Use only tokens in {past_summary} and {last_turns}.
- Output must be a valid text in **bullet-point structure**.
- The **Summary** point should include what the user is intending to ask and mention key points from the last 5 turns and past summary to give an overall picture of the conversation which is useful for the next turn
- If latest turns contradict past summary, add a bullet: **Conflict: true**, otherwise **Conflict: false**.
- Keep the summary concise and to the point.
- Focus on the most recent conversation flow and user intent progression.
- Do not add any extra commentary or explanation.

OUTPUT STRUCTURE (must always follow this order):
- Summary: <1–6 sentence summary covering recent conversation flow>
- Conflict: false  
"""

sum_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
sum_prompt = PromptTemplate.from_template(summarizer_prompt)
_chain = sum_prompt | sum_llm

def summarizer(last_turns: list, past_summary: str = "") -> dict:
    """
    Args:
        last_turns: List of recent turn strings (up to 5)
        past_summary: Previous conversation summary
    """
    # Format the turns into a readable string
    formatted_turns = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(last_turns)])
    
    response = _chain.invoke({
        "past_summary": past_summary or "",
        "last_turns": formatted_turns
    })
    usage = getattr(response, "usage_metadata", {})
    input_token_count = usage.get("input_tokens", 0)
    output_token_count = usage.get("output_tokens", 0)
    total_token_count = usage.get("total_tokens", 0)
    cost = compute_llm_cost(input_token_count, output_token_count)
    return {
        "summary": response.content.strip(),
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "total_tokens": total_token_count,
        "cost": cost
    }