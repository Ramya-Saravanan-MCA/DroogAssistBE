import re
import os
import json
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from calculations.cost_eval import compute_llm_cost
load_dotenv()   


# --- Fast rule-based routing patterns ---
RE_HANDOFF = re.compile(r"\b(human|agent|advisor|call|book|schedule|meeting|appointment|speak|talk)\b", re.I)

def rule_gate(query):
    """Apply deterministic rules to quickly route obvious intents"""
    if RE_HANDOFF.search(query):
        return {"labels": ["handoff_request"], "slots": {}, "source": "rule"}
    return None

# --- LLM-based intent classification ---
ROUTER_SYSTEM = """You are a message router for a banking chatbot.
Return STRICT JSON with keys: labels (array), confidence (0.0-1.0), slots (object), safety_flags (array).

Valid labels: greeting, in_scope_knowledge, out_of_scope, chitchat_smalltalk, handoff_request, safety_risky.

Label Definitions:
- greeting: salutations, thanks, farewells, acknowledgments
- in_scope_knowledge: banking products, account info, loans, deposits, investments, credit cards, fees, interest rates, transfers, payments, financial services
- out_of_scope: unrelated to banking/finance domain (weather, sports, entertainment, etc.)
- chitchat_smalltalk: casual talk without a factual ask (how are you, what's your name, etc.)
- handoff_request: wants a human banker, agent or meeting
- safety_risky: abusive, sensitive/regulated request, or requests for personalized financial advice

Extract slots when present: account_type, product_type, loan_type, card_type, transaction_type, amount, currency, time_period.

IMPORTANT RULES:
1. If the question is a follow-up question (like "what about that?", "can you explain more?", "tell me about it"), assume in_scope_knowledge with confidence 0.9.
2. If context suggests banking/finance but is unclear, prefer in_scope_knowledge over out_of_scope.
3. For ambiguous questions, default to in_scope_knowledge with moderate confidence (0.6-0.7).
4. When user says "yes", "yeah", "yea", "yep", "sure", "okay" etc., treat as continuing the previous conversation thread. Label as in_scope_knowledge with confidence 0.95.
5. When user says "no", "no thanks", "nope" after a banking conversation, treat as polite decline but still in_scope_knowledge with confidence 0.85.


Follow up Examples:
Query: "what about that?"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {}, "safety_flags": []}

Query: "Can you explain more about that?"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {}, "safety_flags": []}

Query: "What about the fees?"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {"product_type": "fees"}, "safety_flags": []}

Query: "Tell me more details"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {}, "safety_flags": []}

Query: "yes"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {}, "safety_flags": []}

Query: "yeah"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.90, "slots": {}, "safety_flags": []}

Query: "no thanks"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.85, "slots": {}, "safety_flags": []}

Query: "nope"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.85, "slots": {}, "safety_flags": []}

Query: "no"
Response: {"labels": ["in_scope_knowledge"], "confidence": 0.85, "slots": {}, "safety_flags": []}

"""

ROUTER_USER_TMPL = 'Query: "{q}"\nReturn JSON only.'

def call_llm_router(query, model_type="groq"):
    if model_type == "openai":
        llm = ChatOpenAI(
            model="gpt-4o-nano",  
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    else:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY", ""),
            temperature=0.0,
        )
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": ROUTER_USER_TMPL.format(q=query)}
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        usage = getattr(response, "usage_metadata", {})
        input_token_count = usage.get("input_tokens", 0)
        output_token_count = usage.get("output_tokens", 0)
        total_token_count = usage.get("total_tokens", 0)
        cost = compute_llm_cost(input_token_count, output_token_count)

        # Handle potential JSON formatting issues
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
            
        result = json.loads(content)
        # Attach token/cost stats for logging
        result.update({
            "router_input_tokens": input_token_count,
            "router_output_tokens": output_token_count,
            "router_total_tokens": total_token_count,
            "router_cost": cost
        })
        return result
    except Exception as e:
        print(f"Error parsing router response: {e}")
        return {
            "labels": ["in_scope_knowledge"], "confidence": 0.5, "slots": {}, "safety_flags": [],
            "router_input_tokens": 0, "router_output_tokens": 0, "router_total_tokens": 0, "router_cost": 0.0
        }

# --- Threshold constants (only confidence-based) ---
TAU_ROUTER_HIGH = 0.80      # LLM router confident

# --- Response composers ---
def reply_greeting():
    return "Hello! I'm your banking assistant. How can I help you with accounts, loans, credit cards, or other banking services today?"

def reply_handoff():
    return "I'd be happy to connect you with a banking representative. Would you prefer a phone call or an in-branch appointment? Please let me know your preferred time."

def reply_safety():
    return "I apologize, but I'm not able to provide personalized financial advice or handle this sensitive request. For your security, please contact our customer service directly at our official helpline."

def reply_oos():
    return "I'm specifically trained to help with banking and financial services queries. Is there something I can assist you with regarding accounts, loans, investments, or other banking products?"

def reply_not_found():
    """Simple, natural response when information is not found"""
    return "I don't have information about that. Could you ask about something else I can help you with?"

def reply_chitchat():
    return "I appreciate the conversation! I'm here primarily to help with banking questions. Is there anything specific about your accounts, transactions, or banking services I can assist with?"

# --- Core routing decision function ---


def route_and_answer(
    query, past_summary, retriever, llm_model, answer_llm, 
    retrieval_mode="hybrid", k_dense=10, k_sparse=10, 
    rrf_k=60, top_k_final=10, doc_filter=None
):
    timing_info = {}
    rule_start = time.perf_counter() 
    ruled = rule_gate(query)
    timing_info["rule_routing_time"] = time.perf_counter() - rule_start
    
    if ruled and "handoff_request" in ruled["labels"]:
        timing_info["llm_routing_time"] = 0.0 
        return {
            "answer": reply_handoff(),
            "intent": ruled,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "handoff", 
            # Router LLM not used: set tokens/cost to zero
            "router_input_tokens": 0,
            "router_output_tokens": 0,
            "router_total_tokens": 0,
            "router_cost": 0.0,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }

    # LLM router for intent classification
    llm_start = time.perf_counter() 
    intent_info = call_llm_router(query, llm_model)
    timing_info["llm_routing_time"] = time.perf_counter() - llm_start
    
    # --- Extract router token/cost stats for logging ---
    router_input_tokens = intent_info.get("router_input_tokens", 0)
    router_output_tokens = intent_info.get("router_output_tokens", 0)
    router_total_tokens = intent_info.get("router_total_tokens", 0)
    router_cost = intent_info.get("router_cost", 0.0)
    
    labels = set(intent_info.get("labels", []))
    confidence = float(intent_info.get("confidence", 0))
    slots = intent_info.get("slots", {})
    safety_flags = intent_info.get("safety_flags", [])
    

    # Check for safety concerns
    if "safety_risky" in labels or safety_flags:
        return {
            "answer": reply_safety(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "safety",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }
    
    # Check for greeting-only intent
    if labels == {"greeting"}:
        return {
            "answer": reply_greeting(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "greeting",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }
    if "chitchat_smalltalk" in labels and confidence >= TAU_ROUTER_HIGH and "in_scope_knowledge" not in labels:
        return {
            "answer": reply_chitchat(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "chitchat",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }
    if "handoff_request" in labels and confidence >= TAU_ROUTER_HIGH:
        return {
            "answer": reply_handoff(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "handoff",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }
    if "out_of_scope" in labels and confidence >= TAU_ROUTER_HIGH:
        return {
            "answer": reply_oos(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "oos",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": 0,
            "main_output_tokens": 0,
            "main_total_tokens": 0,
            "main_llm_cost": 0.0,
        }

    # --- RAG/Answer LLM call and cost tracking ---
    # For any other case (in_scope_knowledge or uncertain) -> Use RAG
    # Perform retrieval with user-selected parameters
    retrieval_start = time.perf_counter()  
    results, retrieval_timings = retriever.retrieve(
        query, 
        mode=retrieval_mode,
        k_dense=k_dense,
        k_sparse=k_sparse,
        rrf_k=rrf_k,
        top_k_final=top_k_final,
        doc_filter=doc_filter
    )
    # Always use RAG if we have results (no retrieval strength check)
    if not results.empty:
        generation_start = time.perf_counter()  
        if answer_llm:
            answer_dict = answer_llm(
                list(results["text"]),
                query,
                llm_model.lower(),
                past_summary,
                chunk_ids=list(results["chunk_id"]) if "chunk_id" in results.columns else None
            )
            # answer_llm returns a dict: {"text", "input_tokens", "output_tokens", "total_tokens", "cost"}
            answer = answer_dict["text"]
            main_input_tokens = answer_dict.get("input_tokens", 0)
            main_output_tokens = answer_dict.get("output_tokens", 0)
            main_total_tokens = answer_dict.get("total_tokens", 0)
            main_llm_cost = answer_dict.get("cost", 0.0)
        else:
            answer = "I found relevant information about your banking query. Let me provide you with the details."
            main_input_tokens = main_output_tokens = main_total_tokens = main_llm_cost = 0

        timing_info["generation_time"] = time.perf_counter() - generation_start
        if "greeting" in labels:
            answer = f"Hello! {answer}"
        return {
            "answer": answer,
            "intent": intent_info,
            "retrieval_results": results,
            "retrieval_strength": 1.0,
            "used_rag": True,
            "timing_info": timing_info,  
            "retrieval_timings": retrieval_timings,
            "answer_decision": "rag_answer",
            "router_input_tokens": router_input_tokens,
            "router_output_tokens": router_output_tokens,
            "router_total_tokens": router_total_tokens,
            "router_cost": router_cost,
            "main_input_tokens": main_input_tokens,
            "main_output_tokens": main_output_tokens,
            "main_total_tokens": main_total_tokens,
            "main_llm_cost": main_llm_cost,
        }

    timing_info["generation_time"] = 0.0
    return {
        "answer": reply_not_found(),
        "intent": intent_info,
        "retrieval_results": results,
        "retrieval_strength": 0.0,
        "used_rag": False,
        "timing_info": timing_info, 
        "retrieval_timings": retrieval_timings,
        "answer_decision": "no_results",
        "router_input_tokens": router_input_tokens,
        "router_output_tokens": router_output_tokens,
        "router_total_tokens": router_total_tokens,
        "router_cost": router_cost,
        "main_input_tokens": 0,
        "main_output_tokens": 0,
        "main_total_tokens": 0,
        "main_llm_cost": 0.0,
    }


