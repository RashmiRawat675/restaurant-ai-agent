"""
Restaurant AI Agent (LangGraph) - single file version.

Flow:
    classify --(not food)--------> not_food            [END, result=fail]
             --(order)------------> order_confirm

    order_confirm --(confirm)-------> cook
                  --(partially)-----> ask_user (pauses, waits for user reply)
                  --(not_available)-> not_available     [END, result=fail]

    ask_user --(user says ok)-----------> cook   (quantity = offered qty)
             --(user names new dish)----> order_confirm
             --(user declines)----------> declined      [END, result=fail]
             --(too many prompts)-------> max_attempts  [END, result=fail]

    cook --(cooked, up to 2 internal tries)---> server
         --(both tries fail)------------------> cook_fail [END, result=fail]

    server --(success)-----------> complete             [END, result=complete]
           --(1st fail)-----------> recook_prep -> cook   (recook + re-serve, ONCE)
           --(2nd fail)-----------> server_fail           [END, result=fail]

Setup:
    pip install -r requirements.txt   (langgraph, groq, python-dotenv, typing_extensions)
    Create a .env file with:
        GROQ_API_KEY=your_key_here
        GROQ_MODEL=llama-3.3-70b-versatile   (optional, this is the default)

Run:
    python restaurant_agent.py            # interactive chat
    python restaurant_agent.py test       # run the 3 required test cases
"""

import os
import sys
import json
import time
import uuid
import random
from typing import TypedDict, Optional, Literal
from typing_extensions import Annotated

from dotenv import load_dotenv
from groq import Groq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================================
# 1. MENU  -- replace with your real dish names / prices / available quantity
# =============================================================================

MENU = {
    "paneer butter masala": {"price": 220, "available": 5},
    "chicken biryani":      {"price": 250, "available": 2},
    "veg fried rice":       {"price": 150, "available": 0},
    "masala dosa":          {"price": 100, "available": 10},
    "gulab jamun":          {"price": 60,  "available": 8},
}


# =============================================================================
# 2. STATE
# =============================================================================
#
# order_attempt / cooked_attempt / server_attempt = FIXED MAXIMUMS (3 / 2 / 2)
# order_tries / cook_tries / server_tries         = running counters
# sim                                             = optional debug hook used
#   only by the test cases to force deterministic cook/server outcomes,
#   e.g. {"cook": "success", "server": "fail"}

class OrderDetails(TypedDict):
    dish: Optional[str]
    quantity: Optional[int]
    available: Optional[int]


Status = Optional[Literal[
    "confirm", "partially", "not_available",
    "cooked", "cook_fail",
    "complete", "server_fail_attempt", "server_fail",
    "retry_order", "declined", "max_attempts",
]]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    order: OrderDetails
    is_food_query: Optional[bool]
    status: Status

    order_attempt: int
    cooked_attempt: int
    server_attempt: int

    order_tries: int
    cook_tries: int
    server_tries: int
    recook: bool

    result: Optional[str]
    result_reason: Optional[str]

    sim: dict


def init_state(user_text: str) -> AgentState:
    """Build a fresh state for a brand new ordering session."""
    return {
        "messages": [{"role": "user", "content": user_text}],
        "order": {"dish": None, "quantity": None, "available": None},
        "is_food_query": None,
        "status": None,
        "order_attempt": 3,
        "cooked_attempt": 2,
        "server_attempt": 2,
        "order_tries": 0,
        "cook_tries": 0,
        "server_tries": 0,
        "recook": False,
        "result": None,
        "result_reason": None,
        "sim": {},
    }


# =============================================================================
# 3. TOOLS  -- order_confirm, cook, server (pure functions, no LLM)
# =============================================================================

def order_confirm(dish: str, quantity: int):
    """
    Returns (status, available_quantity):
      "confirm"       -> full requested quantity is available
      "partially"     -> only part of it is available
      "not_available" -> dish unknown or zero stock
    """
    if not dish:
        return "not_available", 0
    item = MENU.get(dish.lower().strip())
    if item is None:
        return "not_available", 0
    available = item["available"]
    if available <= 0:
        return "not_available", 0
    if available >= quantity:
        return "confirm", quantity
    return "partially", available


def cook(dish: str, quantity: int, sim_mode: str = None) -> bool:
    """Simulates cooking. Returns True (success) / False (fail)."""
    if sim_mode == "success":
        return True
    if sim_mode == "fail":
        return False
    return random.random() > 0.15  # ~85% success rate


def server(dish: str, quantity: int, sim_mode: str = None) -> bool:
    """Simulates serving/delivering the order. Returns True/False."""
    if sim_mode == "success":
        return True
    if sim_mode == "fail":
        return False
    return random.random() > 0.10  # ~90% success rate


# =============================================================================
# 4. LLM UTILITIES  -- Groq calls for NLU (classify) and NLG (reply text)
# =============================================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_client = Groq(api_key=GROQ_API_KEY)


def _chat(messages, temperature: float = 0.2) -> str:
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


CLASSIFY_SYSTEM = """You are the intake assistant for a restaurant ordering system.

Task:
1. Decide if the user's message is a food / restaurant-order related query
   (ordering a dish, mentioning a dish + quantity, accepting or rejecting an
   available quantity that was offered to them). Anything unrelated
   (weather, coding, general chit-chat, other businesses, etc.) is NOT a
   food query.
2. If it IS a food query and expresses a new dish request, extract:
   - dish: the dish name mentioned (lowercase string)
   - quantity: integer quantity requested (default 1 if not stated)
   - intent: "order"
3. If it IS a food query but it's simply the user accepting a previously
   offered reduced quantity (e.g. "yes", "ok fine", "sure go ahead", "that's fine"),
   set intent to "accept" and leave dish/quantity null.
4. If it IS a food query but the user is declining/rejecting the offer with
   no new dish mentioned (e.g. "no thanks", "cancel it"), set intent to "reject".

Respond with ONLY compact JSON, no prose, no markdown fences, exactly:
{"is_food_query": true|false, "intent": "order"|"accept"|"reject"|"other", "dish": string|null, "quantity": integer|null}
"""


def classify_and_extract(user_text: str) -> dict:
    raw = _chat(
        [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": user_text},
        ]
    )
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        data = {"is_food_query": False, "intent": "other", "dish": None, "quantity": None}
    return data


_REPLY_SYSTEM = (
    "You are a friendly restaurant ordering assistant talking directly to the "
    "customer. Write ONE short, natural, polite message for the situation "
    "described below. Do not invent facts beyond what is given. Plain text, "
    "no markdown, 1-3 sentences."
)


def generate_reply(kind: str, **ctx) -> str:
    """Ask the LLM to phrase a natural, user-facing message for a situation."""
    user_prompt = f"Situation: {kind}\nDetails (JSON): {json.dumps(ctx)}\nWrite the message now."
    return _chat(
        [
            {"role": "system", "content": _REPLY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )


# =============================================================================
# 5. GRAPH NODES
# =============================================================================

def _last_user_text(messages) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            return getattr(m, "content", None) or m.get("content")
    return ""


def classify_node(state: AgentState) -> dict:
    order_tries = state.get("order_tries", 0) + 1
    if order_tries > state["order_attempt"]:
        return {"order_tries": order_tries, "status": "max_attempts"}

    text = _last_user_text(state["messages"])
    parsed = classify_and_extract(text)

    update = {"order_tries": order_tries, "is_food_query": bool(parsed.get("is_food_query"))}
    if update["is_food_query"] and parsed.get("intent") == "order" and parsed.get("dish"):
        update["order"] = {
            "dish": parsed["dish"],
            "quantity": parsed.get("quantity") or 1,
            "available": None,
        }
    else:
        update["is_food_query"] = False
    return update


def route_after_classify(state: AgentState) -> str:
    if state.get("status") == "max_attempts":
        return "max_attempts"
    return "order_confirm" if state.get("is_food_query") else "not_food"


def order_confirm_node(state: AgentState) -> dict:
    order = state["order"]
    status, available = order_confirm(order["dish"], order["quantity"])
    return {"status": status, "order": {**order, "available": available}}


def route_after_order_confirm(state: AgentState) -> str:
    return state["status"]  # "confirm" | "partially" | "not_available"


def ask_user_node(state: AgentState) -> dict:
    order = state["order"]
    offer_msg = generate_reply(
        "partial_availability",
        dish=order["dish"],
        requested=order["quantity"],
        available=order["available"],
    )

    user_reply = interrupt({"message": offer_msg, "dish": order["dish"], "available": order["available"]})

    order_tries = state.get("order_tries", 0) + 1
    update = {
        "messages": [
            {"role": "assistant", "content": offer_msg},
            {"role": "user", "content": user_reply},
        ],
        "order_tries": order_tries,
    }
    if order_tries > state["order_attempt"]:
        update["status"] = "max_attempts"
        return update

    parsed = classify_and_extract(user_reply)
    if not parsed.get("is_food_query"):
        update["status"] = "declined"
        return update

    intent = parsed.get("intent")
    if intent == "accept":
        update["order"] = {**order, "quantity": order["available"]}
        update["status"] = "confirm"
    elif intent == "order" and parsed.get("dish"):
        update["order"] = {"dish": parsed["dish"], "quantity": parsed.get("quantity") or 1, "available": None}
        update["status"] = "retry_order"
    else:
        update["status"] = "declined"
    return update


def route_after_ask_user(state: AgentState) -> str:
    status = state.get("status")
    if status == "confirm":
        return "cook"
    if status == "retry_order":
        return "order_confirm"
    if status == "max_attempts":
        return "max_attempts"
    return "declined"


def cook_node(state: AgentState) -> dict:
    order = state["order"]
    max_tries = state["cooked_attempt"]
    sim_mode = state.get("sim", {}).get("cook")

    tries = 0
    success = False
    while tries < max_tries:
        tries += 1
        if cook(order["dish"], order["quantity"], sim_mode=sim_mode):
            success = True
            break

    return {"cook_tries": tries, "status": "cooked" if success else "cook_fail"}


def route_after_cook(state: AgentState) -> str:
    return "server" if state["status"] == "cooked" else "cook_fail"


def server_node(state: AgentState) -> dict:
    order = state["order"]
    tries = state.get("server_tries", 0) + 1
    sim_mode = state.get("sim", {}).get("server")

    ok = server(order["dish"], order["quantity"], sim_mode=sim_mode)
    return {"server_tries": tries, "status": "complete" if ok else "server_fail_attempt"}


def route_after_server(state: AgentState) -> str:
    if state["status"] == "complete":
        return "complete"
    if state["server_tries"] < state["server_attempt"]:
        return "recook"
    return "server_fail"


def recook_prep_node(state: AgentState) -> dict:
    return {"recook": True}


def not_food_node(state: AgentState) -> dict:
    text = generate_reply("not_food_query")
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": "not a food-related query"}


def not_available_node(state: AgentState) -> dict:
    order = state["order"]
    text = generate_reply("dish_not_available", dish=order["dish"])
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": f"'{order['dish']}' is not available"}


def declined_node(state: AgentState) -> dict:
    text = generate_reply("customer_declined_offer", dish=state["order"].get("dish"))
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": "customer did not accept the available quantity"}


def cook_fail_node(state: AgentState) -> dict:
    order = state["order"]
    text = generate_reply("cooking_failed", dish=order["dish"], recook=state.get("recook", False))
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": "cooking failed after max attempts"}


def server_fail_node(state: AgentState) -> dict:
    order = state["order"]
    text = generate_reply("delivery_failed", dish=order["dish"])
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": "serving/delivery failed after max attempts"}


def complete_node(state: AgentState) -> dict:
    order = state["order"]
    text = generate_reply("order_complete", dish=order["dish"], quantity=order["quantity"])
    return {"messages": [{"role": "assistant", "content": text}], "result": "complete",
            "result_reason": "order delivered successfully"}


def max_attempts_node(state: AgentState) -> dict:
    text = generate_reply("max_attempts_exceeded")
    return {"messages": [{"role": "assistant", "content": text}], "result": "fail",
            "result_reason": "maximum number of user attempts exceeded"}


# =============================================================================
# 6. GRAPH ASSEMBLY
# =============================================================================

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify", classify_node)
    g.add_node("order_confirm", order_confirm_node)
    g.add_node("ask_user", ask_user_node)
    g.add_node("cook", cook_node)
    g.add_node("recook_prep", recook_prep_node)
    g.add_node("server", server_node)

    g.add_node("not_food", not_food_node)
    g.add_node("not_available", not_available_node)
    g.add_node("declined", declined_node)
    g.add_node("cook_fail", cook_fail_node)
    g.add_node("server_fail", server_fail_node)
    g.add_node("complete", complete_node)
    g.add_node("max_attempts", max_attempts_node)

    g.add_edge(START, "classify")

    g.add_conditional_edges(
        "classify", route_after_classify,
        {"not_food": "not_food", "order_confirm": "order_confirm", "max_attempts": "max_attempts"},
    )
    g.add_conditional_edges(
        "order_confirm", route_after_order_confirm,
        {"confirm": "cook", "partially": "ask_user", "not_available": "not_available"},
    )
    g.add_conditional_edges(
        "ask_user", route_after_ask_user,
        {"cook": "cook", "order_confirm": "order_confirm", "declined": "declined", "max_attempts": "max_attempts"},
    )
    g.add_conditional_edges(
        "cook", route_after_cook,
        {"server": "server", "cook_fail": "cook_fail"},
    )
    g.add_conditional_edges(
        "server", route_after_server,
        {"complete": "complete", "recook": "recook_prep", "server_fail": "server_fail"},
    )
    g.add_edge("recook_prep", "cook")

    for terminal in ["not_food", "not_available", "declined", "cook_fail", "server_fail", "complete", "max_attempts"]:
        g.add_edge(terminal, END)

    return g.compile(checkpointer=MemorySaver())


# =============================================================================
# 7. INTERACTIVE CLI
# =============================================================================

def _print_assistant_messages(result, already_printed: int) -> int:
    msgs = result["messages"]
    for m in msgs[already_printed:]:
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if role in ("ai", "assistant"):
            print(f"Assistant: {content}")
    return len(msgs)


def run_interactive():
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Welcome! What would you like to order?")
    first_message = input("You: ").strip()

    state = init_state(first_message)
    result = app.invoke(state, config=config)
    printed = _print_assistant_messages(result, 0)

    while "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        print(f"Assistant: {intr.value.get('message')}")
        user_reply = input("You: ").strip()
        result = app.invoke(Command(resume=user_reply), config=config)
        printed = _print_assistant_messages(result, printed)

    print(f"\n--- Session finished: {result.get('result', 'unknown').upper()} ---")
    print(f"Reason: {result.get('result_reason')}")


# =============================================================================
# 8. TEST CASES (exactly 3, as required)
# =============================================================================

def run_case(name: str, first_message: str, sim: dict = None, scripted_replies=None, expected: str = None):
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = init_state(first_message)
    if sim:
        state["sim"] = sim

    print(f"\n=== {name} ===")
    print(f"User: {first_message}")

    result = app.invoke(state, config=config)
    replies = list(scripted_replies or [])

    while "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        print(f"Assistant (offer): {intr.value.get('message')}")
        user_reply = replies.pop(0) if replies else "no thanks"
        print(f"User: {user_reply}")
        result = app.invoke(Command(resume=user_reply), config=config)

    for m in result["messages"]:
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if role in ("ai", "assistant"):
            print(f"Assistant: {content}")

    outcome = result.get("result")
    print(f"RESULT: {outcome} | reason: {result.get('result_reason')}")
    if expected:
        print(f"Expected: {expected} -> {'PASS' if outcome == expected else 'FAIL <-- unexpected!'}")
    return result


def run_test_cases():
    # Test 1: fully available -> should COMPLETE
    run_case(
        "Test 1: Fully available dish",
        "I want 2 plates of masala dosa",
        sim={"cook": "success", "server": "success"},
        expected="complete",
    )

    # Test 2: partially available -> user accepts reduced qty -> should COMPLETE
    run_case(
        "Test 2: Partially available, user accepts reduced quantity",
        "I want 5 plates of chicken biryani",
        sim={"cook": "success", "server": "success"},
        scripted_replies=["yes that's fine, go ahead"],
        expected="complete",
    )

    # Test 3: dish not on the menu / zero stock -> should FAIL (the only failing case)
    run_case(
        "Test 3: Dish not available",
        "Can I get 3 plates of veg fried rice",
        expected="fail",
    )


# =============================================================================
# 9. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_cases()
    else:
        run_interactive()