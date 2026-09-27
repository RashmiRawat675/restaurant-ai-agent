\# Restaurant AI Agent with LangGraph



A backend AI agent for restaurant ordering built with LangGraph and Groq. The project demonstrates stateful agent workflows, conditional routing, human-in-the-loop interaction, retry handling, and failure recovery.



\## Overview



This project simulates a restaurant ordering system where an AI agent understands the customer's request, checks menu availability, handles partial availability, processes the order, simulates cooking and serving, and generates natural-language responses.



The workflow is implemented using LangGraph's `StateGraph` with multiple nodes and conditional transitions.



The project is backend-only and currently runs through an interactive command-line interface.



\## Agent Workflow



```text

&#x20;                   User Request

&#x20;                        |

&#x20;                        v

&#x20;                    Classify

&#x20;                   /        \\

&#x20;            Not Food        Food Order

&#x20;               |                |

&#x20;               v                v

&#x20;           Not Food       Order Confirmation

&#x20;                             /      |       \\

&#x20;                        Available  Partial  Unavailable

&#x20;                           |         |          |

&#x20;                           v         v          v

&#x20;                         Cook     Ask User   Not Available

&#x20;                           |         |

&#x20;                           v         +----> Accept

&#x20;                         Server            |

&#x20;                        /     \\             v

&#x20;                   Success   Failure      Cook

&#x20;                      |        |            |

&#x20;                      v        v            v

&#x20;                   Complete  Recook       Server

&#x20;                               |          /    \\

&#x20;                               +------> Success Failure

&#x20;                                            |       |

&#x20;                                            v       v

&#x20;                                         Complete  Fail



Key Features
LangGraph StateGraph based agent workflow
Stateful order management
LLM-based food query classification
Dish and quantity extraction using Groq
Menu and inventory validation
Full and partial availability handling
Human-in-the-loop interaction using LangGraph interrupt
Resume workflow using Command
Cooking and serving simulation
Retry and maximum-attempt handling
Recooking and re-serving recovery flow
Failure states and result tracking
Groq-powered natural-language response generation
Memory checkpointing using MemorySaver
Interactive CLI mode
Automated test cases with deterministic simulation
LangGraph Architecture

The agent is built using StateGraph.

Main Nodes
classify - identifies whether the request is food-related and extracts dish and quantity
order_confirm - checks menu availability
ask_user - handles partial availability and waits for customer response
cook - simulates food preparation
server - simulates serving/delivery
recook_prep - prepares the workflow for a retry
not_food - handles unrelated requests
not_available - handles unavailable dishes
declined - handles customer rejection
cook_fail - handles repeated cooking failure
server_fail - handles repeated serving failure
complete - completes successful orders
max_attempts - handles excessive user attempts
Human-in-the-Loop

The project uses LangGraph's interrupt() functionality when only part of the requested quantity is available.

For example:

User: I want 5 plates of chicken biryani

Available: 2

Agent: Only 2 plates are available. Would you like to continue?

User: Yes, that's fine.

Agent:
    ↓
Resume workflow
    ↓
Update quantity to 2
    ↓
Cook
    ↓
Server
    ↓
Complete

The workflow is resumed using:

Command(resume=user_reply)

This demonstrates how an agent can pause execution, wait for external input, and continue from the same workflow state.

LLM Integration

Groq is used for two main purposes:

1. Query Classification

The LLM determines whether the user request is food-related and extracts:

Intent
Dish name
Quantity

Example structured output:

{
  "is_food_query": true,
  "intent": "order",
  "dish": "masala dosa",
  "quantity": 2
}
2. Natural Language Generation

The LLM generates short user-facing responses for situations such as:

Partial availability
Dish unavailable
Customer rejection
Cooking failure
Delivery failure
Successful order completion
Menu

The current simulated menu contains:

Dish	Price	Available
Paneer Butter Masala	220	5
Chicken Biryani	250	2
Veg Fried Rice	150	0
Masala Dosa	100	10
Gulab Jamun	60	8
Retry and Failure Handling

The workflow contains bounded retry logic.

Order Attempts

Maximum user/order attempts are controlled using:

order_attempt = 3
Cooking

Cooking can attempt up to:

cooked_attempt = 2
Serving

Serving can retry up to:

server_attempt = 2

If serving fails on the first attempt, the workflow can move through:

Server
  ↓
Recook Preparation
  ↓
Cook
  ↓
Server

If the maximum number of attempts is reached, the workflow terminates with a failure result.

Testing

The project includes three test cases.

Test 1 - Fully Available Dish
User: I want 2 plates of masala dosa

Expected:
complete
Test 2 - Partial Availability
User: I want 5 plates of chicken biryani

Available: 2

User: yes that's fine, go ahead

Expected:
complete
Test 3 - Unavailable Dish
User: Can I get 3 plates of veg fried rice

Available: 0

Expected:
fail

The test cases use simulation controls for deterministic cooking and serving outcomes.

Running the Project

Install the dependencies:

pip install -r Requirement.txt

Create a .env file:

GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

Run the interactive agent:

python project.py

Run the predefined test cases:

python project.py test
Technologies Used
Python
LangGraph
Groq
LLM
StateGraph
TypedDict
Human-in-the-loop
interrupt
Command
MemorySaver
Conditional workflows
State management
JSON
python-dotenv
uv
Project Structure
Project_lang/
├── src/
├── project.py
├── Requirement.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md

