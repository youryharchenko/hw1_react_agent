import json
from typing import Annotated, Literal, TypedDict, cast

import sympy as sp
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ==========================================
# 1. СТРУКТУРОВАНИЙ ВИХІД (Pydantic Model)
# ==========================================
class EvaluationResult(BaseModel):
    """Результат перевірки та оцінки математичної задачі."""

    is_correct_math: bool = Field(
        description="Чи є математичні обчислення в умовах та розв'язку 100% правильними?"
    )
    is_clear_text: bool = Field(
        description="Чи написана умова зрозумілою мовою без росіянізмів та дивних формулювань?"
    )
    status: Literal["PASSED", "REJECTED"] = Field(
        description="Загальний вердикт: PASSED якщо математика і текст OK, інакше REJECTED"
    )
    feedback: str = Field(
        description="Детальний коментар або описи знайдених помилок (якщо є)"
    )


# ==========================================
# 2. TOOLS (Інструмент для SymPy)
# ==========================================
@tool
def verify_math_expression(expression: str, expected_value: str) -> str:
    """
    Перевіряє математичний вираз або рівняння за допомогою SymPy.
    Приклад expression: '1/12' або '(1/2) * 12 * 8'
    Приклад expected_value: '0.08333333333333333' або '48'
    """
    try:
        # Парсимо вираз та очікуване значення через SymPy
        # expr_sym = sp.sympify(expression)
        # expected_sym = sp.sympify(expected_value)

        expr_sym = cast(sp.Expr, sp.sympify(expression))
        expected_sym = cast(sp.Expr, sp.sympify(expected_value))

        # Перевіряємо різницю: expr - expected == 0
        diff = sp.simplify(expr_sym - expected_sym)

        if diff == 0:
            return f"SUCCESS: Вираз '{expression}' повністю збігається з еталоном '{expected_value}'."
        else:
            return f"MISMATCH: Вираз '{expression}' дає {expr_sym}, що НЕ дорівнює очікуваному '{expected_value}'. Різниця: {diff}"
    except Exception as e:
        return f"ERROR: Помилка парсингу SymPy: {str(e)}"


tools = [verify_math_expression]


# ==========================================
# 3. СТАН ГРАФА (AgentState)
# ==========================================
class AgentState(TypedDict):
    # add_messages накопичує історію переписки та результати виконання tools
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 4. ВУЗЛИ ГРАФА (Nodes)
# ==========================================

# Модель Ollama (підтримка tool calling чудова в qwen2.5-coder)
# llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# Вказуємо модель, завантажену в Ollama (наприклад, qwen2.5:14b або qwen2.5:7b)
MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_SERVER_IP = "192.168.2.102"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    num_predict=1024,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
)

# 4.1. LLM з прив'язаними інструментами
llm_with_tools = llm.bind_tools(tools)

# 4.2. LLM зі structured output для фінального кроку
llm_structured = llm.with_structured_output(EvaluationResult)


def agent_node(state: AgentState) -> dict:
    """Вузол Агента: приймає рішення закликати Tool або видати фінальну відповідь."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_execution_node(state: AgentState) -> dict:
    """Вузол виконання інструментів (Ручна або стандартна обробка ToolCalls)."""
    last_message = state["messages"][-1]
    tool_messages = []

    # Співставляємо виклики з прив'язаними функціями
    tools_by_name = {t.name: t for t in tools}

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tools_by_name:
                result = tools_by_name[tool_name].invoke(tool_args)
            else:
                result = f"Error: Tool {tool_name} not found."

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages}


# ==========================================
# 5. УМОВНИЙ ПЕРЕХІД (Router Edge)
# ==========================================
def router_edge(state: AgentState) -> Literal["tools", "generate_structured_output"]:
    """Маршрутизатор: якщо є виклики інструментів — ідемо в tools, інакше — формуємо структуроване рішення."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "generate_structured_output"


def generate_structured_output_node(state: AgentState) -> dict:
    """Фінальний вузол: перетворює всю накопичену історію перевірок у Pydantic структуру."""
    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-експерт. Проаналізуй задачу та результати її математичної перевірки з історії. "
                "Сформуй підсумковий структурований вердикт."
            )
        )
    ] + state["messages"]

    structured_verdict = llm_structured.invoke(prompt)

    # Повертаємо Pydantic-об'єкт у формі повідомлення для збереження в стані
    # return {"messages": [HumanMessage(content=structured_verdict.model_dump_json())]}

    if isinstance(structured_verdict, dict):
        content_str = json.dumps(structured_verdict, ensure_ascii=False)
    elif hasattr(structured_verdict, "model_dump_json"):
        content_str = structured_verdict.model_dump_json()
    else:
        content_str = str(structured_verdict)

    return {"messages": [AIMessage(content=content_str)]}


# ==========================================
# 6. ЗБОРКА LANGGRAPH
# ==========================================
workflow = StateGraph(AgentState)

# Додаємо вузли
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_execution_node)
workflow.add_node("generate_structured_output", generate_structured_output_node)

# Встановлюємо зв'язки
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    router_edge,
    {"tools": "tools", "generate_structured_output": "generate_structured_output"},
)

# Після виконання інструменту повертаємося до агента
workflow.add_edge("tools", "agent")

# Фінальний крок
workflow.add_edge("generate_structured_output", END)

# Компіляція графа
evaluator_graph = workflow.compile()


# ==========================================
# 7. ТЕСТОВИЙ ЗАПУСК
# ==========================================
if __name__ == "__main__":
    # Беремо одну із задач, яку згенерувала ваша Ollama
    test_task = """
    УМОВА: У майстерні виробляють квадратну плитку. Вона має бути розділена на 12 частин, щоб кожна частина мала однакову площу. Яка частина плитки становить одиницю площі?
    МОДЕЛЬ: x = 1/12
    РОЗВ'ЯЗОК: Поділити на 12 частин, x = 1/12.
    ВІДПОВІДЬ: Одина частина плитки становить 1/12.
    """

    system_instruction = SystemMessage(
        content=(
            "Ти — контролер якості математичних задач. "
            "Твоє завдання: обов'язково перевірити математичні обчислення за допомогою інструменту verify_math_expression, "
            "а також оцінити якість мови умови."
        )
    )

    initial_state: AgentState = {
        "messages": [
            system_instruction,
            HumanMessage(content=f"Перевір наступну задачу:\n{test_task}"),
        ]
    }

    print("🚀 Запуск LangGraph Evaluator Agent...\n")

    for chunk in evaluator_graph.stream(initial_state, stream_mode="values"):
        latest_msg = chunk["messages"][-1]
        print(f"[{latest_msg.__class__.__name__}]: {latest_msg.content}")
        if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
            print(f"   🔧 Tool Calls: {latest_msg.tool_calls}")
        print("-" * 60)
