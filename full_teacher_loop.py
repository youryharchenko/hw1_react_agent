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

# Імпортуємо наш зкомпільований evaluator_graph з попереднього файлу
# from evaluator_agent import evaluator_graph
from tools import GeneratedMathProblem

# ==========================================
# 1. МОДЕЛІ ДАНИХ (Pydantic Models)
# ==========================================
# class MathTask(BaseModel):
#     """Структура математичної задачі від генератора."""

#     title: str = Field(description="Назва задачи")
#     grade: int = Field(description="Клас (наприклад, 5)")
#     topic: str = Field(description="Тема (наприклад, fractions)")
#     statement: str = Field(description="Текст умови задачі")
#     math_model: str = Field(
#         description="Математична модель / вираз (наприклад, x = 1/12)"
#     )
#     solution: str = Field(description="Покроковий розв'язок")
#     answer: str = Field(description="Еталонна відповідь")


# ==========================================
# 2. СТАН ЗОВНІШНЬОГО ГРАФА
# ==========================================
class OverallState(TypedDict):
    topic: str
    grade: int
    task: dict | None
    eval_status: str | None
    feedback: str | None
    iterations: int  # Захист від нескінченного циклу
    messages: Annotated[list[BaseMessage], add_messages]


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


# ==========================================
# 3. ВУЗЛИ ЗОВНІШНЬОГО ГРАФА
# ==========================================
MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_SERVER_IP = "192.168.2.102"

llm_gen = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    num_predict=1024,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
)

llm_gen_structured = llm_gen.with_structured_output(GeneratedMathProblem)
llm_with_tools = llm_gen.bind_tools([verify_math_expression])


def agent_node(state: OverallState) -> dict:
    """Вузол Агента: приймає рішення закликати Tool або видати фінальну відповідь."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_execution_node(state: OverallState) -> dict:
    """Вузол виконання інструментів (Ручна або стандартна обробка ToolCalls)."""
    last_message = state["messages"][-1]
    tool_messages = []

    tools = [verify_math_expression]

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


def generate_structured_output_node(state: OverallState) -> dict:
    """Фінальний вузол: перетворює всю накопичену історію перевірок у Pydantic структуру."""
    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-експерт. Проаналізуй задачу та результати її математичної перевірки з історії. "
                "Сформуй підсумковий структурований вердикт."
            )
        )
    ] + state["messages"]

    structured_verdict = llm_gen_structured.invoke(prompt)

    # Повертаємо Pydantic-об'єкт у формі повідомлення для збереження в стані
    # return {"messages": [HumanMessage(content=structured_verdict.model_dump_json())]}

    if isinstance(structured_verdict, dict):
        content_str = json.dumps(structured_verdict, ensure_ascii=False)
    elif hasattr(structured_verdict, "model_dump_json"):
        content_str = structured_verdict.model_dump_json()
    else:
        content_str = str(structured_verdict)

    return {"messages": [AIMessage(content=content_str)]}


def router_edge(state: OverallState) -> Literal["tools", "generate_structured_output"]:
    """Маршрутизатор: якщо є виклики інструментів — ідемо в tools, інакше — формуємо структуроване рішення."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "generate_structured_output"


def generator_node(state: OverallState) -> dict:
    topic = state["topic"]
    grade = state["grade"]
    iterations = state.get("iterations", 0) + 1

    print(f"\n⚙️ [Generator] Спроба #{iterations} сгенерити задачу на тему '{topic}'...")

    # Якщо це перша ітерація — створюємо стартовий промпт
    if not state.get("messages"):
        prompt_msg = HumanMessage(
            content=f"Склади цікаву та зрозумілу математичную задачу для {grade} класу на тему '{topic}'."
        )
        current_messages = [
            SystemMessage(content="Ти — досвідчений вчитель математики."),
            prompt_msg,
        ]
    else:
        # Повторна ітерація: додаємо прохання виправити помилки на основі останнього повідомлення від Evaluator
        current_messages = state["messages"] + [
            HumanMessage(
                content="Виправи вказані зауваження та перепиши задачу повністю."
            )
        ]

    response = llm_gen_structured.invoke(current_messages)
    task_dict = response if isinstance(response, dict) else response.model_dump()

    # Повертаємо оновлений task ТА нове AIMessage в історію
    return {
        "task": task_dict,
        "iterations": iterations,
        "messages": [
            AIMessage(
                content=f"Згенеровано задачу:\n{json.dumps(task_dict, ensure_ascii=False)}"
            )
        ],
    }


def evaluator_node(state: OverallState) -> dict:
    """Вузол оцінки (виклики evaluator_graph з SymPy)."""
    print("🔍 [Evaluator] Перевірка згенерованої задачі...")

    task = state["task"]

    if task:
        task_text = f"""
    НАЗВА: {task.get("title")}
    УМОВА: {task.get("statement")}
    МОДЕЛЬ: {task.get("math_model")}
    РОЗВ'ЯЗОК: {task.get("solution")}
    ВІДПОВІДЬ: {task.get("answer")}
    """
    else:
        task_text = f"Помилка в OverallState {state}"

    system_instruction = SystemMessage(
        content=(
            "Ти — контролер якості математичних задач. "
            "Обов'язково перевір математичні обчислення через verify_math_expression "
            "та оціни зрозумілість мови умови."
        )
    )

    # Запускаємо вкладений subgraph evaluator_graph
    # eval_state: OverallState = {
    #     "messages": [
    #         system_instruction,
    #         HumanMessage(content=f"Перевір задачу:\n{task_text}"),
    #     ]
    # }

    state["messages"] = state["messages"] + [
        system_instruction,
        HumanMessage(content=f"Перевір задачу:\n{task_text}"),
    ]

    res = evaluator_graph.invoke(state)

    # Витягаємо останній structured response (AIMessage)
    last_msg = res["messages"][-1]

    try:
        verdict = json.loads(last_msg.content)
        status = verdict.get("status", "REJECTED")
        feedback = verdict.get("feedback", "Невдалий формат перевірки")
    except Exception:
        status = "REJECTED"
        feedback = f"Помилка парсингу результату перевірки: {last_msg.content}"

    print(f"📊 [Evaluator Verdict]: Status = {status} | Feedback = {feedback}")

    return {"eval_status": status, "feedback": feedback}


# ==========================================
# 4. УМОВНИЙ ПЕРЕХІД (Main Router)
# ==========================================
def main_router(state: OverallState) -> Literal["generator", END]:
    """Маршрутизатор: якщо PASSED або ліміт ітерацій -> END, інакше -> назад у generator."""
    if state["eval_status"] == "PASSED":
        print("✅ Задача успішно пройшла всі перевірки!")
        return END

    if state["iterations"] >= 3:
        print("⚠️ Досягнуто ліміту спроб (3). Зупиняємо цикл.")
        return END

    print("🔄 Відправляємо задачу на доопрацювання генератору...")
    return "generator"


# ==========================================
# 5. ЗБОРКА ЗОВНІШНЬОГО ГРАФА
# ==========================================
main_workflow = StateGraph(OverallState)

main_workflow.add_node("generator", generator_node)
main_workflow.add_node("evaluator", evaluator_node)

main_workflow.add_edge(START, "generator")
main_workflow.add_edge("generator", "evaluator")

main_workflow.add_conditional_edges(
    "evaluator", main_router, {"generator": "generator", END: END}
)

app = main_workflow.compile()


workflow = StateGraph(OverallState)

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
# 6. ТЕСТОВИЙ ЗАПУСК КІЛЬЦЯ
# ==========================================
if __name__ == "__main__":
    initial_input: OverallState = {
        "topic": "fractions",
        "grade": 5,
        "task": None,
        "eval_status": None,
        "feedback": None,
        "iterations": 0,
        "messages": [],
    }

    print("🚀 Запуск повного цикла Generator <-> Evaluator (Feedback Loop)...\n")

    final_output = app.invoke(initial_input)

    print("\n" + "=" * 60)
    print("🎯 ФІНАЛЬНИЙ РЕЗУЛЬТАТ:")
    print("=" * 60)
    print(json.dumps(final_output["task"], ensure_ascii=False, indent=2))
    print(f"СТАТУС: {final_output['eval_status']}")
