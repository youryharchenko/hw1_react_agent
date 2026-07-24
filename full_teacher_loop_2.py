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

# Імпортуємо Pydantic-модель згенерованої задачі з ваших tools/modules
from tools import GeneratedMathProblem


# ==========================================
# 1. МОДЕЛІ ДАНИХ (Pydantic Models)
# ==========================================
class EvaluationResult(BaseModel):
    """Результат перевірки та оцінки математичної задачі."""

    is_correct_math: bool = Field(
        description="Чи є математичні обчислення в умовах та розв'язку правильними?"
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


class OverallState(TypedDict):
    topic: str
    grade: int
    task: dict | None
    eval_status: str | None
    feedback: str | None
    iterations: int
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 2. TOOLS (SymPy Validator)
# ==========================================
@tool
def verify_math_expression(expression: str, expected_value: str) -> str:
    """Перевіряє математичний вираз або рівняння за допомогою SymPy."""
    try:
        expr_sym = cast(sp.Expr, sp.sympify(expression))
        expected_sym = cast(sp.Expr, sp.sympify(expected_value))

        diff = sp.simplify(expr_sym - expected_sym)

        if diff == 0:
            return f"SUCCESS: Вираз '{expression}' повністю збігається з еталоном '{expected_value}'."
        else:
            return f"MISMATCH: Вираз '{expression}' дає {expr_sym}, що НЕ дорівнює очікуваному '{expected_value}'. Різниця: {diff}"
    except Exception as e:
        return f"ERROR: Помилка парсингу SymPy: {str(e)}"


# ==========================================
# 3. НАЛАШТУВАННЯ МОДЕЛЕЙ
# ==========================================
MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_SERVER_IP = "192.168.2.102"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    num_predict=1024,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
)

# Для Генератора — схема задачі
llm_gen_structured = llm.with_structured_output(GeneratedMathProblem)

# Для Оцінювача — схема вердикту та інструменти
llm_eval_structured = llm.with_structured_output(EvaluationResult)
llm_with_tools = llm.bind_tools([verify_math_expression])


# ==========================================
# 4. ВУЗЛИ ТА ГРАФ EVALUATOR (SubGraph)
# ==========================================
def agent_node(state: OverallState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_execution_node(state: OverallState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    tools_by_name = {"verify_math_expression": verify_math_expression}

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
    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-експерт. Проаналізуй задачу та результати її математичної перевірки з історії. "
                "Сформуй підсумковий структурований вердикт у форматі JSON."
            )
        )
    ] + state["messages"]

    # ТУТ використовуємо llm_eval_structured (EvaluationResult)
    structured_verdict = llm_eval_structured.invoke(prompt)

    if isinstance(structured_verdict, dict):
        content_str = json.dumps(structured_verdict, ensure_ascii=False)
    elif hasattr(structured_verdict, "model_dump_json"):
        content_str = structured_verdict.model_dump_json()
    else:
        content_str = str(structured_verdict)

    return {"messages": [AIMessage(content=content_str)]}


def router_edge(state: OverallState) -> Literal["tools", "generate_structured_output"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "generate_structured_output"


eval_workflow = StateGraph(OverallState)
eval_workflow.add_node("agent", agent_node)
eval_workflow.add_node("tools", tool_execution_node)
eval_workflow.add_node("generate_structured_output", generate_structured_output_node)

eval_workflow.add_edge(START, "agent")
eval_workflow.add_conditional_edges(
    "agent",
    router_edge,
    {"tools": "tools", "generate_structured_output": "generate_structured_output"},
)
eval_workflow.add_edge("tools", "agent")
eval_workflow.add_edge("generate_structured_output", END)

evaluator_graph = eval_workflow.compile()


# ==========================================
# 5. ВУЗЛИ ТА ГРАФ MAIN LOOP
# ==========================================
def generator_node(state: OverallState) -> dict:
    topic = state["topic"]
    grade = state["grade"]
    iterations = state.get("iterations", 0) + 1

    print(f"\n⚙️ [Generator] Спроба #{iterations} сгенерити задачу на тему '{topic}'...")

    if iterations == 1:
        system_prompt = (
            "Ти — досвідчений вчитель математики.\n"
            "ОБОВ'ЯЗКОВО заповнюй поле 'canonical_equation' (канонічне рівняння чи вираз зі знаком '='). "
            "Наприклад, 'x = 24 * (1/3)' або 'x = 8'."
        )
        current_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Склади цікаву та зрозумілу математичну задачу для {grade} класу на тему '{topic}'."
            ),
        ]
    else:
        current_messages = state["messages"] + [
            HumanMessage(
                content="Попередня версія відхилена. Виправи вказані зауваження та перепиши задачу повністю."
            )
        ]

    response = llm_gen_structured.invoke(current_messages)
    task_dict = response if isinstance(response, dict) else response.model_dump()

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
    print("🔍 [Evaluator] Перевірка згенерованої задачі...")
    task = state["task"]

    # ТУТ ВИТЯГАЄМО ЗГІДНО З ПОЛЯМИ GeneratedMathProblem!
    if task:
        task_text = f"""
    НАЗВА: {task.get("title")}
    УМОВА: {task.get("problem_statement")}
    РІВНЯННЯ: {task.get("canonical_equation")}
    РОЗВ'ЯЗОК: {task.get("step_by_step_solution")}
    ВІДПОВІДЬ: {task.get("canonical_answer")}
    """
    else:
        task_text = "Задача відсутня"

    eval_input = {
        "messages": [
            SystemMessage(
                content="Ти — контролер якості задач. Обов'язково перевір математичні обчислення через verify_math_expression та оціни мову умови."
            ),
            HumanMessage(content=f"Перевір задачу:\n{task_text}"),
        ]
    }

    res = evaluator_graph.invoke(cast(OverallState, eval_input))
    last_msg = res["messages"][-1]

    try:
        verdict = json.loads(last_msg.content)
        status = verdict.get("status", "REJECTED")
        feedback = verdict.get("feedback", "Неможливо розпарсити вердикт")
    except Exception as e:
        status = "REJECTED"
        feedback = f"Помилка парсингу результату: {str(e)}"

    print(f"📊 [Evaluator Verdict]: Status = {status} | Feedback = {feedback}")

    return {
        "eval_status": status,
        "feedback": feedback,
        "messages": [AIMessage(content=f"Оцінка: {status}. Зауваження: {feedback}")],
    }


def main_router(state: OverallState) -> Literal["generator", END]:
    if state["eval_status"] == "PASSED":
        print("✅ Задача успішно пройшла всі перевірки!")
        return END

    if state["iterations"] >= 3:
        print("⚠️ Досягнуто ліміту спроб (3). Зупиняємо цикл.")
        return END

    print("🔄 Відправляємо задачу на доопрацювання генератору...")
    return "generator"


main_workflow = StateGraph(OverallState)
main_workflow.add_node("generator", generator_node)
main_workflow.add_node("evaluator", evaluator_node)

main_workflow.add_edge(START, "generator")
main_workflow.add_edge("generator", "evaluator")
main_workflow.add_conditional_edges(
    "evaluator", main_router, {"generator": "generator", END: END}
)

app = main_workflow.compile()


# ==========================================
# 6. ЗАПУСК
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
