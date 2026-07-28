import json
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from tools import (
    EvaluationResult,
    GeneratedMathProblem,
    fraction_calculator_tool,
    geometry_2d_tool,
    sympy_solver_tool,
    verify_math_expression,
)


class OverallState(TypedDict):
    topic: str
    grade: int
    task: dict | None
    eval_status: str | None
    feedback: str | None
    iterations: int
    messages: Annotated[list[BaseMessage], add_messages]
    # Додаткові поля для детекції зациклення tool calls
    tool_call_history: list[str]
    eval_steps: int


# ==========================================
# ЗАХИСНІ КОНФІГУРАЦІЇ
# ==========================================
MAX_ITERATIONS = 3  # Максимальна кількість спроб Generator <-> Evaluator
MAX_EVAL_STEPS = 5  # Захист max_steps для внутрішнього циклу Evaluator

# ==========================================
# 2. НАЛАШТУВАННЯ МОДЕЛЕЙ
# ==========================================
MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_SERVER_IP = "192.168.2.102"

TIMEOUT_SEC = 60  # Загальний тайм-аут у секундах

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    num_predict=1024,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
    client_kwargs={"timeout": TIMEOUT_SEC},
)

llm_gen_structured = llm.with_structured_output(GeneratedMathProblem)
llm_eval_structured = llm.with_structured_output(EvaluationResult)

available_tools = [
    verify_math_expression,
    fraction_calculator_tool,
    geometry_2d_tool,
    sympy_solver_tool,
]

llm_with_tools = llm.bind_tools(available_tools)
tools_by_name = {t.name: t for t in available_tools}


# ==========================================
# ВУЗЛИ ТА ГРАФ EVALUATOR (SubGraph)
# ==========================================
def agent_node(state: OverallState) -> dict:
    steps = state.get("eval_steps", 0) + 1
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "eval_steps": steps}


def tool_execution_node(state: OverallState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    history = list(state.get("tool_call_history", []))

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # --- ЗАХИСТ 1: Детекція повторюваних tool calls (Loop Detection) ---
            call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if call_signature in history:
                result = (
                    f"ПОМИЛКА ЗАЦИКЛЕННЯ: Ви вже викликали інструмент {tool_name} "
                    f"з аргументами {tool_args}. Повторний виклик заборонено!"
                )
            else:
                history.append(call_signature)
                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(tool_args)
                else:
                    result = f"Error: Tool {tool_name} not found."

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages, "tool_call_history": history}


def generate_structured_output_node(state: OverallState) -> dict:
    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-експерт. Проаналізуй задачу та результати її математичної перевірки з історії. "
                "Сформуй підсумковий структурований вердикт у форматі JSON."
            )
        )
    ] + state["messages"]

    try:
        structured_verdict = llm_eval_structured.invoke(prompt)

        if isinstance(structured_verdict, dict):
            content_str = json.dumps(structured_verdict, ensure_ascii=False)
        elif isinstance(structured_verdict, BaseModel):
            content_str = structured_verdict.model_dump_json()
        else:
            content_str = str(structured_verdict)
    except Exception as err:
        content_str = json.dumps(
            {
                "status": "REJECTED",
                "feedback": f"Помилка генерації структурованого вердикту: {str(err)}",
            },
            ensure_ascii=False,
        )

    return {"messages": [AIMessage(content=content_str)]}


def router_edge(state: OverallState) -> Literal["tools", "generate_structured_output"]:
    last_message = state["messages"][-1]

    # --- ЗАХИСТ 2: max_steps для підграфу Evaluator ---
    if state.get("eval_steps", 0) >= MAX_EVAL_STEPS:
        print(
            f"⚠️ [Evaluator] Досягнуто max_steps ({MAX_EVAL_STEPS}). Примусовий вихід на генерацію вердикту."
        )
        return "generate_structured_output"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Перевірка чи не було зациклення на останньому кроці
        last_history = state.get("tool_call_history", [])
        for tool_call in last_message.tool_calls:
            sig = f"{tool_call['name']}:{json.dumps(tool_call['args'], sort_keys=True)}"
            if sig in last_history:
                print(
                    "⚠️ [Evaluator] Виявлено повторюваний tool call. Зупинка виклику інструментів."
                )
                return "generate_structured_output"
        return "tools"

    return "generate_structured_output"


# ==========================================
# 4. ВУЗЛИ ТА ГРАФ MAIN LOOP
# ==========================================
def generator_node(state: OverallState) -> dict:
    topic = state["topic"]
    grade = state["grade"]
    iterations = state.get("iterations", 0) + 1

    print(
        f"\n⚙️ [Generator] Спроба #{iterations} згенерувати задачу на тему '{topic}'..."
    )

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
        feedback_str = state.get("feedback", "Причина не вказана")
        current_messages = state["messages"] + [
            HumanMessage(
                content=(
                    f"Попередня версія відхилена оцінювачем з зауваженням:\n"
                    f'"{feedback_str}"\n\n'
                    f"Виправи це зауваження. УВАГА: Не дублюй текст умови в розв'язку! "
                    f"Пиши одразу покрокові дії та обчислення."
                )
            )
        ]

    try:
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
    except ValidationError as err:
        return {
            "eval_status": "REJECTED",
            "feedback": f"Схема JSON не пройшла валідацію: {str(err)}",
            "iterations": iterations,
            "messages": [AIMessage(content=f"Помилка валідації Pydantic: {str(err)}")],
        }


def evaluator_node(state: OverallState) -> dict:
    print("🔍 [Evaluator] Перевірка згенерованої задачі...")
    task = state.get("task")

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

    # Скидаємо прапорці зациклення та кроки для кожного запуску Evaluator
    eval_input = {
        "messages": [
            SystemMessage(
                content=(
                    "Ти — лояльний методист з математики. Твоя головна мета — перевірити МАТЕМАТИЧНУ КОРЕКТНІСТЬ.\n"
                    "Правила оцінювання:\n"
                    "1. Якщо математика та відповідь РЕАЛЬНО правильні — став status='PASSED'.\n"
                    "2. Не відхиляй задачу через незначні стилістичні огріхи або формулювання тексту, якщо суть зрозуміла учню.\n"
                    "3. Став 'REJECTED' ТІЛЬКИ якщо є явна математична помилка, суперечливі дані або повна відсутність розв'язку."
                )
            ),
            HumanMessage(content=f"Перевір задачу:\n{task_text}"),
        ],
        "tool_call_history": [],
        "eval_steps": 0,
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
    if state.get("eval_status") == "PASSED":
        print("✅ Задача успішно пройшла всі перевірки!")
        return END

    # --- ЗАХИСТ 2 (Продовження): max_steps для зовнішнього циклу ---
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        print(f"⚠️ Досягнуто ліміту спроб ({MAX_ITERATIONS}). Зупиняємо цикл.")
        return END

    print("🔄 Відправляємо задачу на доопрацювання генератору...")
    return "generator"


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


main_workflow = StateGraph(OverallState)
main_workflow.add_node("generator", generator_node)
main_workflow.add_node("evaluator", evaluator_node)

main_workflow.add_edge(START, "generator")
main_workflow.add_edge("generator", "evaluator")
main_workflow.add_conditional_edges(
    "evaluator", main_router, {"generator": "generator", END: END}
)

app = main_workflow.compile()
