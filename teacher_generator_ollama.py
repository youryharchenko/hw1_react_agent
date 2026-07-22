import json
from typing import Annotated, Optional, Sequence, TypedDict, cast

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

# =====================================================================
# КРОК 1: Pydantic-схеми
# =====================================================================


class SolveAlgebraicInput(BaseModel):
    """Схема валідації вхідних даних для розв'язання алгебраїчних виразів."""

    expression_str: str = Field(
        ...,
        description="Алгебраїчне рівняння або вираз у форматі Python/SymPy, наприклад 'x**2 - 5*x + 6'. ПІДНЕСЕННЯ ДО СТЕПЕНЯ ТІЛЬКИ ЧЕРЕЗ '**'!",
    )
    variable: str = Field(default="x", description="Змінна для розв'язання")


class GeneratedMathProblem(BaseModel):
    """Фінальна структура математичної задачі."""

    topic: str = Field(..., description="Тема з шкільної математики")
    grade: int = Field(..., description="Клас")
    title: str = Field(..., description="Коротка назва задачі")
    problem_statement: str = Field(..., description="Текст умови задачі")
    canonical_equation: str = Field(..., description="Математична модель/рівняння")
    step_by_step_solution: str = Field(
        ..., description="Покроковий еталонний розв'язок"
    )
    canonical_answer: str = Field(..., description="Фінальна коротка відповідь")


# =====================================================================
# КРОК 2: Інструмент точної математики (SymPy Tool)
# =====================================================================


@tool(args_schema=SolveAlgebraicInput)
def sympy_solver_tool(expression_str: str, variable: str = "x") -> str:
    """
    Точно розв'язує рівняння expression_str = 0 відносно змінної variable за допомогою SymPy.
    """
    try:
        # Автоматична заміна популярної помилки локальних моделей '^' на '**'
        cleaned_expr = expression_str.replace("^", "**")

        var = sp.Symbol(variable)
        expr = sp.sympify(cleaned_expr)
        solutions = sp.solve(expr, var)

        # Фільтруємо дійсні розв'язки для шкільної програми
        real_solutions = [sol for sol in solutions if sol.is_real]

        return json.dumps(
            {
                "status": "success",
                "expression": str(expr),
                "solutions": [str(sol) for sol in real_solutions],
                "raw_solutions": [str(sol) for sol in solutions],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Не вдалося обчислити вираз '{expression_str}': {str(e)}",
            },
            ensure_ascii=False,
        )


tools = [sympy_solver_tool]
tools_by_name = {t.name: t for t in tools}


# =====================================================================
# КРОК 3: Стан Графа (State)
# =====================================================================


class TeacherState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    grade: int
    generated_problem: Optional[GeneratedMathProblem]


# =====================================================================
# КРОК 4: Ініціалізація локальної LLM через Ollama
# =====================================================================

# Вказуємо модель, завантажену в Ollama (наприклад, qwen2.5:14b або qwen2.5:7b)
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.6,
)

llm_with_tools = llm.bind_tools(tools)


# =====================================================================
# КРОК 5: Вузли графа (Nodes)
# =====================================================================


def generate_raw_problem_node(state: TeacherState) -> dict:
    """Вузол 1: Локальна LLM придумує сюжет і викликає SymPy."""
    sys_prompt = SystemMessage(
        content=(
            f"Ти досвідчений вчитель математики української школи. Створи цікаву сюжетну текстову задачу "
            f"для {state['grade']} класу на тему '{state['topic']}'.\n\n"
            f"ВАЖЛИВО:\n"
            f"1. Сформулюй математичне рівняння, яке описує задачу.\n"
            f"2. Використай інструмент `sympy_solver_tool`, щоб розв'язати його.\n"
            f"3. Для виразу у `sympy_solver_tool` використовуй синтаксис Python: степені через '**' (наприклад 'x**2 + 3*x - 10').\n"
            f"4. Пиши ВИКЛЮЧНО літературною українською мовою.\n"
            f"5. Ретельно перевір весь текст на наявність русизмів перед формуванням JSON.\n"
        )
    )

    prompt = HumanMessage(content=f"Згенеруй задачу на тему: {state['topic']}")
    response = llm_with_tools.invoke([sys_prompt, prompt])

    return {"messages": [response]}


def execute_math_tool_node(state: TeacherState) -> dict:
    """Вузол 2: Виконання обчислення у SymPy."""
    last_message = state["messages"][-1]
    tool_responses = []

    # Перевірка на виклики інструментів
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call.get("id", "local_call_id")

            if tool_name in tools_by_name:
                selected_tool = tools_by_name[tool_name]
                observation = selected_tool.invoke(tool_args)

                tool_responses.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_id)
                )

    return {"messages": tool_responses}


def finalize_problem_node(state: TeacherState) -> dict:
    """Вузол 3: Оформлення структурованої задачі на основі відповідей SymPy."""
    structured_llm = llm.with_structured_output(GeneratedMathProblem)

    sys_prompt = SystemMessage(
        content=(
            "Твоє завдання — оформити фінальну задачу на основі попереднього контексту.\n\n"
            "ЗВЕРНИ УВАГУ НА КІНЦЕВУ ВІДПОВІДЬ (`canonical_answer`):\n"
            "1. Не пиши фрази на кшталт 'використаємо SymPy' чи 'за допомогою коду'.\n"
            "2. Візьми ГОТОВІ РОЗВ'ЯЗКИ з результатів інструменту SymPy (ігноруй від'ємні корені, якщо мова про довжину/площу).\n"
            "3. Напиши чітку фінальну відповідь з одиницями виміру (наприклад: 'Ширина — 20 см, довжина — 30 см.')."
        )
    )

    all_messages = [sys_prompt] + list(state["messages"])
    raw_result = structured_llm.invoke(all_messages)
    final_problem: GeneratedMathProblem = cast(
        GeneratedMathProblem, raw_result
    )  # structured_llm.invoke(all_messages)

    return {"generated_problem": final_problem}


# =====================================================================
# КРОК 6: Побудова та запуск графа
# =====================================================================

workflow = StateGraph(TeacherState)

workflow.add_node("generate_raw", generate_raw_problem_node)
workflow.add_node("verify_math", execute_math_tool_node)
workflow.add_node("finalize", finalize_problem_node)

workflow.add_edge(START, "generate_raw")
workflow.add_edge("generate_raw", "verify_math")
workflow.add_edge("verify_math", "finalize")
workflow.add_edge("finalize", END)

teacher_app = workflow.compile()


if __name__ == "__main__":
    test_input: TeacherState = {
        "topic": "Площа прямокутника (задача має зводитися до квадратного рівняння)",
        "grade": 8,
        "messages": [],
        "generated_problem": None,
    }

    print(f"--- Генерація задачі через Ollama ({MODEL_NAME}) ---")

    result = teacher_app.invoke(test_input)
    problem: GeneratedMathProblem = result["generated_problem"]

    print("\n" + "=" * 60)
    print(f"📌 НАЗВА: {problem.title}")
    print(f"📚 КЛАС: {problem.grade} | ТЕМА: {problem.topic}")
    print("=" * 60)
    print(f"\n📝 УМОВА ЗАДАЧІ:\n{problem.problem_statement}")
    print(f"\n📐 МАТЕМАТИЧНА МОДЕЛЬ:\n{problem.canonical_equation}")
    print(f"\n💡 ПОКРОКОВИЙ РОЗВ'ЯЗОК:\n{problem.step_by_step_solution}")
    print(f"\n✅ ЕТАЛОННА ВІДПОВІДЬ:\n{problem.canonical_answer}")
    print("=" * 60)
