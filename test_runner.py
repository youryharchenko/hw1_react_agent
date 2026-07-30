import json
import time

from langgraph.graph.state import RunnableConfig

from agent import OverallState, app, logger_callback

# ── Тест-кейси ──────────────────────────────────────────────────
test_cases = [
    {
        "id": "TC-001",
        "topic": "Арифметика: Проста задача на ділення",
        "grade": 4,
        "expected": "Текст задачі, рішення і відповідь",
        "complexity": "simple",
    },
    {
        "id": "TC-002",
        "topic": "Гометрія: Задача на площу прямкутника",
        "grade": 5,
        "expected": "Текст задачі, рішення і відповідь",
        "complexity": "medium",
    },
    {
        "id": "TC-003",
        "topic": "Алгебра: Задача на лінійне рівняння",
        "grade": 6,
        "expected": "Текст задачі, рішення і відповідь",
        "complexity": "complex",
    },
    {
        "id": "TC-004",
        "topic": "Алгебра: Задача на квадратне рівняння",
        "grade": 7,
        "expected": "Текст задачі, рішення і відповідь",
        "complexity": "complex",
    },
]

if __name__ == "__main__":
    result = []
    for test in test_cases:
        print("\n" + "=" * 60)

        print(
            f"🚀 Test-case # {test['id']} - {test['topic']}(Клас {test['grade']}) - {test['complexity']}"
        )

        initial_input: OverallState = {
            "topic": test["topic"],
            "grade": test["grade"],
            "task": None,
            "eval_status": None,
            "feedback": None,
            "iterations": 0,
            "messages": [],
            "tool_call_history": [],
            "eval_steps": 0,
        }

        start_time = time.time()
        try:
            config: RunnableConfig = {
                "recursion_limit": 25,
                "callbacks": [logger_callback],
            }

            final_output = app.invoke(initial_input, config=config)

            elapsed_time = round(time.time() - start_time, 2)
            print(f"\n⏱️ Час виконання: {elapsed_time} сек.")

            print("\n" + "-" * 60)
            print("🎯 ФІНАЛЬНИЙ РЕЗУЛЬТАТ:")
            print("-" * 60)
            if final_output.get("task"):
                task = final_output["task"]
                print(json.dumps(final_output["task"], ensure_ascii=False, indent=2))
            else:
                task = "Задача не була згенерована"
                print(f"{task}.")
            print(f"СТАТУС: {final_output.get('eval_status')}")
            print(f"FEEDBACK: {final_output.get('feedback')}")

            result.append(
                {
                    "test_id": test["id"],
                    "topic": test["topic"],
                    "grade": test["grade"],
                    "expected": test["expected"],
                    "task": task,
                    "status": final_output.get("eval_status"),
                    "feedback": final_output.get("feedback"),
                    "steps": final_output.get("iterations"),
                    # "tool_calls": tool_calls,
                    "elapsed_sec": int(elapsed_time),
                    "complexity": test["complexity"],
                }
            )

        except Exception as e:
            print(f"\n Виконання перервано за системною помилкою або тайм-аутом: {e}")
            elapsed_time = round(time.time() - start_time, 2)
            result.append(
                {
                    "test_id": test["id"],
                    "topic": test["topic"],
                    "grade": test["grade"],
                    "expected": test["expected"],
                    "status": "ERROR",
                    "feedback": str(e),
                    "elapsed_sec": int(elapsed_time),
                    "complexity": test["complexity"],
                }
            )
        print("\n" + "=" * 60)

    with open("test_results.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
