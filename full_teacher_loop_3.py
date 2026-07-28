import json
import time

from agent import OverallState, app

if __name__ == "__main__":
    initial_input: OverallState = {
        "topic": "Геометрія",
        "grade": 5,
        "task": None,
        "eval_status": None,
        "feedback": None,
        "iterations": 0,
        "messages": [],
        "tool_call_history": [],
        "eval_steps": 0,
    }

    print("🚀 Запуск повного цикла Generator <-> Evaluator (Feedback Loop)...\n")

    # --- ЗАХИСТ 3: Global Timeout ---
    start_time = time.time()
    try:
        # Для LangGraph можна використовувати recursion_limit як додаткову підтримку
        config = {"recursion_limit": 25}

        # Обгортка виклику з перевіркою тайм-ауту
        final_output = app.invoke(initial_input, config=config)

        elapsed_time = round(time.time() - start_time, 2)
        print(f"\n⏱️ Час виконання: {elapsed_time} сек.")

        print("\n" + "=" * 60)
        print("🎯 ФІНАЛЬНИЙ РЕЗУЛЬТАТ:")
        print("=" * 60)
        if final_output.get("task"):
            print(json.dumps(final_output["task"], ensure_ascii=False, indent=2))
        else:
            print("Задача не була згенерована.")
        print(f"СТАТУС: {final_output.get('eval_status')}")

    except Exception as e:
        print(f"\n❌ Виконання перервано за системною помилкою або тайм-аутом: {e}")
