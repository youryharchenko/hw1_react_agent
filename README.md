# ReAct агент у LangGraph для генерації та перевірки задач з математики

## 1. Опис проєкту
Проєкт на базі LangGraph та локальної LLM (Qwen 2.5).

Основна мета системи — автоматизована генерація та багаторівнева методична й математична перевірка навчальних задач (зокрема з математики та геометрії для школярів). Генератор створює задачу із покроковим розв'язком і математичною моделлю, а Оцінювач перевіряє її через SymPy-інструменти та дає структурований фідбек у разі виявлення помилок.

## 2. Доменна задача
- **Домен:** генерація шкільних задач з математики 
- **Що вирішує агент:** Запити  на генерацію задач на дроби, геометрію та рівняння.
- **Інструменти (tools):**
 - `sympy_solver_tool` — розв'язує рівняння відносно змінної за допомогою SymPy.
 - `fraction_calculator_tool` — обчислює дробові вирази
 - `geometry_2d_tool` — обчислює площу або периметр для простих геометричних фігур
 - `verify_math_expression` — перевіряє математичний вираз або рівняння за допомогою SymPy

## 3. Архітектура

**LLM-провайдер**: агенти працюють локально на базі Ollama з моделлю qwen2.5-coder:7b (з налаштованим temperature=0.1 та структурованим виводом через Pydantic).

**Структура графа (LangGraph)**: 

* Зовнішній граф (Main Loop): Працює за схемою Generator -> Evaluator -> Conditional Edge

* Внутрішній підграф (Evaluator Subgraph): Побудований за циклом Agent <-> Tools -> Generate Structured Output. Агент приймає рішення щодо виклику SymPy-інструментів, виконує їх у вузлі tools, після чого формує остаточний JSON-вердикт EvaluationResult.
 
**Застосовані захисні механізми**:

* Обмеження кроків (max_steps): Глобальний ліміт спроб генерування MAX_ITERATIONS = 3 для зовнішнього циклу та MAX_EVAL_STEPS = 5 для  внутрішнього. Додатково застосовано recursion_limit = 25 у конфігу LangGraph. 
 
* Тайм-аут (timeout): Тайм-аут виконання запиту дл LLM TIMEOUT_SEC = 60 сек. 
 
* Детекція зациклення (LoopDetector): Механізм контролю tool_call_history з параметром max_repeats = 1.

## 4. Інструкція запуску
 
1. Клонувати репозиторій: `git clone `.
2. Перейти в директорію: `cd hw1_react_agent`.
3. Встановити залежності: `pip install -r requirements.txt`.
4. Налаштувати адресу сервісу OLLAMA: `OLLAMA_SERVER_IP="..."`.
5. Запустити агента на тест-кейсах: `python test_runner.py`.

## 5. Результати тестування

### Тест інструментів

```
$ pytest -v tools.py
================================================================== test session starts ==================================================================
platform linux -- Python 3.12.9, pytest-9.1.1, pluggy-1.6.0 -- /home/youry/Projects/GoIT/hw1_react_agent/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/youry/Projects/GoIT/hw1_react_agent
plugins: anyio-4.14.2, langsmith-0.10.9
collected 33 items

tools.py::test_valid_input_standard PASSED [  3%]
tools.py::test_autofix_caret_to_power PASSED [  6%]
tools.py::test_custom_variable PASSED [  9%]
tools.py::test_constant_expression_valid PASSED [ 12%]
tools.py::test_invalid_sympy_syntax PASSED [ 15%]
tools.py::test_invalid_variable_identifier PASSED [ 18%]
tools.py::test_mismatched_variable PASSED [ 21%]
tools.py::test_extra_fields_forbidden PASSED [ 24%]
tools.py::test_various_caret_replacements[x^2 + 2*x + 1-x**2 + 2*x + 1] PASSED [ 27%]
tools.py::test_various_caret_replacements[(x + 3)^(2)-(x + 3)**(2)] PASSED [ 30%]
tools.py::test_various_caret_replacements[x**3 - x^2-x**3 - x**2] PASSED [ 33%]
tools.py::test_valid_generated_math_problem PASSED [ 36%]
tools.py::test_autofix_caret_in_canonical_equation PASSED [ 39%]
tools.py::test_invalid_grade_too_high PASSED [ 42%]
tools.py::test_autofix_missing_equality_in_equation PASSED [ 45%]
tools.py::test_valid_evaluation_result_passed PASSED [ 51%]
tools.py::test_valid_evaluation_result_rejected PASSED [ 54%]
tools.py::test_inconsistent_passed_status_when_math_failed PASSED [ 57%]
tools.py::test_inconsistent_passed_status_when_text_failed PASSED [ 60%]
tools.py::test_inconsistent_rejected_status_when_everything_ok PASSED [ 63%]
tools.py::test_too_short_feedback PASSED [ 66%]
tools.py::test_rejected_with_meaningless_feedback PASSED [ 69%]
tools.py::test_extra_fields_forbidden_in_evaluation_result PASSED [ 72%]
tools.py::test_evaluation_status_matrix[True-True-PASSED] PASSED [ 75%]
tools.py::test_evaluation_status_matrix[False-True-REJECTED] PASSED [ 78%]
tools.py::test_evaluation_status_matrix[True-False-REJECTED] PASSED [ 81%]
tools.py::test_evaluation_status_matrix[False-False-REJECTED] PASSED [ 84%]
tools.py::test_sympy_solver_tool_success PASSED [ 87%]
tools.py::test_sympy_solver_tool_error_handling PASSED [ 90%]
tools.py::test_verify_math_expression_success PASSED [ 93%]
tools.py::test_verify_math_expression_mismatch PASSED [ 96%]
tools.py::test_verify_math_expression_invalid_syntax PASSED [100%]

================================================================== 33 passed in 0.41s 
```

Підсумкова таблиця за `test_results.json`:

| Test ID | Складність | Кроків | Tool calls    | Час, мс | Статус    |
|---------|------------|--------|------------------|---------|--------------|
| TC-001 | simple   | 2   | get_weather × 1 | 1200  | ✅ success  |
| TC-002 | medium   | 4   | get_weather × 2 | 2300  | ✅ success  |
| TC-003 | complex  | 7   | get_weather × 3 | 4500  | ⚠️ partial  |

## 6. Аналіз результатів
**Чи правильно агент обирав tools?**
[1-2 абзаци з прикладами правильного та неправильного вибору]

**Де агент помилявся і чому?**
[галюцинації параметрів, неправильний вибір tool, неповна відповідь — наведіть конкретні випадки з trajectory.json]

**Кроки: simple vs complex.**
[порівняння кількості кроків між простими та складними задачами; чи є лінійна залежність від складності]

**Чи досягав агент max_steps або timeout?**
[якщо так — на якому тест-кейсі та чому; якщо ні — наскільки далеко був ліміт]

**Залежність часу виконання від складності.**
[короткий висновок: latency vs complexity]

**Випадки, коли агент НЕ використав tool, хоч мав би.**
[якщо є — приклад запиту, реакція агента та інтерпретація]

## 7. Висновки та обмеження
- Що працює добре: ...
- Що можна покращити: ...
- Відомі обмеження: ...

## 8. Структура файлів
- `tools.py` — Pydantic-схеми та tools
- `agent.py` — LangGraph граф
- `safety.py` — max_steps, timeout, LoopDetector
- `logger.py` — TrajectoryLogger
- `test_runner.py` — тест-кейси
- `trajectory.json` — лог траєкторії (генерується)
- `test_results.json` — результати тестів (генерується)
