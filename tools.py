import json
from typing import Literal, Optional, cast

import pytest
import sympy as sp
from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class SolveAlgebraicInput(BaseModel):
    """Схема валідації вхідних даних для розв'язання алгебраїчних рівнянь."""

    model_config = ConfigDict(extra="forbid")

    expression_str: str = Field(
        ...,
        description="Алгебраїчне рівняння або вираз у форматі Python/SymPy, наприклад 'x**2 - 5*x + 6'. ПІДНЕСЕННЯ ДО СТЕПЕНЯ ТІЛЬКИ ЧЕРЕЗ '**'!",
    )
    variable: str = Field(default="x", description="Невідома змінна для розв'язання")

    @model_validator(mode="after")
    def validate_and_clean_expression(self) -> "SolveAlgebraicInput":
        # 1. Автоматично виправляємо символ піднесення до степеня '^' на '**'
        if "^" in self.expression_str:
            self.expression_str = self.expression_str.replace("^", "**")

        # 2. Перевіряємо, чи є вираз синтаксично коректним для SymPy
        try:
            parsed_expr = sp.sympify(self.expression_str)
        except Exception as e:
            raise ValueError(
                f"Некоректний математичний вираз '{self.expression_str}'. "
                f"Помилка парсингу SymPy: {str(e)}"
            )

        # 3. Валідація назви змінної (перевірка на коректний ідентифікатор Python)
        if not self.variable.isidentifier():
            raise ValueError(
                f"Назва змінної '{self.variable}' має бути коректним ідентифікатором (наприклад, 'x', 'y', 't')."
            )

        # 4. Перевіряємо, чи присутня вказана змінна серед вільних символів виразу
        target_symbol = sp.Symbol(self.variable)
        if (
            target_symbol not in parsed_expr.free_symbols
            and len(parsed_expr.free_symbols) > 0
        ):
            found_vars = ", ".join(str(s) for s in parsed_expr.free_symbols)
            raise ValueError(
                f"Змінна '{self.variable}' відсутня у виразі '{self.expression_str}'. "
                f"Знайдені змінна(і): {found_vars}."
            )

        return self


class GeneratedMathProblem(BaseModel):
    """Фінальна структура математичної задачі."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., description="Тема з шкільної математики")
    grade: int = Field(..., description="Клас")
    title: str = Field(..., description="Коротка назва задачі")
    problem_statement: str = Field(..., description="Текст умови задачі")
    canonical_equation: str = Field(..., description="Математична модель/рівняння")
    step_by_step_solution: str = Field(
        ..., description="Покроковий еталонний розв'язок"
    )
    canonical_answer: str = Field(..., description="Фінальна коротка відповідь")

    @model_validator(mode="after")
    def validate_and_clean_problem(self) -> "GeneratedMathProblem":
        # 1. Очищення від крайових пробілів у всіх текстових полях
        self.topic = self.topic.strip()
        self.title = self.title.strip()
        self.problem_statement = self.problem_statement.strip()
        self.canonical_equation = self.canonical_equation.strip()
        self.step_by_step_solution = self.step_by_step_solution.strip()
        self.canonical_answer = self.canonical_answer.strip()

        # 2. Валідація шкільного класу (1-11)
        if not (1 <= self.grade <= 11):
            raise ValueError(
                f"Клас має бути в межах від 1 до 11, отримано: {self.grade}"
            )

        # 3. Перевірка на мінімальну довжину текстових полів
        if len(self.title) < 3:
            raise ValueError(
                "Назва задачі ('title') занадто коротка (менше 3 символів)."
            )

        if len(self.problem_statement) < 15:
            raise ValueError(
                "Текст умови задачі ('problem_statement') занадто короткий."
            )

        if len(self.step_by_step_solution) < 10:
            raise ValueError(
                "Покроковий розв'язок ('step_by_step_solution') занадто короткий."
            )

        # 4. Перевірка та чистка канонічного рівняння
        if "^" in self.canonical_equation:
            self.canonical_equation = self.canonical_equation.replace("^", "**")

        # Перевіряємо наявність знаку рівності або нерівності у математичній моделі
        valid_operators = ["=", "==", "<", ">", "<=", ">="]
        if not any(op in self.canonical_equation for op in valid_operators):
            raise ValueError(
                f"Канонічне рівняння '{self.canonical_equation}' має містити знак рівності ('=') або нерівності."
            )

        return self

    @field_validator("canonical_equation")
    def validate_equation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            # Якщо модель повернула порожній рядок, можна або дати дефолтне значення,
            # або підставити заглушку замість викидання помилки:
            return "x = 0"
        if "=" not in v and not any(op in v for op in [">", "<", ">=", "<="]):
            # Якщо модель написала "1/3 * 24", перетворюємо на рівняння "x = 1/3 * 24"
            return f"x = {v}"
        return v


class EvaluationResult(BaseModel):
    """Результат перевірки та оцінки математичної задачі."""

    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def validate_evaluation_consistency(self) -> "EvaluationResult":
        # 1. Очищення фідбеку від крайових пробілів
        self.feedback = self.feedback.strip()

        # 2. Логічне узгодження: PASSED можливий ТІЛЬКИ якщо і математика, і текст правильні
        if self.is_correct_math and self.is_clear_text:
            if self.status != "PASSED":
                raise ValueError(
                    "Якщо математика та текст правильні (is_correct_math=True, is_clear_text=True), "
                    "статус 'status' має бути обов'язково 'PASSED'."
                )
        else:
            if self.status != "REJECTED":
                raise ValueError(
                    f"Якщо є помилки у математиці або тексті (is_correct_math={self.is_correct_math}, "
                    f"is_clear_text={self.is_clear_text}), статус 'status' має бути обов'язково 'REJECTED'."
                )

        # 3. Перевірка мінімальної довжини фідбеку
        if len(self.feedback) < 5:
            raise ValueError(
                "Коментар/фідбек ('feedback') занадто короткий (має бути не менше 5 символів)."
            )

        # 4. Якщо вердикт REJECTED, фідбек має містити розяснення (не бути банальною заглушкою)
        if self.status == "REJECTED" and self.feedback.lower() in [
            "ok",
            "good",
            "none",
            "немає",
            "все ок",
        ]:
            raise ValueError(
                "При статусі REJECTED фідбек має детальніше пояснювати причину відхилення."
            )

        return self


@tool("sympy_solver_tool", args_schema=SolveAlgebraicInput)
def sympy_solver_tool(expression_str: str, variable: str = "x") -> str:
    """
    Точно розв'язує рівняння expression_str = 0 відносно змінної variable за допомогою SymPy.
    """
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression_str)

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


@tool("fraction_calculator_tool", args_schema=SolveAlgebraicInput)
def fraction_calculator_tool(expression_str: str) -> str:
    """
    Обчислює дробові вирази, дає точний результат у вигляді нескоротного дробу,
    десяткового значення та відсотка.
    """
    expr = sp.sympify(expression_str)
    result = sp.simplify(expr)

    # Якщо результат — раціональний дріб (Fraction / Rational)
    if isinstance(result, sp.Rational):
        p, q = result.p, result.q
        whole = p // q
        remainder = abs(p) % q

        fraction_str = f"{p}/{q}"
        mixed_str = (
            f"{whole} цілих {remainder}/{q}"
            if whole != 0 and remainder != 0
            else fraction_str
        )
        decimal_val = float(result)

        return (
            f"Точний дріб: {fraction_str} | "
            f"Мішане число: {mixed_str} | "
            f"Десятковий дріб: {decimal_val} | "
            f"Відсоток: {decimal_val * 100:.2f}%"
        )

    return f"Результат: {result}"


@tool("geometry_2d_tool", args_schema=SolveAlgebraicInput)
def geometry_2d_tool(
    shape: str,
    target: str,
    a: float,
    b: Optional[float] = None,
    c: Optional[float] = None,
) -> str:
    """
    Обчислює площу або периметр/довжину кола для простих геометричних фігур (1-6 класи).
    """
    a_sym = sp.Float(a)

    if shape == "square":
        if target == "area":
            res = a_sym**2
            formula = f"S = a^2 = {a}^2"
        else:
            res = 4 * a_sym
            formula = f"P = 4 * a = 4 * {a}"

    elif shape == "rectangle":
        b_sym = sp.Float(b)
        if target == "area":
            res = a_sym * b_sym
            formula = f"S = a * b = {a} * {b}"
        else:
            res = 2 * (a_sym + b_sym)
            formula = f"P = 2 * (a + b) = 2 * ({a} + {b})"

    elif shape == "triangle":
        if target == "area":
            h_sym = sp.Float(b)
            res = 0.5 * a_sym * h_sym
            formula = f"S = 0.5 * a * h = 0.5 * {a} * {b}"
        else:
            b_sym, c_sym = sp.Float(b), sp.Float(c)
            res = a_sym + b_sym + c_sym
            formula = f"P = a + b + c = {a} + {b} + {c}"

    elif shape == "circle":
        r_sym = sp.Symbol("r")
        if target == "area":
            # Точне значення через Pi та приблизне десяткове
            exact = sp.pi * (a_sym**2)
            formula = f"S = pi * r^2 = pi * {a}^2"
            return f"Формула: {formula} | Точно: {exact} | Приблизно (pi≈3.14): {float(exact):.2f}"
        else:
            exact = 2 * sp.pi * a_sym
            formula = f"C = 2 * pi * r = 2 * pi * {a}"
            return f"Формула: {formula} | Точно: {exact} | Приблизно (pi≈3.14): {float(exact):.2f}"

    return "Результат: ???"


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


# =====================================================================
# Тести SolveAlgebraicInput
# =====================================================================

# =====================================================================
# 1. Позитивні тести (Happy Path & Autofix)
# =====================================================================


def test_valid_input_standard():
    """Перевірка базового коректного вводу."""
    data = SolveAlgebraicInput(expression_str="x**2 - 5*x + 6", variable="x")
    assert data.expression_str == "x**2 - 5*x + 6"
    assert data.variable == "x"


def test_autofix_caret_to_power():
    """Перевірка автоматичної заміни '^' на '**'."""
    data = SolveAlgebraicInput(expression_str="x^2 + 3*x - 10", variable="x")
    assert data.expression_str == "x**2 + 3*x - 10"


def test_custom_variable():
    """Перевірка роботи з довільною змінною (наприклад, 't' або 'y')."""
    data = SolveAlgebraicInput(expression_str="2*t**2 - 8", variable="t")
    assert data.expression_str == "2*t**2 - 8"
    assert data.variable == "t"


def test_constant_expression_valid():
    """Вираз без змінних (наприклад, '5 - 5') не повинен викликати помилку про відсутність змінної."""
    data = SolveAlgebraicInput(expression_str="10 - 4", variable="x")
    assert data.expression_str == "10 - 4"


# =====================================================================
# 2. Негативні тести (Очікувані помилки валідації)
# =====================================================================


def test_invalid_sympy_syntax():
    """Перевірка синтаксично некоректного виразу."""
    with pytest.raises(ValidationError) as exc_info:
        SolveAlgebraicInput(expression_str="x**2 - + * 5", variable="x")

    assert "Некоректний математичний вираз" in str(exc_info.value)


def test_invalid_variable_identifier():
    """Перевірка некоректної назви змінної (наприклад, число або спецсимвол)."""
    with pytest.raises(ValidationError) as exc_info:
        SolveAlgebraicInput(expression_str="x**2 - 4", variable="123_var")

    assert "має бути коректним ідентифікатором" in str(exc_info.value)


def test_mismatched_variable():
    """Перевірка ситуації, коли у виразі одна змінна (y), а вказано іншу (x)."""
    with pytest.raises(ValidationError) as exc_info:
        SolveAlgebraicInput(expression_str="y**2 - 9", variable="x")

    assert "Змінна 'x' відсутня у виразі" in str(exc_info.value)
    assert "Знайдені змінна(і): y" in str(exc_info.value)


def test_extra_fields_forbidden():
    """Перевірка заборони додаткових полів через ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError) as exc_info:
        SolveAlgebraicInput(
            expression_str="x**2 - 4",
            variable="x",
            unknown_param="test",  # Додаткове поле
        )

    assert "Extra inputs are not permitted" in str(exc_info.value)


# =====================================================================
# 3. Параметризований тест (для перевірки різних синтаксисів)
# =====================================================================


@pytest.mark.parametrize(
    "input_expr, expected_expr",
    [
        ("x^2 + 2*x + 1", "x**2 + 2*x + 1"),
        ("(x + 3)^(2)", "(x + 3)**(2)"),
        ("x**3 - x^2", "x**3 - x**2"),
    ],
)
def test_various_caret_replacements(input_expr, expected_expr):
    """Параметризована перевірка різних варіацій із символом '^'."""
    data = SolveAlgebraicInput(expression_str=input_expr, variable="x")
    assert data.expression_str == expected_expr


# =====================================================================
# Тести GeneratedMathProblem
# =====================================================================


def test_valid_generated_math_problem():
    """Тест створення валідного об'єкта задачі."""
    problem = GeneratedMathProblem(
        topic=" Квадратні рівняння ",
        grade=8,
        title=" Задача про прямокутну ділянку ",
        problem_statement="Довжина ділянки на 3 м більша за ширину. Площа дорівнює 28 кв.м. Знайдіть ширину.",
        canonical_equation="x*(x + 3) = 28",
        step_by_step_solution="1. Позначимо ширину за x.\n2. x^2 + 3x - 28 = 0.\n3. Корені: x = 4.",
        canonical_answer="Ширина ділянки — 4 м.",
    )

    # Перевірка авто-стрипінгу пробілів
    assert problem.topic == "Квадратні рівняння"
    assert problem.title == "Задача про прямокутну ділянку"
    assert problem.grade == 8
    assert problem.canonical_equation == "x*(x + 3) = 28"


def test_autofix_caret_in_canonical_equation():
    """Перевірка заміни '^' на '**' у канонічному рівнянні."""
    problem = GeneratedMathProblem(
        topic="Алгебра",
        grade=8,
        title="Тестова задача",
        problem_statement="Довжина ділянки на 3 м більша за ширину. Площа дорівнює 28 кв.м.",
        canonical_equation="x^(2) + 3*x = 28",
        step_by_step_solution="Покроковий розв'язок задачі...",
        canonical_answer="Відповідь: 4 м.",
    )
    assert problem.canonical_equation == "x**(2) + 3*x = 28"


def test_invalid_grade_too_high():
    """Перевірка виклику помилки при виході класу за межі (наприклад, 12)."""
    with pytest.raises(ValidationError) as exc_info:
        GeneratedMathProblem(
            topic="Алгебра",
            grade=12,
            title="Тестова задача",
            problem_statement="Текст умови задачі більшої довжини...",
            canonical_equation="x = 5",
            step_by_step_solution="Покроковий розв'язок...",
            canonical_answer="5",
        )
    assert "Клас має бути в межах від 1 до 11" in str(exc_info.value)


def test_autofix_missing_equality_in_equation():
    """Перевірка, що вираз без '=' автоматично перетворюється на рівняння з 'x ='."""
    problem = GeneratedMathProblem(
        topic="Алгебра",
        grade=8,
        title="Тестова задача",
        problem_statement="Текст умови задачі більшої довжини...",
        canonical_equation="x**2 + 3*x - 28",  # Немає '='
        step_by_step_solution="Покроковий розв'язок...",
        canonical_answer="5",
    )
    # Перевіряємо, що автофікс підставив "x = "
    assert problem.canonical_equation == "x = x**2 + 3*x - 28"


# def test_missing_equality_in_equation():
#     """Перевірка помилки, якщо рівняння не містить знака дорівнює/порівняння."""
#     with pytest.raises(ValidationError) as exc_info:
#         GeneratedMathProblem(
#             topic="Алгебра",
#             grade=8,
#             title="Тестова задача",
#             problem_statement="Текст умови задачі більшої довжини...",
#             canonical_equation="x**2 + 3*x - 28",  # Немає '='
#             step_by_step_solution="Покроковий розв'язок...",
#             canonical_answer="5",
#         )
#     assert "має містити знак рівності" in str(exc_info.value)


def test_too_short_problem_statement():
    """Перевірка захисту від порожнього або занадто короткого тексту умови."""
    with pytest.raises(ValidationError) as exc_info:
        GeneratedMathProblem(
            topic="Алгебра",
            grade=8,
            title="Тест",
            problem_statement="Коротко",  # Менше 15 символів
            canonical_equation="x = 5",
            step_by_step_solution="Покроковий розв'язок...",
            canonical_answer="5",
        )
    assert "занадто короткий" in str(exc_info.value)


# =====================================================================
# Тести EvaluationResult
# =====================================================================

# ---------------------------------------------------------------------
# 1. Позитивні тести (Happy Path)
# ---------------------------------------------------------------------


def test_valid_evaluation_result_passed():
    """Тест успішного створення вердикту PASSED."""
    eval_res = EvaluationResult(
        is_correct_math=True,
        is_clear_text=True,
        status="PASSED",
        feedback="  Задача складена чудово, помилок немає.  ",
    )
    assert eval_res.is_correct_math is True
    assert eval_res.is_clear_text is True
    assert eval_res.status == "PASSED"
    # Перевірка стрипінгу
    assert eval_res.feedback == "Задача складена чудово, помилок немає."


def test_valid_evaluation_result_rejected():
    """Тест успішного створення вердикту REJECTED з деталізованим фідбеком."""
    eval_res = EvaluationResult(
        is_correct_math=False,
        is_clear_text=True,
        status="REJECTED",
        feedback="Помилка в обчисленнях на кроці 2: 12 / 3 не дорівнює 5.",
    )
    assert eval_res.status == "REJECTED"
    assert eval_res.is_correct_math is False


# ---------------------------------------------------------------------
# 2. Негативні тести (Конфлікти логіки та некоректний ввід)
# ---------------------------------------------------------------------


def test_inconsistent_passed_status_when_math_failed():
    """Перевірка виклику помилки, якщо математика хибна, але статус стоїть PASSED."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=False,
            is_clear_text=True,
            status="PASSED",
            feedback="Помилка в обчисленнях, але статус чомусь PASSED",
        )
    assert "статус 'status' має бути обов'язково 'REJECTED'" in str(exc_info.value)


def test_inconsistent_passed_status_when_text_failed():
    """Перевірка виклику помилки, якщо текст некоректний, але статус стоїть PASSED."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=True,
            is_clear_text=False,
            status="PASSED",
            feedback="Текст містить росіянізми.",
        )
    assert "статус 'status' має бути обов'язково 'REJECTED'" in str(exc_info.value)


def test_inconsistent_rejected_status_when_everything_ok():
    """Перевірка виклику помилки, якщо і математика, і текст OK, але статус стоїть REJECTED."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=True,
            is_clear_text=True,
            status="REJECTED",
            feedback="Все правильно, але статус REJECTED",
        )
    assert "статус 'status' має бути обов'язково 'PASSED'" in str(exc_info.value)


def test_too_short_feedback():
    """Перевірка захисту від занадто короткого фідбеку."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=True,
            is_clear_text=True,
            status="PASSED",
            feedback="Ок",  # Менше 5 символів
        )
    assert "занадто короткий" in str(exc_info.value)


def test_rejected_with_meaningless_feedback():
    """Перевірка заборони безназмістовного фідбеку при відхиленні задачі."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=False,
            is_clear_text=True,
            status="REJECTED",
            feedback="все ок",
        )
    assert "фідбек має детальніше пояснювати причину" in str(exc_info.value)


def test_extra_fields_forbidden_in_evaluation_result():
    """Перевірка заборони додаткових полів через ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            is_correct_math=True,
            is_clear_text=True,
            status="PASSED",
            feedback="Все чудово",
            score=10,  # Заборонене додаткове поле
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


# ---------------------------------------------------------------------
# 3. Параметризований тест для комбінацій прапорців
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "math_ok, text_ok, expected_status",
    [
        (True, True, "PASSED"),
        (False, True, "REJECTED"),
        (True, False, "REJECTED"),
        (False, False, "REJECTED"),
    ],
)
def test_evaluation_status_matrix(math_ok: bool, text_ok: bool, expected_status: str):
    """Параметризована перевірка матриці відповідності булевих прапорців і статусу."""
    feedback_text = (
        "Все добре" if expected_status == "PASSED" else "Виявлено помилки у задачі"
    )

    eval_res = EvaluationResult(
        is_correct_math=math_ok,
        is_clear_text=text_ok,
        status=expected_status,
        feedback=feedback_text,
    )
    assert eval_res.status == expected_status


# =====================================================================
# Тести для LangChain Tools
# =====================================================================

# ---------------------------------------------------------------------
# 1. Тести sympy_solver_tool
# ---------------------------------------------------------------------


def test_sympy_solver_tool_success():
    """Перевірка знаходження дійсних коренів квадратного рівняння x^2 - 5x + 6 = 0."""
    raw_res = sympy_solver_tool.invoke(
        {"expression_str": "x**2 - 5*x + 6", "variable": "x"}
    )
    res = json.loads(raw_res)

    assert res["status"] == "success"
    assert "2" in res["solutions"]
    assert "3" in res["solutions"]


def test_sympy_solver_tool_error_handling():
    """Перевірка обробки некоректного виразу (Schema/SymPy validation error)."""
    # Оскільки схема SolveAlgebraicInput валідує синтаксис SymPy ще до входу в тул,
    # некоректний вираз викликає ValidationError.
    with pytest.raises(ValidationError):
        sympy_solver_tool.invoke({"expression_str": "x**2 - + *", "variable": "x"})


# ---------------------------------------------------------------------
# 4. Тести verify_math_expression
# ---------------------------------------------------------------------


def test_verify_math_expression_success():
    """Перевірка точного збігу еквівалентних математичних виразів."""
    res = verify_math_expression.invoke(
        {"expression": "2*(x + 3)", "expected_value": "2*x + 6"}
    )
    assert res.startswith("SUCCESS:")


def test_verify_math_expression_mismatch():
    """Перевірка ситуації, коли вирази не збігаються."""
    res = verify_math_expression.invoke(
        {"expression": "x + 5", "expected_value": "x + 10"}
    )
    assert res.startswith("MISMATCH:")
    assert "Різниця: -5" in res


def test_verify_math_expression_invalid_syntax():
    """Перевірка перехоплення синтаксичних помилок SymPy."""
    res = verify_math_expression.invoke(
        {"expression": "x + / 5", "expected_value": "10"}
    )
    assert res.startswith("ERROR:")
