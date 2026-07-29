import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class TrajectoryLogger(BaseCallbackHandler):
    def __init__(self, filepath: str = "trajectory.json"):
        self.filepath = filepath
        self.step_counter = 0
        self._start_times: Dict[UUID, float] = {}
        self._inputs: Dict[UUID, Any] = {}
        self._tool_names: Dict[UUID, str] = {}
        self.steps = []

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_log(self, log_entry: Dict[str, Any]):

        self.steps.append(log_entry)
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.steps, ensure_ascii=False, indent=2) + "\n\n")

    # --- Відстеження Chain / Node ---
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Зберігаємо час початку та вхідні дані за run_id
        self._start_times[run_id] = time.perf_counter()
        self._inputs[run_id] = inputs

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._start_times.pop(run_id, time.perf_counter())
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        node_input = self._inputs.pop(run_id, None)

        # Визначаємо назву вузла (або використовуємо дефолтну)
        node_name = kwargs.get("name") or "Chain/Node"

        self.step_counter += 1

        log_entry = {
            "step_number": self.step_counter,
            "event_type": "node_execution",
            "node_name": node_name,
            "timestamp": self._get_timestamp(),
            "duration_ms": duration_ms,
            "input": self._clean_data(node_input),
            "output": self._clean_data(outputs),
            "tool_calls": [],  # Можна витягнути з outputs, якщо вузол повертає AIMessage з tool_calls
        }

        self._write_log(log_entry)

    # --- Відстеження Tool Calls ---
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        self._inputs[run_id] = input_str
        # Зберігаємо назву інструменту за run_id
        self._tool_names[run_id] = (
            serialized.get("name", "Tool") if serialized else "Tool"
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._start_times.pop(run_id, time.perf_counter())
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        tool_input = self._inputs.pop(run_id, None)

        tool_name = self._tool_names.pop(run_id, "Tool")

        self.step_counter += 1

        log_entry = {
            "step_number": self.step_counter,
            "event_type": "tool_execution",
            "node_name": tool_name,
            "timestamp": self._get_timestamp(),
            "duration_ms": duration_ms,
            "input": tool_input,
            "output": output,
            "tool_calls": [{"name": tool_name, "args": tool_input}],
        }

        self._write_log(log_entry)

    def _clean_data(self, data: Any) -> Any:
        """Серіалізатор для об'єктів, які не конвертуються в JSON за замовчуванням."""
        try:
            # Намагаємося просто дампити, якщо об'єкт базовий
            json.dumps(data)
            return data
        except (TypeError, OverflowError):
            # Перетворюємо складні об'єкти (LangChain Messages тощо) у dict або str
            if hasattr(data, "dict"):
                return data.dict()
            return str(data)
