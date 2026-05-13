"""LLM-based query generation for retrieval and reranking stages."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from skill_flow.config import QueryGenConfig

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Converts verbose task instructions into concise search queries via LLM."""

    def __init__(self, config: QueryGenConfig) -> None:
        self._config = config
        self._backend = os.environ.get("SKILL_FLOW_LLM_BACKEND", "openai").lower()
        self._multi_prompt_template = Path(config.multi_query_prompt).read_text(
            encoding="utf-8"
        )
        load_dotenv()
        from skill_flow.llm import get_llm_client
        self._client = get_llm_client()
        self._cache: dict[str, str | list[str]] = self._load_cache()

    @staticmethod
    def _debug_task_ids() -> set[str]:
        raw = os.environ.get("SKILL_FLOW_DEBUG_QGEN_TASKS", "")
        return {item.strip() for item in raw.split(",") if item.strip()}

    @classmethod
    def _should_debug_task(cls, task_id: str) -> bool:
        if os.environ.get("SKILL_FLOW_DEBUG_QGEN_ALL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        return task_id in cls._debug_task_ids()

    @classmethod
    def _log_debug_response(
        cls,
        kind: str,
        task_id: str,
        attempt: int,
        content: str | None,
    ) -> None:
        if not cls._should_debug_task(task_id):
            return
        logger.warning(
            "QGEN DEBUG [%s] %s attempt %d raw response:\n%s",
            task_id,
            kind,
            attempt,
            content if content else "<empty>",
        )

    @staticmethod
    def _strict_multi_enabled() -> bool:
        return os.environ.get("SKILL_FLOW_QGEN_STRICT_MULTI", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def _cache_path(self) -> Path:
        return Path(self._config.cache_path)

    def _load_cache(self) -> dict[str, str | list[str]]:
        path = self._cache_path
        if path.exists():
            data: dict[str, str | list[str]] = json.loads(
                path.read_text(encoding="utf-8")
            )
            logger.info("Loaded %d cached queries from %s", len(data), path)
            return data
        return {}

    def _save_cache(self) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_cached(value: str | list[str]) -> list[str]:
        """Normalize a cached value to a list of strings (backward compat)."""
        if isinstance(value, str):
            return [value]
        return value

    @staticmethod
    def _clean_query(text: str) -> str:
        """Strip whitespace and surrounding quote characters from LLM output."""
        cleaned = text.strip()
        if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
            cleaned = cleaned[1:-1]
        return cleaned

    def generate(self, task_id: str, instruction: str) -> str:
        """Generate a concise search query from a task instruction.

        Returns a cached result if available, otherwise calls the LLM
        and writes through to the cache file.
        """
        if task_id in self._cache:
            logger.debug("Cache hit for %s", task_id)
            cached = self._cache[task_id]
            if isinstance(cached, list):
                return cached[0]
            return cached

        framed = (
            "[QUERY GENERATION — do NOT solve this task]\n\n"
            f"{instruction}\n\n"
            "[Output: one short search query string, under 200 chars, no explanation]"
        )
        kwargs: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": framed},
            ],
            "max_tokens": self._config.max_tokens,
        }
        if not self._config.model.startswith("gpt-5"):
            kwargs["temperature"] = self._config.temperature
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

        content = response.choices[0].message.content
        self._log_debug_response("single", task_id, 1, content)
        query = self._clean_query(content) if content else instruction

        self._cache[task_id] = query
        self._save_cache()
        logger.info("Generated query for %s: %s", task_id, query)
        return query

    _MAX_RETRIES = 2

    def generate_multi(self, task_id: str, instruction: str) -> list[str]:
        """Generate one or more search queries from a task instruction.

        When ``num_queries == 1`` delegates to :meth:`generate` and wraps
        the result in a list.  When ``num_queries > 1`` makes a single LLM
        call requesting a JSON array of queries.  Retries up to
        ``_MAX_RETRIES`` times if parsing fails.

        Cached values are normalised: legacy ``str`` entries become
        single-element lists so old caches keep working.
        """
        raw_num = self._config.num_queries
        if isinstance(raw_num, list):
            msg = "num_queries must be int for generate_multi, got list"
            raise TypeError(msg)
        num: int = raw_num

        if num <= 1:
            return [self.generate(task_id, instruction)]

        if task_id in self._cache:
            logger.debug("Cache hit (multi) for %s", task_id)
            return self._normalize_cached(self._cache[task_id])

        system_prompt = self._multi_prompt_template.format(num_queries=num)
        queries: list[str] | None = None
        strict_multi = self._strict_multi_enabled()
        max_retries = self._MAX_RETRIES

        framed_multi = (
            "Generate search queries for the following task description. "
            "Return only the numbered queries.\n\n"
            f"{instruction}"
        )
        for attempt in range(1, max_retries + 1):
            temp = self._config.temperature + 0.3 * (attempt - 1)
            kwargs_multi: dict[str, object] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": framed_multi},
                ],
                "max_tokens": self._config.max_tokens,
            }
            if not self._config.model.startswith("gpt-5"):
                kwargs_multi["temperature"] = min(temp, 1.0)
            response = self._client.chat.completions.create(**kwargs_multi)  # type: ignore[call-overload]
            content = response.choices[0].message.content
            self._log_debug_response("multi", task_id, attempt, content)
            parsed = self._parse_multi_response(content, num)
            if parsed is not None:
                completed = self._pad_queries_from_instruction(instruction, parsed, num)
                if len(completed) == num:
                    if len(parsed) != num:
                        logger.info(
                            "Completed %d/%d multi queries from task instruction for %s",
                            len(parsed),
                            num,
                            task_id,
                        )
                    queries = completed
                    break
                if strict_multi:
                    logger.warning(
                        "Strict multi-query mode still missing %d/%d queries for %s",
                        num - len(completed),
                        num,
                        task_id,
                    )
                    continue
                queries = completed if completed else parsed
                break
            logger.warning(
                "Retry %d/%d for %s: LLM returned invalid queries",
                attempt,
                max_retries,
                task_id,
            )

        if queries is None:
            if strict_multi:
                msg = (
                    f"Strict multi-query mode: failed to generate {num} valid multi queries "
                    f"for {task_id} after {max_retries} attempts"
                )
                logger.error(msg)
                raise RuntimeError(msg)
            logger.warning(
                "All retries failed for %s, falling back to %d single-query calls",
                task_id,
                num,
            )
            queries = self._generate_n_single(task_id, instruction, num)

        self._cache[task_id] = queries
        self._save_cache()
        logger.info("Generated %d queries for %s: %s", len(queries), task_id, queries)
        return queries

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    def _generate_n_single(self, task_id: str, instruction: str, num: int) -> list[str]:
        """Generate *num* diverse queries via repeated single-query LLM calls.

        Each call after the first includes previously generated queries in
        the prompt so the LLM avoids repeating itself.  Temperature is
        bumped by 0.3 per call to encourage diversity.
        """
        queries: list[str] = []
        for i in range(num):
            temp = self._config.temperature + 0.3 * i
            user_msg = (
                "[QUERY GENERATION — do NOT solve this task]\n\n"
                f"{instruction}\n\n"
                "[Output: one short search query string, under 200 chars]"
            )
            if queries:
                already = "\n".join(f"- {q}" for q in queries)
                user_msg = (
                    "[QUERY GENERATION — do NOT solve this task]\n\n"
                    f"{instruction}\n\n"
                    f"Already generated (do NOT repeat these):\n{already}\n\n"
                    "[Output: one different short search query, under 200 chars]"
                )
            kwargs_single: dict[str, object] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": self._config.max_tokens,
            }
            if not self._config.model.startswith("gpt-5"):
                kwargs_single["temperature"] = min(temp, 1.0)
            response = self._client.chat.completions.create(**kwargs_single)  # type: ignore[call-overload]
            content = response.choices[0].message.content
            self._log_debug_response("fallback-single", task_id, i + 1, content)
            q = self._clean_query(content) if content else ""
            if q and q not in set(queries):
                queries.append(q)
        logger.info(
            "Single-query fallback for %s: %d / %d requested",
            task_id,
            len(queries),
            num,
        )
        return queries if queries else [instruction]

    _MAX_QUERY_LEN = 500
    _KEYWORD_STOPWORDS = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "each",
        "for",
        "from",
        "get",
        "in",
        "into",
        "of",
        "on",
        "out",
        "the",
        "their",
        "then",
        "to",
        "use",
        "using",
        "with",
    }

    @classmethod
    def _keywordize_step_query(cls, text: str) -> str | None:
        """Convert a procedural step sentence into a compact keyword query."""
        stripped = text.strip().strip('"').strip("'")
        if not stripped:
            return None

        stripped = re.sub(
            r"^(?:output|compute|use|do not|analyze|plan|simulate|load|run|"
            r"only|support|tune|ensure|write|read|apply|generate|check|find|"
            r"detect|evaluate|extract|identify|calculate|parse|look up|look)\s+",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(
            r"\b(?:save|output|write|return)\b.*$",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = stripped.replace("×", " x ")

        raw_tokens = re.findall(r"[A-Za-z0-9_./+-]+", stripped)
        keywords: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            lower = token.lower().strip("._")
            if not lower or lower in cls._KEYWORD_STOPWORDS:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            keywords.append(token.strip())

        if len(keywords) < 3:
            return None
        return " ".join(keywords[:12])

    @classmethod
    def _pad_queries_from_instruction(
        cls,
        instruction: str,
        existing: list[str],
        num: int,
    ) -> list[str]:
        """Fill missing query slots from the task instruction itself.

        This is a local repair step, not an LLM fallback.
        """
        queries = list(existing)
        seen = set(queries)

        def _add(candidate: str | None) -> None:
            if not candidate:
                return
            normalized = candidate.strip().strip('"').strip("'")
            if not normalized or normalized in seen:
                return
            if len(normalized) > cls._MAX_QUERY_LEN:
                return
            seen.add(normalized)
            queries.append(normalized)

        clauses = [c.strip() for c in re.split(r"[\n;.]+", instruction) if c.strip()]
        for clause in clauses:
            _add(cls._keywordize_step_query(clause))
            if len(queries) >= num:
                return queries[:num]

        _add(cls._keywordize_step_query(instruction))
        if len(queries) >= num:
            return queries[:num]

        files = re.findall(
            r"\b[\w./-]+\.(?:json|md|csv|yaml|yml|dxf|stl|mp4|pdf|txt)\b",
            instruction,
            flags=re.IGNORECASE,
        )
        if files:
            _add(" ".join(dict.fromkeys(files)))

        return queries[:num]

    @staticmethod
    def _parse_multi_response(
        content: str | None,
        num: int,
    ) -> list[str] | None:
        """Parse the LLM response as a JSON array of short query strings.

        Returns ``None`` when the response is unparseable or contains
        items that look like task output rather than search queries (e.g.
        JSON objects or strings longer than ``_MAX_QUERY_LEN``).
        """
        if not content:
            return None
        text = QueryGenerator._strip_code_fence(content)

        def _normalize_queries(items: list[object]) -> list[str] | None:
            queries: list[str] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, str):
                    logger.warning(
                        "Non-string item in query array: %s", type(item).__name__
                    )
                    return None
                stripped = item.strip()
                if not stripped:
                    continue
                if len(stripped) > QueryGenerator._MAX_QUERY_LEN:
                    logger.warning(
                        "Query too long (%d chars), likely task output", len(stripped)
                    )
                    return None
                if stripped not in seen:
                    seen.add(stripped)
                    queries.append(stripped)
            return queries[:num] if queries else None

        def _load_json_candidates(raw_text: str) -> object | None:
            candidates: list[str] = [raw_text]
            start = raw_text.find("[")
            end = raw_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidates.append(raw_text[start : end + 1])
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
            return None

        # 步骤型句子过滤：剔除任务描述句，保留关键词型查询。
        _STEP_PREFIXES = re.compile(
            r"^(?:output|compute|use|do not|analyze|plan|simulate|load|run|"
            r"only|support|tune|ensure|write|read|apply|generate|check|find|"
            r"detect|evaluate|extract|identify|calculate|parse|look up|look|"
            r"note:|input data:|constraints:|objective:|output structure:)",
            re.IGNORECASE,
        )

        def _is_search_query(q: str) -> bool:
            """Return True if the string looks like a retrieval keyword query, not a task step."""
            # 过长的句子多为任务步骤描述
            if len(q) > 160:
                return False
            # 以动词短语开头的步骤描述
            if _STEP_PREFIXES.match(q):
                return False
            # markdown 加粗标题
            if q.startswith("**") and q.endswith("**"):
                return False
            # 问句（What/How/Are/Is...）通常不是搜索关键词
            if re.match(r"^(?:what|how|are|is|why|when|where)\b", q, re.IGNORECASE):
                return False
            # 完整句子判断：以句号结尾且超过 6 个词，多为步骤描述
            if q.endswith(".") and len(q.split()) >= 7:
                return False
            # 包含"the ... to ..."结构的完整句
            if re.search(r"\bthe\b.{5,50}\bto\b", q) and len(q.split()) > 8:
                return False
            return True

        def _normalize_queries(items: list[object], filter_steps: bool = False) -> list[str] | None:
            queries: list[str] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, str):
                    logger.warning(
                        "Non-string item in query array: %s", type(item).__name__
                    )
                    return None
                stripped = item.strip().strip("*").strip()
                if not stripped:
                    continue
                if len(stripped) > QueryGenerator._MAX_QUERY_LEN:
                    logger.warning(
                        "Query too long (%d chars), likely task output", len(stripped)
                    )
                    return None
                if filter_steps and not _is_search_query(stripped):
                    continue
                if stripped not in seen:
                    seen.add(stripped)
                    queries.append(stripped)
            return queries[:num] if queries else None

        # 1️⃣ 优先：编号列表（Copilot 自然产出格式）
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        recovered: list[str] = []
        pat = re.compile(r"^(?:[-*•]\s+|\d+[\.)]\s+)(.+)$")
        for ln in lines:
            m = pat.match(ln)
            if m:
                recovered.append(m.group(1).strip().strip('"'))
        if recovered:
            normalized = _normalize_queries(recovered, filter_steps=True)
            if normalized and len(normalized) == num:
                logger.info(
                    "Parsed %d/%d multi queries from numbered list",
                    len(normalized),
                    num,
                )
                return normalized

            salvaged = [
                q
                for item in recovered
                if (q := QueryGenerator._keywordize_step_query(item)) is not None
            ]
            normalized_salvaged = _normalize_queries(salvaged, filter_steps=False)
            if normalized_salvaged:
                logger.info(
                    "Salvaged %d/%d multi queries from numbered step list",
                    len(normalized_salvaged),
                    num,
                )
                return normalized_salvaged

            if normalized:
                logger.info(
                    "Parsed only %d/%d multi queries from numbered list; continuing to JSON fallback",
                    len(normalized),
                    num,
                )

        # 2️⃣ 兜底：JSON 数组（OpenAI 兼容模型）
        parsed = _load_json_candidates(text)
        if parsed is not None:
            if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
                parsed = parsed["queries"]
            if isinstance(parsed, list):
                normalized = _normalize_queries(list(parsed), filter_steps=True)
                if normalized:
                    return normalized

        logger.warning("Failed to parse multi-query response")
        return None
