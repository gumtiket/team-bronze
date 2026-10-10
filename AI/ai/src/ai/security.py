"""Recognized-secret masking for P2. This is not a general secret scanner."""

import ast
import json
import re

from ai.credentials import (
    DUMMY_SECRET as DUMMY_SECRET,
)
from ai.credentials import (
    credential_name as credential_name,
)
from ai.credentials import (
    credential_urls,
    secret_literals,
)
from ai.detectors.repo import RepoView, aliases, qualified

# Config flags such as deploy-spec `secret: true` are not secret values; masking them
# would mark every later diff containing "true"/"false" as sensitive.
CONFIG_SCALARS = frozenset({"true", "false", "yes", "no", "on", "off", "null", "none", "~"})


def node_span(source: str, node: ast.AST) -> tuple[int, int]:
    """AST columns count UTF-8 bytes, including for Korean source literals."""
    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode("utf-8")) for line in lines[: node.lineno - 1]) + node.col_offset
    end = (
        sum(len(line.encode("utf-8")) for line in lines[: node.end_lineno - 1])
        + node.end_col_offset
    )
    return start, end


def edit_nodes(source: str, edits: list[tuple[ast.AST, str]]) -> str:
    ranges = sorted([(node_span(source, node), text) for node, text in edits], reverse=True)
    content = source.encode("utf-8")
    last = len(content) + 1
    for (start, end), replacement in ranges:
        if end > last:
            raise ValueError("overlapping_edits")
        content = content[:start] + replacement.encode("utf-8") + content[end:]
        last = start
    return content.decode("utf-8")


class SourceMasker:
    def __init__(self, repo: RepoView) -> None:
        self._values: set[str] = set()
        self.blocked_files: set[str] = set()
        self._nodes: dict[str, list[ast.Constant]] = {}
        for file, tree in repo.modules():
            secrets = secret_literals(tree)
            imports = aliases(tree)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [
                        t.id
                        if isinstance(t, ast.Name)
                        else t.attr
                        if isinstance(t, ast.Attribute)
                        else ""
                        for t in targets
                    ]
                    value = node.value
                    if (
                        any(credential_name(name) for name in names)
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value
                    ):
                        secrets.append(value)
                if isinstance(node, ast.Call) and qualified(node.func, imports) in {
                    "os.getenv",
                    "os.environ.get",
                }:
                    if (
                        len(node.args) > 1
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and credential_name(node.args[0].value)
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                        and node.args[1].value
                    ):
                        secrets.append(node.args[1])
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values, strict=True):
                        if (
                            isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                            and credential_name(key.value)
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                            and value.value
                        ):
                            secrets.append(value)
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and re.match(
                        r"(?:postgresql|postgres|mysql|mariadb|mssql|oracle)(?:\+\w+)?://",
                        node.value,
                    )
                    and "@" in node.value
                ):
                    secrets.append(node)
            unique = {(n.lineno, n.col_offset): n for n in secrets}
            self._nodes[file] = list(unique.values())
            for node in unique.values():
                self._values.add(node.value)
                if node.value != DUMMY_SECRET:
                    self.blocked_files.add(file)
        # Config values are not sent to the LLM in P2, but avoid diffing credential-like configs.
        for file in repo.files():
            text = repo.read(file)
            urls = credential_urls(text)
            if urls:
                self.blocked_files.add(file)
                self._values.update(urls)
            if file.endswith(".json"):
                try:

                    def collect(value, file=file):
                        if isinstance(value, dict):
                            for key, item in value.items():
                                if credential_name(key) and isinstance(item, str) and item:
                                    self._values.add(item)
                                    self.blocked_files.add(file)
                                collect(item)
                        elif isinstance(value, list):
                            for item in value:
                                collect(item)

                    collect(json.loads(text))
                except (ValueError, RecursionError):
                    pass
            if file.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg")):
                values = re.findall(
                    r"(?im)^\s*[\w.-]*(?:secret|password|token|key)[\w.-]*\s*[:=]\s*['\"]?([^\n]+)",
                    text,
                )
                if values or re.search(r"\w+://[^\s]+@", text):
                    self.blocked_files.add(file)
                for value in values:
                    value = value.strip("'\" ")
                    if value and value.lower() not in CONFIG_SCALARS and not value.isdigit():
                        self._values.add(value)

    def source(self, repo: RepoView, file: str) -> str:
        text = repo.read(file)
        edits = []
        for node in self._nodes.get(file, []):
            literal = ast.get_source_segment(text, node) or ""
            replacement = (
                '"""[REDACTED]' + "\n" * literal.count("\n") + '"""'
                if "\n" in literal
                else '"[REDACTED]"'
            )
            edits.append((node, replacement))
        return self.text(edit_nodes(text, edits))

    def text(self, text: str, *, allow_dummy: bool = False) -> str:
        for value in sorted(self._values, key=len, reverse=True):
            if not (allow_dummy and value == DUMMY_SECRET):
                forms = {
                    value,
                    json.dumps(value, ensure_ascii=False)[1:-1],
                    json.dumps(value, ensure_ascii=True)[1:-1],
                }
                for form in sorted(forms, key=len, reverse=True):
                    text = text.replace(form, "[REDACTED]")
        for url in credential_urls(text):
            text = text.replace(url, "[REDACTED]")
        return text

    def contains_sensitive(self, text: str) -> bool:
        return any(value != DUMMY_SECRET and value in text for value in self._values)

    def summaries(self, repo: RepoView, *, budget: int = 20000) -> dict[str, str]:
        result = {}
        for file, _ in repo.modules():
            source = self.source(repo, file)  # Mask full file before selecting/truncating it.
            if budget <= 0:
                break
            result[file] = source[:budget]
            budget -= len(result[file])
        return result
