#!/usr/bin/env python3
"""从 06-面试题库/ 生成背题库数据。

构建时把 Markdown 答案解析成 HTML，浏览器只需要读取 JSON、随机抽题和展开答案。
这样 quiz 页面不必在客户端再引入 Markdown 解析器，也能保留代码块、表格和 LaTeX。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import markdown


QUESTION_DIR = "06-面试题库"

CATEGORY_META = [
    # id, 显示名, 文件名, 来源标题
    ("algorithm", "算法岗", "01-算法岗高频问答.md", "算法岗高频问答"),
    ("application", "应用岗", "02-应用岗高频问答.md", "应用岗高频问答"),
    ("system-design", "系统设计", "03-系统设计题.md", "系统设计题"),
    ("real-interview", "真实面经", "04-真实面经高频题.md", "真实面经高频题"),
    ("project", "项目深挖", "05-项目深挖与追问.md", "项目深挖与追问"),
    ("coding", "手撕代码", "06-手撕代码题.md", "手撕代码题"),
]

MD_EXTENSIONS = [
    "tables",
    "attr_list",
    "md_in_html",
    "pymdownx.superfences",
    "pymdownx.highlight",
    "pymdownx.inlinehilite",
    "pymdownx.arithmatex",
]
MD_CONFIGS = {
    "pymdownx.highlight": {
        "anchor_linenums": False,
        "pygments_lang_class": True,
    },
    "pymdownx.arithmatex": {"generic": True},
}


def _clean_topic(value: str) -> str:
    """把“一、基础与架构（15 题）”整理成适合展示的短标签。"""
    value = re.sub(r"^\s*(?:[一二三四五六七八九十]+、|\d+(?:\.\d+)*[.、]\s*)", "", value)
    value = re.sub(r"\s*（\s*\d+\s*题\s*）\s*$", "", value)
    return value.strip()


def _render_markdown() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=MD_EXTENSIONS,
        extension_configs=MD_CONFIGS,
        output_format="html5",
    )


def _external_links(html: str) -> str:
    """答案里的外部链接新开标签页，避免打断背题。"""
    return re.sub(
        r'<a href="(https?://[^"]+)"',
        r'<a href="\1" target="_blank" rel="noopener noreferrer"',
        html,
    )


def _html(md: markdown.Markdown, text: str) -> str:
    """渲染一段 Markdown，并保证每次转换的状态是干净的。"""
    md.reset()
    return _external_links(md.convert(text.strip()))


def _make_item(
    md: markdown.Markdown,
    *,
    category: str,
    category_name: str,
    topic: str,
    question: str,
    answer: str,
    source: str,
    source_title: str,
    item_id: str,
) -> dict | None:
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return None
    return {
        "id": item_id,
        "category": category,
        "categoryName": category_name,
        "topic": topic or category_name,
        "question": question,
        "questionHtml": _html(md, question),
        "answerHtml": _html(md, answer),
        "source": source,
        "sourceTitle": source_title,
    }


def parse_numbered_qa(
    path: Path,
    category: str,
    category_name: str,
    source_title: str,
    md: markdown.Markdown,
) -> list[dict]:
    """解析 06-01/02：## 分类 + ### N. 问题 + 正文答案。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    topic = category_name
    current: dict | None = None
    answer_lines: list[str] = []
    items: list[dict] = []
    counter = 0

    def flush() -> None:
        nonlocal current, answer_lines, counter
        if current is not None:
            counter += 1
            item = _make_item(
                md,
                category=category,
                category_name=category_name,
                topic=current["topic"],
                question=current["question"],
                answer="\n".join(answer_lines),
                source=str(path.relative_to(path.parents[1])).replace("\\", "/"),
                source_title=source_title,
                item_id=f"{category}-{counter:03d}",
            )
            if item:
                items.append(item)
        current = None
        answer_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            topic = _clean_topic(line[3:])
            continue
        match = re.match(r"^###\s+(\d+)\.\s+(.+?)\s*$", line)
        if match:
            flush()
            current = {"question": match.group(2).strip(), "topic": topic}
            continue
        if current is not None:
            answer_lines.append(line)

    flush()
    return items


def parse_real_interview(
    path: Path,
    category: str,
    category_name: str,
    source_title: str,
    md: markdown.Markdown,
) -> list[dict]:
    """解析 06-04：## 分类下的三列表格，每行是一道题。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    topic = category_name
    items: list[dict] = []
    counter = 0

    for line in lines:
        if line.startswith("## "):
            topic = _clean_topic(line[3:])
            continue
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") != 4:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3 or not re.fullmatch(r"\d+", cells[0]):
            continue
        counter += 1
        item = _make_item(
            md,
            category=category,
            category_name=category_name,
            topic=topic,
            question=cells[1],
            answer=cells[2],
            source=str(path.relative_to(path.parents[1])).replace("\\", "/"),
            source_title=source_title,
            item_id=f"{category}-{counter:03d}",
        )
        if item:
            items.append(item)

    return items


def parse_system_design(
    path: Path,
    category: str,
    category_name: str,
    source_title: str,
    md: markdown.Markdown,
) -> list[dict]:
    """解析 06-03：## 题目 N：标题 + 整段设计答案。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict] = []
    current_title: str | None = None
    answer_lines: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current_title, answer_lines, counter
        if current_title is not None:
            counter += 1
            item = _make_item(
                md,
                category=category,
                category_name=category_name,
                topic="系统设计",
                question=f"系统设计：{current_title}",
                answer="\n".join(answer_lines),
                source=str(path.relative_to(path.parents[1])).replace("\\", "/"),
                source_title=source_title,
                item_id=f"{category}-{counter:03d}",
            )
            if item:
                items.append(item)
        current_title = None
        answer_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            match = re.match(r"^##\s*题目\s*\d+\s*[：:]\s*(.+?)\s*$", line)
            if match:
                current_title = match.group(1).strip()
            continue
        if current_title is not None:
            answer_lines.append(line)

    flush()
    return items


def parse_project_qa(
    path: Path,
    category: str,
    category_name: str,
    source_title: str,
    md: markdown.Markdown,
) -> list[dict]:
    """解析 06-05：只取带参考答案的 **Q：** / ### Qn： 小节。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    topic = category_name
    current_question: str | None = None
    answer_lines: list[str] = []
    items: list[dict] = []
    counter = 0

    def flush() -> None:
        nonlocal current_question, answer_lines, counter
        if current_question is not None:
            counter += 1
            item = _make_item(
                md,
                category=category,
                category_name=category_name,
                topic=topic,
                question=current_question,
                answer="\n".join(answer_lines),
                source=str(path.relative_to(path.parents[1])).replace("\\", "/"),
                source_title=source_title,
                item_id=f"{category}-{counter:03d}",
            )
            if item:
                items.append(item)
        current_question = None
        answer_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            topic = _clean_topic(line[3:])
            continue

        q_match = re.match(r"^\*\*Q[：:]\s*(.+?)\*\*\s*$", line)
        if q_match:
            flush()
            current_question = q_match.group(1).strip()
            continue

        q_match = re.match(r"^###\s+Q\d+\s*[：:]\s*(.+?)\s*$", line, re.IGNORECASE)
        if q_match:
            flush()
            current_question = q_match.group(1).strip()
            continue

        if line.startswith("### "):
            # 3.2 重点题参考要点之类的分节标题，遇到时结束上一题。
            flush()
            continue

        if current_question is not None:
            answer_lines.append(line)

    flush()
    return items


def parse_coding(
    path: Path,
    category: str,
    category_name: str,
    source_title: str,
    md: markdown.Markdown,
) -> list[dict]:
    """解析 06-06：## N. 标题 + 要求/关键点/代码。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict] = []
    current_title: str | None = None
    answer_lines: list[str] = []
    counter = 0

    def flush() -> None:
        nonlocal current_title, answer_lines, counter
        if current_title is not None:
            counter += 1
            body = "\n".join(answer_lines)
            requirement = ""
            for line in answer_lines:
                match = re.match(r"^\*\*要求\*\*\s*[：:]\s*(.+?)\s*$", line)
                if match:
                    requirement = match.group(1).strip()
                    break
            if current_title == "通用算法题清单":
                question = "通用算法题常考哪些题？分别的关键点和复杂度是什么？"
            elif requirement:
                question = f"手写：{current_title}。{requirement}"
            else:
                question = f"手写：{current_title}"
            item = _make_item(
                md,
                category=category,
                category_name=category_name,
                topic=current_title,
                question=question,
                answer=body,
                source=str(path.relative_to(path.parents[1])).replace("\\", "/"),
                source_title=source_title,
                item_id=f"{category}-{counter:03d}",
            )
            if item:
                items.append(item)
        current_title = None
        answer_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            match = re.match(r"^##\s+(\d+)\.\s+(.+?)\s*$", line)
            if match:
                number = int(match.group(1))
                title = match.group(2).strip()
                # 第 13 节是答题模板，不是可抽的题目；第 12 节是通用题清单。
                if number <= 12:
                    current_title = title
            continue
        if current_title is not None:
            answer_lines.append(line)

    flush()
    return items


def build_quiz(docs_dir: Path, content_dir: Path | None = None) -> dict:
    """生成 docs/assets/quiz-data.json，返回统计信息。"""
    content_dir = content_dir or docs_dir.parent.parent
    question_dir = content_dir / QUESTION_DIR
    if not question_dir.exists():
        raise FileNotFoundError(f"缺少题库目录：{question_dir}")

    md = _render_markdown()
    items: list[dict] = []

    for category, category_name, filename, source_title in CATEGORY_META:
        path = question_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少题库文件：{path}")

        if category in {"algorithm", "application"}:
            parsed = parse_numbered_qa(path, category, category_name, source_title, md)
        elif category == "real-interview":
            parsed = parse_real_interview(path, category, category_name, source_title, md)
        elif category == "system-design":
            parsed = parse_system_design(path, category, category_name, source_title, md)
        elif category == "project":
            parsed = parse_project_qa(path, category, category_name, source_title, md)
        elif category == "coding":
            parsed = parse_coding(path, category, category_name, source_title, md)
        else:  # pragma: no cover - CATEGORY_META 已经穷举
            parsed = []

        if not parsed:
            raise ValueError(f"题库解析结果为空：{path}")
        items.extend(parsed)
        print(f"[quiz] {category_name}: {len(parsed)} 道")

    ids = {item["id"] for item in items}
    if len(ids) != len(items):
        raise ValueError("题库 id 出现重复，请检查生成逻辑")

    categories = [
        {"id": "all", "name": "全部题库", "count": len(items)},
    ]
    for category, category_name, _, _ in CATEGORY_META:
        count = sum(1 for item in items if item["category"] == category)
        categories.append({"id": category, "name": category_name, "count": count})

    payload = {
        "generated": date.today().isoformat(),
        "total": len(items),
        "categories": categories,
        "questions": items,
    }

    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output = assets_dir / "quiz-data.json"
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    if not output.exists() or output.read_text(encoding="utf-8") != text:
        output.write_text(text, encoding="utf-8")
    print(f"[quiz] 共 {len(items)} 道题 -> {output}")
    return {"total": len(items), "path": output, "categories": categories}
