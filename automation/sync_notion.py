from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


NOTION_VERSION = "2022-06-28"
MAX_TEXT_LENGTH = 1900


def chunk_text(text: str, size: int = MAX_TEXT_LENGTH) -> list[str]:
    if not text:
        return []

    return [text[i : i + size] for i in range(0, len(text), size)]


def rich_text(content: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": {
                "content": content,
            },
        }
    ]


def text_block(block_type: str, content: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": block_type,
        block_type: {
            "rich_text": rich_text(content),
        },
    }


def code_block(content: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": rich_text(content),
            "language": "plain text",
        },
    }


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = markdown.splitlines()

    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return

        table_text = "\n".join(table_lines)
        for part in chunk_text(table_text):
            blocks.append(code_block(part))

        table_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()

            if in_code:
                code_text = "\n".join(code_lines)
                for part in chunk_text(code_text):
                    blocks.append(code_block(part))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
            continue

        flush_table()

        if not stripped:
            continue

        if stripped.startswith("### "):
            blocks.append(text_block("heading_3", stripped[4:]))
        elif stripped.startswith("## "):
            blocks.append(text_block("heading_2", stripped[3:]))
        elif stripped.startswith("# "):
            blocks.append(text_block("heading_1", stripped[2:]))
        elif stripped.startswith("- "):
            blocks.append(text_block("bulleted_list_item", stripped[2:]))
        elif stripped[:3].rstrip(".").isdigit() and ". " in stripped:
            blocks.append(
                text_block(
                    "numbered_list_item",
                    stripped.split(". ", 1)[1],
                )
            )
        else:
            for part in chunk_text(stripped):
                blocks.append(text_block("paragraph", part))

    flush_table()

    if code_lines:
        code_text = "\n".join(code_lines)
        for part in chunk_text(code_text):
            blocks.append(code_block(part))

    return blocks


class NotionClient:
    def __init__(self, token: str) -> None:
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = requests.request(
            method,
            url,
            headers=self.headers,
            timeout=30,
            **kwargs,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def list_children(self, page_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor

            data = self.request(
                "GET",
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                params=params,
            )

            results.extend(data.get("results", []))

            if not data.get("has_more"):
                break

            cursor = data.get("next_cursor")

        return results

    def clear_page(self, page_id: str) -> None:
        for block in self.list_children(page_id):
            self.request(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{block['id']}",
                json={"archived": True},
            )

    def append_blocks(
        self,
        page_id: str,
        blocks: list[dict[str, Any]],
    ) -> None:
        for index in range(0, len(blocks), 100):
            batch = blocks[index : index + 100]

            self.request(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                json={"children": batch},
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN이 없습니다.")

    page_env = (
        "https://app.notion.com/p/CPU-Benchmark-Report-3b04a399b7bb804f842ec3ba3310488c"
        if args.target == "cpu"
        else "https://app.notion.com/p/GPU-Benchmark-Report-3b04a399b7bb802d99e3cc73d42db0ac"
    )

    page_id = os.environ.get(page_env)
    if not page_id:
        raise RuntimeError(f"{page_env}가 없습니다.")

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    markdown = report_path.read_text(encoding="utf-8")

    title = (
        "CPU Benchmark Report"
        if args.target == "cpu"
        else "GPU Benchmark Report"
    )

    updated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    blocks = [
        text_block("heading_1", title),
        text_block("paragraph", f"자동 갱신: {updated_at}"),
        {
            "object": "block",
            "type": "divider",
            "divider": {},
        },
        *markdown_to_blocks(markdown),
    ]

    client = NotionClient(token)
    client.clear_page(page_id)
    client.append_blocks(page_id, blocks)

    print(f"{args.target.upper()} 보고서를 Notion에 업로드했습니다.")


if __name__ == "__main__":
    main()