from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


NOTION_VERSION = "2022-06-28"
MAX_TEXT_LENGTH = 1900


# --------------------------------------------------
# 기본 텍스트 처리
# --------------------------------------------------

def chunk_text(
    text: str,
    size: int = MAX_TEXT_LENGTH,
) -> list[str]:
    if not text:
        return []

    return [
        text[i : i + size]
        for i in range(0, len(text), size)
    ]


def rich_text(content: str) -> list[dict[str, Any]]:
    if not content:
        return []

    return [
        {
            "type": "text",
            "text": {
                "content": part,
            },
        }
        for part in chunk_text(content)
    ]


def text_block(
    block_type: str,
    content: str,
) -> dict[str, Any]:
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


# --------------------------------------------------
# Markdown Table -> Notion Table
# --------------------------------------------------

def parse_markdown_table_row(line: str) -> list[str]:
    """
    Markdown table 한 행을 cell 목록으로 변환한다.

    예:
    | GPU | RTX 3090 |

    ->
    ["GPU", "RTX 3090"]
    """

    line = line.strip()

    if line.startswith("|"):
        line = line[1:]

    if line.endswith("|"):
        line = line[:-1]

    return [
        cell.strip()
        for cell in line.split("|")
    ]


def is_markdown_table_separator(
    cells: list[str],
) -> bool:
    """
    Markdown 표의 구분선인지 확인한다.

    예:
    | --- | --- |
    | :--- | ---: |
    | :---: | --- |
    """

    if not cells:
        return False

    return all(
        re.fullmatch(r":?-{3,}:?", cell.strip()) is not None
        for cell in cells
    )


def markdown_table_block(
    table_lines: list[str],
) -> dict[str, Any] | None:
    """
    Markdown table을 Notion native table block으로 변환한다.
    """

    if not table_lines:
        return None

    rows: list[list[str]] = []

    for line in table_lines:
        cells = parse_markdown_table_row(line)

        # | --- | --- | 부분은 실제 데이터가 아니므로 제외
        if is_markdown_table_separator(cells):
            continue

        rows.append(cells)

    if not rows:
        return None

    # 가장 열이 많은 행을 기준으로 table width 결정
    column_count = max(
        len(row)
        for row in rows
    )

    notion_rows: list[dict[str, Any]] = []

    for row in rows:
        # 행마다 열 개수가 다르면 빈 셀로 채운다.
        padded_row = row + [""] * (
            column_count - len(row)
        )

        notion_rows.append(
            {
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        rich_text(cell)
                        for cell in padded_row
                    ],
                },
            }
        )

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": column_count,

            # 첫 번째 데이터 행을 Header로 사용
            "has_column_header": True,

            # 첫 번째 열 자체는 Header로 취급하지 않음
            "has_row_header": False,

            "children": notion_rows,
        },
    }


# --------------------------------------------------
# Markdown -> Notion blocks
# --------------------------------------------------

def markdown_to_blocks(
    markdown: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    lines = markdown.splitlines()

    in_code = False

    code_lines: list[str] = []
    table_lines: list[str] = []

    # ----------------------------------------------
    # Table flush
    # ----------------------------------------------

    def flush_table() -> None:
        nonlocal table_lines

        if not table_lines:
            return

        table = markdown_table_block(
            table_lines
        )

        if table is not None:
            blocks.append(table)

        table_lines = []

    # ----------------------------------------------
    # Code flush
    # ----------------------------------------------

    def flush_code() -> None:
        nonlocal code_lines

        if not code_lines:
            return

        code_text = "\n".join(
            code_lines
        )

        for part in chunk_text(code_text):
            blocks.append(
                code_block(part)
            )

        code_lines = []

    # ----------------------------------------------
    # Markdown line parser
    # ----------------------------------------------

    for line in lines:
        stripped = line.strip()

        # 코드 블록 시작 / 종료
        if stripped.startswith("```"):
            flush_table()

            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True

            continue

        # 코드 블록 내부
        if in_code:
            code_lines.append(line)
            continue

        # Markdown Table
        if (
            stripped.startswith("|")
            and stripped.endswith("|")
        ):
            table_lines.append(line)
            continue

        # Table이 끝났으면 Notion Table 생성
        flush_table()

        # 빈 줄
        if not stripped:
            continue

        # Heading 3
        if stripped.startswith("### "):
            blocks.append(
                text_block(
                    "heading_3",
                    stripped[4:],
                )
            )

        # Heading 2
        elif stripped.startswith("## "):
            blocks.append(
                text_block(
                    "heading_2",
                    stripped[3:],
                )
            )

        # Heading 1
        elif stripped.startswith("# "):
            blocks.append(
                text_block(
                    "heading_1",
                    stripped[2:],
                )
            )

        # Bullet list
        elif stripped.startswith("- "):
            blocks.append(
                text_block(
                    "bulleted_list_item",
                    stripped[2:],
                )
            )

        # Numbered list
        elif re.match(
            r"^\d+\.\s+",
            stripped,
        ):
            content = re.sub(
                r"^\d+\.\s+",
                "",
                stripped,
            )

            blocks.append(
                text_block(
                    "numbered_list_item",
                    content,
                )
            )

        # 일반 Paragraph
        else:
            for part in chunk_text(
                stripped
            ):
                blocks.append(
                    text_block(
                        "paragraph",
                        part,
                    )
                )

    # 문서 끝에 Table이 남아있을 수 있음
    flush_table()

    # 코드 블록이 닫히지 않았더라도 내용은 저장
    if code_lines:
        flush_code()

    return blocks


# --------------------------------------------------
# Notion API Client
# --------------------------------------------------

class NotionClient:
    def __init__(
        self,
        token: str,
    ) -> None:
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

        if response.content:
            return response.json()

        return {}

    # ----------------------------------------------
    # Page children 조회
    # ----------------------------------------------

    def list_children(
        self,
        page_id: str,
    ) -> list[dict[str, Any]]:
        results: list[
            dict[str, Any]
        ] = []

        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "page_size": 100
            }

            if cursor:
                params[
                    "start_cursor"
                ] = cursor

            data = self.request(
                "GET",
                (
                    "https://api.notion.com/"
                    f"v1/blocks/{page_id}/children"
                ),
                params=params,
            )

            results.extend(
                data.get(
                    "results",
                    [],
                )
            )

            if not data.get("has_more"):
                break

            cursor = data.get(
                "next_cursor"
            )

        return results

    # ----------------------------------------------
    # 기존 페이지 내용 삭제
    # ----------------------------------------------

    def clear_page(
        self,
        page_id: str,
    ) -> None:
        children = self.list_children(
            page_id
        )

        for block in children:
            self.request(
                "PATCH",
                (
                    "https://api.notion.com/"
                    f"v1/blocks/{block['id']}"
                ),
                json={
                    "archived": True
                },
            )

    # ----------------------------------------------
    # 새 Block 추가
    # ----------------------------------------------

    def append_blocks(
        self,
        page_id: str,
        blocks: list[dict[str, Any]],
    ) -> None:

        # Notion API는 한 요청당 최대 100개의
        # top-level block을 전송하도록 분할
        for index in range(
            0,
            len(blocks),
            100,
        ):
            batch = blocks[
                index : index + 100
            ]

            self.request(
                "PATCH",
                (
                    "https://api.notion.com/"
                    f"v1/blocks/{page_id}/children"
                ),
                json={
                    "children": batch
                },
            )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=[
            "cpu",
            "gpu",
        ],
        required=True,
    )

    parser.add_argument(
        "--report",
        required=True,
    )

    args = parser.parse_args()

    # ----------------------------------------------
    # Notion Token
    # ----------------------------------------------

    token = os.environ.get(
        "NOTION_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "NOTION_TOKEN이 없습니다."
        )

    # ----------------------------------------------
    # CPU / GPU Notion 페이지 분리
    # ----------------------------------------------

    page_env = (
        "NOTION_CPU_PAGE_ID"
        if args.target == "cpu"
        else "NOTION_GPU_PAGE_ID"
    )

    page_id = os.environ.get(
        page_env
    )

    if not page_id:
        raise RuntimeError(
            f"{page_env}가 없습니다."
        )

    # ----------------------------------------------
    # Report 읽기
    # ----------------------------------------------

    report_path = Path(
        args.report
    )

    if not report_path.exists():
        raise FileNotFoundError(
            report_path
        )

    markdown = (
        report_path.read_text(
            encoding="utf-8"
        )
    )

    # ----------------------------------------------
    # 페이지 제목
    # ----------------------------------------------

    title = (
        "CPU Benchmark Report"
        if args.target == "cpu"
        else "GPU Benchmark Report"
    )

    updated_at = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    # ----------------------------------------------
    # Markdown -> Notion blocks
    # ----------------------------------------------

    blocks = [
        text_block(
            "heading_1",
            title,
        ),
        text_block(
            "paragraph",
            f"자동 갱신: {updated_at}",
        ),
        {
            "object": "block",
            "type": "divider",
            "divider": {},
        },
        *markdown_to_blocks(
            markdown
        ),
    ]

    # ----------------------------------------------
    # Notion 업데이트
    # ----------------------------------------------

    client = NotionClient(
        token
    )

    client.clear_page(
        page_id
    )

    client.append_blocks(
        page_id,
        blocks,
    )

    print(
        f"{args.target.upper()} "
        "보고서를 Notion에 업로드했습니다."
    )


if __name__ == "__main__":
    main()