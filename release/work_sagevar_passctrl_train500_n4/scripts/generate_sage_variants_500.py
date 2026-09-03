#!/usr/bin/env python3
"""Generate 500 SAGE-style training profiles from 100 seed profiles.

This script does not modify training code. It reads the fixed SAGE benchmark
profiles, asks an OpenAI-compatible API to create surface-level variants, and
writes a JSONL training pool.

The API key is intentionally read from the environment instead of being stored
in this file:

    export IDEALAB_API_KEY="YOUR_KEY"
    python3 scripts/generate_sage_variants_500.py \
      --input SAGE-en/profile/simulator_profile.jsonl \
      --output RLVER/data/train_profile_sage_variants_500.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("id", "player", "scene", "main_cha", "cha_group", "task")


SYSTEM_PROMPT = """你是SAGE心理咨询来访者Profile扩充助手。
你的任务是基于原始profile生成同源变体，用于RLVER/SAGE情绪支持强化学习训练。

必须遵守：
1. 必须保留原始profile的核心矛盾、hidden theme、情绪大类、整体难度区间。
2. 只能改年龄、职业、姓名、生活细节、说话风格、事件细碎细节和表达方式。
3. 禁止新增完全不同的主冲突，禁止改变hidden theme，禁止改变来访者真正诉求。
4. 禁止把profile简化成cooperation/emotion/trust等标量设定。
5. 每个变体仍然必须是完整SAGE人物剧本，包含完整人物画像、背景事件、困境、不同emotion下的反应、NPC贴合/偏离hidden theme时的反应。
6. 不要直接复制原文，要生成同源但文本和具体情节细节不同的变体。
7. 输出语言必须和原始profile保持一致：原始profile是英文就输出英文，原始profile是中文就输出中文。
8. 禁止生成极端化剧情：来访者诉求不能极易满足，也不能完全无法达成；整体难度应保持在原始profile附近的中等水平。
9. 同一批变体之间不得高度雷同，人物职业、生活背景、具体事件细节、表达风格都要有明显差异。
10. scene里必须显式保留原始SAGE结构，尤其是NPC回复贴合/偏离hidden theme时emotion上升/下降的两种反应。
11. 输出必须是纯净合法JSON数组，不要输出Markdown代码块、注释、解释、额外文本或乱码。
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise ValueError(f"{path}:{line_no} missing required keys: {missing}")
            rows.append(row)
    return rows


def load_existing_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row.get("source_id")
            variant_index = row.get("variant_index")
            if source_id is not None and variant_index is not None:
                keys.add((str(source_id), int(variant_index)))
    return keys


def make_user_prompt(seed: dict[str, Any], variants_per_profile: int) -> str:
    seed_json = json.dumps(seed, ensure_ascii=False, indent=2)
    return f"""请基于下面这一条SAGE原始profile，生成{variants_per_profile}个RLVER式同源表层变体。

关键要求:
- 原始profile只作为测试基准种子，不要原样放入输出。
- 原始seed仅用于线下评测，生成的变体专门用于RL训练，训练变体和测试seed不能混用。
- 每个变体必须保留原始hidden theme/task：{seed["task"]}
- 每个变体必须保留原始main_cha和cha_group的心理类型含义。
- 每个变体所属情感支持大类必须和原始seed完全一致，不能从学业漂移到亲密关系、人际关系漂移到职场压力等跨主题变化。
- 每个变体必须改写人物身份、姓名、年龄、职业、爱好、表达风格、事件细节。
- 所有人物经历、事件细节和表达方式不能与原始seed出现大段重合，整体改写程度至少达到60%；不要只替换姓名或职业。
- 每个变体必须保持完整SAGE剧本结构，不能只写摘要；英文输出时player建议不少于650个英文字符，scene建议不少于1800个英文字符。
- 每个变体的scene里必须包含：
  1. 事件起因
  2. 事件经过/时间线/子事件（英文输出时请显式写成 Timeline / process / how the situation unfolded:）
  3. 主要冲突和冲突内在原因
  4. 尝试过但失败的解决方案
  5. 当前面对的问题
  6. emotion高/中/低时的反应
  7. NPC回复贴合hidden theme时emotion如何上升
  8. NPC回复偏离hidden theme时emotion如何下降
- 上述scene八大模块、高/中/低emotion反应、NPC贴合/偏离hidden theme的情绪升降逻辑缺一不可；缺任意一块都属于无效样本，必须重写完整内容。
- 为了保持SAGE结构稳定，如果原始profile是英文，请在scene中显式保留下面五个小标题或非常接近的等价表达：
  - When the character's emotion is high (calm, relaxed):
  - When the character's emotion is average (impatient, disappointed):
  - When the character's emotion is low (agitated, irritable, despairing):
  - If the NPC's reply matches the hidden theme (emotion rises):
  - If the NPC's reply strays from the hidden theme (emotion falls):
  如果原始profile是中文，请使用对应中文含义的小标题。
- 同一批变体之间也要保持多样性，不能5条都是相同职业、相同关系结构、相同事件细节。
- 如果原始profile有first_talk，请也为每个变体生成自然的first_talk；如果没有，可以省略。
- 若无法稳定输出{variants_per_profile}条同源变体，且本次请求数量不少于4条，则至少产出4条有效完整剧本；如果本次请求少于4条，则产出请求的全部数量；不要为了凑数填充无效、残缺或重复内容。

输出格式：
只输出一个纯净JSON数组，优先长度为{variants_per_profile}。
数组中每个对象必须包含这些字段：
{{
  "player": "...完整人物画像...",
  "scene": "...完整背景剧本...",
  "first_talk": "...可选，若生成则为自然开场白..."
}}

不要输出id、source_id、variant_index；这些字段会由程序统一生成。
不要输出Markdown代码块、注释、解释、额外文本、换行符乱码或任何JSON数组之外的内容。

原始profile如下：
{seed_json}
"""


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_variants(text: str) -> list[dict[str, Any]]:
    text = strip_json_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if isinstance(data, dict) and "variants" in data:
        data = data["variants"]
    if not isinstance(data, list):
        raise ValueError("API response must be a JSON array")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every variant must be a JSON object")
    return data


def normalize_variant(
    raw: dict[str, Any],
    seed: dict[str, Any],
    seed_index: int,
    variant_index: int,
) -> dict[str, Any]:
    player = str(raw.get("player", "")).strip()
    scene = str(raw.get("scene", "")).strip()
    if not player or not scene:
        raise ValueError("Variant missing non-empty player or scene")

    row: dict[str, Any] = {
        "id": f"sagevar-{seed_index:03d}-{variant_index:02d}-{uuid.uuid4().hex[:8]}",
        "source_id": seed["id"],
        "variant_index": variant_index,
        "player": player,
        "scene": scene,
        # Keep these aligned with the source seed to avoid accidental drift.
        "main_cha": seed["main_cha"],
        "cha_group": seed["cha_group"],
        "task": seed["task"],
    }

    first_talk = str(raw.get("first_talk", "")).strip()
    if first_talk:
        row["first_talk"] = first_talk

    if "topic" in seed:
        row["topic"] = seed["topic"]

    return row


def validate_variant(row: dict[str, Any], seed: dict[str, Any]) -> None:
    player = row["player"]
    scene = row["scene"]
    if len(player) < 500:
        raise ValueError("player is too short; likely not a complete persona")
    if len(scene) < 1500:
        raise ValueError("scene is too short; likely not a complete SAGE script")

    scene_lower = scene.lower()
    required_any_groups = [
        ("trigger/cause", ["事件起因", "起因", "triggered", "trigger", "cause"]),
        (
            "timeline/process",
            [
                "事件经过",
                "时间线",
                "timeline",
                "unfolded",
                "phase",
                "what happened",
                "how it unfolded",
                "how the situation unfolded",
                "process",
                "sequence of events",
                "series of events",
                "over time",
                "developed over time",
                "situation developed",
                "development of the situation",
                "sub-event",
            ],
        ),
        ("conflict", ["主要冲突", "冲突", "conflict"]),
        (
            "failed solution",
            ["失败", "尝试过", "unsuccessful", "failed", "tried", "attempted", "did not work", "didn't work"],
        ),
        ("current problem", ["当前", "current", "facing"]),
        (
            "emotion high",
            [
                "emotion高",
                "emotion is high",
                "emotion high",
                "emotion state is high",
                "emotional state is high",
                "character's emotion is high",
                "character emotion is high",
                "when the character's emotion is high",
                "when the character emotion is high",
                "high (calm",
                "calm, relaxed",
            ],
        ),
        (
            "emotion low",
            [
                "emotion低",
                "emotion is low",
                "emotion low",
                "emotion state is low",
                "emotional state is low",
                "character's emotion is low",
                "character emotion is low",
                "when the character's emotion is low",
                "when the character emotion is low",
                "low (agitated",
                "agitated, irritable",
                "despairing",
            ],
        ),
    ]
    missing = [
        label
        for label, options in required_any_groups
        if not any(option.lower() in scene_lower for option in options)
    ]
    if missing:
        raise ValueError(f"scene missing SAGE structural parts: {missing}")

    hidden_theme_groups = [
        ("hidden-theme match", ["贴合", "符合", "matches the hidden", "matches hidden", "match the hidden", "aligns with", "fits the hidden"]),
        ("hidden-theme deviate", ["偏离", "不贴合", "strays from the hidden", "strays from hidden", "deviates from", "deviate", "misaligns"]),
        ("emotion rises", ["emotion上升", "情绪上升", "emotion rises", "emotion rise", "emotion increases", "emotion improves", "emotion goes up"]),
        ("emotion falls", ["emotion下降", "情绪下降", "emotion falls", "emotion fall", "emotion decreases", "emotion drops", "emotion worsens", "emotion goes down"]),
    ]
    hidden_theme_missing = [
        label
        for label, options in hidden_theme_groups
        if not any(option.lower() in scene_lower for option in options)
    ]
    if hidden_theme_missing:
        raise ValueError(
            f"scene missing SAGE structural parts: ['hidden-theme fit/deviate']; details={hidden_theme_missing}"
        )

    # Cheap anti-copy guard: the first 240 chars should not be identical.
    if player[:240] == str(seed.get("player", ""))[:240]:
        raise ValueError("player starts with identical source text")
    if scene[:240] == str(seed.get("scene", ""))[:240]:
        raise ValueError("scene starts with identical source text")


def call_with_retries(
    client: Any,
    *,
    model: str,
    seed: dict[str, Any],
    variants_per_profile: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    prompt = make_user_prompt(seed, variants_per_profile)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            content = chat_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            variants = parse_variants(content)
            min_expected = min(4, variants_per_profile)
            if len(variants) < min_expected or len(variants) > variants_per_profile:
                raise ValueError(
                    f"Expected {min_expected}-{variants_per_profile} variants, got {len(variants)}"
                )
            for raw in variants:
                probe = {
                    "player": str(raw.get("player", "")).strip(),
                    "scene": str(raw.get("scene", "")).strip(),
                }
                validate_variant(probe, seed)
            return variants
        except Exception as exc:  # noqa: BLE001 - keep generation resumable.
            last_error = exc
            if attempt < retries:
                sleep_for = retry_sleep * attempt
                print(
                    f"[warn] API/parse failed for seed {seed['id']} "
                    f"(attempt {attempt}/{retries}): {exc}; sleep {sleep_for:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_for)
    raise RuntimeError(f"Failed after {retries} attempts: {last_error}") from last_error


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def generate_for_seed(
    *,
    client: Any,
    args: argparse.Namespace,
    seed_index: int,
    total_seeds: int,
    seed: dict[str, Any],
    needed_indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = max(1, min(args.batch_variants, args.variants_per_profile))
    for start in range(0, len(needed_indices), batch_size):
        batch_indices = needed_indices[start : start + batch_size]
        print(
            f"[call] seed {seed_index:03d}/{total_seeds:03d} "
            f"{seed['id']} need={batch_indices}"
        )
        raw_variants = call_with_retries(
            client,
            model=args.model,
            seed=seed,
            variants_per_profile=len(batch_indices),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        for variant_index, raw in zip(batch_indices, raw_variants):
            rows.append(normalize_variant(raw, seed, seed_index, variant_index))

    if not rows:
        raise RuntimeError(
            f"API returned no new usable variants for seed {seed['id']}; "
            "rerun will retry missing indices."
        )
    return rows


def normalize_api_key() -> str:
    api_key = os.environ.get("IDEALAB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set IDEALAB_API_KEY to your Idealab key, e.g. sk-..."
        )
    # Idealab expects the AK without the leading sk- prefix.
    if api_key.startswith("sk-"):
        api_key = api_key[3:]
    return api_key


def build_client(base_url: str) -> Any:
    api_key = normalize_api_key()
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)
    except ModuleNotFoundError:
        return {"api_key": api_key, "base_url": base_url.rstrip("/")}


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    if isinstance(client, dict):
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{client['base_url']}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {client['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        return payload["choices"][0]["message"].get("content") or ""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 500 SAGE-style profile variants from 100 seed profiles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("SAGE-en/profile/simulator_profile.jsonl"),
        help="Seed SAGE profile JSONL. Expected to contain 100 profiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RLVER/data/train_profile_sage_variants_500.jsonl"),
        help="Output JSONL path for generated training profiles.",
    )
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument(
        "--base-url",
        default="https://idealab.alibaba-inc.com/api/openai/v1",
    )
    parser.add_argument("--variants-per-profile", type=int, default=5)
    parser.add_argument(
        "--batch-variants",
        type=int,
        default=5,
        help="Generate at most this many variants per API call while still targeting variants-per-profile total variants per seed.",
    )
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between seeds.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API calls. Use 1 for serial generation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N seeds.")
    parser.add_argument("--variant-start-index", type=int, default=1, help="First variant_index to generate for each seed, inclusive.")
    parser.add_argument("--variant-end-index", type=int, default=None, help="Last variant_index to generate for each seed, inclusive. Defaults to variants-per-profile.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle seed order.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and print the first prompt without calling API.",
    )
    args = parser.parse_args()

    seeds = load_jsonl(args.input)
    if args.limit is not None:
        seeds = seeds[: args.limit]
    indexed_seeds = list(enumerate(seeds, 1))
    if args.shuffle:
        random.shuffle(indexed_seeds)

    variant_end_index = args.variant_end_index or args.variants_per_profile
    if args.variant_start_index < 1 or variant_end_index > args.variants_per_profile or args.variant_start_index > variant_end_index:
        raise ValueError("Invalid variant index range")
    selected_variant_indices = list(range(args.variant_start_index, variant_end_index + 1))
    target_count = len(seeds) * len(selected_variant_indices)
    print(f"[info] seeds={len(seeds)} variants_per_profile={args.variants_per_profile}")
    print(f"[info] selected variant indices={selected_variant_indices[0]}-{selected_variant_indices[-1]}")
    print(f"[info] target output rows for this run={target_count}")
    print(f"[info] output={args.output}")

    if args.dry_run:
        print("\n===== SYSTEM PROMPT =====\n")
        print(SYSTEM_PROMPT)
        print("\n===== USER PROMPT EXAMPLE =====\n")
        print(make_user_prompt(seeds[0], args.variants_per_profile))
        return 0

    client = build_client(args.base_url)
    existing = load_existing_keys(args.output)
    if existing:
        print(f"[info] resume enabled: found {len(existing)} existing generated rows")

    total_written = 0
    write_lock = Lock()

    seed_jobs: list[tuple[int, dict[str, Any], list[int]]] = []
    for seed_index, seed in indexed_seeds:
        needed_indices = [
            i
            for i in selected_variant_indices
            if (str(seed["id"]), i) not in existing
        ]
        if not needed_indices:
            print(f"[skip] seed {seed_index:03d} {seed['id']} already complete")
            continue
        seed_jobs.append((seed_index, seed, needed_indices))

    if args.workers <= 1:
        for seed_index, seed, needed_indices in seed_jobs:
            rows = generate_for_seed(
                client=client,
                args=args,
                seed_index=seed_index,
                total_seeds=len(seeds),
                seed=seed,
                needed_indices=needed_indices,
            )
            append_jsonl(args.output, rows)
            for row in rows:
                existing.add((str(row["source_id"]), int(row["variant_index"])))
            total_written += len(rows)
            print(f"[ok] wrote {len(rows)} rows for seed {seed_index:03d}")

            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        print(f"[info] workers={args.workers} pending_seed_jobs={len(seed_jobs)}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    generate_for_seed,
                    client=client,
                    args=args,
                    seed_index=seed_index,
                    total_seeds=len(seeds),
                    seed=seed,
                    needed_indices=needed_indices,
                ): seed_index
                for seed_index, seed, needed_indices in seed_jobs
            }
            for future in as_completed(futures):
                seed_index = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    print(f"[error] seed {seed_index:03d} failed and will be retried by a later resume run: {exc}")
                    continue
                with write_lock:
                    append_jsonl(args.output, rows)
                    for row in rows:
                        existing.add((str(row["source_id"]), int(row["variant_index"])))
                    total_written += len(rows)
                print(f"[ok] wrote {len(rows)} rows for seed {seed_index:03d}")

    print(f"[done] newly_written={total_written} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
