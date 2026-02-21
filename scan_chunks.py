#!/usr/bin/env python3
import json
from pathlib import Path

p = Path(r"data/chunks/isaw_paper1.jsonl")

top = []  # list of (len_chars, line_no, chunkId, articleId)
with p.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = row.get("text", "") or ""
        top.append((len(text), i, row.get("chunkId"), row.get("articleId")))

top.sort(reverse=True, key=lambda x: x[0])

print(f"File: {p}")
print("Top 10 longest chunks (chars):")
for L, line_no, chunk_id, article_id in top[:20]:
    print(f"  {L:>7} chars | line {line_no:>5} | {article_id} | {chunk_id}")
