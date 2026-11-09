#!/usr/bin/env python3
"""promote_to_brain.py — nah-remember → nah-brain consolidation sink (MEM-01).

Closes the measured gap between nah-remember (PostgreSQL transactional memory)
and nah-knows / nah-brain (institutional Second Brain corpus).

Pipeline:
    nah-remember (Postgres nah_remember.semantic_memories & nah_remember.procedures)
        -> THIS SCRIPT (markdown export, idempotent, per-item file)
        -> nah-brain/raw/nah-remember/mem-<id8>-<slug>.md (Semantic)
        -> nah-brain/raw/nah-remember/proc-<id8>-<slug>.md (Layer 4 Procedural)
        -> reindex-keepfresh.sh (launchd WatchPaths on nah-brain/, 6h floor)
        -> nah_knows.chunks (bge-m3 1024-dim; hybrid semantic+BM25 retrieval)

Usage:
    python3 promote_to_brain.py [--dry-run] [--check] [--all]
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import asyncpg

DB = os.getenv("NAH_DB", "postgresql://human:@127.0.0.1/notahuman")
VAULT_DIR = Path(os.getenv("NAH_VAULT_DIR", "/Users/human/notahuman/nah-brain/raw/nah-remember"))
IMPORTANCE_LEGACY_MIN = 5.0  # 1-10 scale
IMPORTANCE_UNIT_MIN = 0.90  # 0-1 scale
SLUG_MAX = 48


def promotable(score) -> bool:
    """The 'learned' rule: threshold on either importance scale."""
    if score is None:
        return False
    s = float(score)
    return s >= IMPORTANCE_LEGACY_MIN or (s <= 1.0 and s >= IMPORTANCE_UNIT_MIN)


def slugify(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words)[:SLUG_MAX].strip("-")
    return slug or "item"


def body_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def render_memory(md) -> dict:
    mid = str(md["id"])
    short = mid[:8]
    content = md["content"]
    h = body_hash(content)
    fname = f"mem-{short}-{slugify(content)}.md"
    tags = md["tags"]
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = [tags]
    tag_lines = "\n".join(f"  - {t}" for t in (tags or []) if t) or "  - memory"
    created = md["created_at"].isoformat() if md["created_at"] else ""
    updated = md["updated_at"].isoformat() if md["updated_at"] else ""
    fm = [
        "---",
        f'title: "nah-remember: {slugify(content)}"',
        "type: memory-export",
        "source: nah-remember.semantic_memories",
        f"memory_id: {mid}",
        f"memory_type: {md['memory_type'] or 'semantic'}",
        f"importance: {md['importance_score']}",
        f"content_date: {created}",
        f"export_hash: {h}",
        "tags:",
        "  - memory",
        "  - nah-remember",
        tag_lines,
        "---",
        "",
        "> Promoted from nah-remember episodic store by promote_to_brain.py (MEM-01).",
        f"> Created {created} · updated {updated} · importance {md['importance_score']}",
        "",
        content,
        "",
    ]
    return {"path": VAULT_DIR / fname, "text": "\n".join(fm), "hash": h, "id": mid}


def render_procedure(proc) -> dict:
    pid = str(proc["id"])
    short = pid[:8]
    name = proc["name"]
    desc = proc.get("description") or ""
    steps = proc.get("steps") or []
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except Exception:
            steps = [steps]
    h = body_hash(f"{name}:{desc}:{json.dumps(steps)}")
    fname = f"proc-{short}-{slugify(name)}.md"

    triggers = proc.get("trigger_conditions") or []
    if isinstance(triggers, str):
        try:
            triggers = json.loads(triggers)
        except Exception:
            triggers = [triggers]
    trigger_lines = "\n".join(f"  - {t}" for t in (triggers or []) if t) or "  - procedure"

    steps_md = "\n".join(
        f"{i + 1}. {s.get('action') if isinstance(s, dict) else s}" for i, s in enumerate(steps)
    )
    created = proc["created_at"].isoformat() if proc["created_at"] else ""
    updated = proc["updated_at"].isoformat() if proc["updated_at"] else ""
    fm = [
        "---",
        f'title: "Procedure: {name}"',
        "type: procedure-export",
        "source: nah-remember.procedures",
        f"procedure_id: {pid}",
        f"procedure_type: {proc.get('procedure_type', 'workflow')}",
        f"success_rate: {proc.get('success_rate', 1.0)}",
        f"execution_count: {proc.get('execution_count', 0)}",
        f"content_date: {created}",
        f"export_hash: {h}",
        "tags:",
        "  - procedure",
        "  - playbook",
        "  - nah-remember",
        trigger_lines,
        "---",
        "",
        "> Crystallized Layer 4 Procedure from nah-remember.",
        f"> Created {created} · updated {updated} · success_rate {proc.get('success_rate', 1.0)} · executions {proc.get('execution_count', 0)}",
        "",
        f"## {name}",
        "",
        desc,
        "",
        "### Executable Steps",
        steps_md,
        "",
    ]
    return {"path": VAULT_DIR / fname, "text": "\n".join(fm), "hash": h, "id": pid}


def existing_hash(path: Path):
    if not path.exists():
        return None
    m = re.search(r"^export_hash: ([0-9a-f]{16})$", path.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if promotions are stale (health probe)",
    )
    ap.add_argument("--all", action="store_true", help="ignore threshold; export every live memory")
    args = ap.parse_args()

    try:
        conn = await asyncpg.connect(DB, timeout=15)
    except Exception as e:
        print(f"FAIL db connect {DB}: {e}", file=sys.stderr)
        return 2

    try:
        mem_rows = await conn.fetch(
            "SELECT id, content, memory_type, importance_score, created_at, updated_at, "
            "tags, archived FROM nah_remember.semantic_memories"
        )
        proc_rows = await conn.fetch(
            "SELECT id, name, procedure_type, description, steps, trigger_conditions, "
            "execution_count, success_rate, created_at, updated_at, status "
            "FROM nah_remember.procedures WHERE status = 'active'"
        )
    finally:
        await conn.close()

    want = []
    # 1. Semantic memories
    for r in mem_rows:
        live = not r["archived"]
        ok = args.all or promotable(r["importance_score"])
        if live and ok:
            want.append(render_memory(dict(r)))

    # 2. Procedural memories
    for p in proc_rows:
        want.append(render_procedure(dict(p)))

    want_by_name = {w["path"].name: w for w in want}
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    on_disk_mem = {p.name for p in VAULT_DIR.glob("mem-*.md")}
    on_disk_proc = {p.name for p in VAULT_DIR.glob("proc-*.md")}
    on_disk = on_disk_mem | on_disk_proc

    added = updated = skipped = removed = 0
    for w in want:
        cur = existing_hash(w["path"])
        if cur == w["hash"] and w["path"].name in on_disk:
            skipped += 1
            continue
        action = "update" if w["path"].name in on_disk else "add"
        if args.dry_run:
            print(f"[dry-run] {action}: {w['path'].name}")
        else:
            w["path"].write_text(w["text"], encoding="utf-8")
        if action == "add":
            added += 1
        else:
            updated += 1

    for stale_name in sorted(on_disk - set(want_by_name)):
        if args.dry_run:
            print(f"[dry-run] remove (demoted/archived/deleted): {stale_name}")
        else:
            (VAULT_DIR / stale_name).unlink()
        removed += 1

    print(
        f"promote_to_brain: promotable={len(want)} (mems={len(mem_rows)}, procs={len(proc_rows)}) "
        f"added={added} updated={updated} skipped={skipped} removed={removed} vault={VAULT_DIR}"
    )

    if args.check:
        missing = [w for w in want if existing_hash(w["path"]) != w["hash"]]
        if missing:
            print(
                f"CHECK FAILED: {len(missing)} promotable items not current on disk "
                f"(pipeline stale; run promote_to_brain.py)",
                file=sys.stderr,
            )
            return 1
        print("CHECK OK: every promotable item is present and current in the vault")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
