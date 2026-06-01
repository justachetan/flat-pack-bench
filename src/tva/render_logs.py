#!/usr/bin/env python3
"""
render_logs.py

Usage:
    python render_logs.py path/to/logs.jsonl output.html
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import shutil
import uuid
from pathlib import Path
from textwrap import dedent


def load_events(log_path: Path, ctx: "RenderContext"):
    events = []
    with log_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            record = entry.get("record") or {}
            extra = record.get("extra") or {}
            stage = extra.get("stage") or "unknown"
            action = extra.get("event") or record.get("function") or ""
            timestamp = (record.get("time") or {}).get("repr") or entry.get("text", "").split(" | ")[0]
            message = record.get("message") or entry.get("text") or ""
            file_path = extra.get("file_path")
            mime = infer_mime(file_path, extra.get("mime"))
            rel_asset, _ = ctx.prepare_asset(file_path)
            meta = extra.get("meta") or {}
            events.append(
                {
                    "stage": stage,
                    "action": action,
                    "timestamp": timestamp,
                    "message": message,
                    "file_path": file_path,
                    "asset_href": rel_asset,
                    "mime": mime,
                    "meta": meta,
                    "raw": record,
                }
            )
    return events


TEXT_FALLBACK_SUFFIXES = {".py", ".json", ".md", ".yaml", ".yml", ".txt", ".log"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


class RenderContext:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.asset_cache: dict[Path, Path] = {}
        self.rel_index: dict[str, Path] = {}

    def prepare_asset(self, path_str: str | None) -> tuple[str | None, Path | None]:
        href_str, original = resolve_asset_path(path_str)
        if not href_str or not original or not original.exists():
            return None, original
        if original.is_dir():
            return None, original
        src_key = original.resolve()
        if src_key in self.asset_cache:
            target = self.asset_cache[src_key]
        else:
            if original.parent == self.report_dir:
                target = original
            else:
                target = self._copy_into_report(original)
            self.asset_cache[src_key] = target
        rel_path = str(target.relative_to(self.report_dir))
        self.rel_index[rel_path] = target
        return rel_path, target

    def _copy_into_report(self, src: Path) -> Path:
        resolved = src.resolve()
        stem = src.stem
        suffix = src.suffix
        candidate = self.report_dir / f"{stem}{suffix}"
        counter = 1
        while candidate.exists() and candidate.resolve() != resolved:
            candidate = self.report_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        if not candidate.exists():
            shutil.copy2(src, candidate)
        return candidate

    def absolute_from_relative(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        if rel in self.rel_index:
            return self.rel_index[rel]
        return self.report_dir / rel


def infer_mime(file_path: str | None, declared: str | None) -> str | None:
    if declared:
        return declared
    if not file_path:
        return None
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime
    suffix = Path(file_path).suffix.lower()
    if suffix in TEXT_FALLBACK_SUFFIXES:
        return "text/plain"
    return None


def slugify(name: str, existing: dict[str, str]) -> str:
    base = re.sub(r"\W+", "_", name.strip() or "unknown").strip("_")
    if not base:
        base = "stage"
    if base[0].isdigit():
        base = f"_{base}"
    candidate = base
    counter = 2
    while candidate in existing.values():
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def build_mermaid(events):
    if not events:
        return "graph TD\n    Empty[\"No events\"]"
    stage_ids: dict[str, str] = {}
    lines = ["graph LR"]
    last_stage = None
    for ev in events:
        stage = ev["stage"]
        if stage not in stage_ids:
            stage_ids[stage] = slugify(stage, stage_ids)
            lines.append(f'    {stage_ids[stage]}["{html.escape(stage)}"]')
        if last_stage and last_stage != stage:
            label = ev["action"] or ev["message"]
            label = (label[:60] + "…") if len(label) > 60 else label
            lines.append(
                f'    {stage_ids[last_stage]} -->|{html.escape(label)}| {stage_ids[stage]}'
            )
        last_stage = stage
    return "\n".join(lines)


def resolve_asset_path(path_str: str | None) -> tuple[str | None, Path | None]:
    if not path_str:
        return None, None
    raw_path = Path(path_str)
    if raw_path.exists():
        return path_str, raw_path
    fallback = Path.cwd() / path_str
    if fallback.exists():
        return str(fallback), fallback
    return path_str, raw_path


def render_asset_section(ev: dict, ctx: RenderContext) -> str:
    rel_href = ev.get("asset_href")
    if not rel_href:
        return ""

    abs_path = ctx.absolute_from_relative(rel_href)
    href = html.escape(rel_href)
    pieces = [
        f'<p class="path-label"><code>{href}</code> · '
        f'<a href="{href}" target="_blank" rel="noopener">Open file</a></p>'
    ]

    if not abs_path or not abs_path.exists():
        pieces.append('<p class="warn">File not found when generating report.</p>')
        return "\n".join(pieces)

    mime = ev.get("mime") or ""
    suffix = abs_path.suffix.lower()

    if mime.startswith("image/"):
        pieces.append(
            f'<div class="asset asset-image"><img src="{href}" loading="lazy" alt=""></div>'
        )
    elif mime.startswith("video/"):
        pieces.append(
            f'<details class="asset"><summary>View video</summary>'
            f'<video src="{href}" controls preload="metadata"></video></details>'
        )
    elif mime.startswith("text/") or suffix in TEXT_FALLBACK_SUFFIXES:
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover
            pieces.append(f'<p class="warn">Unable to read file contents: {html.escape(str(exc))}</p>')
        else:
            limit = 20000
            display = content[:limit]
            if len(content) > limit:
                display += "\n… (truncated for display)"
            pieces.append(
                f'<details class="asset"><summary>View file contents</summary>'
                f'<pre>{html.escape(display)}</pre></details>'
            )

    return "\n".join(pieces)


def render_image_bubble(path_str: str, ctx: RenderContext) -> str:
    rel_href, path = ctx.prepare_asset(path_str)
    if not rel_href or not path or not path.exists():
        return f'{html.escape(path_str)} <span class="warn">(not found)</span>'
    href = html.escape(rel_href)
    bubble_id = f"img-{uuid.uuid4().hex}"
    return (
        f'<span class="image-popover">'
        f'<button type="button" class="image-link" data-popover-open="{bubble_id}">'
        f'{html.escape(Path(rel_href).name)}</button>'
        f'<span class="image-bubble" id="{bubble_id}" role="dialog" aria-modal="false">'
        f'<button type="button" class="close" data-popover-close="{bubble_id}" aria-label="Close preview">×</button>'
        f'<img src="{href}" alt="">'
        f'<a class="open-original" href="{href}" target="_blank" rel="noopener">Open original</a>'
        f'</span>'
        f'</span>'
    )


def render_meta_value(value, ctx: RenderContext) -> str:
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{render_meta_value(v, ctx)}</td></tr>"
            for k, v in sorted(value.items())
        )
        return f'<table class="meta-nested">{rows}</table>'
    if isinstance(value, (list, tuple, set)):
        items = "".join(f"<li>{render_meta_value(item, ctx)}</li>" for item in value)
        return f'<ul class="meta-list">{items}</ul>'
    if isinstance(value, str):
        suffix = Path(value).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return render_image_bubble(value, ctx)
        rel_href, _ = ctx.prepare_asset(value)
        if rel_href:
            return html.escape(rel_href)
        return html.escape(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value))


def render_meta_rows(meta: dict, ctx: RenderContext) -> str:
    return "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{render_meta_value(v, ctx)}</td></tr>"
        for k, v in sorted(meta.items())
    )


def render_html(events, mermaid_src: str, output_path: Path, ctx: RenderContext):
    cards_html = []
    for ev in events:
        meta_html = ""
        if ev["meta"]:
            rows = render_meta_rows(ev["meta"], ctx)
            meta_html = f"<details><summary>meta</summary><table>{rows}</table></details>"
        raw_json = html.escape(json.dumps(ev["raw"], indent=2, default=str))
        asset_html = render_asset_section(ev, ctx)
        cards_html.append(
            dedent(
                f"""
                <article class="event">
                  <header>
                    <span class="badge">{html.escape(ev["stage"])}</span>
                    <strong>{html.escape(ev["action"] or ev["message"])}</strong>
                    <time>{html.escape(ev["timestamp"])}</time>
                  </header>
                  <p>{html.escape(ev["message"])}</p>
                  {asset_html}
                  {meta_html}
                  <details>
                    <summary>raw log record</summary>
                    <pre>{raw_json}</pre>
                  </details>
                </article>
                """
            ).strip()
        )
    cards = "\n".join(cards_html) or "<p>No events found.</p>"

    html_doc = dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>TVA Agent Run</title>
          <link rel="stylesheet" href="https://unpkg.com/@picocss/pico@2/css/pico.min.css">
          <script type="module">
            import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.mjs";
            mermaid.initialize({{ startOnLoad: true, theme: "base", themeVariables: {{ primaryColor: "#2563eb" }} }});
          </script>
          <style>
            body {{ max-width: 1100px; margin: auto; padding: 2rem; }}
            .mermaid {{ background: #f8fafc; border-radius: 0.75rem; padding: 1rem; }}
            .event {{ border-left: 3px solid #2563eb; padding-left: 1.25rem; margin-bottom: 1.5rem; position: relative; }}
            .badge {{ text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.08em; background: #e0e7ff; padding: 0.2rem 0.6rem; border-radius: 999px; }}
            .event header {{ display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 0.6rem; }}
            .event time {{ color: #475569; font-size: 0.85rem; }}
            .asset {{ margin-top: 0.75rem; }}
            .asset-image img {{ width: min(560px, 100%); border-radius: 0.75rem; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35); }}
            .event details.asset video {{ width: min(640px, 100%); margin-top: 0.5rem; border-radius: 0.75rem; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35); }}
            .warn {{ color: #b91c1c; }}
            table {{ width: 100%; font-size: 0.9rem; }}
            th {{ text-align: left; width: 25%; color: #475569; }}
            td {{ vertical-align: top; }}
            .meta-nested {{ width: 100%; margin: 0.25rem 0; font-size: 0.85rem; }}
            .meta-list {{ margin: 0.25rem 0; padding-left: 1.2rem; }}
            .path-label {{ font-size: 0.85rem; color: #475569; margin: 0.25rem 0; }}
            .path-label code {{ background: #0f172a; color: #e2e8f0; padding: 0.35rem 0.6rem; border-radius: 0.5rem; overflow-wrap: anywhere; }}
            iframe {{ width: 100%; height: 320px; border: 1px solid #cbd5f5; border-radius: 0.5rem; margin-top: 0.5rem; }}
            pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 0.5rem; font-size: 0.85rem; }}
            .image-popover {{ position: relative; display: inline-block; }}
            .image-link {{ background: none; border: none; color: #1d4ed8; padding: 0; cursor: pointer; text-decoration: underline; font: inherit; }}
            .image-link:hover {{ color: #1e3a8a; }}
            .image-backdrop {{ position: fixed; inset: 0; background: rgba(15, 23, 42, 0.55); z-index: 60; display: none; }}
            .image-backdrop.is-active {{ display: block; }}
            .image-bubble {{ display: none; position: fixed; z-index: 70; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #fff; border: 1px solid #cbd5f5; box-shadow: 0 35px 70px rgba(15, 23, 42, 0.45); border-radius: 1rem; padding: 1.5rem; width: min(90vw, 900px); max-height: 90vh; overflow: auto; }}
            .image-bubble.is-active {{ display: block; }}
            .image-bubble img {{ width: 100%; max-height: 80vh; object-fit: contain; border-radius: 0.75rem; }}
            .image-bubble .close {{ position: absolute; top: 0.3rem; right: 0.6rem; background: none; border: none; color: #475569; font-size: 1.25rem; cursor: pointer; }}
            .image-bubble .open-original {{ display: inline-block; margin-top: 0.5rem; font-size: 0.85rem; }}
          </style>
        </head>
        <body>
          <h1>TVA Agent Timeline</h1>
          <section>
            <h2>Control Flow (Mermaid)</h2>
            <pre class="mermaid">
{html.escape(mermaid_src)}
            </pre>
          </section>
          <section>
            <h2>Event Timeline</h2>
            {cards}
          </section>
          <div class="image-backdrop" id="image-backdrop"></div>
          <script>
            const backdrop = document.getElementById('image-backdrop');
            const closeAllPopovers = () => {{
              document.querySelectorAll('.image-bubble.is-active').forEach((el) => el.classList.remove('is-active'));
              if (backdrop) {{
                backdrop.classList.remove('is-active');
              }}
            }};
            document.addEventListener('click', (event) => {{
              const openBtn = event.target.closest('[data-popover-open]');
              if (openBtn) {{
                event.preventDefault();
                const id = openBtn.getAttribute('data-popover-open');
                const bubble = document.getElementById(id);
                if (bubble) {{
                  const isActive = bubble.classList.contains('is-active');
                  closeAllPopovers();
                  if (!isActive) {{
                    bubble.classList.add('is-active');
                    if (backdrop) {{
                      backdrop.classList.add('is-active');
                    }}
                  }}
                }}
                return;
              }}
              const closeBtn = event.target.closest('[data-popover-close]');
              if (closeBtn) {{
                event.preventDefault();
                const id = closeBtn.getAttribute('data-popover-close');
                const bubble = document.getElementById(id);
                if (bubble) {{
                  bubble.classList.remove('is-active');
                }}
                if (backdrop) {{
                  backdrop.classList.remove('is-active');
                }}
                return;
              }}
              if (backdrop && event.target === backdrop) {{
                closeAllPopovers();
                return;
              }}
              if (!event.target.closest('.image-popover') && !event.target.closest('.image-bubble')) {{
                closeAllPopovers();
              }}
            }});
          </script>
        </body>
        </html>
        """
    ).strip()

    output_path.write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Render TVA agent logs into HTML.")
    parser.add_argument("log_path", type=Path, help="Path to logs.jsonl")
    parser.add_argument("output_path", type=Path, help="Path for the generated HTML report")
    args = parser.parse_args()

    report_dir = args.output_path.parent
    ctx = RenderContext(report_dir=report_dir)
    events = load_events(args.log_path, ctx)
    mermaid_src = build_mermaid(events)
    render_html(events, mermaid_src, args.output_path, ctx)


if __name__ == "__main__":
    main()
