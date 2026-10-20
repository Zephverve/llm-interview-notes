#!/usr/bin/env python3
"""同步 Markdown 源文件并构建 / 预览 MkDocs 站点。

用法：
    python 网站/build.py              # 同步 + 构建到 网站/site
    python 网站/build.py --serve      # 同步 + 本地预览（含手机同网访问）
    python 网站/build.py --serve --no-watch
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent
CONTENT = WEBSITE.parent
DOCS = WEBSITE / "docs"
ASSETS = WEBSITE / "assets"
INDEX_TEMPLATE = WEBSITE / "index.template.md"


def iter_sources():
    """遍历内容目录中的 Markdown，排除网站构建目录和隐藏目录。"""
    for path in sorted(CONTENT.rglob("*.md")):
        try:
            rel = path.relative_to(CONTENT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == WEBSITE.name:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path, rel


def sync(clean: bool = False) -> None:
    """把源 Markdown 复制到 docs/，并把 README 映射成首页片段。"""
    if clean and DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "_snippets").mkdir(parents=True, exist_ok=True)

    expected: set[Path] = set()
    updated = 0

    for src, rel in iter_sources():
        if rel.as_posix() == "README.md":
            dest = DOCS / "_snippets" / "README.md"
        else:
            dest = DOCS / rel
        expected.add(dest.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or src.read_bytes() != dest.read_bytes():
            shutil.copy2(src, dest)
            updated += 1

    if not INDEX_TEMPLATE.exists():
        raise FileNotFoundError(f"缺少首页模板：{INDEX_TEMPLATE}")
    index_dest = DOCS / "index.md"
    expected.add(index_dest.resolve())
    if not index_dest.exists() or INDEX_TEMPLATE.read_bytes() != index_dest.read_bytes():
        shutil.copy2(INDEX_TEMPLATE, index_dest)
        updated += 1

    if ASSETS.exists():
        shutil.copytree(ASSETS, DOCS / "assets", dirs_exist_ok=True)

    # 清理源文件已删除后遗留的页面
    for old in DOCS.rglob("*.md"):
        if old.resolve() not in expected:
            old.unlink()
            updated += 1

    # 清理空目录（保留两个特殊目录）
    keep = {DOCS / "_snippets", DOCS / "assets"}
    for directory in sorted(
        (p for p in DOCS.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if directory in keep:
            continue
        if not any(directory.iterdir()):
            directory.rmdir()

    print(f"[sync] {len(expected)} 个页面，{updated} 个文件有更新 -> {DOCS}")


def run_mkdocs(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "mkdocs", *args]
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(WEBSITE))


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def start_watcher() -> object | None:
    """监听源 Markdown 变化，自动同步到 docs/，触发 mkdocs 重建。"""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("[watch] 未安装 watchdog，跳过自动同步。修改源文件后请重新运行。")
        return None

    class Handler(FileSystemEventHandler):
        def __init__(self) -> None:
            self.timer: threading.Timer | None = None
            self.lock = threading.Lock()

        def on_any_event(self, event) -> None:
            if getattr(event, "is_directory", False):
                return
            path = Path(event.src_path).resolve()
            if path.suffix.lower() != ".md":
                return
            if path == WEBSITE or WEBSITE in path.parents:
                return
            with self.lock:
                if self.timer is not None:
                    self.timer.cancel()
                self.timer = threading.Timer(0.35, self.run_sync)
                self.timer.daemon = True
                self.timer.start()

        def run_sync(self) -> None:
            try:
                sync(clean=False)
            except Exception as exc:  # noqa: BLE001 - 预览时只提示，不退出
                print(f"[watch] 同步失败：{exc}")

    observer = Observer()
    observer.daemon = True
    observer.schedule(Handler(), str(CONTENT), recursive=True)
    observer.start()
    print(f"[watch] 正在监听 {CONTENT} 下的 Markdown 变化")
    return observer


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 / 预览大模型面试资料站点")
    parser.add_argument("--serve", action="store_true", help="启动本地预览服务器")
    parser.add_argument("--host", default="0.0.0.0", help="预览监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="预览端口，默认 8000")
    parser.add_argument("--no-watch", action="store_true", help="预览时不自动同步源文件")
    args = parser.parse_args()

    if not args.serve:
        sync(clean=True)
        code = run_mkdocs(["build", "--clean"])
        if code == 0:
            (WEBSITE / "site" / ".nojekyll").write_text("", encoding="utf-8")
        return code

    sync(clean=True)
    observer = None if args.no_watch else start_watcher()
    print()
    print(f"[preview] 本机：http://127.0.0.1:{args.port}/")
    print(f"[preview] 同一 Wi-Fi 手机：http://{lan_ip()}:{args.port}/")
    print("[preview] 按 Ctrl+C 停止")
    print()
    try:
        return run_mkdocs(
            ["serve", "--dev-addr", f"{args.host}:{args.port}"]
        )
    except KeyboardInterrupt:
        return 0
    finally:
        if observer is not None:
            observer.stop()
            observer.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
