#!/usr/bin/env python3
"""Repair the hook commands that `ruflo init` writes into .claude/settings.json.

WHY THIS EXISTS
---------------
ruflo (via @claude-flow/cli) generates hooks in this shape:

    cmd /c "IF EXIST "%CLAUDE_PROJECT_DIR%\\.claude\\helpers\\X" \\
        (node "%CLAUDE_PROJECT_DIR%\\.claude\\helpers\\X" sub) \\
        ELSE (node "%USERPROFILE%\\.claude\\helpers\\X" sub)"

The JS template emits \\" believing it escapes, but inside a JS template
literal that collapses to a bare quote, leaving nested unescaped quotes. The
command gets fragmented and pieces of it -- and of tool arguments -- are
created as 0-byte files in the working directory (observed: cmd, ', $c,
each_symbol, {corr!r}, hook-handler, ruflo).

The template is byte-identical in @claude-flow/cli 3.10.2, 3.41.2 and 3.42.4,
so upgrading does not help. Fixing locally does -- and `ruflo init` will undo
it, which is why this script is checked in.

THE CORRECT FORM USES BASH SYNTAX, NOT CMD SYNTAX
-------------------------------------------------
The hook executor is bash, not cmd.exe, so %VAR% is NOT expanded. A fix that
keeps %VAR% produces:

    Cannot find module 'Y:\\proj\\%USERPROFILE%\\.claude\\helpers\\X'

which disables every hook while appearing to fix the stray files. Do not
reintroduce %VAR%.

    project: node "${CLAUDE_PROJECT_DIR:-.}/.claude/helpers/X" sub
    global:  node "$USERPROFILE/.claude/helpers/X" sub

The ${VAR:-.} fallback keeps project hooks working whether or not
CLAUDE_PROJECT_DIR is exported; the CWD is already the project root.

VERIFYING A FIX
---------------
Test with a command containing shell metacharacters -- $(...), quotes, braces.
Plain `echo` reproduces nothing and gives a FALSE PASS. Also watch for the
Stop hook on session exit: it fires auto-memory-hook.mjs and surfaces
MODULE_NOT_FOUND if the paths are wrong.

USAGE
-----
    python tools/fix_ruflo_hooks.py                 # cwd project + global
    python tools/fix_ruflo_hooks.py --scan Y:/Projects
    python tools/fix_ruflo_hooks.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Matches any of the three shapes the template has taken, and pulls out the
# helper script name plus the subcommand.
SCRIPT_RE = re.compile(r"helpers[\\/]([A-Za-z0-9_.-]+\.(?:cjs|mjs))")
SUB_RE = re.compile(r"\.(?:cjs|mjs)\"?\s+([a-z][a-z0-9-]*)")
USER_VAR_RE = re.compile(r"%USERPROFILE%|\$USERPROFILE", re.I)
PROJ_VAR_RE = re.compile(r"%CLAUDE_PROJECT_DIR%|\$CLAUDE_PROJECT_DIR", re.I)

PROJECT_TPL = 'node "${{CLAUDE_PROJECT_DIR:-.}}/.claude/helpers/{script}" {sub}'
GLOBAL_TPL = 'node "$USERPROFILE/.claude/helpers/{script}" {sub}'


def desired(command: str, file_scope: str) -> str | None:
    """Return the correct command for this hook, or None if unrecognised.

    Scope is decided carefully. The original probe form

        cmd /c "IF EXIST "%CLAUDE_PROJECT_DIR%\\..." (node ...) ELSE (node "%USERPROFILE%\\..." ...)"

    is PROJECT scope with a user-scope fallback, so it mentions %USERPROFILE%
    too. Matching on the mere presence of that variable misclassifies every
    probe-form hook as user scope -- which is precisely the case this script
    exists to repair. So:

      * probe form (contains IF EXIST) -> follow the file's own scope
      * otherwise, whichever variable actually appears

    A hook command with no backslash before `helpers` also reaches here, in
    the `"$VAR$/.claude/helpers/..."` form -- same rules apply.
    """
    script_m = SCRIPT_RE.search(command)
    sub_m = SUB_RE.search(command)
    if not script_m or not sub_m:
        return None
    script, sub = script_m.group(1), sub_m.group(1)

    if "IF EXIST" in command:
        scope = file_scope
    else:
        has_user = bool(USER_VAR_RE.search(command))
        has_proj = bool(PROJ_VAR_RE.search(command))
        if has_user and not has_proj:
            scope = "user"
        elif has_proj and not has_user:
            scope = "project"
        else:
            scope = file_scope

    template = GLOBAL_TPL if scope == "user" else PROJECT_TPL
    return template.format(script=script, sub=sub)


def is_clean(command: str) -> bool:
    return not ("cmd /c" in command or "%" in command) and "$" in command


def scope_of(path: Path) -> str:
    """User scope for ~/.claude/settings.json, project scope for anything else."""
    return "user" if path.parent == Path.home() / ".claude" else "project"


def fix_file(path: Path, dry_run: bool) -> tuple[int, int, list[str], bool]:
    """Returns (changed, already_ok, notes, existed)."""
    if not path.exists():
        return 0, 0, [], False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0, 0, [f"!! unparseable, left alone: {path} ({exc})"], True

    file_scope = scope_of(path)

    changed = 0
    ok = 0
    notes: list[str] = []

    for event, items in data.get("hooks", {}).items():
        for item in items:
            for hook in item.get("hooks", []):
                cmd = hook.get("command")
                if not isinstance(cmd, str):
                    continue
                if is_clean(cmd):
                    ok += 1
                    continue
                new = desired(cmd, file_scope)
                if new is None:
                    notes.append(f"!! unrecognised shape, left alone [{event}]: {cmd[:90]}")
                    continue
                if new != cmd:
                    hook["command"] = new
                    changed += 1
                    notes.append(f"   [{event} / {item.get('matcher', '-')}] -> {new}")

    if changed and not dry_run:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak-{stamp}")
        shutil.copy2(path, backup)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Re-read: never leave a file we cannot parse.
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded == data, f"round-trip mismatch: {path}"
        notes.append(f"   backup: {backup.name}")

    return changed, ok, notes, True


def scan_roots(roots: list[str]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for candidate in base.rglob("settings.json"):
            if ".claude" in candidate.parts:
                found.append(candidate)
    return sorted(set(found))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=os.getcwd(),
                    help="project root (default: cwd); its .claude/settings.json is fixed")
    ap.add_argument("--global-only", action="store_true", help="skip the project file")
    ap.add_argument("--skip-global", action="store_true", help="skip ~/.claude/settings.json")
    ap.add_argument("--scan", nargs="*", metavar="ROOT",
                    help="also scan these roots (default when bare: Y:/Projects Y:/代码)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets: list[Path] = []
    if not args.global_only:
        targets.append(Path(args.project) / ".claude" / "settings.json")
    if not args.skip_global:
        targets.append(Path.home() / ".claude" / "settings.json")
    if args.scan is not None:
        roots = args.scan or ["Y:/Projects", "Y:/代码"]
        targets.extend(scan_roots(roots))

    seen: set[Path] = set()
    ordered = [p for p in targets if not (p in seen or seen.add(p))]

    total_changed = 0
    for path in ordered:
        changed, ok, notes, existed = fix_file(path, args.dry_run)
        total_changed += changed
        if not existed:
            print(f"[skip] {path}  (absent)")
            continue
        status = "DRY" if args.dry_run else "fix"
        print(f"[{status}] {path}  [{scope_of(path)} scope]")
        print(f"        rewritten={changed}  already-ok={ok}")
        for line in notes:
            print(line)

    print()
    if total_changed:
        verb = "would rewrite" if args.dry_run else "rewrote"
        print(f"Total: {verb} {total_changed} hook command(s).")
        print("Verify with a command containing metacharacters, e.g.:")
        print("""  c=$(echo x); echo "brace {a,b}" 'quote' "1e-12)" && ls -a | wc -l""")
        print("A plain `echo` proves nothing -- it reproduces no stray files either way.")
    else:
        print("Nothing to do: every hook already uses the bash form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
