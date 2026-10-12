from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger: logging.Logger = logging.getLogger("prntscraper")


def aria2c(
    urls_and_options: list[tuple[str, dict[str, str]]],
    output_dir: Path,
) -> bool:
    aria2c_path = shutil.which("aria2c") or shutil.which("aria2c.exe")
    if not aria2c_path:
        logger.error("[red]aria2c not installed or not in PATH[/red]")
        return False

    input_lines: list[str] = []
    for url, opts in urls_and_options:
        input_lines.append(url)
        for key, value in opts.items():
            input_lines.append(f"  {key}={value}")
        input_lines.append("")
    input_text = "\n".join(input_lines)

    cmd = [
        aria2c_path,
        "--input-file=-",
        "--dir",
        str(output_dir),
        "-x",
        "16",
        "-s",
        "16",
        "-j",
        "8",
        "-k",
        "1M",
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--human-readable=true",
        "--summary-interval=0",
        "--download-result=hide",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as error:
        logger.error(f"[red]aria2c failed:[/red] {error}")
        return False

    if result.returncode != 0:
        logger.error(f"[red]aria2c exited with code {result.returncode}[/red]")
        if result.stderr:
            logger.error(result.stderr)
        return False

    logger.info("[green]aria2c completed successfully[/green]")
    return True
