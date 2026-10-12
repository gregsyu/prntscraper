from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import requests
import typer
from bs4 import BeautifulSoup
from rich.logging import RichHandler
from rich.progress import track

from download import aria2c
from images import (
    LABELS,
    SCREENSHOT_THRESHOLD,
    classify_image,
    gen_code,
    is_image_valid,
    load_model,
)

app = typer.Typer()
BASE_URL = "https://prnt.sc"
HEADERS = {"User-Agent": "Mozilla/5.0"}
logger: logging.Logger = logging.getLogger("prntscraper")


def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
    )
    logger = logging.getLogger("prntscraper")
    logger.setLevel(level)
    for name in ("urllib3", "requests", "http.client", "PIL", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


@app.command()
def download(
    max_downloads: Annotated[
        int,
        typer.Option(
            "--max-downloads",
            "-m",
            help="Number of screenshots to attempt downloading.",
        ),
    ] = 100,
    directory: Annotated[
        str, typer.Option("--directory", "-d", help="Directory to store images.")
    ] = "sss",
    max_attempts_multiplier: Annotated[
        int,
        typer.Option(
            "--max-attempts-multiplier",
            "-a",
            help="How many attempts per desired image.",
        ),
    ] = 5,
    ignore_invalid_images: Annotated[
        bool,
        typer.Option(
            "--ignore-invalid-images",
            "-i",
            help="Ignores invalid images from downloading.",
        ),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", "-v", help="Enable debug output.")
    ] = False,
) -> None:
    """Download screenshots from `prnt.sc`."""
    global logger
    logger = setup_logging(debug)
    output_dir = Path(directory).resolve()
    output_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)
    downloads: list[tuple[str, dict[str, str]]] = []
    max_attempts = max_downloads * max_attempts_multiplier
    successful = attempts = 0

    while successful < max_downloads and attempts < max_attempts:
        attempts += 1
        code = gen_code()
        url = f"{BASE_URL}/{code}"
        try:
            page = session.get(url, timeout=10)
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            img = soup.find("img", {"id": "screenshot-image"})
            if not img:
                logger.warning(f"[yellow]No screenshot:[/yellow] `{code}`")
                continue

            img_url = img["src"]
            if ignore_invalid_images:
                try:
                    head = session.head(img_url, timeout=6, allow_redirects=True)
                    match head.status_code:
                        case 200:
                            pass
                        case 404:
                            logger.debug(f"Image 404 not found: `{code}` -> {img_url}")
                            continue
                        case _:
                            logger.debug(
                                f"Unexpected status {head.status_code} for image: {code}"
                            )
                            continue
                except requests.RequestException as error:
                    logger.debug(f"HEAD check failed for `{code}`: {error}")
                    continue

            downloads.append((img_url, {"out": f"{code}.png"}))
            successful += 1
        except requests.exceptions.RequestException as error:
            logger.error(f"[red]Network error on {code}:[/red] `{error}`")
            return
        except Exception:
            logger.exception(f"[red]Unexpected error:[/red] `{code}`")
            return

    if not downloads:
        logger.warning("[yellow]No valid images found.[/yellow]")
        return
    if not aria2c(downloads, output_dir):
        logger.warning(
            "[yellow]aria2c failed -> consider installing it or using slow mode[/yellow]"
        )


@app.command()
def filter(
    filter_image: Annotated[
        str, typer.Option("--filter-image", "-f", help="Image type to keep.")
    ] = LABELS[1],
    directory: Annotated[
        str,
        typer.Option("--directory", "-d", help="Directory containing image to filter."),
    ] = "sss",
    debug: Annotated[
        bool, typer.Option("--debug", "-v", help="Enable debug output.")
    ] = False,
) -> None:
    """Filter images in a directory using CLIP classification."""
    global logger
    logger = setup_logging(debug)
    load_model()
    files = sorted(Path(directory).glob("*.png"))
    for file in track(files, description="Filtering..."):
        result = classify_image(file)
        if result[filter_image] < SCREENSHOT_THRESHOLD:
            logger.warning(f"[yellow]Removing image:[/yellow] `{file}`")
            file.unlink()
        else:
            logger.info(f"[green]Keeping image:[/green] `{file}`")


@app.command()
def cleanup(
    directory: Annotated[
        str, typer.Option("--directory", "-d", help="Directory containing images.")
    ] = "sss",
    debug: Annotated[
        bool, typer.Option("--debug", "-v", help="Enable debug output.")
    ] = False,
) -> None:
    """Remove invalid image files from a directory."""
    global logger
    logger = setup_logging(debug)
    files = sorted(Path(directory).resolve().glob("*"))
    invalid_files = 0
    for file in track(files, description="Checking..."):
        if not is_image_valid(file):
            logger.warning(f"[yellow]Removing invalid file:[/yellow] {file.name}")
            file.unlink()
            invalid_files += 1
    logger.info(f"[green]Removed {invalid_files}/{len(files)}[/green]")


if __name__ == "__main__":
    app()
