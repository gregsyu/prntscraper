from __future__ import annotations

import logging
from pathlib import Path
from random import randrange, choices
from string import ascii_lowercase
from typing import Dict
import shutil
import subprocess

import requests
from bs4 import BeautifulSoup
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track
import torch
import clip
import typer

app = typer.Typer()

BASE_URL = "https://prnt.sc"
HEADERS = {"User-Agent": "Mozilla/5.0"}
LABELS = ["screenshot", "photograph", "painting"]
SCREENSHOT_THRESHOLD = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = None
PREPROCESS = None
TEXT_TOKENS = None


def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=False,
                show_time=True,
                show_level=True,
            )
        ],
    )
    logger = logging.getLogger("prntscraper")
    logger.setLevel(level)

    # Silencing
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("http.client").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


logger: logging.Logger = logging.getLogger("prntscraper")


def aria2c(
    urls_and_options: list[tuple[str, dict]], output_dir: Path, debug: bool = False
) -> bool:
    aria2c_path = shutil.which("aria2c") or shutil.which("aria2c.exe")
    if not aria2c_path:
        logger.error("[red]aria2c not installed or not in PATH[/red]")
        return False

    input_lines = list()
    for url, opts in urls_and_options:
        input_lines.append(url)
        for k, v in opts.items():
            input_lines.append(f"  {k}={v}")
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
        "--summary-interval=1",  # if not debug else
    ]

    try:
        result = subprocess.run(
            cmd, input=input_text, check=False, text=True
        )
        if result.returncode != 0:
            logger.error(f"[red]aria2c exited with code {result.returncode}[/red]")
            logger.error(result.stderr)
            return False
        logger.info("[green]aria2c completed successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[red]aria2c failed (code {e.returncode})")
        logger.error(e.stderr)
        return False


def load_model(debug: bool = False) -> None:
    global MODEL, PREPROCESS, TEXT_TOKENS

    if MODEL is None:
        logger.debug("[cyan]Loading CLIP model...[/cyan]")
        MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
        TEXT_TOKENS = clip.tokenize(LABELS).to(DEVICE)


def gen_code() -> str:
    alphabet = ascii_lowercase
    result = (
        "u"
        + choices(alphabet, k=1)[0]
        + str(randrange(10))
        + "".join(choices(alphabet, k=3))
    )
    return result


def is_image_valid(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def classify_image(path: Path) -> Dict[str, float]:
    load_model()

    image = PREPROCESS(Image.open(path)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits_per_image, _ = MODEL(image, TEXT_TOKENS)
        probs = logits_per_image.softmax(dim=-1)

    return dict(zip(LABELS, probs[0].cpu().numpy()))


@app.command()
def download(
    max_downloads: int = typer.Option(
        100,
        "--max-downloads",
        "-m",
        help="Number of screenshots to attempt downloading.",
    ),
    directory: str = typer.Option(
        "sss", "--directory", "-d", help="Directory to store images."
    ),
    max_attempts_multiplier: int = typer.Option(
        5, "--max-attempts-multiplier", help="How many attempts per desired image"
    ),
    debug: bool = typer.Option(False, "--debug", "-v", help="Enable debug output."),
) -> None:
    """
    Download screenshots from `prnt.sc`.
    Optionally filter images by classification.
    """
    global logger
    logger = setup_logging(debug)

    output_dir = Path(directory).resolve()
    output_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    downloads: list[tuple[str, dict]] = list()
    max_attempts = max_downloads * max_attempts_multiplier
    successful = 0
    attempts = 0

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

            # checks if image is valid
            try:
                head = session.head(img_url, timeout=6, allow_redirects=True)
                match head.status_code:
                    case 200:
                        pass
                    case 404:
                        logger.debug(f"Image 404 not found: `{code}` -> {img_url}")
                        continue
                    case _:
                        logger.debug(f"Unexpected status {head.status_code} for image: {code}")
                        continue
            except requests.RequestException as e:
                logger.debug(f"HEAD check failed for `{code}`: {e}")
                continue

            options = {
                "out": f"{code}.png",
            }
            downloads.append((img_url, options))
            successful += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"[red]Network error on {code}:[/red] `{e}`")
            return
        except Exception as e:
            logger.exception(f"[red]Unexpected error:[/red] `{code}`: `{e}`")
            return
    if not downloads:
        logger.warning("[yellow]No valid images found.[/yellow]")
        return

    success = aria2c(urls_and_options=downloads, output_dir=output_dir, debug=debug)

    if not success:  # fallback
        logger.warning(
            "[yellow]aria2c failed -> consider installing it or using slow mode[/yellow]"
        )
        return

    for file in output_dir.glob("*"):
        if not is_image_valid(file):
            logger.error(f"[red]Removing invalid file:[/red] {file.name}")
            file.unlink()
            continue
    # NOTE: webbrowser.open_new_tab(url)


@app.command()
def filter(
    filter_image: str = typer.Option(
        LABELS[1], "--filter-image", "-f", help="Image type to keep."
    ),
    directory: str = typer.Option(
        "sss", "--directory", "-d", help="Directory containing image to filter."
    ),
    debug: bool = typer.Option(False, "--debug", "-v", help="Enable debug output."),
) -> None:
    """
    Filter images in a directory using openai's CLIP classification.
    """
    global logger
    logger = setup_logging(debug)

    load_model()

    path = Path(directory)
    files = sorted(path.glob("*"))

    for file in track(files, description="Filtering..."):
        if not is_image_valid(file):
            logger.error(f"[red]Removing invalid image:[/red] `{file}`")
            file.unlink()
            continue

        result = classify_image(file)

        if result[filter_image] < SCREENSHOT_THRESHOLD:
            logger.warning(f"[yellow]Removing image:[/yellow] `{file}`")
            file.unlink()
        else:
            logger.info(f"[green]Keeping image:[/green] `{file}`")


if __name__ == "__main__":
    app()
