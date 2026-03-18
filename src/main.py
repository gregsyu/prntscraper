from __future__ import annotations

import logging
from pathlib import Path
from random import randrange, choices
from string import ascii_lowercase
from typing import Dict

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
    filter: str = typer.Option(
        None,
        "--filter",
        "-f",
        help="Keep only images classified as this label (screenshot, photograph, painting.)",
    ),
    debug: bool = typer.Option(False, "--debug", "-v", help="Enable debug output."),
) -> None:
    """
    Download screenshots from `prnt.sc`.
    Optionally filter images by classification.
    """
    global logger
    logger = setup_logging(debug)

    output_dir = Path(directory)
    output_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    for _ in track(range(max_downloads), description="Downloading..."):
        code = gen_code()
        url = f"{BASE_URL}/{code}"

        # for item in gen_urls(n):
        # code = item["code"]
        # url = item["url"]

        try:
            page = session.get(url, timeout=10)
            soup = BeautifulSoup(page.text, "html.parser")
            img = soup.find("img", {"id": "screenshot-image"})

            if not img:
                logger.warning(f"[yellow]No screenshot:[/yellow] `{code}`")
                continue

            img_url = img["src"]
            response = session.get(img_url, timeout=10)

            # if "image" not in response.headers.get("Content-Type", ""):
            # if debug:
            # print(f"[yellow]Invalid image:[/yellow] {code}")
            # continue

            file_path = output_dir / f"{code}.png"

            with open(file_path, "wb") as f:
                f.write(response.content)
            if not is_image_valid(file_path):
                logger.error(f"[red]Invalid file removed:[/red] `{code}`")
                file_path.unlink()
            if filter:
                result = classify_image(file_path)
                if result.get(filter, 0) < SCREENSHOT_THRESHOLD:
                    logger.warning(f"[yellow]Filtered out:[/yellow] `{file_path}`")
                    file_path.unlink()
            logger.info(f"[green]Downloaded:[/green] `{code}`")
        except requests.exceptions.RequestException as e:
            logger.error(f"[red]Network error on {code}:[/red] `{e}`")
        except Exception as e:
            logger.exception(f"[red]Unexpected error:[/red] `{code}`: `{e}`")
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
