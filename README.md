# prntscraper

A command-line tool to download random screenshots from **prnt.sc** (lightshot) and optionally classify/filter them using OpenAI's CLIP model.

Features:
- Downloads images using `aria2c` (multi-connection, resumable, very fast) when available
- Optional post-download filtering with CLIP (keep only screenshots / photographs)
- Automatic cleanup of invalid/corrupted images
- Pretty console output with `rich`

## Requirements

- Python 3.8+
- aria2c - install via your package manager:
  - Ubuntu/Debian: `sudo apt install aria2`
  - macOS: `brew install aria2`
  - Windows: download from https://aria2.github.io/ or use winget/choco
- Python packages (install with `pip install -r requirements.txt`):

```text
typer
rich
requests
beautifulsoup4
pillow
torch
clip
```

## Installation
```bash
git clone https://github.com/gregsyu/prntscraper.git
cd prntscraper
pip install -r requirements.txt
# or: pip install typer rich requests beautifulsoup4 pillow torch git+https://github.com/openai/CLIP.git
```

## Usage
```bash
python3 main.py --help
```

### 1. Download screenshots

```bash
# Download ~100 valid screenshots
python main.py download -m 100 -d my_screenshots

# Try harder to get exactly 200 images (up to 1000 attempts)
python main.py download -m 200 -a 5

# Download to custom folder, HEAD check for faster collection
python main.py download -m 150 -d screenshots --ignore-invalid-images
```

### 2. Filter existing images by type (using CLIP)

```bash
# Keep only screenshots, remove others
python main.py filter -f screenshot -d my_screenshots

# Keep only photographs (default)
python main.py filter -d my_screenshots
```

### 3. Clean invalid/corrupted images

```bash
python main.py cleanup -d my_screenshosts
```

## Options

```
download:
  -m, --max-downloads       Number of valid images to collect (default: 100)
  -d, --directory           Output directory (default: sss)
  -a, --max-attempts-multiplier  Attempts per desired image (default: 5)
  -i, --ignore-invalid-images  Skip HEAD check (faster but may include dead links)

filter:
  -f, --filter-image        Type to keep: screenshot | photograph | painting (default: photograph)
  -d, --directory           Directory to filter

cleanup:
  -d, --directory           Directory to clean
```

## How it works (brief)
1. Generates random prnt.sc codes (u?0???)
2. Fetches page -> extracts image URL
3. (optional) HEAD check to skip 404s
4. Collects until enough valid images found
5. Downloads with aria2c (parallel, multi-connection) or warns if missing
6. (optional) Runs CLIP classification to keep only desired category
7. Validates downloaded files (removes corrupted ones)
