from random import randrange, choices
from string import ascii_lowercase
from rich import print
from bs4 import BeautifulSoup
import requests
import webbrowser
import os

def gen_url() -> str:
    alphabet = ascii_lowercase
    result = (
        "u" +
        choices(alphabet, k=1)[0] +
        str(randrange(10)) +
        ''.join(choices(alphabet, k=3))
    )
    return {"code": result, "url": f"https://prnt.sc/{result}"}


def download_sss(num: int, debug: bool = False):
    os.makedirs("sss", exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    urls = [gen_url() for _ in range(num)]

    for item in urls:
        try:
            r = requests.get(item["url"], headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            img = soup.find("img", {"id": "screenshot-image"})
            if not img:
                print(f"No screenshot: `{item[code]}`")
                continue

            img_url = img["src"]
            img_data = requests.get(img_url, headers=headers).content

            with open(f'sss/{item["code"]}.png', "wb") as f:
                f.write(img_data)

            if debug:
                print("Downloaded:", item["code"])
        except Exception as e:
            print("Error:", item["code"], e)
    # url = gen_url()
    # print(f"Opening `{url}`")
    #webbrowser.open_new_tab(url)


def main():
    download_sss(10, debug=True)


if __name__ == "__main__":
    main()

# ai to check forbidden images
