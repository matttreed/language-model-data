from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def extract_text_from_html_bytes(html: bytes) -> str:
    encoding = detect_encoding(html)
    html_string = html.decode(encoding)
    return extract_plain_text(html_string)


if __name__ == "__main__":
    text = open("test.txt", "r").read().encode("utf-8")
    print(extract_text_from_html_bytes(text))