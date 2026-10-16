import re
import unicodedata

PAD = "_"   # filler symbol, number 0. Short sentences get padded with it
EOS = "~"   # "end of sentence" symbol. The model learns to say "I'm done" with it
CHARS = " !',-.:;?abcdefghijklmnopqrstuvwxyz"
SYMBOLS = [PAD, EOS] + list(CHARS)
CHAR_TO_ID = {c: i for i, c in enumerate(SYMBOLS)}   # e.g. {"_":0, "~":1, " ":2, ...}

def clean_text(text):
    # Curly quotes need converting FIRST, or the ASCII step below would delete them
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    # "é" -> "e". NFKD splits "é" into "e" + an accent mark; encoding to ASCII drops the accent
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()                              # "Hello" -> "hello"
    text = re.sub(r"[^a-z !',\-.:;?]", " ", text)    # anything not in our alphabet becomes a space
    return re.sub(r"\s+", " ", text).strip()         # collapse repeated spaces

def text_to_ids(text):
    return [CHAR_TO_ID[c] for c in clean_text(text)] + [CHAR_TO_ID[EOS]]
