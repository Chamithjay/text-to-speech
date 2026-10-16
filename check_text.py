import unicodedata

line = next(line for line in open("data/LJSpeech-1.1/metadata.csv", encoding="utf8")
            if line.startswith("LJ001-0001|"))
parts = line.rstrip("\n").split("|")
print("number of fields:", len(parts))

for name, field in (("text", parts[1]), ("normalized_text", parts[2])):
    i = field.find("arts")
    seg = field[i - 12:i]
    print(name, "->", repr(seg))
    for ch in seg:
        if ord(ch) > 127:
            print("   non-ASCII:", hex(ord(ch)), unicodedata.name(ch, "?"))
