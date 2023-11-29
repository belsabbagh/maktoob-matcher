from arabicstopwords import arabicstopwords as stp


def remove_symbols(text):
    return "".join(c for c in text if c.isalpha())


def remove_english(text):
    return "".join(c for c in text if not c.isascii())


def preprocess_text(text):
    words = text.split()
    # words = [word for word in words if not stp.is_stop(word)]
    words = [remove_symbols(word) for word in words]
    words = [remove_english(word) for word in words]
    return " ".join(w.strip() for w in words if w.strip() != "")
