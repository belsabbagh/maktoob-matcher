from arabicstopwords import arabicstopwords as stp


def remove_symbols(text):
    return "".join(c for c in text if c.isalpha())


def preprocess_text(text):
    words = [word for word in text.split() if not stp.is_stop(word)]
    words = [remove_symbols(word) for word in words]
    return " ".join(words)
