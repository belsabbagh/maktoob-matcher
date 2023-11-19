import os

FOLDER = "data/raw/Authorship attribution data"
AUTHOR = 2
TEXT_LIMIT = 50

def make_latex_row(author, title, text, limit=TEXT_LIMIT):
    return f"""{author} & \\begin{{arabtext}}{title}\\end{{arabtext}} & \\begin{{arabtext}}{text[:limit]+'...'}\\end{{arabtext}} \\\\ \\hline"""


if __name__ == "__main__":
    for author in range(0, 9):
        author_folder = os.path.join(FOLDER, f"a{author}")
        rows = []
        for f in os.listdir(author_folder):
            if f.endswith(".txt"):
                with open(os.path.join(author_folder, f), "r", encoding="utf-8") as file:
                    text = file.read()
                # props filename example: a1_0.properties
                # text filename example: sample0.txt
                article_id = f.split(".")[0].removeprefix("sample")
                propsfile = f"a{author}_{article_id}.properties"
                with open(os.path.join(author_folder, propsfile), "r", encoding="utf-8") as file:
                    props = file.read()
                props = props.split("\n")
                title = props[2].split("=")[1]
                rows.append(make_latex_row(author, title, text))
        print("\n".join(rows))