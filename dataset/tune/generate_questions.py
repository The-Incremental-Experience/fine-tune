import json
import os
from typing import List, Dict

import cohere

from dataset.tune.data_models import ProcessedDataModel, ProcessedDatum
from bs4 import BeautifulSoup

JSON_DIR_IN = "../data"
JSON_OUT = "processed.json"
CO_KEY = "Kwi3nCYBE9ihcpvY8TNa3DTsCe0rKGGXqOnmrVrh"

co = cohere.Client(CO_KEY)


def process_raw_and_save(
    processed: Dict[str, ProcessedDataModel], raw: Dict[str, ProcessedDataModel]
):
    """
    Take title-text pairs and add questions to all if missing, saves incrementally

    @param processed: already stored data that might not have generated questions
    @param raw: raw data
    """

    data = raw | processed
    o: ProcessedDataModel
    for i, _ in enumerate(data.items()):
        title, o = _
        if len(o.questions) > 0:
            print(
                f"{process_raw_and_save}: skipping processed, {(i + 1)} out of {len(raw)} finished"
            )
            continue

        o.questions = generate_questions_for_text(o.text)

        # save
        processed[o.title] = o
        with open(JSON_OUT, "w") as f:
            json.dump({title: o.dict() for title, o in processed.items()}, f)

        print(
            f"{process_raw_and_save}: generated, {(i + 1)} out of {len(raw)} finished"
        )


def generate_questions_for_text(tx: str) -> List[str]:
    """Generate three questions for given text with cohere"""

    prompt = """You are reading a book about Psychology. Write three questions based on the chapter below."""
    prompt += "\n\n"
    prompt += tx

    print(f"{generate_questions_for_text}: generating cohere response")
    response = co.generate(
        model="command-xlarge-20221108",
        prompt=prompt,
        max_tokens=200,
        temperature=0.1,
        truncate="LEFT",
        k=0,
        p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods="NONE",
    )

    out: List[str] = []
    for line in response[0].text.splitlines():
        question_starts = ["1. ", "2. ", "3. "]
        if line[:3] in question_starts:
            out.append(line[3:])

    return out


processed_data = []
for directory in os.walk(JSON_DIR_IN):

    for book_name in list(directory)[2]:

        if ".json" not in book_name:
            continue

        book = json.load(open(f"{JSON_DIR_IN}/{book_name}", "r"))

        title = book["title"]
        creator = book["creator"]

        print("Book title: ", title)

        for chapter_key in book["chapters"]:
            html = book["chapters"][chapter_key]

            if len(html) < 2500:
                print("Chapter too short or too long: ", chapter_key)
                continue

            # todo: filter out properly
            lower_html = html.lower()
            if (
                "contents" in lower_html
                or "introduction" in lower_html
                or "references" in lower_html
                or "resources" in lower_html
                or "foreword" in lower_html
                or "preface" in lower_html
                or "epilogue" in lower_html
                or "squibs" in lower_html
            ):
                print("Skip Contents")
                continue

            soup = BeautifulSoup(html)
            text = soup.get_text()

            try:
                text = text.strip()
            except Exception as exc:
                print("Failed to strip, ignore chapter: ", chapter_key)
                continue

            chapter_title = None

            try:
                chapter_title = soup.section["title"]
            except Exception as exc:
                print(
                    "Failed to get chapter title from section 0",
                    chapter_key,
                )

            if not chapter_title:
                try:
                    chapter_title = soup.h1.get_text()
                except Exception as exc:
                    print(
                        "Failed to get chapter title from header",
                        chapter_key,
                    )

            if not chapter_title:
                try:
                    chapter_title = soup.find({"h2": {"class": "chap_ttl"}}).get_text()
                except Exception as exc:
                    print(
                        "Failed to get chapter title from h2 chap_ttl",
                        chapter_key,
                    )

            if not chapter_title:
                try:
                    chapter_title = soup.h2.get_text()
                except Exception as exc:
                    print(
                        "Failed to get chapter title from h2, ignore chapter: ",
                        chapter_key,
                    )
                    continue

            # todo: make cute and clean batches, not crude ಠ_ಠ
            lines = text.split("\n")
            batches = [
                lines[i] for i in range(0, len(lines), 10) if len(lines[i]) > 200
            ]

            for batch in batches:
                try:
                    questions = generate_questions_for_text(batch)
                except Exception as exc:
                    print(
                        "Failed to generate questions for prompt, ignore it: ",
                        batch,
                        repr(exc),
                    )
                    continue

                processed_data.append(
                    ProcessedDataModel(
                        title=title,
                        creator=creator,
                        chapter=chapter_title,
                        text=batch,
                        questions=questions,
                    ),
                )

js = ProcessedDatum.parse_obj(processed_data).json()
print("Processed this many", len(processed_data))
open("lala.json", "w").write(js)
