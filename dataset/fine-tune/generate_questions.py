import json
from typing import List, Dict

import cohere

from dataset.data_models import ProcessedDataModel

JSON_IN = "raw.json"
JSON_OUT = "processed.json"
CO_KEY = "Kwi3nCYBE9ihcpvY8TNa3DTsCe0rKGGXqOnmrVrh"


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


# load and parse
co = cohere.Client(CO_KEY)
raw_pairs: Dict[str, str] = json.load(open(JSON_IN))
processed_pairs: Dict[str, dict] = json.load(open(JSON_OUT))
raw_data = {
    title: ProcessedDataModel(title=title, text=text)
    for title, text in raw_pairs.items()
}
processed_data = {
    title: ProcessedDataModel.parse_obj(o) for title, o in processed_pairs.items()
}

process_raw_and_save(processed_data, raw_data)
