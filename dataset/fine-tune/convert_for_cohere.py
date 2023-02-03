import json
from typing import Dict

from dataset.data_models import ProcessedDataModel

JSON_IN = "processed.json"
TXT_OUT = "for_cohere.txt"

SEPARATOR = "--"


def sample_from_data(data: ProcessedDataModel) -> str:
    out = ""
    for q in data.questions:
        out += f"question: {q}\ntext: {data.text}\n{SEPARATOR}\n\n"

    return out


processed_data: Dict[str, dict] = json.load(open(JSON_IN, "r"))

data_text = ""
for o in processed_data.values():
    o = ProcessedDataModel.parse_obj(o)

    data_text += sample_from_data(o)

# remove trailing separator
data_text = data_text[:-2]

with open(TXT_OUT, "w") as f:
    print(data_text)
    f.write(data_text)
