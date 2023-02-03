import json
from typing import Dict

json_out = "raw.json"
txt_in = "raw_sample_text.txt"

with open(txt_in, "r", encoding="utf-8") as f:
    raw_sample_title = f.readline().strip()

    # skip empty line
    f.readline()

    raw_sample_txt = f.read().replace("\n", " ")

with open(json_out, "r") as f:
    data: Dict[str, str] = json.load(f)

data[raw_sample_title] = raw_sample_txt
with open(json_out, "w") as f:
    json.dump(data, f)
