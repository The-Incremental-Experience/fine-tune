import json
import numpy
from annoy import AnnoyIndex
from typing import List, Dict

import cohere

from dataset.tune.data_models import ProcessedDataModel

JSON_IN = "../tune/lala.json"
INDEX_OUT = "index.ann"
JSON_OUT = "keke.json"
MODEL = "large"
api_key = "wilJVepgbMNVHebtIy8hYVnAQhvoJu5Qkp9UQEW2"
co = cohere.Client(api_key)

with open(JSON_IN, "r") as f:
    data: List[dict] = json.load(f)

ordered_processed_data = [ProcessedDataModel.parse_obj(o) for o in data]
ordered_processed_data = [o for o in ordered_processed_data if len(o.questions) > 0]

# ordered map of questions to sources
ordered_question_to_source_map: List[Dict[str, str]] = []
for o in ordered_processed_data:
    source = f"{o.title}, {o.chapter} by {o.creator}"
    for q in o.questions:
        ordered_question_to_source_map.append({"question": q, "source": source})

ordered_embeddings = numpy.array(
    co.embed(texts=[o["question"] for o in ordered_question_to_source_map], model=MODEL, truncate="RIGHT").embeddings)

search_index = AnnoyIndex(ordered_embeddings.shape[1], "angular")

for i in range(len(ordered_embeddings)):
    search_index.add_item(i, ordered_embeddings[i])

search_index.build(n_trees=10)
search_index.save(INDEX_OUT)
with open(JSON_OUT, "w") as f:
    json.dump(ordered_question_to_source_map, f)

# test

sample_question = "What are some emotions that narcissists hide?"

question_embedding = numpy.array(co.embed(texts=[sample_question], model=MODEL).embeddings)[0]

similar_item_ids = search_index.get_nns_by_vector(question_embedding, 10,
                                                  include_distances=True)

for i in range(len(similar_item_ids[0])):
    index = similar_item_ids[0][i]
    distance = similar_item_ids[1][i]
    print(ordered_question_to_source_map[index]["question"], distance)
