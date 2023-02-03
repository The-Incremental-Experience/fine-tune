import json
import numpy
from annoy import AnnoyIndex
from typing import List

import cohere

from dataset.tune.data_models import ProcessedDataModel

JSON_IN = "../tune/lala.json"
INDEX_OUT = "index.ann"
CUSTOM_MODEL = "medium"
api_key = "wilJVepgbMNVHebtIy8hYVnAQhvoJu5Qkp9UQEW2"
co = cohere.Client(api_key)


with open(JSON_IN, "r") as f:
    data: List[dict] = json.load(f)

ordered_processed_data = [ProcessedDataModel.parse_obj(o) for o in data[:2]]
ordered_processed_data = [o for o in ordered_processed_data if len(o.questions) > 0]

questions = [f"{i+1}. {o.questions[0]}" for i, o in enumerate(ordered_processed_data)]

questions = ["How fast does a cougar run?"]
print(questions)

# todo: take all, not just first
ordered_embeddings = numpy.array(
    co.embed(texts=questions, model=CUSTOM_MODEL, truncate="RIGHT").embeddings)
#
# search_index = AnnoyIndex(ordered_embeddings.shape[1], "angular")
#
# for i in range(len(ordered_embeddings)):
#     search_index.add_item(i, ordered_embeddings[i])
#
# search_index.build(n_trees=10)
# search_index.save(INDEX_OUT)
#
# ordered_sources = []
# for o in ordered_processed_data:
#     kek = f"{o.title}, {o.chapter} by {o.creator}"
#     ordered_sources.append(kek)
#
# # test
#
# sample_question = "What are some emotions that narcissists bury?"
#
# question_embedding = numpy.array(co.embed(texts=[sample_question], model=CUSTOM_MODEL).embeddings)[0]
#
# similar_item_ids = search_index.get_nns_by_vector(question_embedding, 10,
#                                                   include_distances=True)
#
# for (i, distance) in similar_item_ids:
#     print(ordered_sources[i], distance)
