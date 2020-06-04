from typing import Iterable

import numpy as np


def reduce_mean_list(ls):
    ''' arverage the mutiple list'''
    if len(ls) == 1:
        return ls[0]
    for item in ls[1:]:
        for index, value in enumerate(item):
            ls[0][index] += value
    return [value / len(ls) for value in ls[0]]


def reduce_max_list(ls):
    if len(ls) == 1:
        return ls[0]
    max_ls = ls[0]
    for item in ls[1:]:
        for index, value in enumerate(item):
            if value > max_ls[index]:
                max_ls[index] = value
    return max_ls


def align_features(features: Iterable[dict], mode: str):
    context_chunk = []
    for context_features in features:
        num_token = len(context_features["features"])
        orig_to_tok_map = [id_ for id_ in context_features["orig_to_tok_map"] if id_ != 0] + [num_token - 1]
        # embeddings = []
        word_pieces_embs = []

        for token_id, feature in enumerate(context_features["features"]):
            if mode == "first" and token_id in orig_to_tok_map[:-1]:
                context_chunk.append(np.array(feature["layers"][0]["values"]))

            if mode == "mean" and token_id in orig_to_tok_map[1:]:  # merage before word pieces
                context_chunk.append(np.array(reduce_mean_list(word_pieces_embs)))
                word_pieces_embs = []  # clean word pieces

            if mode == "max" and token_id in orig_to_tok_map[1:]:
                context_chunk.append(np.array(reduce_max_list(word_pieces_embs)))
                word_pieces_embs = []

            if token_id > 0 and token_id < num_token - 1:  # CLS and SEP are not necessary
                word_pieces_embs.append(feature["layers"][0]["values"])

        yield tuple(context_chunk)
        context_chunk = []
