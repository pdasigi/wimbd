import argparse
import json
import os
from typing import List
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
import spacy
from tqdm import tqdm

from wimbd.es import (count_documents_for_each_phrase, es_init, get_documents_containing_phrases)

SPACY_MODEL = spacy.load("en_core_web_lg")

INDICES = [
    "tulu-v2-sft-mixture",
    "aya_dataset",
    "coconot-sft",
    "codefeedback-filtered-instruction",
    "code-feedback",
    "daring-anteater",
    "metamath-qa",
    "no_robots",
    "openassistant-guanaco",
    "sciriff-train-mix-science",
    "sharegpt-cleaned",
    "slimorca",
    "table-gpt-all-trained",
    "ultrafeedback_binarized",
    "webinstructsub",
    "wildchat-1m-full-gpt4-only",
    "wizardlm_evol_instruct_v2_196k",
]

EVAL_SETS = [
    # (dataset, subset, split, fields)
    ("cais/mmlu", "all", "test", ["question"]),
    ("openai/openai_humaneval", None, "test", ["prompt"]),
    ("openai/gsm8k", "main", "test", ["question"]),
    ("ucinlp/drop", None, "validation", ["passage", "question"]),
    ("lighteval/MATH", "all", "test", ["problem"]),
    ("google/IFEval", None, "train", ["prompt"]),
    ("akariasai/PopQA", None, "test", ["subj", "prop", "obj"]),
    ("tatsu-lab/alpaca_eval", None, "eval", ["instruction"])
]



def get_ngrams(string: str, n: int):
    doc = SPACY_MODEL(string)
    ngrams = [doc[i:i+n].text for i in range(len(doc) - n + 1)]
    return ngrams


def contamination_percentage(index_name: str, corpus_name: str, sub_corpus_name: str = None, split: str = "train", fields: List[str] = ["text"], report_file: str = None):
    
    path = (Path(__file__).parent / ".." / "..").resolve() / "es_config.yml"
    es = es_init(path, timeout=180)
    
    dataset = load_dataset(corpus_name, sub_corpus_name)
    string_list = list(zip(*[dataset[split][x] for x in fields]))
    string_list = [list(x) for x in string_list]

    presence = [x > 0 for x in count_documents_for_each_phrase(index_name, string_list, batch_size=60, es=es, all_phrases=True)]
    counts = sum(presence)
    percentage = counts / len(string_list)

    if report_file is not None:
        matches = []
        for string, string_presence in zip(string_list, presence):
            if string_presence:
                matches.append({"instance": string})

        with open(report_file, "w") as outfile:
            json.dump(
                {
                    "contamination": percentage,
                    "fields": fields,
                    "index_name": index_name,
                    "matches": matches,
                },
                outfile
            )

    return percentage


def partial_contamination_percentage(index_name: str, ngram_size: int, corpus_name: str, sub_corpus_name: str = None, split: str = "train", fields: List[str] = ["text"], report_file: str = None):
    
    path = (Path(__file__).parent / ".." / "..").resolve() / "es_config.yml"
    es = es_init(path, timeout=180)
    
    dataset = load_dataset(corpus_name, sub_corpus_name)
    instance_list = list(zip(*[dataset[split][x] for x in fields]))
    instance_list = [list(x) for x in instance_list]
    ngram_index = defaultdict(list)
    for i, instance in enumerate(instance_list):
        for instance_field in instance:
            for ngram in get_ngrams(instance_field, ngram_size):
                ngram_index[ngram].append(i)

    most_frequent_ngram, coverage_list = sorted(ngram_index.items(), key=lambda x: len(x), reverse=True)[0]
    print(f"Most frequent n-gram: '{most_frequent_ngram}'")
    print(f"\tCoverage: {100 * len(coverage_list) / len(instance_list):.2f}%")

    instance_contamination = defaultdict(list)
    ngram_list = list(ngram_index.keys())
    ngram_presence = [x > 0 for x in count_documents_for_each_phrase(index_name, ngram_list, batch_size=60, es=es)]
    for ngram, presence in tqdm(zip(ngram_list, ngram_presence)):
        if presence:
            for instance_id in ngram_index[ngram]:
                instance_contamination[instance_id].append(ngram)

    percentage = len(instance_contamination) / len(instance_list)
    if report_file is not None:
        matches = [{"instance": instance_list[instance_id], "matching_grams": ngrams} for instance_id, ngrams in instance_contamination.items()]
        with open(report_file, "w") as outfile:
            json.dump(
                {
                    "contamination": percentage,
                    "fields": fields,
                    "ngram_size": ngram_size,
                    "index_name": index_name,
                    "matches": matches,
                },
                outfile,
            )

    return percentage
    

if __name__ == "__main__":
    parse = argparse.ArgumentParser("")
    parse.add_argument("--ngram_size", type=int)
    parse.add_argument("--output_dir", type=str)

    args = parse.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_data = []
    for index_name in INDICES:
        print(f"Querying {index_name}")
        output_data.append([])
        for dataset, subset, split, fields in EVAL_SETS:
            report_file = os.path.join(args.output_dir, f"{index_name}_{dataset}.json".replace("/", "_"))
            print(f"\tChecking for contamination with {dataset} in fields {fields}; see {report_file} for a detailed report")
            if args.ngram_size is None:
                percentage = contamination_percentage(index_name, dataset, subset, split, fields, report_file)
            else:
                percentage = partial_contamination_percentage(index_name, args.ngram_size, dataset, subset, split, fields, report_file)
            print(f"\t\tContamination percentage: {percentage}")
            output_data[-1].append(percentage)

    output_file = os.path.join(args.output_dir, "contamination_results.tsv")
    print(f"TSV file with all results: {output_file}")
    with open(output_file, "w") as outfile:
        print("\t" + "\t".join(ev[0] for ev in EVAL_SETS), file=outfile)
        for index_name, results in zip(INDICES, output_data):
            print(index_name + "\t" + "\t".join([f"{p:.4f}" for p in results]), file=outfile)