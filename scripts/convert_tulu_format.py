import argparse
import json
import gzip
import os
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if token is not None:
    login(token=token)


def get_formatted_data_from_conversation(dataset, extra_fields, name_prefix):
    formatted_data = []
    for i, data in tqdm(enumerate(dataset)):
        text_entries = None
        for field_name in ["messages", "conversations", "conversation"]:
            if field_name in data:
                text_entries = data[field_name]
                break
        if text_entries is None:
            raise RuntimeError
        formatted_entry = []
        for entry in text_entries:
            role = None
            for role_field in ["role", "from"]:
                if role_field in entry:
                    role = entry[role_field].upper()
                    break
            if role is None:
                raise RuntimeError
            content = None
            for content_field in ["content", "value"]:
                if content_field in entry:
                    content = entry[content_field]
                    break
            if content is None:
                raise RuntimeError

            formatted_entry.append(f"{role}: {content}")
        
        if len(extra_fields) == 0 or extra_fields[0] == "none":
            output_data = {
                "id": f"{name_prefix}_{i}",
                "text": "\n\n".join(formatted_entry),
            }
            formatted_data.append(output_data)
        elif extra_fields[0] == "all":
            data['text'] = "\n\n".join(formatted_entry)
            if "id" not in data:
                data["id"] = f"{name_prefix}_{i}"
            formatted_data.append(data)
        else:
            output_data = {
                "id": f"{name_prefix}_{i}",
                "text": "\n\n".join(formatted_entry),
            }
            for field in extra_fields:
                output_data[field] = data[field]
            formatted_data.append(output_data)

    return formatted_data


def get_formatted_data_from_query_response(dataset, name_prefix):
    formatted_data = []
    for i, data in tqdm(enumerate(dataset)):
        formatted_entry = [
            f"QUERY: {data['query']}",
            f"ANSWER: {data['answer']}"
        ]
        data["text"] = "\n\n".join(formatted_entry)
        if "id" not in data:
            data["id"] = f"{name_prefix}_{i}"
        formatted_data.append(data)

    return formatted_data


def get_formatted_data_from_preferences(dataset, extra_fields, name_prefix):
    formatted_data = []
    for i, data in tqdm(enumerate(dataset)):
        chosen_text_entries = data["chosen"]
        rejected_text_entries = data["rejected"]
        assert len(chosen_text_entries) == 2
        assert len(rejected_text_entries) == 2
        assert chosen_text_entries[0]["content"] == rejected_text_entries[0]["content"]
        user_role = chosen_text_entries[0]["role"].upper()
        assistant_role = chosen_text_entries[1]["role"].upper()
        formatted_entry = []
        formatted_entry.append(f"{user_role}: {chosen_text_entries[0]['content']}")
        formatted_entry.append(f"CHOSEN {assistant_role}: {chosen_text_entries[1]['content']}")
        formatted_entry.append(f"REJECTED {assistant_role}: {rejected_text_entries[1]['content']}")
        
        if len(extra_fields) == 0 or extra_fields[0] == "none":
            output_data = {
                "id": f"{name_prefix}_{i}",
                "text": "\n\n".join(formatted_entry),
            }
            formatted_data.append(output_data)
        elif extra_fields[0] == "all":
            data['text'] = "\n\n".join(formatted_entry)
            if "id" not in data:
                data["id"] = f"{name_prefix}_{i}"
            formatted_data.append(data)
        else:
            output_data = {
                "id": f"{name_prefix}_{i}",
                "text": "\n\n".join(formatted_entry),
            }
            for field in extra_fields:
                output_data[field] = data[field]
            formatted_data.append(output_data)

    return formatted_data


def convert_text_format(dataset_name, split_name, output_folder_path, extra_fields):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    dataset = load_dataset(dataset_name, split=split_name)
    name_prefix = dataset_name.split("/")[-1].lower()
    file_name =  f"{name_prefix}.jsonl.gz"
    output_file_path = os.path.join(output_folder_path, file_name)
    
    if "chosen" in dataset[0] and "rejected" in dataset[0]:
        formatted_data = get_formatted_data_from_preferences(dataset, extra_fields, name_prefix)
    elif "messages" in dataset[0] or "conversation" in dataset[0]:
        formatted_data = get_formatted_data_from_conversation(dataset, extra_fields, name_prefix)
    elif "query" in dataset[0] and "answer" in dataset[0]:
        formatted_data = get_formatted_data_from_query_response(dataset, name_prefix)
    
    with gzip.open(output_file_path, 'wt', encoding='utf-8') as outfile:
        for entry in tqdm(formatted_data):
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save huggingface dataset to jsonl.gz files"
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--extra_fields", type=str, nargs="+", default=["all"])
    args = parser.parse_args()

    output_folder_path = args.out_path
    
    convert_text_format(args.dataset_name, args.split_name, output_folder_path, args.extra_fields)
