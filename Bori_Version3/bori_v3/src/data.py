import torch
from datasets import load_dataset, interleave_datasets
from itertools import chain

def _load_with_split_fallback(path, split, streaming, **kwargs):
    """Try loading with the requested split; if it doesn't exist, try common alternatives."""
    try:
        return load_dataset(path, split=split, streaming=streaming, **kwargs)
    except ValueError as e:
        if "Unknown split" not in str(e):
            raise
        for alt in ["train_sft", "train_gen", "train", "test_sft", "test"]:
            if alt == split:
                continue
            try:
                ds = load_dataset(path, split=alt, streaming=streaming, **kwargs)
                print(f"  Split '{split}' not found → using '{alt}' instead.")
                return ds
            except (ValueError, Exception):
                continue
        raise

def _load_multiple_datasets(dataset_names, split, streaming, probabilities=None, seed=42):
    if isinstance(dataset_names, str):
        dataset_names = [d.strip() for d in dataset_names.split(',')]
        
    if isinstance(probabilities, str):
        probabilities = [float(p.strip()) for p in probabilities.split(',')]
        
    datasets = []
    for name in dataset_names:
        name = name.strip()
        filter_col = None
        filter_val = None
        
        if "?" in name:
            name, query = name.split("?", 1)
            if "=" in query:
                filter_col, filter_val = query.split("=", 1)
                filter_col = filter_col.strip()
                filter_val = filter_val.strip()
        
        if ":" in name:
            path, config = name.split(":", 1)
            print(f"Loading dataset {path} with configuration '{config}'...")
            ds = _load_with_split_fallback(path, split, streaming, name=config)
        else:
            print(f"Loading dataset {name}...")
            ds = _load_with_split_fallback(name, split, streaming)
            
        if filter_col and filter_val:
            print(f"Filtering dataset '{name}' where '{filter_col}' == '{filter_val}'...")
            ds = ds.filter(lambda x, col=filter_col, val=filter_val: str(x.get(col)) == val)
            
        datasets.append(ds)
    
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        unified_datasets = []
        for ds in datasets:
            available_keys = []
            try:
                if hasattr(ds, "features") and ds.features:
                    available_keys = list(ds.features.keys())
                else:
                    sample_iter = iter(ds)
                    sample = next(sample_iter)
                    available_keys = list(sample.keys())
            except Exception:
                pass
            
            text_col = "text"
            for alt in ["text", "content", "body", "markdown", "document"]:
                if alt in available_keys:
                    text_col = alt
                    break
            
            # Map to a unified single-column schema containing only "text" to avoid Arrow schema conflicts
            ds_unified = ds.map(
                lambda x, tc=text_col: {"text": x.get(tc, "") if x.get(tc) is not None else ""}, 
                remove_columns=available_keys if available_keys else None
            )
            unified_datasets.append(ds_unified)
            
        dataset = interleave_datasets(unified_datasets, probabilities=probabilities, seed=seed)
    
    if streaming:
        buf_size = 10000 if torch.cuda.is_available() else 500
        dataset = dataset.shuffle(seed=seed, buffer_size=buf_size)
    return dataset

def get_packed_dataset(tokenizer, dataset_name, text_column="text", max_seq_length=4096, split="train", streaming=True, probabilities=None, seed=42):
    dataset = _load_multiple_datasets(dataset_name, split, streaming, probabilities, seed=seed)
    
    available_keys = []
    try:
        sample_iter = iter(dataset)
        first_sample = next(sample_iter)
        available_keys = list(first_sample.keys())
        if text_column not in available_keys:
            for alt in ["content", "body", "markdown", "document"]:
                if alt in available_keys:
                    text_column = alt
                    print(f"Auto-detected text column name: '{text_column}'")
                    break
            else:
                text_column = available_keys[0]
                print(f"Warning: Default text column not found. Falling back to: '{text_column}'")
    except Exception:
        pass
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=available_keys if available_keys else None,
    )

    def group_texts(examples):
        tokenizer_keys = [k for k in ["input_ids", "attention_mask", "token_type_ids"] if k in examples]
        concatenated_examples = {k: list(chain(*examples[k])) for k in tokenizer_keys}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
            
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    packed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )
    
    return packed_dataset

def get_sft_dataset(dataset_name, tokenizer, split="train", max_seq_length=2048, probabilities=None):
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )

    # Parse multiple datasets if comma-separated
    if isinstance(dataset_name, str):
        dataset_names = [d.strip() for d in dataset_name.split(',')]
    else:
        dataset_names = dataset_name

    if isinstance(probabilities, str):
        probabilities = [float(p.strip()) for p in probabilities.split(',')]

    def format_chat(example):
        if "conversations" in example and isinstance(example["conversations"], list):
            messages = []
            for msg in example["conversations"]:
                role = msg.get("from", "user")
                if role in ["human", "user"]:
                    mapped_role = "user"
                elif role in ["gpt", "assistant", "chatgpt"]:
                    mapped_role = "assistant"
                elif role in ["system"]:
                    mapped_role = "system"
                else:
                    mapped_role = "user"
                messages.append({"role": mapped_role, "content": msg.get("value", "")})
        elif "instruction" in example and "output" in example:
            user_content = example["instruction"]
            if "input" in example and example["input"]:
                user_content += "\n" + example["input"]
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]}
            ]
        elif "messages" in example:
            messages = example["messages"]
        elif "text" in example and isinstance(example["text"], str) and "<usr>" in example["text"] and "<bot>" in example["text"]:
            text_val = example["text"]
            parts = text_val.split("<bot>")
            if len(parts) >= 2:
                usr_part = parts[0].replace("<usr>", "").strip()
                bot_part = "<bot>".join(parts[1:]).strip()
                messages = [
                    {"role": "user", "content": usr_part},
                    {"role": "assistant", "content": bot_part}
                ]
            else:
                messages = [{"role": "user", "content": text_val}]
        elif "text" in example and isinstance(example["text"], str):
            # Already formatted text, keep as-is
            return {"text": example["text"]}
        else:
            raise ValueError(f"Dataset formatting not recognized.")
            
        messages = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    formatted_datasets = []
    for name in dataset_names:
        name = name.strip()
        filter_col = None
        filter_val = None
        
        if "?" in name:
            name, query = name.split("?", 1)
            if "=" in query:
                filter_col, filter_val = query.split("=", 1)
                filter_col = filter_col.strip()
                filter_val = filter_val.strip()
        
        if ":" in name:
            path, config = name.split(":", 1)
            print(f"Loading SFT dataset {path} with configuration '{config}'...")
            ds = _load_with_split_fallback(path, split, streaming=False, name=config)
        else:
            print(f"Loading SFT dataset {name}...")
            ds = _load_with_split_fallback(name, split, streaming=False)
            
        if filter_col and filter_val:
            print(f"Filtering SFT dataset '{name}' where '{filter_col}' == '{filter_val}'...")
            ds = ds.filter(lambda x, col=filter_col, val=filter_val: str(x.get(col)) == val)
        
        # Get features to remove all original columns and prevent schema mismatches
        available_keys = []
        if hasattr(ds, "features") and ds.features:
            available_keys = list(ds.features.keys())
        
        print(f"Formatting SFT dataset '{name}' using chat template...")
        ds_formatted = ds.map(format_chat, remove_columns=available_keys)
        formatted_datasets.append(ds_formatted)

    if len(formatted_datasets) == 1:
        dataset = formatted_datasets[0]
    else:
        print(f"Interleaving SFT datasets with probabilities: {probabilities}")
        dataset = interleave_datasets(formatted_datasets, probabilities=probabilities, seed=42)

    # Filter out examples with empty or trivially short text to prevent
    # 0-token sequences from crashing the model during training.
    dataset = dataset.filter(lambda x: len(x.get("text", "").strip()) > 10)
    
    return dataset
