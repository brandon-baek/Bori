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
        # Try common training split names
        for alt in ["train_sft", "train_gen", "train", "test_sft", "test"]:
            if alt == split:
                continue
            try:
                ds = load_dataset(path, split=alt, streaming=streaming, **kwargs)
                print(f"  Split '{split}' not found → using '{alt}' instead.")
                return ds
            except (ValueError, Exception):
                continue
        raise  # Re-raise original if no alternative works

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
        
        # Check for query filtering, e.g. path:config?col=val or path?col=val
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
            # Capture filter_col and filter_val correctly in lambda closure
            ds = ds.filter(lambda x, col=filter_col, val=filter_val: str(x.get(col)) == val)
            
        datasets.append(ds)
    
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        # Interleave allows streaming from multiple sources at once according to the probabilities
        # Passing seed ensures deterministic interleaving order
        dataset = interleave_datasets(datasets, probabilities=probabilities, seed=seed)
    
    if streaming:
        # Shuffle the stream deterministically
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    return dataset

def get_packed_dataset(tokenizer, dataset_name, text_column="text", max_seq_length=4096, split="train", streaming=True, probabilities=None, seed=42):
    """
    Loads one or more datasets and packs multiple short documents into a single max_seq_length context window.
    This is highly efficient for pre-training.
    """
    dataset = _load_multiple_datasets(dataset_name, split, streaming, probabilities, seed=seed)
    
    # Auto-detect the correct text column in case it's named 'content', 'body', etc.
    available_keys = []
    try:
        # Peek at the first sample in the stream to read its keys
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
                # Fallback to first available key
                text_column = available_keys[0]
                print(f"Warning: Default text column not found. Falling back to: '{text_column}'")
    except Exception:
        # Fallback to default if stream peeking fails
        pass
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=available_keys if available_keys else None,
    )

    def group_texts(examples):
        # Concatenate all texts. Only process tokenizer-generated keys
        # to avoid unpacking metadata columns like language_score (floats) or id (strings)
        tokenizer_keys = [k for k in ["input_ids", "attention_mask", "token_type_ids"] if k in examples]
        concatenated_examples = {k: list(chain(*examples[k])) for k in tokenizer_keys}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        
        # Drop the small remainder, we could add padding if the model supported it
        # instead of this drop, you can customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
            
        # Split by chunks of max_len.
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
    """
    Helper for formatting a conversational dataset (like ShareGPT, Alpaca, or standard Messages) for SFT.
    Dynamically normalizes different structures to the standard chat template.
    """
    dataset = _load_multiple_datasets(dataset_name, split, streaming=False, probabilities=probabilities)
    
    def format_chat(example):
        # 1. ShareGPT format detection (e.g. SlimOrca-Dedup)
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
        
        # 2. Alpaca format detection (e.g. jojo0217/korean_safe_conversation)
        elif "instruction" in example and "output" in example:
            user_content = example["instruction"]
            if "input" in example and example["input"]:
                user_content += "\n" + example["input"]
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]}
            ]
            
        # 3. Standard Hugging Face messages format detection
        elif "messages" in example:
            messages = example["messages"]
            
        else:
            raise ValueError(f"Dataset formatting not recognized. Keys present: {list(example.keys())}. "
                             "Please ensure your dataset follows Alpaca, ShareGPT, or standard ChatML formatting.")
            
        # Sanitize: ensure all message content is a string (some datasets have None)
        messages = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]

        # Apply the chat template defined in the tokenizer
        example["text"] = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return example
        
    dataset = dataset.map(format_chat)
    return dataset
