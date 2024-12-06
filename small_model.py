import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from datasets import load_dataset


def get_model_and_tokenizer():
    config = GPTNeoConfig(
        num_heads=4,
        num_layers=6,
        hidden_size=128,
        intermediate_size=512,
        max_position_embeddings=2048,
        attention_types=[[['global', 'local'], 3]],
        attention_layers=["global", "local", "global", "local", "global", "local"],
    )

    model_gpt_neo = GPTNeoForCausalLM(config)
    tokenizer_gpt_neo = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # print the model architecture and parameter count in millions
    # print(model_gpt_neo)
    print(f"Number of parameters in model: {model_gpt_neo.num_parameters() / 1_000_000}M")
    return model_gpt_neo, tokenizer_gpt_neo


def get_wiki_dataset():
    # ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    print(ds)
    return ds


if __name__ == '__main__':
    model, tokenizer = get_model_and_tokenizer()
    get_wiki_dataset()

