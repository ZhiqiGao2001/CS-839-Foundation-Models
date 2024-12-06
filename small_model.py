import random
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from datasets import load_dataset

from dream_trainer import DreamTrainer


def get_model_and_tokenizer_gpt_neo(config=None):
    if config is None:
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
    data_collator_gpt_neo = DataCollatorForLanguageModeling(tokenizer=tokenizer_gpt_neo, mlm=False)
    # print(model_gpt_neo)
    print(f"Number of parameters in model: {model_gpt_neo.num_parameters() / 1_000_000:.2f}M")
    return model_gpt_neo, tokenizer_gpt_neo, data_collator_gpt_neo


def get_wiki_dataset(tokenizer_, small=False):
    if small:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    ds_train, ds_eval, ds_test = ds["train"], ds["validation"], ds["test"]

    def tokenize_function(examples):
        return tokenizer_(examples["text"], truncation=True, max_length=2048, padding="max_length")

    train_tokenized = ds_train.map(tokenize_function, batched=True, num_proc=4)
    eval_tokenized = ds_eval.map(tokenize_function, batched=True, num_proc=4)
    test_tokenized = ds_test.map(tokenize_function, batched=True, num_proc=4)

    return train_tokenized, eval_tokenized, test_tokenized


def initialize_training(model_to_train, tokenizer_, data_collator_, train_dataset_, eval_dataset_, hyperparameters=None, dream=False):
    training_args = TrainingArguments(
        output_dir="trash",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        save_total_limit=2,
        report_to="wandb",
        load_best_model_at_end=True,
        logging_steps=10,
        **hyperparameters
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.1,
    )
    if dream:
        print("Using DreamTrainer")
        trainer = DreamTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset_,
            eval_dataset=eval_dataset_,
            callbacks=[early_stopping_callback],
            data_collator=data_collator_,
            processing_class=tokenizer_,
        )
    else:
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset_,
            eval_dataset=eval_dataset_,
            callbacks=[early_stopping_callback],
            data_collator=data_collator_,
            processing_class=tokenizer_,
        )

    trainer.train()


if __name__ == '__main__':
    # config = GPTNeoConfig(
    #     num_heads=4,
    #     num_layers=6,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     max_position_embeddings=2048,
    #     attention_types=[[['global', 'local'], 3]],
    #     attention_layers=["global", "local", "global", "local", "global", "local"],
    # )

    model, tokenizer, data_collator = get_model_and_tokenizer_gpt_neo()
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset, test_dataset = get_wiki_dataset(
        tokenizer,
        # small=True
    )

    hps = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
    }

    initialize_training(
        model,
        tokenizer,
        data_collator,
        train_dataset,
        eval_dataset,
        hps,
        # dream=True,
    )



