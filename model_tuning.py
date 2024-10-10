import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

os.environ["HF_DATASETS_CACHE"] = "./cache"

dataset = load_dataset("OpenAssistant/oasst1", split="train").filter(lambda x: x['lang'] == 'en')

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    per_device_train_batch_size=3,  
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=False,
    no_cuda=True, =
    logging_dir='./logs',
    logging_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=collate_fn
)

trainer.train()

model.save_pretrained("./opt-350m-oasst1")
tokenizer.save_pretrained("./opt-350m-oasst1")
