# Documentation for Fine-tuning OPT-350M on OpenAssistant/oasst1 Dataset

### 1. **Environment Setup and Dataset Caching**
```python
import os
os.environ["HF_DATASETS_CACHE"] = "./cache"
```
- **Purpose**: This line sets the cache directory where the Hugging Face `datasets` library will store downloaded datasets. 
- **Why**: Caching helps avoid re-downloading the dataset each time you run the script.
- **Possible Changes**: Ensure that the cache directory has enough storage space, especially when working with large datasets. You can adjust the directory to a drive with more space if needed.

### 2. **Loading the Dataset**
```python
from datasets import load_dataset
dataset = load_dataset("OpenAssistant/oasst1", split="train").filter(lambda x: x['lang'] == 'en')
```
- **Purpose**: Loads the `OpenAssistant/oasst1` dataset and filters out only the English language entries.
- **Why**: We filter the dataset to focus on English text data, as the model is intended to be fine-tuned on English tasks.
- **Possible Changes**: 
   - If you need to train the model on other languages, remove or modify the `.filter(lambda x: x['lang'] == 'en')` line.
   - You can also split the dataset into training and validation sets if needed by using `load_dataset(..., split='train[:80%]')` and `load_dataset(..., split='train[80%:]')`.

### 3. **Loading the Model and Tokenizer**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```
- **Purpose**: Loads the `facebook/opt-350m` model and its corresponding tokenizer.
- **Why**: The tokenizer converts the text into a format (token IDs) that the model can understand. The model will be fine-tuned on the tokenized input.
- **Possible Changes**: You can switch to a larger or smaller model depending on your hardware limitations or performance requirements (e.g., `facebook/opt-125m` for lighter models or `facebook/opt-1.3b` for more powerful ones).

### 4. **Adding Special Tokens and Resizing Embeddings**
```python
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
```
- **Purpose**: Adds a padding token `[PAD]` to the tokenizer and resizes the model’s token embeddings accordingly.
- **Why**: Models like OPT do not have a padding token by default, which can lead to tokenization errors if the dataset contains sequences of different lengths. Padding ensures all sequences are the same length during batch processing.
- **Possible Changes**: 
   - If you’re working with variable-length sequences, this step is crucial. If not, you can skip adding a padding token.
   - You could also add other special tokens like `[CLS]` or `[SEP]` if your task requires them.

### 5. **Tokenizing the Dataset**
```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
- **Purpose**: Tokenizes the text data by converting text to token IDs and ensuring all sequences are of the same length (`max_length=512`).
- **Why**: Tokenization prepares the data for the model by converting raw text into numerical format (token IDs). Padding and truncation ensure that sequences are the same length for batch processing.
- **Possible Changes**: 
   - You can adjust `max_length` to be shorter (e.g., `256`) to reduce memory usage, especially on systems with limited RAM or GPU memory.
   - Instead of `padding="max_length"`, you can use `padding="longest"` to pad each batch to the length of the longest sequence, reducing unnecessary padding.

### 6. **Data Collation Function**
```python
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```
- **Purpose**: This function organizes the input data into tensors, making them ready for input into the model.
- **Why**: The data needs to be batched into tensors that the model can process in parallel. The `labels` are set to be the same as `input_ids`, as this is a causal language modeling task (where the model predicts the next token).
- **Possible Changes**: 
   - You can adjust this function to handle more complex tasks, such as adding different types of masks or labels if you’re doing sequence classification or another type of task.
   - If memory is an issue, consider using dynamic padding (only padding to the longest sequence in each batch) rather than static `max_length` padding.

### 7. **Training Arguments**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Adjust to fit within your 16GB RAM
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=False,  # Turn off mixed precision since training on CPU
    no_cuda=False,  # Use GPU if available
    logging_dir='./logs',
    logging_steps=1000
)
```
- **Purpose**: These arguments control the training process, including where to save the model, the batch size, number of epochs, and logging settings.
- **Why**: These settings are essential for controlling how the training proceeds. For instance, `per_device_train_batch_size=4` ensures that the model trains with four samples per batch on each GPU (or CPU if `no_cuda=True`).
- **Possible Changes**: 
   - **Batch size**: You can increase the batch size if your system allows it (i.e., sufficient RAM or GPU memory). A larger batch size may speed up training.
   - **Mixed Precision (fp16)**: If you have a supported GPU, you can enable `fp16=True` for mixed-precision training to save memory and potentially speed up training.
   - **Epochs**: If you want more thorough training, you can increase `num_train_epochs` to 5 or 10. Monitor validation loss to avoid overfitting.
   - **CUDA**: If you’re only using CPU, leave `no_cuda=True`. If you have GPU resources, set `no_cuda=False` to enable GPU utilization.

### 8. **Trainer Initialization**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=collate_fn
)
```
- **Purpose**: Initializes the Hugging Face `Trainer` class, which simplifies the training process by managing the training loop, gradient computation, and model updates.
- **Why**: Using `Trainer` makes it easier to handle model fine-tuning without manually writing the training loop.
- **Possible Changes**: You can add a validation dataset (e.g., `eval_dataset=...`) to monitor performance during training.

### 9. **Training the Model**
```python
trainer.train()
```
- **Purpose**: Starts the training process.
- **Why**: This line triggers the actual fine-tuning of the model using the defined arguments, dataset, and `collate_fn`.
- **Possible Changes**: You can save intermediate checkpoints and monitor validation metrics during training by configuring `evaluation_strategy` and `save_strategy`.

### 10. **Saving the Model and Tokenizer**
```python
model.save_pretrained("./opt-350m-oasst1")
tokenizer.save_pretrained("./opt-350m-oasst1")
```
- **Purpose**: Saves the fine-tuned model and tokenizer to the specified directory for future use.
- **Why**: After training, you want to save the model so that it can be reused for inference or further training.
- **Possible Changes**: You can specify different save directories or use `push_to_hub` to directly push the model to the Hugging Face Model Hub.
