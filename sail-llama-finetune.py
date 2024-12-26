import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import os
from huggingface_hub import login, HfFolder
import sys
import json
from torch.utils.data import Dataset
import pandas as pd

class SAILDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        print("Loading dataset...")
        # Load the JSON data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Dataset loaded with {len(self.data)} examples")
        # Print first record structure for debugging
        if len(self.data) > 0:
            print("\nFirst record structure:")
            for k, v in self.data[0].items():
                print(f"{k}: {type(v)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert inputs to string if it's a dictionary or list
        inputs = item['inputs']
        if isinstance(inputs, (dict, list)):
            inputs = json.dumps(inputs)
        
        # Format the text in a structured way
        text = f"""Type: {item['type']}
Code with QRT: {item['code_with_qrt']}
Name: {item['name']}
Instruction: {item['instruction']}
Inputs: {inputs}
Code: {item['code']}
Schema: {item['reduced_schema']}"""
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Remove the batch dimension and return
        return {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0]
        }

def setup_auth():
    """Setup authentication for Hugging Face and verify access"""
    print("Setting up Hugging Face authentication...")
    
    # First check if already logged in
    if HfFolder.get_token() is not None:
        print("Found existing Hugging Face token")
        
        # Verify access to the model
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                filename="config.json",
                repo_type="model"
            )
            print("Successfully verified access to Meta Llama 3 model!")
            return
        except Exception as e:
            print("\nError: Could not access Meta Llama 3 model.")
            print("Please make sure you have been granted access at: https://huggingface.co/meta-llama")
            print("You need to:")
            print("1. Request access at https://ai.meta.com/resources/models-and-libraries/llama-downloads/")
            print("2. Link your approval to Hugging Face at https://huggingface.co/meta-llama")
            print(f"\nError details: {str(e)}")
            sys.exit(1)
    
    # Check environment variable
    token = os.environ.get('HF_TOKEN')
    
    if not token:
        print("\nPlease enter your Hugging Face token.")
        print("You can find your token at: https://huggingface.co/settings/tokens")
        token = input("Token: ").strip()
    
    try:
        # Login to Hugging Face
        login(token)
        print("Successfully authenticated with Hugging Face!")
            
    except Exception as e:
        print(f"Error during authentication: {str(e)}")
        sys.exit(1)

def prepare_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    print(f"Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. Requested access at https://ai.meta.com/resources/models-and-libraries/llama-downloads/")
        print("2. Linked your approval to Hugging Face at https://huggingface.co/meta-llama")
        print("3. Logged in with the correct Hugging Face token")
        sys.exit(1)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Setup authentication first
    setup_auth()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model()
    
    # Create dataset using custom Dataset class
    dataset = SAILDataset("sail_code_samples.json", tokenizer)
    
    # Split into train/validation (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    training_args = TrainingArguments(
        output_dir="./sail-llama-finetuned",
        num_train_epochs=2,  # was 3
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=0.3,
        torch_compile=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train()
