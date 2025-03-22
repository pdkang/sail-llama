# sail-llama
Finetune Llama 3 with Appian SAIL code (~12000 items) and run locally with a single Nvidia GPU with 24G VRAM.

Fine-tuning Technology in brief:

QLoRA (4-bit quantization with LoRA), final GGUF model is around 16GB in size<br>
Base model: Meta-Llama-3-8B-Instruct<br>
LoRA config: rank=4, alpha=16, targeting attention layers<br>
Training: 2 epochs, batch size=1, gradient accumulation=32<br>

Merging & Conversion Process:

Merge: PEFT to combine LoRA weights with base model<br>
Convert: llama.cpp tools to create GGUF format<br>
Output: f16 precision GGUF file for local inference<br>

Notable Features:

Memory-efficient with CPU merging and garbage collection<br>
Custom dataset handler for structured code data<br>
Final output compatible with LMStudio and llama.cpp<br>

===============================================================================
Finetuning Process Overview

The process consists of two main phases:
1. Finetuning with LoRA (in sail-llama-finetune.py)
2. Merging and Converting (in merge_and_convert_sail.py)
   
Phase 1: Finetuning with LoRA
The sail-llama-finetune.py script handles the actual finetuning process:

1. Authentication Setup: Verifies access to the Meta Llama 3 model on Hugging Face
2. Model Preparation: Loads the base model with 4-bit quantization
3. LoRA Configuration: Sets up Low-Rank Adaptation parameters
4. Dataset Preparation: Loads and processes the SAIL code samples
5. Training: Finetunes the model using the Hugging Face Trainer API
6. Saving: Saves the resulting LoRA weights

Phase 2: Merging and Converting
The merge_and_convert_sail.py script takes the finetuned LoRA weights and:
1. Merges LoRA Weights: Combines the LoRA weights with the base model
2. Sets up llama.cpp: Clones and builds the llama.cpp repository
3. Converts to GGUF: Transforms the merged model to GGUF format for efficient inference


Technologies Used

The finetuning process leverages several key technologies:

1. Parameter-Efficient Fine-Tuning (PEFT)
- LoRA (Low-Rank Adaptation): Instead of updating all model parameters, LoRA adds small trainable "adapter" layers to specific attention components (q_proj, k_proj, v_proj, o_proj)
- The configuration uses rank=4, alpha=16, and dropout=0.05

2. Quantization
- 4-bit Quantization: Uses the BitsAndBytes library to reduce memory requirements
- Specifically uses NF4 (normalized float 4) quantization with double quantization

3. Training Framework
- Hugging Face Transformers: For model loading and training infrastructure
- PyTorch: As the underlying deep learning framework
- Gradient Checkpointing: To reduce memory usage during training
- Mixed Precision Training (fp16): For faster training

4. Optimization Parameters
- Learning Rate: 2e-4 with cosine scheduler
- Batch Size: 1 per device with gradient accumulation of 32 steps
- Training Duration: 2 epochs
- Optimizer: AdamW with beta1=0.9, beta2=0.999, epsilon=1e-8
- Gradient Clipping: 0.3 max norm

5. Model Conversion
- llama.cpp: For efficient inference on consumer hardware
- GGUF Format: The modern replacement for GGML format, allowing for optimized inference
