# llamasail
Finetune Llama 3 with Appian SAIL code (~12000 items) and run locally with a single Nvidia GPU with 24G VRAM.

Fine-tuning Technology:

QLoRA (4-bit quantization with LoRA), final GGUF model is around 16GB in size.</B>
Base model: Meta-Llama-3-8B-Instruct
LoRA config: rank=4, alpha=16, targeting attention layers
Training: 2 epochs, batch size=1, gradient accumulation=32

Merging & Conversion Process:

Merge: PEFT to combine LoRA weights with base model
Convert: llama.cpp tools to create GGUF format
Output: f16 precision GGUF file for local inference

Notable Features:

Memory-efficient with CPU merging and garbage collection
Custom dataset handler for structured code data
Final output compatible with LLMStudio and llama.cpp
