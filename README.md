# llamasail
Finetune Llama 3 with Appian SAIL code (~12000 items) and run locally with a single Nvidia GPU with 24G VRAM.

Fine-tuning Technology:

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
Final output compatible with LLMStudio and llama.cpp<br>
