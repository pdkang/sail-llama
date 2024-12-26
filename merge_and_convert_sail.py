import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os
import subprocess
import sys
import shutil
import gc

def merge_lora_weights(
    base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    lora_weights_path="./sail-llama-finetuned",
    output_dir="./sail-finetuned-merged"
):
    print(f"Loading fine-tuned model from {lora_weights_path}")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Load model on CPU
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_weights_path,
            device_map="cpu",  # Force CPU
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        # Clear memory after merging
        del model
        gc.collect()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        print("Loading tokenizer from base model")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)
        
        # Clear memory after saving
        del merged_model
        gc.collect()
        
        print(f"Merge completed successfully!")
        return output_dir
        
    except Exception as e:
        print(f"Error during merging: {str(e)}")
        gc.collect()
        sys.exit(1)

def setup_llamacpp():
    """Setup llama.cpp repository"""
    print("\nSetting up llama.cpp...")
    
    if os.path.exists("llama.cpp"):
        shutil.rmtree("llama.cpp")
    
    subprocess.run([
        "git", "clone",
        "https://github.com/ggerganov/llama.cpp.git"
    ], check=True)
    
    os.chdir("llama.cpp")
    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    
    print("Building llama.cpp...")
    subprocess.run(["cmake", ".."], check=True)
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], check=True)
    
    os.chdir("../..")

def convert_to_gguf(merged_model_path, output_name="sail-merged.gguf", outtype="f16"):
    """Convert the merged model to GGUF format"""
    print(f"\nConverting merged model to GGUF format with {outtype} precision...")
    
    try:
        subprocess.run([
            "python", 
            "llama.cpp/convert_hf_to_gguf.py",
            merged_model_path,
            "--outfile", output_name,
            "--outtype", outtype
        ], check=True)
        
        print(f"Conversion complete! Model saved as: {output_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def main():
    try:
        # Step 1: Merge LoRA weights
        print("Step 1: Merging LoRA weights with base model")
        merged_model_path = merge_lora_weights(
            base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_weights_path="./sail-llama-finetuned",
            output_dir="./sail-finetuned-merged"
        )
        
        # Step 2: Setup llama.cpp
        print("\nStep 2: Setting up llama.cpp")
        setup_llamacpp()
        
        # Step 3: Convert to GGUF
        print("\nStep 3: Converting to GGUF format")
        convert_to_gguf(merged_model_path, "sail-merged-f16.gguf", "f16")
        
        print("\nProcess completed successfully!")
        print("Created model version:")
        print("1. sail-merged-f16.gguf (16-bit precision)")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        gc.collect()
        sys.exit(1)

if __name__ == "__main__":
    main()
