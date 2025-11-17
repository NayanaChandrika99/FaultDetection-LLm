"""
LLM Setup and Configuration
Loads LLaMA-3-8B-Instruct with QLoRA for explanation generation.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict
import warnings


def setup_llm(
    model_name: str = "meta-llama/Llama-3-8B-Instruct",
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto",
    use_cache: bool = True
):
    """
    Setup LLM with QLoRA configuration.
    
    Args:
        model_name: HuggingFace model identifier
        load_in_4bit: Use 4-bit quantization (default: True)
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        device_map: Device mapping strategy
        use_cache: Use KV cache for generation
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading LLM: {model_name}")
    print(f"4-bit quantization: {load_in_4bit}")
    
    # Configure quantization
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=use_cache,
    )
    
    # Configure LoRA (for potential fine-tuning)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print(f"Model loaded successfully")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    return model, tokenizer, lora_config


def generate_explanation(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True
) -> str:
    """
    Generate explanation using the LLM.
    
    Args:
        model: Loaded language model
        tokenizer: Tokenizer
        system_prompt: System instruction
        user_prompt: User query/data
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Use sampling (vs greedy)
    
    Returns:
        Generated text string
    """
    # Format prompt for chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


class LLMExplainer:
    """
    Wrapper class for LLM-based explanation generation.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        load_in_4bit: bool = True,
        temperature: float = 0.8,
        max_new_tokens: int = 512
    ):
        """
        Initialize LLM explainer.
        
        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Use 4-bit quantization
            temperature: Generation temperature
            max_new_tokens: Max tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.lora_config = setup_llm(
            model_name=model_name,
            load_in_4bit=load_in_4bit
        )
    
    def explain(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate explanation.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Optional override temperature
        
        Returns:
            Generated explanation string
        """
        temp = temperature if temperature is not None else self.temperature
        
        return generate_explanation(
            model=self.model,
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temp,
            max_new_tokens=self.max_new_tokens
        )
    
    def free_memory(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
        print("Model memory freed")


def estimate_memory_requirements(load_in_4bit: bool = True) -> Dict[str, str]:
    """
    Estimate GPU memory requirements.
    
    Args:
        load_in_4bit: Whether using 4-bit quantization
    
    Returns:
        Dict with memory estimates
    """
    if load_in_4bit:
        return {
            'model_size': '~4-5 GB (4-bit)',
            'generation_overhead': '~2-3 GB',
            'total_recommended': '8-10 GB',
            'min_gpu': 'T4 16GB / RTX 3060 12GB',
        }
    else:
        return {
            'model_size': '~16 GB (fp16)',
            'generation_overhead': '~4-6 GB',
            'total_recommended': '24+ GB',
            'min_gpu': 'A100 40GB / RTX 4090 24GB',
        }

