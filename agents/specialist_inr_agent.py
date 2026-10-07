"""
The INR Specialist served locally: Llama 3.2 3B in 4-bit on your GPU plus the LoRA adapter
trained by scripts/train_specialist_inr.py (models/specialist_inr/). The USD Specialist on
Modal (agents/specialist_agent.py, pricer_service.py) is unchanged and used only in
usd_legacy mode.

It is loaded by the Ensemble Agent only when ensemble.inr.specialist in settings.yaml is
above 0. It needs a CUDA GPU (about 2.5 GB of memory) and the Hugging Face licence for
meta-llama/Llama-3.2-3B.
"""

import re
from typing import Optional

from agents.agent import Agent
from agents.items import inr_prompt

ADAPTER = "models/specialist_inr"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_rupees(answer: str) -> Optional[float]:
    """The first number in the model's answer ("17249\\n\\nSamsung..." -> 17249.0)."""
    match = _NUMBER.search((answer or "").split("\n")[0])
    return float(match.group().replace(",", "")) if match else None


class SpecialistInrAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self, adapter: str = ADAPTER, base_model: str = BASE_MODEL, model=None, tokenizer=None):
        self.log("Specialist Agent is loading Llama 3.2 3B (4-bit) with the INR adapter")
        if model is None:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.pad_token = tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant, device_map="cuda:0")
            model = PeftModel.from_pretrained(base, adapter)
            model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.log("Specialist Agent is ready")

    def price(self, description: str) -> Optional[float]:
        """Estimate the typical Indian selling price in rupees, or None if the answer has no number."""
        import torch

        batch = self.tokenizer(inr_prompt(description[:400]), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=8, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        result = parse_rupees(answer)
        self.log(f"Specialist Agent estimates Rs {result:,.0f}" if result else "Specialist Agent gave no number")
        return result
