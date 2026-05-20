"""
Moondream2 VLM client.
Loads the model once, exposes a query(image, prompt) method.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch


class MoondreamClient:
    MODEL_ID = "vikhyatk/moondream2"
    REVISION = "2024-08-26"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MoondreamClient] Loading model on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, revision=self.REVISION
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            revision=self.REVISION,
        ).to(self.device)
        self.model.eval()
        print("[MoondreamClient] Ready.")

    def query(self, pil_image: Image.Image, prompt: str) -> str:
        enc_image = self.model.encode_image(pil_image)
        answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        return answer
