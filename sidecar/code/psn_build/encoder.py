# Provenance: sidecar-arc fork sidecar-b (PSN foundation -- Hopfield build). Source: experiments/sidecar_arc/sidecar-b/sidecar-b_encoder.py
"""Text encoder: raw text -> dense embedding vector via sentence-transformers."""

import torch
from transformers import AutoModel, AutoTokenizer
from .config import PSNConfig


class TextEncoder:
    """Wraps all-MiniLM-L6-v2 to produce 384d sentence embeddings.

    Uses raw transformers + mean pooling (no sentence-transformers dependency).
    The model is loaded from the configured hf_cache_dir, or HuggingFace's
    default cache location if hf_cache_dir is not set.
    """

    def __init__(self, config: PSNConfig):
        self.config = config
        self.device = config.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.hf_cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.hf_cache_dir,
        ).to(self.device).eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text string to a 384d embedding.

        Args:
            text: any natural language string

        Returns:
            embedding: [d_embedding] normalized dense vector
        """
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
        token_embeddings = outputs.last_hidden_state  # [1, seq_len, 384]
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts).squeeze(0)  # [384]

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, dim=0)

        return embedding

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode multiple texts.

        Args:
            texts: list of strings

        Returns:
            embeddings: [batch, d_embedding] normalized dense vectors
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts

        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        return embeddings
