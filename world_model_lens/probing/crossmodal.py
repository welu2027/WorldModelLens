"""Cross-modal probing for vision-language world models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CrossModalResult:
    """Result of cross-modal probing."""

    alignment_score: float
    shared_concepts: List[str]
    retrieval_accuracy: float


class CrossModalProber:
    """Probe cross-modal representations in world models."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize cross-modal prober."""
        self.device = device or _get_device()
        self.clip_model = None
        self.clip_processor = None

    def load_clip(self) -> None:
        """Load CLIP model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for CrossModalProber")

        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            self.clip_model.to(self.device)

    def align_vision_language(
        self,
        vm_latents: torch.Tensor,
        images: torch.Tensor,
        captions: List[str],
    ) -> float:
        """Compute alignment between world model latents and CLIP features.

        Args:
            vm_latents: World model latent tensor [N, D].
            images: Image tensor [N, C, H, W].
            captions: List of caption strings.

        Returns:
            Alignment score.
        """
        self.load_clip()

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(images.to(self.device))
            text_inputs = self.clip_processor(text=captions, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)

            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            vm_latents = F.normalize(vm_latents.to(self.device), dim=1)

            image_text_sim = (image_features @ text_features.T).diag()
            vm_text_sim = (vm_latents @ text_features.T).mean(dim=0)

            alignment = (image_text_sim.mean() + vm_text_sim.mean()) / 2

        return float(alignment.item())

    def find_shared_concepts(
        self,
        vm_latents: torch.Tensor,
        images: torch.Tensor,
        concept_candidates: List[str],
    ) -> List[Tuple[str, float]]:
        """Find concepts shared between world model and CLIP.

        Args:
            vm_latents: World model latents [N, D].
            images: Image tensor [N, C, H, W].
            concept_candidates: Candidate concept strings.

        Returns:
            List of (concept, score) sorted by score.
        """
        self.load_clip()

        with torch.no_grad():
            vm_latents = F.normalize(vm_latents.to(self.device), dim=1)

            text_inputs = self.clip_processor(
                text=concept_candidates, return_tensors="pt", padding=True
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=1)

            similarities = vm_latents @ text_features.T
            concept_scores = similarities.mean(dim=0)

            concept_score_list = [
                (concept, float(score.item()))
                for concept, score in zip(concept_candidates, concept_scores)
            ]

        return sorted(concept_score_list, key=lambda x: x[1], reverse=True)

    def crossmodal_retrieval(
        self,
        vm_latents: torch.Tensor,
        text_queries: List[str],
    ) -> List[List[int]]:
        """Perform cross-modal retrieval from latents to text.

        Args:
            vm_latents: World model latents [N, D].
            text_queries: Text query strings.

        Returns:
            List of retrieved indices for each query.
        """
        self.load_clip()

        with torch.no_grad():
            vm_latents = F.normalize(vm_latents.to(self.device), dim=1)

            text_inputs = self.clip_processor(text=text_queries, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=1)

            similarities = vm_latents @ text_features.T
            retrieved_indices = similarities.argmax(dim=0)

        return [[int(idx)] for idx in retrieved_indices]


def align_multimodal(
    vm_latents: torch.Tensor,
    vlm_features: torch.Tensor,
) -> Dict[str, float]:
    """Align world model latents with vision-language model features.

    Args:
        vm_latents: World model latents [N, D].
        vlm_features: VLM features [N, D].

    Returns:
        Alignment metrics.
    """
    vm_norm = F.normalize(vm_latents, dim=1)
    vlm_norm = F.normalize(vlm_features, dim=1)

    cosine_sim = (vm_norm * vlm_norm).sum(dim=1).mean()

    euclidean_dist = (vm_latents - vlm_features).norm(dim=1).mean()

    return {
        "cosine_similarity": float(cosine_sim.item()),
        "euclidean_distance": float(euclidean_dist.item()),
    }
