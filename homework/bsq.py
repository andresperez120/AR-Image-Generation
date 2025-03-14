import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        # Create layers for projection
        self.project_down = torch.nn.Linear(embedding_dim, codebook_bits)
        self.project_up = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Preserve original shape
        original_shape = x.shape
        # Reshape to 2D for linear layer
        x = x.reshape(-1, original_shape[-1])
        # Project down
        x = self.project_down(x)
        # L2 normalize
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
        # Convert to -1/1 using differentiable sign
        x = diff_sign(x)
        # Restore original shape with new feature dimension
        return x.reshape(*original_shape[:-1], self._codebook_bits)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        # Preserve original shape
        original_shape = x.shape
        # Reshape to 2D for linear layer
        x = x.reshape(-1, self._codebook_bits)
        # Project up
        x = self.project_up(x)
        # Restore original shape
        return x.reshape(*original_shape[:-1], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (2 ** torch.arange(x.size(-1), device=x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits, device=x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First encode the image using the parent's encoder
        features = super().encode(x)
        # Then convert to indices using BSQ
        return self.bsq.encode_index(features)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # Convert indices back to features using BSQ
        decoded_features = self.bsq.decode_index(x)
        # Then decode using parent's decoder
        return super().decode(decoded_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # First encode using parent's encoder
        features = super().encode(x)
        # Then apply BSQ encoding
        return self.bsq.encode(features)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # First decode BSQ features
        decoded_features = self.bsq.decode(x)
        # Then decode using parent's decoder
        return super().decode(decoded_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        """
        # First get reconstructed image
        reconstructed = self.decode(self.encode(x))
        
        # Then compute codebook usage statistics
        with torch.no_grad():  # Don't track gradients for monitoring
            indices = self.encode_index(x)
            cnt = torch.bincount(indices.flatten(), minlength=2**self.codebook_bits)
        
        # Return reconstruction and monitoring info
        return reconstructed, {
            "cb0": (cnt == 0).float().mean().detach(),  # unused codes
            "cb2": (cnt <= 2).float().mean().detach(),  # rarely used codes
            "cb10": (cnt <= 10).float().mean().detach(),  # moderately used codes
        }
