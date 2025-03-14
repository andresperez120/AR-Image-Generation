import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        # Token embedding layer
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Transformer encoder layer with self-attention
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,  # Multi-head attention with 8 heads
            dim_feedforward=512,  # Size of the feedforward network
            dropout=0.1,
            batch_first=True  # Important for our sequence format
        )
        
        # Output projection to token probabilities
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)
        
        # Save parameters for later use
        self.d_latent = d_latent
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) of integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        """
        # Print shape for debugging
        print("Input tensor shape:", x.shape)
        
        # If we get tokenized data from BSQPatchAutoEncoder, it will be (B, h, w)
        # If we get raw image data, we need to tokenize it first
        if len(x.shape) == 4:  # Raw image data (B, H, W, C)
            raise ValueError("Expected tokenized input (B, h, w), got raw image data (B, H, W, C). Please tokenize the data first using BSQPatchAutoEncoder.")
        elif len(x.shape) == 2:  # Single image tokens (h, w)
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Now x should be (B, h, w)
        B, h, w = x.shape
        seq_len = h * w
        
        # Flatten spatial dimensions into sequence
        x_flat = x.reshape(B, -1)  # Shape: (B, seq_len)
        
        # Create input sequence by shifting: [0, x1, x2, ..., xN-1]
        # Target sequence will be: [x1, x2, ..., xN]
        x_input = torch.nn.functional.pad(x_flat[:, :-1], (1, 0), value=0)  # Add start token (0)
        
        # Embed the tokens
        embedded = self.embedding(x_input)  # Shape: (B, seq_len, d_latent)
        
        # Create causal mask to ensure autoregressive property
        # Each position can only attend to previous positions
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Apply transformer with causal mask
        transformed = self.transformer(embedded, src_mask=mask)
        
        # Project to token probabilities
        logits = self.output_proj(transformed)  # Shape: (B, seq_len, n_tokens)
        
        # Reshape to match required output format (B, h, w, n_tokens)
        logits = logits.reshape(B, h, w, -1)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        
        # Initialize sequence with start tokens
        generated = torch.zeros((B, h * w), dtype=torch.long, device=device)
        
        # Generate tokens autoregressively
        for i in range(h * w):
            # Get current sequence
            current_seq = generated[:, :i+1]
            
            # Pad sequence to full length
            x_input = torch.nn.functional.pad(current_seq, (0, h*w - i-1), value=0)
            
            # Embed tokens
            embedded = self.embedding(x_input)
            
            # Create causal mask
            mask = torch.nn.Transformer.generate_square_subsequent_mask(h * w).to(device)
            
            # Get transformer output
            transformed = self.transformer(embedded, src_mask=mask)
            
            # Get next token probabilities (only for current position)
            logits = self.output_proj(transformed[:, i, :])
            
            # Sample next token
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            
            # Add to generated sequence
            generated[:, i] = next_token
        
        # Reshape to image format (B, h, w)
        return generated.reshape(B, h, w)
