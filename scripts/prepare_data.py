"""
Dataset Preparation for Toroidal Attention Training

Prepares two types of datasets:
1. Synthetic periodic data - to validate circular wrapping benefits
2. OpenWebText subsets - for realistic language modeling

Creates 128-token windows with overlap for efficient training.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class PeriodicSequenceDataset(Dataset):
    """
    Synthetic dataset with periodic token sequences.

    Generates sequences with repeating patterns to test toroidal attention's
    ability to exploit periodicity. The circular wrapping should help the model
    recognize these patterns more efficiently than linear attention.

    Args:
        vocab_size (int): Size of vocabulary
        seq_len (int): Sequence length
        period (int): Repetition period
        n_samples (int): Number of samples to generate
        noise_prob (float): Probability of random token (adds noise)
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        seq_len: int = 128,
        period: int = 32,
        n_samples: int = 1000,
        noise_prob: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.period = period
        self.n_samples = n_samples
        self.noise_prob = noise_prob

        # Generate samples
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[torch.Tensor]:
        """Generate periodic sequences."""
        samples = []

        for _ in range(self.n_samples):
            # Create a base pattern
            base_pattern = torch.randint(0, self.vocab_size, (self.period,))

            # Repeat pattern to fill sequence
            n_repeats = (self.seq_len + self.period - 1) // self.period
            sequence = base_pattern.repeat(n_repeats)[:self.seq_len]

            # Add structured noise but preserve periodicity inside periods
            if self.noise_prob > 0:
                # Flip a small subset of entire period blocks to maintain high intra-period similarity
                n_blocks = max(1, self.seq_len // self.period)
                flip_mask = torch.rand(n_blocks) < (self.noise_prob * 0.3)
                for b in torch.nonzero(flip_mask, as_tuple=False).flatten():
                    start = int(b) * self.period
                    end = min(start + self.period, self.seq_len)
                    sequence[start:end] = torch.randint(0, self.vocab_size, (end - start,))
                # Add token-level noise at lower rate for realism
                token_noise_prob = self.noise_prob * 0.2
                if token_noise_prob > 0:
                    noise_mask = torch.rand(self.seq_len) < token_noise_prob
                    noise_tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
                    sequence = torch.where(noise_mask, noise_tokens, sequence)

            samples.append(sequence)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.samples[idx]

        # Create input and target (shifted by 1 for language modeling)
        input_ids = sequence[:-1]
        labels = sequence[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class SinusoidalDataset(Dataset):
    """
    Dataset based on quantized sinusoidal functions.

    Maps continuous sinusoidal values to discrete tokens, creating
    naturally periodic sequences that toroidal attention should handle well.
    """

    def __init__(
        self,
        vocab_size: int = 256,  # Use smaller vocab for cleaner mapping
        seq_len: int = 128,
        n_samples: int = 1000,
        n_frequencies: int = 3,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.n_frequencies = n_frequencies

        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[torch.Tensor]:
        """Generate sinusoidal sequences."""
        samples = []

        for _ in range(self.n_samples):
            # Random frequency and phase
            freq = np.random.uniform(0.5, 3.0, size=self.n_frequencies)
            phase = np.random.uniform(0, 2*np.pi, size=self.n_frequencies)

            # Generate positions
            t = np.linspace(0, 4*np.pi, self.seq_len)

            # Sum sinusoids
            signal = np.zeros(self.seq_len)
            for f, p in zip(freq, phase):
                signal += np.sin(f * t + p)

            # Normalize to [0, 1]
            signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

            # Quantize to vocabulary
            tokens = (signal * (self.vocab_size - 1)).astype(np.int64)
            sequence = torch.from_numpy(tokens)

            samples.append(sequence)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.samples[idx]

        input_ids = sequence[:-1]
        labels = sequence[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class OpenWebTextDataset(Dataset):
    """
    OpenWebText dataset with fixed-length windows.

    Loads OpenWebText from Hugging Face and creates 128-token windows
    with 50% overlap for efficient training on realistic text.

    Args:
        tokenizer: Hugging Face tokenizer
        seq_len (int): Sequence length (default 128)
        n_samples (int): Number of samples to create
        split (str): Dataset split ('train' or 'validation')
        overlap (float): Overlap ratio between windows (0 to 1)
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 128,
        n_samples: int = 10000,
        split: str = 'train',
        overlap: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.split = split
        self.overlap = overlap
        self.stride = int(seq_len * (1 - overlap))

        print(f"Loading OpenWebText ({split})...")
        self.dataset = load_dataset('openwebtext', split=split, streaming=True)

        print(f"Creating {n_samples} samples with seq_len={seq_len}, stride={self.stride}...")
        self.samples = self._create_samples()

    def _create_samples(self) -> List[torch.Tensor]:
        """Create fixed-length samples from OpenWebText."""
        samples = []

        iterator = iter(self.dataset)
        pbar = tqdm(total=self.n_samples, desc="Tokenizing")

        current_tokens = []

        while len(samples) < self.n_samples:
            try:
                # Get next text
                example = next(iterator)
                text = example['text']

                # Tokenize
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                current_tokens.extend(tokens)

                # Extract windows
                while len(current_tokens) >= self.seq_len + 1:  # +1 for target
                    window = current_tokens[:self.seq_len + 1]
                    samples.append(torch.tensor(window, dtype=torch.long))
                    current_tokens = current_tokens[self.stride:]  # Slide window

                    pbar.update(1)

                    if len(samples) >= self.n_samples:
                        break

            except StopIteration:
                print("Reached end of dataset")
                break

        pbar.close()

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.samples[idx]

        # Split into input and target
        input_ids = sequence[:-1]  # All but last token
        labels = sequence[1:]      # All but first token

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def create_dataloaders(
    dataset_type: str,
    tokenizer,
    seq_len: int = 128,
    batch_size: int = 8,
    n_train: int = 1000,
    n_val: int = 200,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        dataset_type (str): 'periodic', 'sinusoidal', or 'openwebtext'
        tokenizer: Tokenizer (required for openwebtext)
        seq_len (int): Sequence length
        batch_size (int): Batch size
        n_train (int): Number of training samples
        n_val (int): Number of validation samples
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"\nCreating {dataset_type} dataloaders...")

    if dataset_type == 'periodic':
        train_dataset = PeriodicSequenceDataset(
            seq_len=seq_len,
            n_samples=n_train,
            **dataset_kwargs
        )
        val_dataset = PeriodicSequenceDataset(
            seq_len=seq_len,
            n_samples=n_val,
            **dataset_kwargs
        )

    elif dataset_type == 'sinusoidal':
        train_dataset = SinusoidalDataset(
            seq_len=seq_len,
            n_samples=n_train,
            **dataset_kwargs
        )
        val_dataset = SinusoidalDataset(
            seq_len=seq_len,
            n_samples=n_val,
            **dataset_kwargs
        )

    elif dataset_type == 'openwebtext':
        train_dataset = OpenWebTextDataset(
            tokenizer=tokenizer,
            seq_len=seq_len,
            n_samples=n_train,
            split='train',
            **dataset_kwargs
        )
        # For validation, use a subset (OpenWebText doesn't have official val split)
        val_dataset = OpenWebTextDataset(
            tokenizer=tokenizer,
            seq_len=seq_len,
            n_samples=n_val,
            split='train',  # Use train split but different samples
            **dataset_kwargs
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simpler debugging
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


def save_dataset_samples(dataset: Dataset, save_path: Path, n_samples: int = 10):
    """Save sample sequences for inspection."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        samples.append({
            'input_ids': sample['input_ids'].tolist(),
            'labels': sample['labels'].tolist(),
        })

    with open(save_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {save_path}")


def main():
    """Demo of dataset creation."""

    print("=" * 60)
    print("Toroidal Attention Dataset Preparation Demo")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test periodic dataset
    print("\n[1/3] Periodic Dataset")
    print("-" * 60)
    train_loader, val_loader = create_dataloaders(
        dataset_type='periodic',
        tokenizer=tokenizer,
        seq_len=128,
        batch_size=4,
        n_train=100,
        n_val=20,
        period=32,
        noise_prob=0.1,
    )

    # Show sample
    batch = next(iter(train_loader))
    print(f"Sample batch shape: {batch['input_ids'].shape}")
    print(f"First sequence (first 32 tokens): {batch['input_ids'][0, :32].tolist()}")

    # Test sinusoidal dataset
    print("\n[2/3] Sinusoidal Dataset")
    print("-" * 60)
    train_loader, val_loader = create_dataloaders(
        dataset_type='sinusoidal',
        tokenizer=tokenizer,
        seq_len=128,
        batch_size=4,
        n_train=100,
        n_val=20,
    )

    batch = next(iter(train_loader))
    print(f"Sample batch shape: {batch['input_ids'].shape}")

    # Test OpenWebText (commented out as it requires download)
    # print("\n[3/3] OpenWebText Dataset")
    # print("-" * 60)
    # train_loader, val_loader = create_dataloaders(
    #     dataset_type='openwebtext',
    #     tokenizer=tokenizer,
    #     seq_len=128,
    #     batch_size=4,
    #     n_train=100,
    #     n_val=20,
    # )

    print("\n✓ Dataset preparation demo complete!")


if __name__ == "__main__":
    main()

