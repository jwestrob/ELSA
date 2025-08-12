"""
Protein language model embedding system for ELSA.

Supports ESM2 and ProtT5 models with GPU acceleration and CPU fallback.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Iterator, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import logging
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.console import Console
from enum import Enum

from .params import PLMConfig

logger = logging.getLogger(__name__)
console = Console()


class AggregationStrategy(Enum):
    """Different strategies for aggregating sliding window embeddings."""
    MAX_POOL = "max_pool"          # Element-wise maximum (preserves strongest features)
    MEAN_POOL = "mean_pool"        # Element-wise average (balanced representation)
    WEIGHTED_MEAN = "weighted_mean" # Length-weighted average (longer windows weighted more)
    ATTENTION_POOL = "attention"    # Learned attention weights (future enhancement)


@dataclass
class ProteinSequence:
    """A protein sequence with metadata."""
    sample_id: str
    contig_id: str
    gene_id: str
    start: int
    end: int
    strand: int  # +1 or -1
    sequence: str
    
    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass  
class ProteinEmbedding:
    """A protein embedding with metadata."""
    sample_id: str
    gene_id: str
    embedding: np.ndarray
    sequence_length: int


class DeviceManager:
    """Manages device selection and memory for PLM inference."""
    
    def __init__(self, device_preference: str = "auto"):
        self.device = self._select_device(device_preference)
        self.max_memory_gb = self._get_max_memory()
        
    def _select_device(self, preference: str) -> torch.device:
        """Select optimal device for inference."""
        if preference == "auto":
            if torch.backends.mps.is_available():
                console.print(f"✓ MPS available, using GPU acceleration")
                return torch.device("mps")
            elif torch.cuda.is_available():
                console.print(f"✓ CUDA available, using GPU acceleration")
                return torch.device("cuda")
            else:
                console.print(f"⚠️  Using CPU (no GPU available)")
                return torch.device("cpu")
        else:
            console.print(f"✓ Using specified device: {preference}")
            return torch.device(preference)
    
    def _get_max_memory(self) -> float:
        """Get available memory in GB."""
        if self.device.type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device.type == "mps":
            # M4 Max has unified memory - be conservative
            return 32.0  # Use 32GB of the 48GB to leave room for system
        else:
            return 8.0  # Conservative for CPU
    
    def optimize_batch_size(self, target_aa: int, model_size: str) -> int:
        """Optimize batch size based on available memory and model size."""
        # Rough memory estimates (in GB per 1000 AA)
        memory_per_1k_aa = {
            "esm2_t12": 0.5,   # ESM2-35M (smaller model)
            "esm2_t33": 1.2,   # ESM2-650M  
            "prot_t5": 2.0     # ProtT5-XL
        }
        
        base_memory = memory_per_1k_aa.get(model_size, 1.5)
        max_aa = int((self.max_memory_gb * 0.8) / base_memory * 1000)
        
        # MPS has issues with very large batches - limit to smaller sizes
        if self.device.type == "mps":
            max_aa = min(max_aa, 4000)  # Very conservative limit for MPS
            
        return min(target_aa, max_aa)


class ESM2Embedder:
    """ESM2 protein language model embedder with sliding window support."""
    
    def __init__(self, config: PLMConfig, device_manager: DeviceManager, 
                 window_size: int = 1024, overlap: int = 256, 
                 aggregation: AggregationStrategy = AggregationStrategy.MAX_POOL):
        self.config = config
        self.device_manager = device_manager
        self.window_size = window_size
        self.overlap = overlap
        self.aggregation = aggregation
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load ESM2 model and tokenizer."""
        import esm
        
        model_name = {
            "esm2_t12": "esm2_t12_35M_UR50D",
            "esm2_t33": "esm2_t33_650M_UR50D"
        }.get(self.config.model)
        
        if not model_name:
            raise ValueError(f"Unknown ESM2 model: {self.config.model}")
            
        console.print(f"Loading {model_name} on {self.device_manager.device}...")
        
        # Load model
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.tokenizer = self.alphabet.get_batch_converter()
        
        self.model = self.model.to(self.device_manager.device)
        if self.config.fp16 and self.device_manager.device.type != "cpu":
            self.model = self.model.half()
        
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.embed_dim
        
        console.print(f"✓ Loaded {model_name} (dim={self.embedding_dim})")
        console.print(f"Sliding window: {self.window_size} AA, overlap: {self.overlap} AA")
        console.print(f"Aggregation strategy: {self.aggregation.value}")
    
    def _should_use_sliding_window(self, sequence: str, threshold: int = None) -> bool:
        """Determine if sequence needs sliding window processing."""
        if threshold is None:
            threshold = self.window_size
        # Account for special tokens and safety margin
        effective_limit = threshold - 10  # Safety margin for special tokens
        return len(sequence) > effective_limit
    
    def _embed_long_sequence_with_sliding_window(self, sequence: str, gene_id: str) -> np.ndarray:
        """Process long sequences using sliding window approach with overlap."""
        # 1. Create overlapping windows
        windows = []
        start = 0
        while start < len(sequence):
            end = min(start + self.window_size, len(sequence))
            window = sequence[start:end]
            windows.append(window)
            
            if end == len(sequence):
                break
            start += (self.window_size - self.overlap)
        
        # 2. Generate embeddings for each window
        window_embeddings = []
        window_lengths = []
        
        with torch.no_grad():
            for i, window in enumerate(windows):
                try:
                    # Process individual window
                    batch_data = [(f"{gene_id}_w{i}", window)]
                    batch_labels, batch_strs, batch_tokens = self.tokenizer(batch_data)
                    batch_tokens = batch_tokens.to(self.device_manager.device)
                    
                    if self.config.fp16 and self.device_manager.device.type != "cpu":
                        with torch.autocast(device_type=self.device_manager.device.type):
                            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                    else:
                        results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                    
                    # Extract window embedding
                    representations = results["representations"][self.model.num_layers]
                    seq_repr = representations[0, 1:len(window)+1]  # Skip BOS/EOS
                    window_emb = seq_repr.mean(dim=0).cpu().numpy()
                    
                    window_embeddings.append(window_emb)
                    window_lengths.append(len(window))
                    
                    # Clean up immediately
                    if self.device_manager.device.type in ['cuda', 'mps']:
                        del batch_tokens, results
                        if self.device_manager.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        elif self.device_manager.device.type == 'mps':
                            torch.mps.empty_cache()
                            
                except Exception as e:
                    logger.error(f"Failed to process window {i} of {gene_id}: {e}")
                    # Use zero embedding for failed window
                    window_embeddings.append(np.zeros(self.embedding_dim))
                    window_lengths.append(len(window))
        
        # 3. Aggregate window embeddings using specified strategy
        if not window_embeddings:
            logger.error(f"No valid windows processed for {gene_id}")
            return np.zeros(self.embedding_dim)
            
        if self.aggregation == AggregationStrategy.MAX_POOL:
            # Element-wise maximum preserves strongest features
            aggregated_embedding = np.maximum.reduce(window_embeddings)
        elif self.aggregation == AggregationStrategy.MEAN_POOL:
            # Simple average
            aggregated_embedding = np.mean(window_embeddings, axis=0)
        elif self.aggregation == AggregationStrategy.WEIGHTED_MEAN:
            # Weight by window length
            weights = np.array(window_lengths) / sum(window_lengths)
            aggregated_embedding = np.average(window_embeddings, axis=0, weights=weights)
        else:
            # Default to max pooling
            logger.warning(f"Unknown aggregation strategy {self.aggregation}, using max pooling")
            aggregated_embedding = np.maximum.reduce(window_embeddings)
        
        return aggregated_embedding
    
    def embed_batch(self, sequences: List[ProteinSequence]) -> List[ProteinEmbedding]:
        """Embed a batch of protein sequences with sliding window support."""
        if not sequences:
            return []
        
        # Separate short and long sequences for different processing
        short_sequences = [seq for seq in sequences if not self._should_use_sliding_window(seq.sequence)]
        long_sequences = [seq for seq in sequences if self._should_use_sliding_window(seq.sequence)]
        
        embeddings = []
        
        # Process short sequences normally
        if short_sequences:
            embeddings.extend(self._embed_normal_batch(short_sequences))
        
        # Process long sequences with sliding window
        for seq in long_sequences:
            try:
                aggregated_emb = self._embed_long_sequence_with_sliding_window(seq.sequence, seq.gene_id)
                embedding = ProteinEmbedding(
                    sample_id=seq.sample_id,
                    gene_id=seq.gene_id,
                    embedding=aggregated_emb,
                    sequence_length=seq.length
                )
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed long sequence {seq.gene_id}: {e}")
                # Use zero embedding for failed sequences
                embedding = ProteinEmbedding(
                    sample_id=seq.sample_id,
                    gene_id=seq.gene_id,
                    embedding=np.zeros(self.embedding_dim),
                    sequence_length=seq.length
                )
                embeddings.append(embedding)
        
        return embeddings
    
    def _embed_normal_batch(self, sequences: List[ProteinSequence]) -> List[ProteinEmbedding]:
        """Embed a batch of normal-length sequences."""
        # Prepare batch data (no truncation needed for short sequences)
        batch_data = [(seq.gene_id, seq.sequence) for seq in sequences]
        
        batch_labels, batch_strs, batch_tokens = self.tokenizer(batch_data)
        batch_tokens = batch_tokens.to(self.device_manager.device)
        
        # Forward pass
        with torch.no_grad():
            if self.config.fp16 and self.device_manager.device.type != "cpu":
                with torch.autocast(device_type=self.device_manager.device.type):
                    results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            else:
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
        
        # Extract embeddings (mean pooling over sequence length)
        embeddings = []
        representations = results["representations"][self.model.num_layers]
        
        for i, seq in enumerate(sequences):
            # Skip BOS/EOS tokens (positions 1:-1)
            seq_repr = representations[i, 1:len(seq.sequence)+1]
            
            # Mean pooling
            pooled = seq_repr.mean(dim=0).cpu().numpy()
            
            embedding = ProteinEmbedding(
                sample_id=seq.sample_id,
                gene_id=seq.gene_id,
                embedding=pooled,
                sequence_length=seq.length
            )
            embeddings.append(embedding)
        
        return embeddings


class ProtT5Embedder:
    """ProtT5 protein language model embedder."""
    
    def __init__(self, config: PLMConfig, device_manager: DeviceManager):
        self.config = config
        self.device_manager = device_manager
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load ProtT5 model and tokenizer."""
        from transformers import T5Tokenizer, T5EncoderModel
        
        model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        
        console.print(f"Loading ProtT5 on {self.device_manager.device}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name)
        
        self.model = self.model.to(self.device_manager.device)
        if self.config.fp16 and self.device_manager.device.type != "cpu":
            self.model = self.model.half()
            
        self.model.eval()
        
        # Get embedding dimension  
        self.embedding_dim = self.model.config.d_model
        
        console.print(f"✓ Loaded ProtT5 (dim={self.embedding_dim})")
    
    def embed_batch(self, sequences: List[ProteinSequence]) -> List[ProteinEmbedding]:
        """Embed a batch of protein sequences."""
        if not sequences:
            return []
            
        # Prepare sequences (ProtT5 expects space-separated amino acids)
        batch_sequences = [" ".join(seq.sequence) for seq in sequences]
        
        # Tokenize
        inputs = self.tokenizer(
            batch_sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device_manager.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            if self.config.fp16 and self.device_manager.device.type != "cpu":
                with torch.autocast(device_type=self.device_manager.device.type):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # Extract embeddings (mean pooling over sequence length)
        embeddings = []
        hidden_states = outputs.last_hidden_state
        
        for i, seq in enumerate(sequences):
            # Get attention mask for proper pooling
            attention_mask = inputs["attention_mask"][i]
            seq_hidden = hidden_states[i][attention_mask.bool()]
            
            # Mean pooling
            pooled = seq_hidden.mean(dim=0).cpu().numpy()
            
            embedding = ProteinEmbedding(
                sample_id=seq.sample_id,
                gene_id=seq.gene_id,
                embedding=pooled,
                sequence_length=seq.length
            )
            embeddings.append(embedding)
        
        return embeddings


class ProteinEmbedder:
    """Main protein embedding interface with model switching."""
    
    def __init__(self, config: PLMConfig, window_size: int = 1024, overlap: int = 256,
                 aggregation: AggregationStrategy = AggregationStrategy.MAX_POOL):
        self.config = config
        self.device_manager = DeviceManager(config.device)
        
        # CLEAR GPU/CPU STATUS
        device_type = self.device_manager.device.type.upper()
        if device_type == "MPS":
            console.print(f"🚀 [bold green]USING GPU (Apple Metal)[/bold green]")
        elif device_type == "CUDA":
            console.print(f"🚀 [bold green]USING GPU (NVIDIA CUDA)[/bold green]")
        else:
            console.print(f"🐌 [bold red]USING CPU (SLOW!)[/bold red]")
        
        # Initialize the appropriate embedder
        if config.model.startswith("esm2"):
            self.embedder = ESM2Embedder(config, self.device_manager, window_size, overlap, aggregation)
        elif config.model == "prot_t5":
            self.embedder = ProtT5Embedder(config, self.device_manager)
        else:
            raise ValueError(f"Unsupported model: {config.model}")
        
        # Optimize batch size (more conservative for sliding window)
        base_batch_size = self.device_manager.optimize_batch_size(
            config.batch_amino_acids, config.model
        )
        # Reduce batch size to account for sliding window memory overhead
        self.batch_size_aa = max(1000, base_batch_size // 2)
        
        console.print(f"Optimized batch size: {self.batch_size_aa} amino acids (sliding window aware)")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedder.embedding_dim
    
    def create_batches(self, sequences: List[ProteinSequence]) -> Iterator[List[ProteinSequence]]:
        """Create batches based on amino acid count."""
        current_batch = []
        current_aa = 0
        
        for seq in sequences:
            if current_aa + seq.length > self.batch_size_aa and current_batch:
                yield current_batch
                current_batch = [seq]
                current_aa = seq.length
            else:
                current_batch.append(seq)
                current_aa += seq.length
        
        if current_batch:
            yield current_batch
    
    def embed_sequences(self, sequences: List[ProteinSequence]) -> Iterator[ProteinEmbedding]:
        """Embed protein sequences in optimized batches."""
        total_sequences = len(sequences)
        total_aa = sum(seq.length for seq in sequences)
        
        console.print(f"Embedding {total_sequences:,} proteins ({total_aa:,} amino acids)")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[blue]{task.fields[aa_per_sec]} AA/sec"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Embedding proteins...", 
                total=total_sequences,
                aa_per_sec=0
            )
            
            processed_aa = 0
            
            for batch in self.create_batches(sequences):
                batch_aa = sum(seq.length for seq in batch)
                
                # Embed batch
                embeddings = self.embedder.embed_batch(batch)
                
                # Yield embeddings
                for embedding in embeddings:
                    yield embedding
                
                # Update progress
                processed_aa += batch_aa
                aa_per_sec = int(processed_aa / (progress.get_time() + 1e-6))
                
                progress.update(
                    task, 
                    advance=len(batch),
                    aa_per_sec=f"{aa_per_sec:,}"
                )


def parse_fasta_to_proteins(fasta_path: Path, sample_id: str) -> List[ProteinSequence]:
    """Parse FASTA file and extract protein sequences."""
    proteins = []
    
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        # Simple gene ID from FASTA header
        gene_id = f"{sample_id}_{record.id}"
        
        # For now, assume all sequences are on positive strand
        # In real implementation, this would come from GFF or gene calling
        protein = ProteinSequence(
            sample_id=sample_id,
            contig_id=record.id,
            gene_id=gene_id,
            start=0,  # Placeholder
            end=len(record.seq),  # Placeholder
            strand=1,
            sequence=str(record.seq)
        )
        proteins.append(protein)
    
    return proteins


if __name__ == "__main__":
    # Test embedding functionality
    from .params import ELSAConfig
    
    config = ELSAConfig()
    embedder = ProteinEmbedder(config.plm)
    
    print(f"Device: {embedder.device_manager.device}")
    print(f"Max memory: {embedder.device_manager.max_memory_gb:.1f} GB")
    print(f"Embedding dim: {embedder.embedding_dim}")