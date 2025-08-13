"""
MGE (Mobile Genetic Element) masking for weighted sketching.

Identifies and masks codewords associated with repetitive/mobile genetic elements
based on PFAM domain annotations.
"""

import yaml
import re
from typing import Set, List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MGEPattern:
    """Pattern for identifying MGE-associated PFAM domains."""
    pattern: str
    pattern_type: str  # 'exact', 'prefix', 'regex'
    description: str
    category: str  # 'transposon', 'integrase', 'phage', 'plasmid', etc.


class MGEMask:
    """
    Manages masking of MGE-associated codewords based on PFAM domain patterns.
    """
    
    def __init__(self, patterns: List[MGEPattern] = None):
        """
        Initialize MGE mask.
        
        Args:
            patterns: List of MGE patterns to match against
        """
        self.patterns = patterns or self._get_default_patterns()
        self.compiled_regexes = {}
        
        # Compile regex patterns for efficiency
        for pattern in self.patterns:
            if pattern.pattern_type == 'regex':
                try:
                    self.compiled_regexes[pattern.pattern] = re.compile(pattern.pattern, re.IGNORECASE)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern.pattern}': {e}")
    
    def _get_default_patterns(self) -> List[MGEPattern]:
        """Get default set of MGE-associated PFAM patterns."""
        return [
            # Transposases and insertion sequences
            MGEPattern("Transposase", "prefix", "Transposase families", "transposon"),
            MGEPattern("DDE_Tnp", "prefix", "DDE transposase superfamily", "transposon"),
            MGEPattern("IS", "prefix", "Insertion sequence families", "transposon"),
            MGEPattern("Tn", "prefix", "Transposon families", "transposon"),
            
            # Integrases
            MGEPattern("Integrase", "prefix", "Integrase families", "integrase"),
            MGEPattern("Phage_integrase", "exact", "Phage integrase", "phage"),
            MGEPattern("Recombinase", "prefix", "Site-specific recombinases", "integrase"),
            
            # Phage structural proteins
            MGEPattern("Phage", "prefix", "Phage-related proteins", "phage"),
            MGEPattern("Tail_", "prefix", "Phage tail proteins", "phage"),
            MGEPattern("Head_", "prefix", "Phage head proteins", "phage"),
            MGEPattern("Portal", "prefix", "Phage portal proteins", "phage"),
            MGEPattern("Terminase", "prefix", "Phage terminases", "phage"),
            
            # Plasmid-associated
            MGEPattern("Rep_", "prefix", "Replication proteins", "plasmid"),
            MGEPattern("Mob", "prefix", "Mobilization proteins", "plasmid"),
            MGEPattern("TraG", "exact", "Conjugal transfer protein", "plasmid"),
            
            # CRISPR and defense systems
            MGEPattern("Cas", "prefix", "CRISPR-associated proteins", "defense"),
            MGEPattern("CRISPR", "prefix", "CRISPR-related", "defense"),
            
            # Restriction-modification systems
            MGEPattern("Methylase", "prefix", "Methyltransferases", "defense"),
            MGEPattern("Endonuclease", "prefix", "Restriction endonucleases", "defense"),
            
            # Generic mobile element patterns (regex)
            MGEPattern(r".*[Tt]ranspos.*", "regex", "General transposition", "transposon"),
            MGEPattern(r".*[Mm]obile.*", "regex", "Mobile genetic elements", "mobile"),
            MGEPattern(r".*[Ii]nsertion.*", "regex", "Insertion elements", "transposon"),
        ]
    
    def matches_mge_pattern(self, pfam_domain: str) -> Optional[MGEPattern]:
        """
        Check if a PFAM domain matches any MGE pattern.
        
        Args:
            pfam_domain: PFAM domain identifier or description
            
        Returns:
            Matching MGEPattern if found, None otherwise
        """
        for pattern in self.patterns:
            if pattern.pattern_type == 'exact':
                if pattern.pattern.lower() == pfam_domain.lower():
                    return pattern
            
            elif pattern.pattern_type == 'prefix':
                if pfam_domain.lower().startswith(pattern.pattern.lower()):
                    return pattern
            
            elif pattern.pattern_type == 'regex':
                regex = self.compiled_regexes.get(pattern.pattern)
                if regex and regex.search(pfam_domain):
                    return pattern
        
        return None
    
    def get_mge_codewords(self, codeword_to_pfam: Dict[int, List[str]]) -> Set[int]:
        """
        Identify codewords associated with MGE patterns.
        
        Args:
            codeword_to_pfam: Mapping from codeword_id to list of PFAM domains
            
        Returns:
            Set of codeword IDs that should be masked
        """
        mge_codewords = set()
        category_counts = {}
        
        for codeword_id, pfam_domains in codeword_to_pfam.items():
            for domain in pfam_domains:
                matching_pattern = self.matches_mge_pattern(domain)
                if matching_pattern:
                    mge_codewords.add(codeword_id)
                    
                    # Track category statistics
                    category = matching_pattern.category
                    category_counts[category] = category_counts.get(category, 0) + 1
                    
                    logger.debug(f"Masked codeword {codeword_id}: {domain} -> {matching_pattern.description}")
                    break  # One match per codeword is enough
        
        logger.info(f"Identified {len(mge_codewords)} MGE-associated codewords for masking")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} codewords")
        
        return mge_codewords
    
    def apply_mask_weights(self, codeword_weights: Dict[int, float], 
                          mge_codewords: Set[int],
                          mask_strategy: str = 'zero') -> Dict[int, float]:
        """
        Apply MGE masking to codeword weights.
        
        Args:
            codeword_weights: Original codeword weights
            mge_codewords: Set of codeword IDs to mask
            mask_strategy: 'zero' (set to 0) or 'downweight' (multiply by 0.1)
            
        Returns:
            Masked codeword weights
        """
        masked_weights = codeword_weights.copy()
        
        for codeword_id in mge_codewords:
            if codeword_id in masked_weights:
                if mask_strategy == 'zero':
                    masked_weights[codeword_id] = 0.0
                elif mask_strategy == 'downweight':
                    masked_weights[codeword_id] *= 0.1
                else:
                    raise ValueError(f"Unknown mask strategy: {mask_strategy}")
        
        return masked_weights
    
    def get_statistics(self, codeword_to_pfam: Dict[int, List[str]]) -> Dict[str, int]:
        """Get statistics about MGE masking coverage."""
        mge_codewords = self.get_mge_codewords(codeword_to_pfam)
        
        category_stats = {}
        total_domains = 0
        mge_domains = 0
        
        for codeword_id, pfam_domains in codeword_to_pfam.items():
            total_domains += len(pfam_domains)
            
            for domain in pfam_domains:
                matching_pattern = self.matches_mge_pattern(domain)
                if matching_pattern:
                    mge_domains += 1
                    category = matching_pattern.category
                    category_stats[category] = category_stats.get(category, 0) + 1
        
        return {
            'total_codewords': len(codeword_to_pfam),
            'mge_codewords': len(mge_codewords),
            'mge_fraction': len(mge_codewords) / len(codeword_to_pfam) if codeword_to_pfam else 0,
            'total_domains': total_domains,
            'mge_domains': mge_domains,
            'mge_domain_fraction': mge_domains / total_domains if total_domains else 0,
            **{f'{cat}_codewords': count for cat, count in category_stats.items()}
        }


def load_mge_mask(config_path: Optional[Path] = None) -> MGEMask:
    """
    Load MGE mask from configuration file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file with custom patterns
        
    Returns:
        Configured MGEMask instance
    """
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            patterns = []
            for pattern_config in config.get('mge_patterns', []):
                patterns.append(MGEPattern(
                    pattern=pattern_config['pattern'],
                    pattern_type=pattern_config.get('type', 'exact'),
                    description=pattern_config.get('description', ''),
                    category=pattern_config.get('category', 'unknown')
                ))
            
            logger.info(f"Loaded {len(patterns)} MGE patterns from {config_path}")
            return MGEMask(patterns)
            
        except Exception as e:
            logger.error(f"Failed to load MGE mask config from {config_path}: {e}")
            logger.info("Using default MGE patterns")
    
    return MGEMask()  # Use default patterns


def create_default_mge_config(output_path: Path) -> None:
    """Create a default MGE masking configuration file."""
    mask = MGEMask()
    
    config = {
        'mge_patterns': [
            {
                'pattern': pattern.pattern,
                'type': pattern.pattern_type,
                'description': pattern.description,
                'category': pattern.category
            }
            for pattern in mask.patterns
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Created default MGE mask configuration at {output_path}")


if __name__ == "__main__":
    # Example usage and testing
    mask = MGEMask()
    
    # Test some example PFAM domains
    test_domains = [
        "Transposase_7",
        "DDE_Tnp_1",
        "Phage_integrase",
        "ABC_tran",  # Should not match
        "Cas1",
        "Methylase_N",
        "Random_domain"  # Should not match
    ]
    
    print("Testing MGE pattern matching:")
    for domain in test_domains:
        match = mask.matches_mge_pattern(domain)
        if match:
            print(f"  {domain} -> {match.category} ({match.description})")
        else:
            print(f"  {domain} -> No match")
    
    # Create example config
    create_default_mge_config(Path("mge_mask_config.yaml"))