#\!/bin/bash
# Test PFAM annotation integration
echo '🧬 Testing PFAM annotation pipeline...'

# Clean up existing results
rm -rf elsa_index/pfam_annotations/
echo '✓ Cleaned up existing PFAM results'

# Run just the PFAM annotation part
python -c "
from elsa.pfam_annotation import run_pfam_annotation_pipeline
from elsa.params import load_config
from elsa.manifest import ELSAManifest
from pathlib import Path

config = load_config('elsa.config.yaml')
manifest = ELSAManifest('elsa_index')

print('🔬 Running PFAM annotation...')
result = run_pfam_annotation_pipeline(
    config, manifest, Path('elsa_index'), 
    threads=8, force_annotation=True
)

if result:
    print(f'✅ PFAM annotation completed\!')
    print(f'   Successful samples: {result.get("successful_samples", 0)}')
    print(f'   Total hits: {result.get("total_hits", 0)}')
else:
    print('❌ PFAM annotation failed')
"

