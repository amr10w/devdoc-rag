# 1. Install the package in editable mode
pip install -e .

# 2. Run directly via CLI command
devdoc-ground-truth --num-samples 40

# Or run via standard python module syntax
python -m src.evaluation.generate_ground_truth --num-samples 40