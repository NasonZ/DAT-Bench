# Should move to intergration tests
"""
Quick test of triangular matrix visualization with pre-computed distances.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from divergent_bench.visualization import triangular_matrix

# Example data from the image
words = [
    "innovation",
    "stomata", 
    "modulo",
    "chorizo",
    "dialectics",
    "cello",
    "chic"
]

# Pre-computed pairwise distances (from the image)
# These are in the order of itertools.combinations(words, 2)
distances = [
    99,           # innovation-stomata
    102, 79,      # innovation-modulo, stomata-modulo
    107, 97, 89,  # innovation-chorizo, stomata-chorizo, modulo-chorizo
    86, 82, 78, 93,  # innovation-dialectics, stomata-dialectics, modulo-dialectics, chorizo-dialectics
    95, 95, 95, 98, 99,  # innovation-cello, stomata-cello, modulo-cello, chorizo-cello, dialectics-cello
    83, 108, 99, 91, 100, 93  # innovation-chic, stomata-chic, modulo-chic, chorizo-chic, dialectics-chic, cello-chic
]

# Create the visualization
fig = triangular_matrix(
    words=words,
    distances=distances,
    title="Example DAT Result - Score: 93.72",
    figsize=(8, 8),
    show_values=True,
    cmap='Greens'  # Use green shading like the original
)

# Save the figure
output_path = Path("visualizations/test_triangular_matrix_example.png")
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved triangular matrix to {output_path}")
plt.close(fig)

print("✅ Triangular matrix visualization test complete!")