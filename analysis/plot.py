import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data
data = {
    'Model': ['gemma2-9b', 'gemma2-27b', 'llama31-8b', 'llama31-70b', 
              'ministral-8b', 'mistral-small', 'ministral-14b-r', 
              'qwen3-4b', 'qwen3-8b', 'qwen3-30b', 'qwen3-32b'],
    'Family': ['Gemma', 'Gemma', 'Llama', 'Llama', 
               'Mistral', 'Mistral', 'Mistral', 
               'Qwen', 'Qwen', 'Qwen', 'Qwen'],
    'AIR_Romanian': [0.89, 0.87, 0.92, 1.03, 
                     0.99, 0.96, 1.00, 
                     0.72, 0.94, 1.00, 0.85],
    'Hallucination_Rate': [14, 12, 6, 28, 
                           60, 16, 26, 
                           2, 6, 14, 27],
    'Size_B': [9, 27, 8, 70, 8, 22, 14, 4, 8, 30, 32]
}

df = pd.DataFrame(data)

# Set scientific style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(8, 6))

# Plot
palette = {'Gemma': '#e67e22', 'Llama': '#2980b9', 'Mistral': '#27ae60', 'Qwen': '#8e44ad'}

# CORRECTED CALL: Use 'size' and 'sizes' parameters
ax = sns.scatterplot(
    data=df, 
    x='Hallucination_Rate', 
    y='AIR_Romanian', 
    hue='Family', 
    palette=palette,
    size='Size_B',        # Maps column to size automatically
    sizes=(50, 400),      # Sets the min and max bubble size (pixels)
    alpha=0.8,
    edgecolor='black'
)

# Add Threshold Line (EEOC 0.80)
plt.axhline(y=0.80, color='red', linestyle='--', linewidth=1.5, label='EEOC Violation (<0.80)')

# Annotate outliers
plt.annotate('Ministral-8B\n(High Hallucination, Fair)', 
             xy=(60, 0.99), xytext=(40, 0.92),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=9)

plt.annotate('Qwen3-4B\n(Low Hallucination, Biased)', 
             xy=(2, 0.72), xytext=(5, 0.65),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=9)

# Labels and Limits
plt.xlabel('Visa Hallucination Rate (%)', fontsize=11)
plt.ylabel('Adverse Impact Ratio (AIR)', fontsize=11)
plt.ylim(0.60, 1.10)
plt.xlim(-5, 70)

# Legend adjustments
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Model Family')

plt.tight_layout()
plt.savefig('visa_wall_scatter.png', dpi=300, bbox_inches='tight')
print("Plot generated: visa_wall_scatter.pdf")