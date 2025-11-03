"""
Prepare Disease Outbreak Dataset for Training
Creates labeled dataset from disease_outbreaks_minimal.csv
Generates both outbreak (label=1) and non-outbreak (label=0) examples
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("="*80)
print("DISEASE OUTBREAK DATASET PREPARATION")
print("="*80)

# Load the outbreak data
print("\n1. Loading outbreak data...")
df = pd.read_csv('data/raw/disease_outbreaks_minimal.csv')

print(f"   ✓ Loaded {len(df)} outbreak records")
print(f"   ✓ Diseases: {df['Disease'].nunique()}")
print(f"   ✓ Countries: {df['Country'].nunique()}")
print(f"   ✓ Regions: {df['who_region'].nunique()}")

# Create outbreak texts (positive examples - label=1)
print("\n2. Creating outbreak signal texts (label=1)...")

outbreak_templates = [
    "Multiple cases of {disease} reported in {country}",
    "Health officials confirm {disease} outbreak in {region}, {country}",
    "{disease} cases surge in {country} region",
    "Outbreak alert: {disease} spreading in {country}",
    "Health authorities investigate {disease} cluster in {country}",
    "Rising {disease} cases detected in {country}, {region}",
    "{disease} outbreak declared in {country}",
    "Epidemic warning: {disease} affecting {country}",
    "Public health emergency: {disease} in {country}",
    "{disease} infections increase across {country}",
    "Disease surveillance detects {disease} in {country}",
    "{disease} outbreak response activated in {country}",
    "Unusual spike in {disease} cases reported in {country}",
    "Health officials respond to {disease} in {region}, {country}",
    "{disease} spreading rapidly in {country}",
    "Hospital admissions for {disease} rise in {country}",
    "{disease} cases confirmed in multiple areas of {country}",
    "Health alert issued for {disease} in {country}",
    "{disease} outbreak investigation underway in {country}",
    "Emergency measures for {disease} in {country}",
]

outbreak_texts = []
for _, row in df.iterrows():
    template = random.choice(outbreak_templates)
    text = template.format(
        disease=row['Disease'],
        country=row['Country'],
        region=row['unsd_region']
    )
    outbreak_texts.append({
        'text': text,
        'label': 1,
        'disease': row['Disease'],
        'country': row['Country'],
        'region': row['unsd_region'],
        'who_region': row['who_region'],
        'year': row['Year']
    })

print(f"   ✓ Created {len(outbreak_texts)} outbreak texts")

# Create non-outbreak texts (negative examples - label=0)
print("\n3. Creating non-outbreak texts (label=0)...")

countries = df['Country'].unique()
regions = df['unsd_region'].unique()
diseases = df['Disease'].unique()

non_outbreak_templates = [
    "Health ministry announces vaccination campaign in {country}",
    "New medical facility inaugurated in {country}",
    "Healthcare workers receive training in {country}",
    "{country} celebrates World Health Day",
    "Medical research breakthrough announced in {country}",
    "Public health conference held in {region}",
    "Government increases healthcare budget for {country}",
    "Health awareness program launched in {country}",
    "Medical supplies distributed to {country}",
    "Healthcare infrastructure upgraded in {country}",
    "Routine immunization program continues in {country}",
    "Health screening initiative in {country}",
    "Medical college announces admissions in {country}",
    "Healthcare symposium scheduled in {region}",
    "Wellness program promotes fitness in {country}",
    "Health insurance expansion in {country}",
    "Telemedicine services introduced in {country}",
    "Medical tourism grows in {country}",
    "Healthcare quality improvement in {country}",
    "Public health system modernization in {country}",
    "Preventive healthcare campaign in {country}",
    "Medical equipment donation to {country}",
    "Health ministry reports routine statistics for {country}",
    "Hospital capacity expansion in {country}",
    "Healthcare professionals honored in {country}",
    "Medical conference discusses innovations in {region}",
    "Nutrition awareness week in {country}",
    "Health checkup camps organized in {country}",
    "Medical education reform in {country}",
    "Healthcare partnerships announced for {country}",
]

# Generate same number of non-outbreak examples as outbreak examples
non_outbreak_texts = []
for i in range(len(outbreak_texts)):
    template = random.choice(non_outbreak_templates)
    country = random.choice(countries)
    region = random.choice(regions)
    
    text = template.format(
        country=country,
        region=region
    )
    
    non_outbreak_texts.append({
        'text': text,
        'label': 0,
        'disease': 'None',
        'country': country,
        'region': region,
        'who_region': 'N/A',
        'year': 2025
    })

print(f"   ✓ Created {len(non_outbreak_texts)} non-outbreak texts")

# Combine datasets
print("\n4. Combining and shuffling dataset...")
all_data = outbreak_texts + non_outbreak_texts
df_final = pd.DataFrame(all_data)

# Shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   ✓ Total samples: {len(df_final)}")
print(f"   ✓ Outbreak samples (label=1): {(df_final['label']==1).sum()}")
print(f"   ✓ Non-outbreak samples (label=0): {(df_final['label']==0).sum()}")

# Display sample
print("\n5. Sample data:")
print("-"*80)
print("\nOutbreak examples (label=1):")
for i, row in df_final[df_final['label']==1].head(3).iterrows():
    print(f"   • {row['text']}")

print("\nNon-outbreak examples (label=0):")
for i, row in df_final[df_final['label']==0].head(3).iterrows():
    print(f"   • {row['text']}")

# Save processed dataset
output_path = 'data/processed/epidemic_data.csv'
df_final.to_csv(output_path, index=False)

print(f"\n6. Saving dataset...")
print(f"   ✓ Saved to: {output_path}")

# Display statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
print(f"\nTotal Records: {len(df_final)}")
print(f"Features: {df_final.columns.tolist()}")
print(f"\nClass Distribution:")
print(df_final['label'].value_counts())
print(f"\nBalance Ratio: {(df_final['label']==0).sum() / (df_final['label']==1).sum():.2f}")

print(f"\nTop 10 Diseases (in outbreak examples):")
disease_counts = df_final[df_final['label']==1]['disease'].value_counts().head(10)
for disease, count in disease_counts.items():
    print(f"   {disease}: {count}")

print(f"\nTop 10 Countries:")
country_counts = df_final['country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"   {country}: {count}")

print(f"\nWHO Regions:")
region_counts = df_final[df_final['label']==1]['who_region'].value_counts()
for region, count in region_counts.items():
    print(f"   {region}: {count}")

print("\n" + "="*80)
print("✓ DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\nYou can now train models using: {output_path}")
print("\nNext steps:")
print("  1. Run: python src/models/train_all.py")
print("  2. Or open: notebooks/epiwatch_training.ipynb")
print("\n" + "="*80)
