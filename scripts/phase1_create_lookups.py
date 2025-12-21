"""
Phase 1, Task 1.4: Create Lookup Tables
Creates reference tables for race filtering and scoring.
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
LOOKUPS_DIR = DATA_DIR / 'processed' / 'lookups'
INPUT_FILE = DATA_DIR / 'processed' / 'all_gb_races_cleaned.parquet'

# Create lookups directory
LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)


def create_distance_bands_lookup():
    """Create distance band reference table."""
    print("Creating distance bands lookup...")
    
    distance_bands = pd.DataFrame([
        {'band_name': 'Sprint', 'min_furlongs': 5.0, 'max_furlongs': 6.99, 
         'description': 'Sprint races (5-6.99f)', 'typical_distance': '5f-6f'},
        {'band_name': 'Mile', 'min_furlongs': 7.0, 'max_furlongs': 8.99,
         'description': 'Mile races (7-8.99f)', 'typical_distance': '7f-8f'},
        {'band_name': 'Middle', 'min_furlongs': 9.0, 'max_furlongs': 12.99,
         'description': 'Middle distance (9-12.99f)', 'typical_distance': '10f-12f'},
        {'band_name': 'Long', 'min_furlongs': 13.0, 'max_furlongs': 25.0,
         'description': 'Long distance (13f+)', 'typical_distance': '14f+'}
    ])
    
    output_file = LOOKUPS_DIR / 'distance_bands.csv'
    distance_bands.to_csv(output_file, index=False)
    
    print(f"  Created {len(distance_bands)} distance bands")
    print(f"  Saved to: {output_file}")
    
    return distance_bands


def create_class_hierarchy_lookup():
    """Create class hierarchy reference table."""
    print("\nCreating class hierarchy lookup...")
    
    class_hierarchy = pd.DataFrame([
        {'class_name': 'Class 1', 'quality_rank': 7, 'tier': 'Premium', 
         'typical_prize_min': 15000, 'description': 'Highest class - Group/Listed races'},
        {'class_name': 'Class 2', 'quality_rank': 6, 'tier': 'High',
         'typical_prize_min': 8000, 'description': 'High quality handicaps and conditions races'},
        {'class_name': 'Class 3', 'quality_rank': 5, 'tier': 'Standard',
         'typical_prize_min': 6000, 'description': 'Competitive handicaps'},
        {'class_name': 'Class 4', 'quality_rank': 4, 'tier': 'Standard',
         'typical_prize_min': 4000, 'description': 'Moderate handicaps'},
        {'class_name': 'Class 5', 'quality_rank': 3, 'tier': 'Low',
         'typical_prize_min': 3000, 'description': 'Lower class handicaps'},
        {'class_name': 'Class 6', 'quality_rank': 2, 'tier': 'Low',
         'typical_prize_min': 2000, 'description': 'Selling/claiming races'},
        {'class_name': 'Class 7', 'quality_rank': 1, 'tier': 'Low',
         'typical_prize_min': 1500, 'description': 'Lowest class - sellers'}
    ])
    
    output_file = LOOKUPS_DIR / 'class_hierarchy.csv'
    class_hierarchy.to_csv(output_file, index=False)
    
    print(f"  Created {len(class_hierarchy)} class levels")
    print(f"  Saved to: {output_file}")
    
    return class_hierarchy


def create_course_tiers_lookup(df):
    """Create course tier classification based on race volume and quality."""
    print("\nCreating course tiers lookup...")
    
    # Get course-level statistics
    course_stats = df.groupby('course_clean').agg({
        'race_id': 'nunique',  # Total unique races
        'prize_clean': 'median',  # Median prize
        'class_clean': lambda x: (x == 'Class 1').sum() + (x == 'Class 2').sum()  # High quality races
    }).reset_index()
    
    course_stats.columns = ['course', 'total_races', 'median_prize', 'high_class_races']
    
    # Define premium courses (Group 1 venues)
    premium_courses = [
        'Ascot', 'Newmarket', 'York', 'Doncaster', 'Goodwood', 
        'Chester', 'Sandown', 'Epsom', 'Haydock'
    ]
    
    # Assign tiers
    def assign_tier(row):
        if row['course'] in premium_courses:
            return 'Premium'
        elif row['high_class_races'] >= 10 or (row['median_prize'] > 5000 and row['total_races'] > 50):
            return 'Major'
        else:
            return 'Standard'
    
    course_stats['tier'] = course_stats.apply(assign_tier, axis=1)
    
    # Add surface type (majority surface for each course)
    surface_mode = df.groupby('course_clean')['surface_clean'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown').reset_index()
    surface_mode.columns = ['course', 'primary_surface']
    
    course_tiers = course_stats.merge(surface_mode, on='course')
    
    # Reorder columns
    course_tiers = course_tiers[['course', 'tier', 'primary_surface', 'total_races', 'median_prize', 'high_class_races']]
    
    output_file = LOOKUPS_DIR / 'course_tiers.csv'
    course_tiers.to_csv(output_file, index=False)
    
    print(f"  Created tiers for {len(course_tiers)} courses")
    print(f"  Premium: {(course_tiers['tier'] == 'Premium').sum()} courses")
    print(f"  Major: {(course_tiers['tier'] == 'Major').sum()} courses")
    print(f"  Standard: {(course_tiers['tier'] == 'Standard').sum()} courses")
    print(f"  Saved to: {output_file}")
    
    return course_tiers


def create_surface_types_lookup():
    """Create surface types reference table."""
    print("\nCreating surface types lookup...")
    
    surface_types = pd.DataFrame([
        {'surface_name': 'Turf', 'going_dependent': True, 'surface_code': 'T',
         'description': 'Grass surface - performance varies with going'},
        {'surface_name': 'All-Weather', 'going_dependent': False, 'surface_code': 'AW',
         'description': 'Synthetic surface - consistent conditions year-round'}
    ])
    
    output_file = LOOKUPS_DIR / 'surface_types.csv'
    surface_types.to_csv(output_file, index=False)
    
    print(f"  Created {len(surface_types)} surface types")
    print(f"  Saved to: {output_file}")
    
    return surface_types


def create_going_codes_lookup():
    """Create going (track condition) codes reference."""
    print("\nCreating going codes lookup...")
    
    going_codes = pd.DataFrame([
        {'going_code': 'Firm', 'numeric_value': 1, 'description': 'Fast, hard ground'},
        {'going_code': 'Good to Firm', 'numeric_value': 2, 'description': 'Firm with some give'},
        {'going_code': 'Good', 'numeric_value': 3, 'description': 'Ideal conditions'},
        {'going_code': 'Good to Soft', 'numeric_value': 4, 'description': 'Some softness'},
        {'going_code': 'Soft', 'numeric_value': 5, 'description': 'Soft, yielding ground'},
        {'going_code': 'Heavy', 'numeric_value': 6, 'description': 'Very soft, slow ground'},
        {'going_code': 'Standard', 'numeric_value': 3, 'description': 'All-Weather standard'},
        {'going_code': 'Standard to Slow', 'numeric_value': 4, 'description': 'AW slightly slower'},
        {'going_code': 'Slow', 'numeric_value': 5, 'description': 'AW slow conditions'}
    ])
    
    output_file = LOOKUPS_DIR / 'going_codes.csv'
    going_codes.to_csv(output_file, index=False)
    
    print(f"  Created {len(going_codes)} going codes")
    print(f"  Saved to: {output_file}")
    
    return going_codes


def main():
    """Main lookup table creation workflow."""
    print("=" * 60)
    print("PHASE 1: CREATE LOOKUP TABLES")
    print("=" * 60)
    
    # Load cleaned data
    print(f"\nLoading cleaned data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    
    # Create lookup tables
    distance_bands = create_distance_bands_lookup()
    class_hierarchy = create_class_hierarchy_lookup()
    course_tiers = create_course_tiers_lookup(df)
    surface_types = create_surface_types_lookup()
    going_codes = create_going_codes_lookup()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("LOOKUP TABLES CREATED")
    print(f"{'=' * 60}")
    print(f"\nAll lookup tables saved to: {LOOKUPS_DIR}")
    print("\nFiles created:")
    print("  - distance_bands.csv (4 bands)")
    print("  - class_hierarchy.csv (7 classes)")
    print(f"  - course_tiers.csv ({len(course_tiers)} courses)")
    print("  - surface_types.csv (2 types)")
    print("  - going_codes.csv (9 codes)")
    
    print(f"\n{'=' * 60}")
    print("LOOKUP CREATION COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("1. Build aggregation scripts (Tasks 1.5-1.6)")
    print("2. Run exploratory analysis (Task 1.7)")


if __name__ == "__main__":
    main()
