#!/usr/bin/env python3
"""Extract recent 2025 racing fixtures from .ics file (12/17/2025 to 12/31/2025).

Writes: data/processed/recent_2025_fixtures.csv
"""
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, date

INPUT = Path("data/raw/2025_racing_fixture_list.ics")
OUT = Path("data/processed/recent_2025_flat_fixtures.csv")
START_DATE = date(2025, 12, 17)
END_DATE = date(2025, 12, 31)


def parse_ics_file(filepath):
    """Parse .ics file and extract race events, expanding multi-course entries"""
    try:
        from icalendar import Calendar
    except ImportError:
        print("Installing icalendar package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "icalendar"])
        from icalendar import Calendar
    
    events = []
    
    with open(filepath, 'rb') as f:
        cal = Calendar.from_ical(f.read())
        
        for component in cal.walk():
            if component.name == "VEVENT":
                # Extract event details
                summary = str(component.get('summary', ''))
                dtstart = component.get('dtstart')
                
                if dtstart:
                    # Handle both date and datetime objects
                    if hasattr(dtstart.dt, 'date'):
                        event_date = dtstart.dt.date()
                    else:
                        event_date = dtstart.dt
                    
                    # Filter by date range
                    if START_DATE <= event_date <= END_DATE:
                        # Parse the summary which contains multiple courses
                        courses_info = parse_multi_course_summary(summary, event_date)
                        events.extend(courses_info)
    
    return events


def parse_multi_course_summary(summary, event_date):
    """Parse summary containing multiple courses and racing types
    
    Format: 'Course1 (Type1) / Course2 (Type2) / ...'
    Returns list of individual course events
    """
    import re
    
    weekday = event_date.strftime('%A')
    courses = []
    
    # Split by ' / ' to get individual course entries
    course_entries = summary.split(' / ')
    
    for entry in course_entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Parse "Course Name (Time Type)" format
        # e.g., "Kempton Park (Floodlit Flat)" or "Newbury (Afternoon Jump)"
        match = re.match(r'(.+?)\s*\((.+?)\s+(Flat|Jump)\)', entry)
        
        if match:
            course_name = match.group(1).strip()
            time_desc = match.group(2).strip()  # "Afternoon", "Floodlit", "Matinee"
            race_type = match.group(3).strip()  # "Flat" or "Jump"
            
            # Determine surface from course name or type
            # AWT courses (known all-weather tracks)
            awt_courses = ['Chelmsford City', 'Kempton Park', 'Lingfield Park', 
                          'Newcastle', 'Southwell', 'Wolverhampton']
            surface = 'AWT' if any(awt in course_name for awt in awt_courses) else 'Turf'
            
            # Determine time
            if 'floodlit' in time_desc.lower() or 'evening' in time_desc.lower():
                time = 'Evening'
            elif 'matinee' in time_desc.lower() or 'morning' in time_desc.lower():
                time = 'Morning'
            else:
                time = 'Afternoon'
            
            courses.append({
                'Date': event_date,
                'Weekday': weekday,
                'Course': course_name,
                'Time': time,
                'CourseGroup': '',  # Not available in .ics
                'Region': '',  # Not available in .ics
                'Code': race_type,  # Flat or Jump
                'Surface': surface,
                'Type': 'Racecourse/Normal'
            })
    
    return courses


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        sys.exit(2)
    
    print(f"Parsing {INPUT}...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    events = parse_ics_file(INPUT)
    
    if not events:
        print("No events found in date range")
        sys.exit(0)
    
    # Create DataFrame
    df = pd.DataFrame(events)
    
    # Filter to Flat races only (exclude Jump)
    original_count = len(df)
    df = df[df['Code'] == 'Flat'].copy()
    print(f"\nFiltered to Flat races only: {len(df)} races (excluded {original_count - len(df)} Jump races)")
    
    # Load course metadata from 2026 fixtures to fill in Region and CourseGroup
    try:
        fixtures_2026 = pd.read_csv('data/processed/bha_2026_all_courses_class1-4.csv')
        course_lookup = fixtures_2026[['Course', 'Region', 'CourseGroup']].drop_duplicates()
        
        # Merge to add Region and CourseGroup
        df = df.merge(course_lookup, on='Course', how='left', suffixes=('', '_lookup'))
        df['Region'] = df['Region_lookup'].fillna('')
        df['CourseGroup'] = df['CourseGroup_lookup'].fillna('')
        df = df.drop(['Region_lookup', 'CourseGroup_lookup'], axis=1)
        
        print(f"Added Region/CourseGroup metadata for {df['Region'].notna().sum()} courses")
    except Exception as e:
        print(f"Warning: Could not load course metadata: {e}")
    
    df = df.sort_values('Date')
    
    # Save to CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    
    print(f"\nExtracted {len(df)} Flat race events")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique courses: {df['Course'].nunique()}")
    print(f"\nWrote to {OUT}")
    
    # Show sample
    print("\nFirst 10 events:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
