import re
from pathlib import Path

html_file = 'debug_race_page_odds.html'
if not Path(html_file).exists():
    print(f"File {html_file} not found")
    exit(1)

with open(html_file, encoding='utf-8') as f:
    content = f.read()

print(f'File size: {len(content):,} bytes\n')

# Look for actual displayed text with horses
print('=== Looking for visible text content ===')

# Remove script and style tags
clean_content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.I)
clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.I)

# Find text near odds
odds_contexts = re.findall(r'.{100}([0-9]+/[0-9]+|[0-9]+\.[0-9]{2}).{100}', clean_content)
print(f'Found {len(odds_contexts)} odds contexts')
if odds_contexts:
    # Show unique contexts
    unique = list(set(odds_contexts))[:5]
    for ctx in unique:
        # Clean up HTML
        ctx_clean = re.sub(r'<[^>]+>', ' ', ctx)
        ctx_clean = re.sub(r'\s+', ' ', ctx_clean).strip()
        if len(ctx_clean) > 50:
            print(f'  {ctx_clean[:150]}')

# Look for data attributes that might have JSON
print('\n=== Checking data attributes ===')
data_attrs = re.findall(r'data-[a-z-]+="(\{[^"]+\})"', content)
print(f'Found {len(data_attrs)} data attributes with JSON')
if data_attrs:
    for attr in data_attrs[:3]:
        if 'odds' in attr.lower() or 'horse' in attr.lower():
            print(f'  {attr[:200]}')

# Look for window variables with data
print('\n=== Checking window variables ===')
window_vars = re.findall(r'window\.__[A-Z_]+__\s*=\s*(\{[^;]+);', content, re.DOTALL)
print(f'Found {len(window_vars)} window variables')
for var in window_vars:
    if 'odds' in var.lower() or 'runner' in var.lower() or 'horse' in var.lower():
        print(f'  Contains race data: {var[:200]}...')

# Look for specific class patterns
print('\n=== Checking for odds comparison table classes ===')
odds_comp_classes = re.findall(r'class="([^"]*(?:odds|comparison|bookmaker|runner|horse)[^"]*)"', content, re.I)
unique_classes = sorted(set(odds_comp_classes))
print(f'Found {len(unique_classes)} unique related classes:')
for cls in unique_classes[:20]:
    print(f'  {cls}')
