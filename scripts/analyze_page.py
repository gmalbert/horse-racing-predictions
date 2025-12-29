import re

with open('debug_racing_post.html', encoding='utf-8') as f:
    content = f.read()

# Look for race/card related classes
matches = re.findall(r'class="([^"]*(?:race|card|runner|horse|odds)[^"]*)"', content, re.I)
classes = sorted(set(matches))

print(f"Found {len(classes)} unique race-related classes:")
for cls in classes[:50]:
    print(f"  {cls}")

# Check for JavaScript frameworks
print("\n\nJavaScript check:")
print(f"  React: {'react' in content.lower()}")
print(f"  Vue: {'vue' in content.lower()}")
print(f"  Angular: {'angular' in content.lower()}")

# Check for data in scripts
scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL | re.I)
print(f"\n  Found {len(scripts)} script tags")

# Look for JSON data
json_data = re.findall(r'window\.__[A-Z_]+__\s*=\s*(\{.*?\});', content[:50000], re.DOTALL)
print(f"  Found {len(json_data)} potential data objects")
