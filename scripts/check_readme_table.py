"""
Simple validator to check the Top-10 features markdown table in README.md
Exits with code 0 on success, 2 on failure.
"""
import sys
from pathlib import Path

readme = Path(__file__).parent.parent / "README.md"
if not readme.exists():
    print("README.md not found")
    sys.exit(2)

text = readme.read_text(encoding='utf-8')
start_marker = "### Top 10 Model Features (Leak-free)"
if start_marker not in text:
    print("Top-10 features section not found in README.md")
    sys.exit(2)

section = text.split(start_marker, 1)[1]
# Stop at first double-newline followed by 'Notes:' or end
stop_tokens = ["Notes:", "\n\n"]
# Find table start (first '|' after marker)
idx = section.find("|")
if idx == -1:
    print("Markdown table not found after section header")
    sys.exit(2)
# Extract lines until an empty line
lines = section[idx:].splitlines()
# Keep only lines that look like table rows (contain | and not only ---)
table_lines = []
for ln in lines:
    if ln.strip() == "":
        break
    if '|' in ln:
        table_lines.append(ln.rstrip())
    else:
        break

if len(table_lines) < 3:
    print("Table seems too short or malformed")
    sys.exit(2)

# Validate header separator (second line) contains ---
if not ('---' in table_lines[1]):
    print("Table header separator missing or malformed:")
    print(table_lines[0:3])
    sys.exit(2)

# Validate each data row has 3 columns (two '|' separators at minimum between first and last)
for i, row in enumerate(table_lines[2:], start=3):
    # Count number of '|' characters
    pipe_count = row.count('|')
    if pipe_count < 2:
        print(f"Row {i} malformed (not enough columns): {row}")
        sys.exit(2)

print("README Top-10 features table looks well-formed.")
sys.exit(0)
