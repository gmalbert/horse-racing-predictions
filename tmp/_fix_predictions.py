"""Fix corrupted predictions.py by removing the orphaned US Racing block."""
import re

with open('predictions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all occurrences of 'def fetch_racecards'
positions = [m.start() for m in re.finditer(r'\ndef fetch_racecards\(', content)]
print(f"Found {len(positions)} occurrences of 'def fetch_racecards' at chars: {positions}")

# Find the End US Racing Tab marker
end_marker_pos = content.find('# \u2500\u2500 End US Racing Tab')
print(f"End marker at char: {end_marker_pos}")

if len(positions) >= 2 and end_marker_pos != -1:
    # positions[0] = corrupt fetch_racecards
    # positions[1] = real fetch_racecards (after the end marker)
    # Remove from positions[0] through the end marker line + trailing newlines
    end_of_marker_line = content.find('\n', end_marker_pos) + 1
    # Skip any blank lines after the marker
    while end_of_marker_line < len(content) and content[end_of_marker_line] == '\n':
        end_of_marker_line += 1
    
    print(f"Removing chars {positions[0]} to {end_of_marker_line}")
    print(f"Removed text starts with: {repr(content[positions[0]:positions[0]+100])}")
    print(f"Text after removal starts with: {repr(content[end_of_marker_line:end_of_marker_line+100])}")
    
    fixed = content[:positions[0]] + '\n' + content[end_of_marker_line:]
    
    with open('predictions.py', 'w', encoding='utf-8') as f:
        f.write(fixed)
    print("Fixed! Verifying syntax...")
    
    import ast
    try:
        ast.parse(fixed)
        print("Syntax OK")
    except SyntaxError as e:
        print(f"Syntax error: {e}")
else:
    print(f"ERROR: expected 2+ fetch_racecards and an end marker. positions={positions}, end_marker={end_marker_pos}")
