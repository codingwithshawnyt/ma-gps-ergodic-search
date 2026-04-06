import sys
sys.path.insert(0, '.')

file_path = 'MAGPS/MARL_gym_envs/ergodic_search.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("First few problematic lines:")
for i in range(100, 120):
    if i < len(lines):
        line = lines[i].rstrip()
        spaces = len(line) - len(line.lstrip())
        print(f"{i+1}: spaces={spaces}, content={repr(line[:50] if len(line) > 50 else line)}")

print("\nChecking specific area:")
for i in range(94, 115):
    if i < len(lines):
        print(f"{i+1}: {repr(lines[i].rstrip()[:60])}")
