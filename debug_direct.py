import sys
import os

log_file = 'build_log_direct.txt'
if not os.path.exists(log_file):
    print("Log file not found")
    sys.exit(1)

content = ""
for enc in ['utf-16', 'utf-8', 'cp932']:
    try:
        with open(log_file, 'r', encoding=enc) as f:
            content = f.read()
            if content: break
    except:
        continue

print(f"Content length: {len(content)}")
print(content[:1000]) # Print first 1000 chars
if len(content) > 1000:
    print("... (truncated) ...")
    print(content[-1000:]) # Print last 1000 chars
