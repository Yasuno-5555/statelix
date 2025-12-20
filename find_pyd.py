import os

with open('pyd_list.txt', 'w') as f:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyd'):
                f.write(os.path.abspath(os.path.join(root, file)) + '\n')
