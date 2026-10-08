with open('dataset_baseline.py', 'r') as f:
    text = f.read()

text = text.replace('"""\ndef main():', 'def main():')
text = text.replace('=================================")\n"""', '=================================")')

with open('dataset_baseline.py', 'w') as f:
    f.write(text)
