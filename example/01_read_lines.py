from main import file

files = file("data/kftt.ja")
lines = files.lines()

datsset = lines.create()

for line in datsset:
    print(line)
