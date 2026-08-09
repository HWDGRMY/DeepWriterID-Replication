import os, glob
BASE = r'F:\DeepWriterID-Replication\data\raw'
test_files = glob.glob(BASE + '/**/*Test*.wptt', recursive=True)
writers = []
ids = set()
for f in test_files:
    wid = os.path.basename(f).split('-')[0]
    writers.append(wid)
    ids.add(wid)
print(f'测试集文件数: {len(test_files)}')
print(f'测试集作者数: {len(ids)}')
from collections import Counter
counts = Counter(writers)
duplicates = {k: v for k, v in counts.items() if v > 1}
print(f'重复出现的作者ID及次数: {duplicates}')