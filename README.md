# flowder

simple (fast) dataloader for machine learning.

# installation

```sh
pip install git+https://github.com/moajo/flowder.git
```

### require

- Python 3.6
- tqdm
- pytorch(optional)
- pandas(optional)
- PIL(optional)

# example

```python
from flowder import lines, mapped, directory, filtered
from flowder.utils.csv import csv
from flowder.utils.image import to_image

# create data source
lines_source = lines("text.txt")

# map with the pipe like unix shell
line_length = lines_source | mapped(lambda l: len(l))

# data source is iterable
for line in lines_source[:10]:
    print(line)

# zip with operator '*'
for line, l in lines_source * line_length:
    print(line, "length:", l)

# supported csv/json/...
for annotation in csv("anno.csv"):
    print(annotation)

# load directory files as image
image_source = directory("imgs") | filtered(lambda x: x[-4:] == ".png") | to_image()

# index access
first_img = image_source[0]

```

# DataModel

## Source

反復可能なオブジェクト。map/filter などの演算によって連鎖します。
これらの計算は基本的に遅延評価され、必要な部分だけがメモリ上にロードされます。

```python
from flowder import lines, mapped
source = lines("big_data.txt") | mapped(complex_process) | filtered(predicate)

for data in source:
  pass # 値はイテレーション毎に遅延評価されます

# ファイルにキャッシュを作成できます。Sourceは連鎖中のすべてのパラメータからhashを計算し、一致性を確認します。
file_cache = source.cache("processed_data")

for data in file_cache:
  pass # キャッシュの値が使用されます
```

## PipeLine
Sourceへの変換を表すオブジェクトです。 `|` 演算子でSourceを変換します。

```python
from flowder import mapped, filtered, from_items, flat_mapped, from_array

source1 = from_items(*range(100))  # 0,1,2,3,,,

# Sourceを変換
double = lambda a: a * 2
mapped_source = source1 | mapped(double)
for v in mapped_source:
    print(v)  # 0,2,4,6,,,

# フィルタ
odd = lambda a: a % 2 == 1
mapped_source = source1 | filtered(odd)
for v in mapped_source:
    print(v)  # 1,3,5,7,,,

# PipeLineの連結
pipe = mapped(double) | filtered(lambda a: a > 20)
for v in source1 | pipe:
    print(v)  # 20,22,24,,,

# 流れるデータの一部をフィルタ
data = source1 | mapped(lambda i: {"index": i, "is_odd": i % 2 == 1})
for v in data | {"is_odd": filtered(lambda is_odd: not is_odd)}:
    print(v["index"])  # 0,2,4,6,,,
```

## Aggregator

```python
from flowder.pipes import split, add_eos, add_sos
from flowder.processors import VocabBuilder

from flowder import lines

# スペース区切りのテキストソース
en_source = lines("train.en") | split()
ja_source = lines("train.ja") | split()

# Vocabの構築
en = VocabBuilder("src", max_size=24000)
ja = VocabBuilder("trg", max_size=24000)
en_source >> en
ja_source >> ja

# wordのindexへの変換、その他の前処理
en_source |= en.numericalizer
ja_source |= ja.numericalizer | add_eos() | add_sos()

for en, ja in en_source * ja_source:
    print(en, ja)
```

## Iterator

マルチプロセスで非同期にデータをロードするイテレータを提供します。
バッチの作成や前処理を別プロセスで行うので、メインプロセスのイテレーションが高速化します。
また、開始時にデータをまとめてロードする必要がなくなります。
BucketIterator はシーケンスデータ用のイテレータで、バッチ内のシーケンスの長さが近くなるようにして計算効率を上げます。

```python
batch_transforms = [
    sort(sort_key),
    default_create_batch(),
    tensor_pad_sequence(("src", "trg"), include_length=True),
]
train_iter = flowder.create_bucket_iterator(
    train,
    args.batch_size,
    sort_key=sort_key,
    batch_transforms=batch_transforms,
    device=device,
)

test_iter = flowder.create_iterator(
    test,
    args.batch_size,
    shuffle=True,
    batch_transforms=batch_transforms,
    device=device,
)
```

# benchmark

前処理：テキストをトークナイズして wordIndex に変換、<sos>/<eos>の挿入、バッチ内のデータの長さをできるだけ揃えて(BucketIterator)ソートして padding
Vocab は事前に作成済み
flowder は事前読み込みなしで学習中に非同期でデータ生成
torchtext は途中まで前処理済みデータを pickle でキャッシュして、学習前にロードする
学習タスクは比較の安定性のため、疑似タスクとしてバッチ毎に`time.sleep(0.01)`している

### flowder

```
[flowder.Dataset]preprocess is not needed for any fields
train epoch:1: 100%|███████████████████████████████████████████| 2222/2222 [00:24<00:00, 90.34it/s]
TEST[epoch:1]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 61.55it/s]
train epoch:2: 100%|███████████████████████████████████████████| 2222/2222 [00:23<00:00, 95.34it/s]
TEST[epoch:2]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 67.35it/s]
train epoch:3: 100%|███████████████████████████████████████████| 2222/2222 [00:23<00:00, 95.49it/s]
TEST[epoch:3]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 65.16it/s]
train epoch:4: 100%|███████████████████████████████████████████| 2222/2222 [00:23<00:00, 95.83it/s]
TEST[epoch:4]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 66.54it/s]
train epoch:5: 100%|███████████████████████████████████████████| 2222/2222 [00:23<00:00, 95.42it/s]
TEST[epoch:5]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 65.87it/s]
iteration-time: 118.43958020210266
88.32s user 10.04s system 76% cpu 2:08.31 total
```

### torchtext

```
train epoch:1: 100%|███████████████████████████████████████████| 2222/2222 [00:31<00:00, 70.99it/s]
TEST[epoch:1]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 65.46it/s]
train epoch:2: 100%|███████████████████████████████████████████| 2222/2222 [00:30<00:00, 71.77it/s]
TEST[epoch:2]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 67.12it/s]
train epoch:3: 100%|███████████████████████████████████████████| 2222/2222 [00:30<00:00, 71.78it/s]
TEST[epoch:3]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 66.63it/s]
train epoch:4: 100%|███████████████████████████████████████████| 2222/2222 [00:30<00:00, 71.81it/s]
TEST[epoch:4]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 67.25it/s]
train epoch:5: 100%|███████████████████████████████████████████| 2222/2222 [00:30<00:00, 71.94it/s]
TEST[epoch:5]: 100%|███████████████████████████████████████████████| 10/10 [00:00<00:00, 68.14it/s]
iteration-time: 155.81213545799255s
51.78s user 1.74s system 32% cpu 2:46.48 total
```

# more example

```python
from flowder.pipes import select, add_sos, add_eos
from flowder.processors import VocabBuilder

from flowder import lines, filtered
from flowder.source.depend_func import depend


def filter_by_data_length(max_data_length):
    # 依存変数の指定
    @depend(max_data_length)
    def wrapper(k):
        return 0 < len(k) <= max_data_length

    return wrapper


train_src, train_trg = lines("train_src.txt"), lines("train_trg.txt")
dev_src, dev_trg = lines("dev_src.txt"), lines("dev_trg.txt")

# pattern matching pipeline
filt = filtered(filter_by_data_length(50))
filtered_data = (train_src * train_trg) | (split(), split()) | (filt, filt)
filtered_data = filtered_data.cache("name")

# build vocab
src = VocabBuilder("src", max_size=24000)
trg = VocabBuilder("trg", max_size=24000)
(filtered_data | select(0)) >> src
(filtered_data | select(1)) >> trg

# preprocess pipeline
train_src = filtered_data | select(0) | src.numericalizer
train_trg = filtered_data | select(1) | trg.numericalizer | add_sos() | add_eos()
train_data = train_src * train_trg
val_data = (dev_src * dev_trg) | \
           (split(), split()) | \
           (src.numericalizer, trg.numericalizer) | \
           (None, (add_sos() | add_eos()))

```
