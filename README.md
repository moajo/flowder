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
from flowder import file, zip_source, directory

# create data source
lines_source = file("text.txt").lines()

# data source is iterable
for line in lines_source:
  print(line)

# support zip/map/filter
for text, annotation in zip_source(lines_source, file("anno.csv").csv()):
  print(text, annotation)

# load directory files as image
image_source = directory("imgs").filter(lambda x: x[-4:]==".png").image()

# index access
first_img = image_source[0]
```

# DataModel
## Source
Sourceは反復可能なオブジェクトで、map/filterなどの演算によって連鎖します。
これらの計算はすべて遅延評価され、必要な部分のみがメモリ上に読み込まれます。
```python
source = file("big_data.txt").lines().map(complex_process).filter(predicate)

# 計算結果をファイルにキャッシュできます。計算は最初にiterが呼ばれたタイミングで行われます。
file_cache = source.file_cache("processed_data")

# file_cache.load()　# 手動で計算のタイミングを指定できます

for data in file_cache:
  pass # 2回目の実行以降は、キャッシュの値が使用されます
```
## Field
対象となるSourceを指定し、値全体の統計量を計算するような前処理や、それを使ったデータの変換を管理します。
Fieldオブジェクトはデータ項目ごとに用意され、マージして後述するDatasetオブジェクトを作成します。
TextFieldはテキスト用のFieldのプリセットで、語彙生成とキャッシュ、sos/eosの挿入とindex化ができます。
```python
train_en_loader = file("train.en").lines()
train_ja_loader = file("train.ja").lines()
en_vocab_processor = BuildVocab(
    cache_file=".tmp/cache_en",
    max_size=VOCAB_SIZE,
)
ja_vocab_processor = BuildVocab(
    additional_special_token=[sos_token, eos_token],
    cache_file=".tmp/cache_ja",
    max_size=VOCAB_SIZE,
)
src = TextField("src",
                train_en_loader,
                vocab_processor=en_vocab_processor)
trg = TextField("trg",
                train_ja_loader,
                eos_token=eos_token,
                sos_token=sos_token,
                vocab_processor=ja_vocab_processor)
```

## Dataset
Datasetは複数のFieldを束ね、Sourceの依存グラフをもとに計算を最適化します。
```python
ds = create_dataset(len(train_en_loader), src, trg)
ds.preprocess()# Fieldの前処理の実行
# ds.map(func).filter(pred) # DatasetはSourceでもあります
```
## Iterator
マルチプロセスで非同期にデータをロードするイテレータを提供します。
バッチの作成や前処理を別プロセスで行うので、メインプロセスのイテレーションが高速化します。
また、開始時にデータをまとめてロードする必要がなくなります。
BucketIteratorはシーケンスデータ用のイテレータで、バッチ内のシーケンスの長さが近くなるようにして計算効率を上げます。
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
前処理：テキストをトークナイズしてwordIndexに変換、<sos>/<eos>の挿入、バッチ内のデータの長さをできるだけ揃えて(BucketIterator)ソートしてpadding
Vocabは事前に作成済み
flowderは事前読み込みなしで学習中に非同期でデータ生成
torchtextは途中まで前処理済みデータをpickleでキャッシュして、学習前にロードする

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
"""`file1.txt` is tab sepalated parallel corpus
世界 World
こんにちは hello
"""

from flowder import file

# open file and get lines iterator
source = file("data_foreach_line.txt").lines()

for line in source:
  print(line) #print 2 line

split = source.split("\t") # convert item: str->tuple(str)
ja = split.item[0] # convert for each item
en = split.item[1]

for sent in ja:
  print(sent) # 世界->こんにちは

vocab = BuildVocab(
  additional_special_token=["<sos>"],
  cache_file=".tmp/en_vocab"
) # custom vocab builder with auto cacheing

src = TextField("src", ja, tokenizer=lambda s:s.split()) # custom tokenizer
trg = TextField("trg", en, lowercase=True, vocab=vocab, sos_token="<sos>") # there are more options

data_size = len(ja) # obtain length
ds = create_dataset(data_size, src, trg)

it = flowder.create_iterator( # loading data async in background process
    ds,
    batch_size=10,
    shuffle=True,
)

for data in it:
  print(data) # {"src": [2], "trg": [2]}
  print([vocab.itos[i] for i in data["trg]]) # ["world"]

```
