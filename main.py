from moajo_tool.utils import measure_time

from abstracts import Dataset
from utils import file, create_dataset
from fields import TextField
from iterator import Iterator
from processors import BuildVocab
from sources import Source, TextFileSource, ZipSource
import torchtext as txt


def main():
    gitignore = file("example/data/kftt.ja")
    hoge = gitignore.create()
    # for k in hoge:
    #     print(k)
    ls = gitignore.lines()
    # ds = ls.create()
    # for l in ds:
    #     print(l)
    #
    # #splitによるMapのテスト
    # spl = ls.split()
    # for l in spl:
    #     print(l)
    #
    # f = TextField(ls)
    # ds = ls.create(f)
    # ds.preprocess()
    # for l in ds:
    #     print(l)

    # f = TextField(ls)
    # f.name="hogehoge"
    # ds = ls.create(f,return_raw_value_for_single_data=False)
    # ds.preprocess()
    # for l in ds:
    #     print(l)
    #
    # test_iter = Iterator(ds, 2, shuffle=False, repeat=False,
    #                               sort_key=lambda a: len(a[0]),
    #                               sort_within_batch=True,
    #                               device=-1,
    #                               )

    # for batch in test_iter:
    #     print(batch)

    # @measure_time()
    # def hoge():
    #     src = txt.data.Field(include_lengths=True)
    #     trg = txt.data.Field(include_lengths=True)
    #     ds = txt.datasets.TranslationDataset(path="../___main/DATA/kftt/kyoto_tokenized.", exts=("en", "ja"),
    #                                          fields=[('src', src), ('trg', trg)])
    #     src.build_vocab(ds)
    #     trg.build_vocab(ds)
    #     return src,trg,ds
    #
    # dd = hoge()

    kftt_ja = file("../___main/DATA/kftt/kyoto_tokenized.en").lines()
    kftt_en = file("../___main/DATA/kftt/kyoto_tokenized.ja").lines()
    # zipped = zip_source(kftt_en, kftt_ja)

    src = TextField("src", kftt_ja, vocab_processor=BuildVocab(cache_file="./tmp/hogehoge_src"), include_length=True)
    trg = TextField("trg", kftt_en, vocab_processor=BuildVocab(cache_file="./tmp/hogehoge_trg"), include_length=True)
    # ds = zipped.create(src, trg)
    ds = create_dataset(len(kftt_ja), src, trg)
    ds.preprocess()
    # ds.load_to_memory()

    it = Iterator(ds, 100, sort_key=lambda a: len(a[0]), shuffle=True)
    for i, d in enumerate(it):
        print(i)

    # データ
    # src = Field(include_lengths=True)  # でーたの各項目に対してどう前処理してどうロードするかを定める。
    # trg = Field(include_lengths=True, eos_token="<eos>", init_token="<sos>")
    # data = file("file").lines().split("\t")  # 指定ファイル
    #
    # # 型
    # f = file("tokenized.en")  # ファイル
    # f.lines()  # 各行 lineSet型
    #
    # # 具体例 kftt 別ファイル
    # kftt = file("tokenized.en").lines() | file("tokenized.ja").lines()  # linesetはzip演算子|でtuplesetになる
    # for en, ja in kftt:
    #     pass  # 各行のタプルイテレーション
    # src = Field(include_lengths=True, use_vocab=True)
    # src = Field(preprocess=tokenize(), process=normalize() | build_vocab(max_size=1000), postprocess=indexing())
    # trg = Field(include_lengths=True, use_vocab=True)
    # kftt.item[0] >> src
    # kftt.item[1] >> trg
    # ds = kftt.create_dataset()  # 非同期iterable lengthと__iter__がある
    #
    # # 具体例 aspec　特殊記号区切り
    # kftt = file("train-1.txt").lines().split("|||")  # tupleset
    # src = Field(include_lengths=True, use_vocab=True, tokenizer="mecab_hoghoge")  # トーク内座指定。デフォルトはspace split?
    # trg = Field(include_lengths=True, use_vocab=True)
    # kftt.item[3] >> src
    # kftt.item[4] >> trg
    # ds = kftt.create_dataset()  # 非同期iterable lengthと__iter__がある
    #
    # # 具体例 openSUB　複雑怪奇
    # linkGrps = xml("en-fr.xml").children("linkGrp")  # linkGrpのセット(xmlとして読んで特定の名前の小要素のセットにする。)
    # fromDocs, toDocs = linkGrps.sub("fromDoc", "toDoc")  # linkGrpをmapして分岐
    # m = fromDocs.only_exist_file().open("gzip").xml() | toDocs.only_exist_file().open("gzip").xml()
    # fromDocs = m.item(0)
    # toDocs = m.item(1)
    #
    # links = linkGrps.children("link")  # 「linkのセット」のセットになる
    # xtargets = links.get("xtargets").split(";").filter(lambda t, f: t != "" and f != "")  # [strのtupleのセット]のセット
    # fs = xtargets.item(0).split(" ")
    # ts = xtargets.item(1).split(" ")
    # # TODO 未完・・・
    #
    # # 具体例 celebA　画像ファイルとtsvアノテーション
    # anno = tsv("list_attr_celeba.txt")
    # img_files = anno.item(0)  # strのセット
    # # a = map_zip(
    # #     directory("img_align_celeba").files(),
    # #     img_files,
    # # )
    # imgs = img_files.map(lambda a: "img_align_celeba/" + a).open_img()  # imgのセット
    # f = Field(process=mean(), postprocess=whitening())
    # imgs >> f
    # ds = imgs.create_datsset()
    #
    # # 具体例 タイタニック　csv
    # ds = csv("train.csv").create()


if __name__ == '__main__':
    main()
