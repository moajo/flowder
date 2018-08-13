import torch
from moajo_tool.utils import measure_time
from torch.utils.data.dataloader import default_collate

from moajoloader.abstracts import Field
from moajoloader.iterator import create_bucket_iterator
from moajoloader.utils import file, create_dataset, directory, collect
from moajoloader.fields import TextField
# from iterator import Iterator
from moajoloader.processors import BuildVocab
from moajoloader.sources import Source, TextFileSource, ZipSource, ImageSource
import torchtext as txt
import torchnet as tnt
from tqdm import tqdm


def main():
    # gitignore = file("example/data/kftt.ja")
    # hoge = gitignore.create()
    # for k in hoge:
    #     print(k)
    # ls = gitignore.lines()
    # ds = ls.create()
    # for l in ds:
    #     print(l)
    #
    # # splitによるMapのテスト
    # spl = ls.split()
    # for l in spl:
    #     print(l)
    #
    # f = TextField("hoge",ls)
    # f.name = "hogehoge"
    # ds = ls.create(f, return_as_tuple=False)
    # ds.preprocess()
    # for l in ds:
    #     print(l)

    # test_iter = Iterator(ds, 2, shuffle=False, repeat=False,
    #                      sort_key=lambda a: len(a[0]),
    #                      sort_within_batch=True,
    #                      device=-1,
    #                      )
    #
    # for batch in test_iter:
    #     print(batch)

    @measure_time()
    def hoge():
        src = txt.data.Field(include_lengths=True)
        trg = txt.data.Field(include_lengths=True)
        ds = txt.datasets.TranslationDataset(path="../___main/DATA/kftt/kyoto_tokenized.", exts=("en", "ja"),
                                             fields=[('src', src), ('trg', trg)])
        src.build_vocab(ds)
        trg.build_vocab(ds)
        return src, trg, ds

    @measure_time()
    def hoge_2():
        kftt_ja = file("../___main/DATA/kftt/kyoto_tokenized.en").lines()
        kftt_en = file("../___main/DATA/kftt/kyoto_tokenized.ja").lines()
        src = TextField("src", kftt_ja, vocab_processor=BuildVocab(cache_file="./tmp/hogehoge_src"))
        trg = TextField("trg", kftt_en, vocab_processor=BuildVocab(cache_file="./tmp/hogehoge_trg"))
        ds = create_dataset(len(kftt_ja), src, trg, return_as_tuple=False)
        ds.preprocess()
        return ds

    # hoge()
    ds = hoge_2()

    # kwargs = {'num_workers': 1, 'pin_memory': False}
    kwargs = {}

    train_iter = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, collate_fn=lambda s: s, **kwargs)
    it2 = create_bucket_iterator(ds, 100, lambda x: len(x["src"]), )

    @measure_time()
    def hoge2():
        print("2 Iterator start")
        c = 0
        for i, d in tqdm(enumerate(it2)):
            c += len(d)
        print("end 2", c)

    @measure_time()
    def hoge3():
        print("3 raw start")
        c = 0
        for i, d in tqdm(enumerate(ds)):
            c += 1
        print("end 3", c)

    @measure_time()
    def hoge4():
        print("4 dl start")
        c = 0
        for i, d in tqdm(enumerate(train_iter)):
            c += len(d)
        print("end 4", c)

    # hoge3()
    # hoge2()
    hoge4()
    hoge2()
    # hoge4()

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
    # 具体例 celebA　画像ファイルとtsvアノテーション
    d = directory("example/data/celebA/img_align_celeba")
    d.create()[0]
    anno = file("example/data/celebA/list_attr_celeba.txt").csv(header=1, sep="\s+")
    assert len(anno) == 8
    # for v in anno.item["5_o_Clock_Shadow"]:
    #     print(v)
    # img_files = anno.item(0)  # strのセット
    # a = map_zip(
    #     directory("img_align_celeba").files(),
    #     img_files,
    # )
    files = anno.item[0]
    imgs = collect(files, d.item.name, d).to(ImageSource)
    f = Field(process=mean(), postprocess=whitening())
    ds = imgs.create_datsset()


if __name__ == '__main__':
    main()
