# はじめに

```{only} html
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-numpy](https://img.shields.io/badge/Made%20with-NumPy-1f425f.svg)](https://numpy.org/)
[![made-with-matplotlib](https://img.shields.io/badge/Made%20with-Matplotlib-1f425f.svg)](https://matplotlib.org/)
[![made-with-scikit-learn](https://img.shields.io/badge/Made%20with-scikit--learn-1f425f.svg)](https://scikit-learn.org/)
[![made-with-tensorflow](https://img.shields.io/badge/Made%20with-TensorFlow-1f425f.svg)](https://tensorflow.org/)
![Colab](https://colab.research.google.com/assets/colab-badge.svg)
%[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamada-kd/binds-training/blob/main)
```

## この教材について
この教材は未踏スケールデータアナリティクスセンターと創薬等先端技術支援基盤プラットフォーム（Basis for Supporting innovative Drug Discovery and Life Science Research（BINDS））に関連する方向けに，**Python** の使い方，基本的な**機械学習法**の実装の仕方，**深層学習法**の実装の仕方を紹介する教材です．

:::{panels}
:container:
:column: col-lg-6 px-2 py-2
:card:

---
**対象者**
^^^
これからデータ科学を用いた解析技術を身につけたい人．プログラミングの経験，機械学習に関する知識はないことを前提にしています．

---
**Python**
^^^
プログラミング言語 [Python](https://www.python.org/) の利用方法を紹介します．Python は機械学習法を実装するための便利なライブラリを有する，データ科学を利用した研究開発に便利な言語です．

```python
#!/usr/bin/env python3

def main():
    print("Hello world")
     
if __name__ == "__main__":
    main()
```

---
**scikit-learn**
^^^
[scikit-learn](https://scikit-learn.org/) は機械学習法を実現するためのライブラリです．決定木，サポートベクトルマシン，クラスタリング等の様々な方法を利用することができます．scikit-learn は利用方法がとても簡単でそれらをコマンド一発で実行することができます．機械学習アルゴリズムはそれぞれ性質の異なるものですが，scikit-learn には基本的な書き方があり，それさえ学んでしまえば，ある機械学習法で何らかのデータを解析するプログラムを別の機械学習法でデータを解析するプログラムに書き換えることが容易にできます．深層学習法のライブラリとしては TensorFlow や PyTorch がありますが，それらを使う必要がない場合にはこのライブラリをまずは使ってみることは良い選択肢のひとつかもしれません．

---
**TensorFlow**
^^^
[TensorFlow](https://www.tensorflow.org/) は世界で最も利用されている深層学習フレームワークです．scikit-learn ではできない複雑なニューラルネットワーク構造を実現することができます．TensorFlow はひとつのフレームワークなのですが，コーディングをする際にいくつかの書き方があります．Sequential な書き方，Functional な書き方，Subclassing な書き方です．Sequential な書き方はとても簡単にニューラルネットワークを実現することができます．しかし，拡張性が高くありません．Subclassing API は最も柔軟な書き方が可能です．習得は大して難しくありません．習得の難易度とニューラルネットワークに対する理解を得られる度合いや柔軟にネットワークを構築できる利点を天秤にかけたときに，Subclassing API を最初に学習した方が得られるものが多いと思い，これを紹介します．Subclassing API は [PyTorch](https://pytorch.org/)（元々は [Chainer](https://chainer.org/) の書き方 = define by run）とほぼ同じ書き方です．

:::

### このコンテンツで学ぶこと
このコンテンツの目的は，ウェブベースの計算環境である Jupyter Notebook（このウェブページを形作っているもの）を利用して，Python や機械学習ライブラリの基本的な動作を習得することです．このコンテンツは東北大学大学院情報科学研究科のプログラミング初学者向けの授業「ビッグデータスキルアップ演習（Big Data Skillup Training）」の内容の一部を日本語の e-learning コンテンツとして再構築したものです．
```{note}
つまり，このコンテンツに間違いが含まれていても山田はまったく悪くなくて元の作成者が悪いです．
```
### この環境について
Jupyter Notebook は Python を実行するための環境です．メモを取りながら Python のコードを実行することができます．この環境は，Python プログラムがコマンドライン上で実行される実際の環境とは少し異なるのですが，Python プログラムがどのように動くかということを簡単に確認しながら学習することができます．

## 教材の利用方法
この教材は Google Colaboratory（グーグルコラボラトリー）を利用して作られています．グーグルコラボラトリーは Jupyter Notebook のファイルをウェブブラウザから使えるように Google が用意してくれたアプリです．各ページの上部にロケットのアイコン <i class="fa fa-rocket" aria-hidden="true"></i> があるのでこれをクリックして各ページのファイルを Google Colaboratory 上で開いて利用してください．

### GPU の利用方法

グーグルコラボラトリーで GPU を利用するには上のメニューの「ランタイム」から「ランタイムのタイプを変更」と進み，「ハードウェアアクセラレータ」の「GPU」を選択します

### 開始前に行うこと

```{hint}
グーグルコラボラトリー自体の一番上の「ファイル」をクリックし，さらにポップアップで出てくる項目から「ドライブにコピーを保存」をクリックし，自身のグーグルドライブにこのウェブページ全体のソースを保存します（グーグルのアカウントが必要です）．こうすることによって，自分で書いたプログラムを実行することができるようになります．また，メモ等を自由に以下のスペースに追加することができるようになります．
```

### 進め方

上から順番に読み進めます．Python のコードが書かれている場合は実行ボタンをクリックして実行します．コード内の値を変えたり，関数を変えたりして挙動を確かめてみてください．

### コードセル

コードセルとは，Python のコードを書き込み実行するためのセルです．以下のような灰色のボックスで表示されていますす．ここにコードを書きます．実行はコードセルの左に表示される「実行ボタン」をクリックするか，コードセルを選択した状態で `Ctrl + Enter` を押します．環境によっては行番号が表示されていると思いますので注意してください（行番号の数字はプログラムの構成要素ではありません）．

```python
print("This is a code cell.")
```
