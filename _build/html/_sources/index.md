# はじめに

```{only} html
%[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chokkan/mlnote/blob/main/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-numpy](https://img.shields.io/badge/Made%20with-NumPy-1f425f.svg)](https://numpy.org/)
[![made-with-matplotlib](https://img.shields.io/badge/Made%20with-Matplotlib-1f425f.svg)](https://matplotlib.org/)
[![made-with-scikit-learn](https://img.shields.io/badge/Made%20with-scikit--learn-1f425f.svg)](https://scikit-learn.org/)
[![made-with-tensorflow](https://img.shields.io/badge/Made%20with-TensorFlow-1f425f.svg)](https://tensorflow.org/)
```

## この教材について
この教材は創薬等先端技術支援基盤プラットフォーム（Basis for Supporting innovative Drug Discovery and Life Science Research（BINDS））に関連する方向けに，**Python** の使い方，基本的な**機械学習法**の実装の仕方，**深層学習法**の実装の仕方を紹介する教材です．

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
プログラミング言語 [Python](https://www.python.org/) の利用方法を紹介します．Pythonには機械学習法を実装するための便利なライブラリを利用することができるデータ科学を利用した解析に便利な言語です．

```python
#!/usr/bin/env python3

def main():
    print("Hello world")
     
if __name__ == "__main__":
    main()
```

---
**Scikit-learn**
^^^
解説する．

---
**TensorFlow**
^^^
ほげ

:::

### このコンテンツで学ぶこと
このコンテンツの目的は，ウェブベースの計算環境である Jupyter Notebook（このウェブページを形作っているもの）を利用して，Python の基本的な動作を習得することです．このコンテンツは東北大学大学院情報科学研究のプログラミング初学者向けの授業「ビッグデータスキルアップ演習（Big Data Skillup Training）」の内容の一部を日本語の e-learning コンテンツとして再構築したものです．
```{note}
つまり，このコンテンツに間違いが含まれていても山田はまったく悪くなくて元の作成者が悪いです．
```
### この環境について
Jupyter Notebook は Python を実行するための環境です．メモを取りながら Python のコードを実行することができます．この環境は，Python プログラムがコマンドライン上で実行される実際の環境とは少し異なるのですが，Python プログラムがどのように動くかということを簡単に確認しながら学習することができます．

### Google Colaboratory
Google Colaboratory（グーグルコラボラトリー）は Jupyter Notebook のファイルをウェブブラウザから使えるように Google が用意してくれたアプリです．各ページに以下のようなアイコンがあるのでこれをクリックして各ページのファイルを Google Colaboratory 上で開いて利用してください．

ロケットアイコン <i class="fa fa-rocket" aria-hidden="true"></i> を押す

<i class="fa fa-rocket" aria-hidden="true"></i>

![Colab](https://colab.research.google.com/assets/colab-badge.svg)
<i class=\"fa fa-rocket\" aria-hidden=\"true\"></i>

### グーグルコラボラトリーでの GPU の利用方法

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
