#!/usr/bin/env python
# coding: utf-8

# # 演習問題

# ## はじめに

# この演習問題集に関する説明をします．

# ### 問題について

# この演習問題集では，簡単に答えが見つからない，回答者に難易度の設定すら委ねるような問題がいくつか含まれています．この講義の対象者がデータ科学的な解析技術を有しないものの，自身の専門を持つ研究者であるからです．データ科学は問題解決のアプローチに過ぎず，既に研究をされている研究者なら解くべき問題に対する調査や取り組み方法を持っていると思ったからこのようにしました．解答のためのコードセルは適宜増やしてください．

# ### 配点と採点基準

# 配点は各問題についてそれぞれ 10 点です．採点者は以下のような採点基準で点数をつけます．
# 
# | 点 | 基準 |
# | --- | --- |
# | 10 | この解答が良い評価を得ないのであれば二度と採点者を引き受けない． |
# | 9 | この解答が良い評価が与えられるために戦う． |
# | 8 | この解答は良い評価を得るべきである． |
# | 7 | この解答は良い評価を得るべきであるが，場合によって悪い評価を得たしても許容可能． |
# | 6 | この解答が良い評価を得てほしいものの，そうでなくても問題ない． |
# | 5 | この解答が良い評価を得てほしくないものの，そうでなくても問題ない． |
# | 4 | この解答は良い評価を得ないべきであるが，場合によって良い評価を得たとしても許容可能． |
# | 3 | この解答は良い評価を得ないべきである． |
# | 2 | この解答に良い評価を与えられないために戦う． |
# | 1 | この解答が良い評価を得るのだとしたら二度と採点者を引き受けない． |
# | 0 | 解答がないため採点できない． |
# 

# ## プログラミングの基礎
# 
# 

# 「プログラミングの基礎」で触れた事柄に関する問題集です．講義資料に載っていない事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 1-1 行列の掛け算

# 以下の行列の掛け算を NumPy 等のライブラリを用いずに計算するためのプログラムを書いてください．
# 
# $
#   \left[
#     \begin{array}{ccc}
#       1 & 2 & 3 \\
#       3 & 4 & 5\\
#     \end{array}
#   \right]
#   \times
#   \left[
#     \begin{array}{cc}
#       5 & 6 \\
#       6 & 7 \\
#       7 & 8 \\
#     \end{array}
#   \right]
# $

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{note}
# 行列の掛け算なんてライブラリ使わずにやることなんてないからなんて不毛な問題なんだ，と思うかもしれないのですが，プログラミング初心者がこれをできるようになるということは `for` の使い方と動き方をしっかり理解していることの確認となり，その観点からは有意義です．
# ```

# ### 1-2 ファイルの処理

# シェルのコマンドに `wget` というものがありますが，これを使うとインターネット上から情報を取得することができます．Wikipedia の「334」という項目は以下の URL に載っています．
# 
# https://ja.wikipedia.org/wiki/334
# 
# この URL にアクセスして表示されるテキストの中に `334` という値が何個あるか数えるプログラムを作ってください．

# 最初に，`wget` で URL にアクセスしてその情報全部を `wikipedia-334.txt` というファイル名で現在のディレクトリに保存してください．

# In[ ]:


# ここにシェルのコマンドを書く．


# ```{hint}
# オプションコマンド `-O` を利用するとファイル名を指定して保存できます．
# ```

# 次に，このファイルを Python で読んで `334` という文字列が何個あるかカウントして，カウントした結果だけを出力してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# ファイルがどこにあるかわからない人は「カレントディレクトリ」という概念を調べてみましょう．
# ```

# ## scikit-learn を利用した機械学習
# 
# 

# 「scikit-learn を利用した機械学習」で触れた事柄に関する問題集です．これも講義資料に載っていない事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 2-1 探索的データ解析

# 探索的データ解析とはこれから利用しようとするデータセットを様々な角度から観察してデータの性質を把握しようとする行為のことです．英語では exploratory data analysis（EDA）と呼ばれます．ここでは，scikit-learn から利用可能な breast cancer wisconsin (diagnostic) dataset なるデータセットを利用します．以下のように読み込むことができます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer

def main():
    dBreastCancer = load_breast_cancer()
    print(dBreastCancer)

if __name__ == "__main__":
    main()


# ```{hint}
# なんとなく紹介したくなくて触れなかったのですが，`pandas` ていうライブラリのデータフレームというものの使い方を調べるとこのデータを上手に扱えるようになると思います．
# ```

# 最も簡単な解析はデータセットのインスタンスサイズを知ることであったり，各アトリビュートの意味を知ることであったり，教師データの性質を知ることであったりするでしょう．次に，各アトリビュートと教師データの関連性を図で表現することが考えられます．

# 最初に，何か図を利用しない解析を行う何らかのプログラムを書いてください．思いつく限りの解析をたくさん行ってください．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer

def main():
    dBreastCancer = load_breast_cancer()
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# 平均値，中央値，最頻値，最大値，最小値とか色々ありますよね
# ```

# 次に，何か図を描画することでデータの性質を把握するためのプログラムを書いてください．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def main():
    dBreastCancer = load_breast_cancer()
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# 図示すると変数間の関連性を理解しやすいですよね．
# ```

# ### 2-2 人工知能の性能比較

# データセット breast cancer wisconsin (diagnostic) dataset は肺の良性腫瘍と悪性腫瘍に関するデータを含んでいます．この分類器を決定木を用いて構築してファイルとして保存してください．このとき，以下のことをやってください．
# 
# *   元のデータセットの 20% をテストデータセットとする．
# *   乱数の種を固定する．
# *   グリッドサーチとクロスバリデーションでハイパーパラメータを最適化する．
# 
# 
# 

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# 上で保存した人工知能を新たに呼び出して，テストデータセットを利用していくつかの評価指標を用いて性能の評価値を出力してください．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# 勾配ブースティングと呼ばれる機械学習モデルを利用して予測器を構築し，上と同様のテストデータセットで同様の評価指標を用いて性能の評価値を出力してください．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_breast_cancer

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# 決定木の性能と勾配ブースティングの性能の差に意味のある差がありそうなのかについて，適切な統計検定法を用いて，有意水準 1% で評価してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# 対応のない t 検定，対応のある t 検定，ウィルコクソンの順位和検定，ウィルコクソンの符号順位検定，ダネット検定，チューキー・クレーマー検定，スティール検定，スティール・ドゥワス検定．たくさん検定法があってその比較に適した方法を選ばなければなりませんよね．フィッシャーの直接確率検定とかありますよね．
# ```

# ### 2-3 計算の速さの比較

# 以下のようにすると MNIST というデータセットをダウンとロードして読み込むことができます．ここでは MNIST が何であるかに触れませんが，10 個のクラス分類のためのデータセットであり，各インスタンスは 784 の長さのベクトルであり，公式から提供されている学習データセットのサイズは 60000 個であることだけお知らせします．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    (learnX, learnT), (textX, testT) = tf.keras.datasets.mnist.load_data()

if __name__ == "__main__":
    main()


# このデータセットは以下のようにすることで簡単に scikit-learn で利用することができます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

def main():
    (learnX, learnT), (textX, testT) = tf.keras.datasets.mnist.load_data()
    learnX = learnX.reshape(-1, 784)
    predictor = DecisionTreeClassifier()
    predictor.fit(learnX, learnT)

if __name__ == "__main__":
    main()


# 利用する機械学習モデルを変えるとその評価性能が変わるのと同じように，利用する機械学習モデルによって学習に要する計算時間も変わります．各モデルの計算時間はデータセットのサイズはデータセットに含まれるインスタンスのサイズや実装の方法に依存して変わります．例えば，サポートベクトルマシンはサンプルサイズの 3 乗に依存して計算時間が増加します．計算時間の観点から自身が解析したいデータセットの特性に合わせて機械学習モデルを選択することもあり得るでしょう．この問題では，サポートベクトルマシンを含めて少なくとも 3 個の機械学習モデルを自由に選択してそれらを MNIST を用いて学習させてください．テストデータセットにおける評価値と共にそれぞれの方法が学習に要した時間を出力してください．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    (learnX, learnT), (textX, testT) = tf.keras.datasets.mnist.load_data()
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{note}
# 良い予測性能が出るように工夫することも忘れないでくださいね．ただし，それぞれのモデルを同じ基準で比較しなければなりませんよね．
# ```

# ```{note}
# ここではデータセットとして MNIST を利用しました．MNIST のデータセットのサイズが大きいため計算時間に差が出やすいためです．
# ```

# ## 深層学習
# 
# 

# 「深層学習」で触れた事柄に関する問題集です．またしても講義資料にギリ載っているか載っていないような事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 3-1 性能向上テクニック

# 講義資料では MLP を MNIST で学習させて，その性能評価をしました．MNIST はとても簡単な問題なので，エポックを経ると簡単に評価値が良くなりました．ここでは普通には予測性能を高めることが難しい不均衡データセットを用います．正例と負例のインスタンスサイズが大きく違うデータセットのことです．このようなデータを scikit-learn を用いて以下のように生成します．生成されるデータの `wholeX` はインプットデータで，`wholeT` はターゲットデータです．

# In[ ]:


#!/usr/bin/env python3
from sklearn.datasets import make_classification

def main():
    wholeX, wholeT = make_classification(n_samples=50000, n_features=200, n_classes=2, weights = [0.96, 0.04], flip_y=0, random_state=0)
    print(wholeX)

if __name__ == "__main__":
    main()


# ここでは 50000 個のデータを生成していますが，ここから 10000 個だけテストデータセットとして用いて良い予測性能が出るように MLP を成長させてください．MLP は TensorFlow で実装してください．また，評価基準には ACC，MCC，F1 スコアを含んでください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# ナイーブにはコスト関数に重みをつけるという方法が考えられます．
# ```

# ### 3-2 感情分析

# インターネット上の映画レビューのデータセット IMDb と呼ばれるものを TensorFlow から簡単にダウンロードして利用することができます．以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    (learnX,learnT), (testX,testT) = tf.keras.datasets.imdb.load_data()

if __name__ == "__main__":
    main()


# このデータセットを用いて何らかのニューラルネットワークを学習させ，テストデータセットでその性能を評価してください．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    (learnX,learnT), (testX,testT) = tf.keras.datasets.imdb.load_data()
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# 上のデータセットは既にトークナイズされたデータでしたが，Hugging Face を利用すれば元のデータをダウンロードできるのでした．以下のようにします．

# In[ ]:


get_ipython().system(' pip3 install datasets')


# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset

def main():
    imdb = load_dataset("imdb")
    print(imdb["train"][0])

if __name__ == "__main__":
    main()


# このデータセットに対して Hugging Face を利用して感情分析の予測結果を計算し，また，その予測性能の評価値を計算して，上で作ったモデルとの比較をしてください．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset

def main():
    imdb = load_dataset("imdb")
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# どうでしょう．「Hugging Face めちゃすげー」てなりますかね．
# ```

# ### 3-3 アテンションの活用

# サンプルデータセットとして利用してきた MNIST の各インスタンスは画像データではありますが，それらは 784 個の実数の要素からなるベクトルでもあります．アテンション機構によって扱いやすいデータの形式をしています．アテンション機構を含む何らかのニューラルネットワークを利用して MNIST の分類器を構築してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# 同程度のパラメータサイズを持つ MLP による予測器も同時に構築して，同程度の予測性能に達するまでの時間を比較してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# やっぱ MLP て使い方次第じゃ最強なんすわ．
# ```

# ### 3-4 普通の CGAN の実装

# 教材において CGAN の実装をしたときには WGAN-gp のコスト関数を利用しましたが，最も基本的な GAN と同じコスト関数を利用して CGAN を書き直してください．WGAN-gp を応用した場合の CGAN と比べて性能が落ちていることを目て見て確認してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# 基本的な GAN を少しずつ変えていく方が WGAN-gp のコスト関数を利用した CGAN を少しずつ変えるやり方より簡単でしょう．
# ```

# ### 3-5 強化学習法の実装

# 以下のような 5 行 9 列の行列を考えます．このグリッドワールド上で何らかのオブジェクト動かしてゴール（G）を目指すこととします．オブジェクトはフィールド上のグリッド線で囲まれたどこかの位置に存在できることとします．オブジェクトは上下左右に動かせるものとします．ただし，フィールドは壁に囲まれており，オブジェクトはこのグリッドワールドの外には出られません．また，フィールドには障害物（X）があり，障害物が存在する位置にオブジェクトは行けないこととします．オブジェクトは最初ゴールでも障害物でもないオブジェクトが移動可能なマス（S）に位置しているものとします．このオブジェクトをうまく移動させてゴールまで導くエージェントを何らかの強化学習法を用いて作ってください．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/largeField.svg?raw=1" width="100%" />

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# 深層 Q ネットワークを利用しても良いし，ニューラルネットワークを用いない Q 学習法でも他の方法でも何でも良いです．
# ```

# ```{note}
# 終わりです．
# ```
