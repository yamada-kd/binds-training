#!/usr/bin/env python
# coding: utf-8

# # 教師なし学習法

# #1. <font color="Crimson">はじめに</font>
# 

# ##1-1. <font color="Crimson">このコンテンツで学ぶこと</font>

# Scikit-learn とは様々な機械学習法を Python から利用するためのライブラリです．サポートベクトルマシン，ニューラルネットワーク，決定木等の様々な機械学習法が実装されています．それらをコマンド一発で実行することができます．機械学習アルゴリズムはそれぞれ性質の異なるものですが，Scikit-learn には基本的な書き方があり，それさえ学んでしまえば，ある機械学習法で何らかのデータを解析するプログラムを別の機械学習法でデータを解析するプログラムに書き換えることが容易にできます．深層学習法のライブラリとしては TensorFlow や PyTorch がありますが，それらを使う必要がない場合にはこのライブラリをまずは使ってみることは良い選択肢のひとつかもしれません．
# 
# 

# ##1-2. <font color="Crimson">コンパニオン</font>

# このコンテンツには，受講者の理解を助けるためのコンパニオンがいます．以下のものは「サミー先生」です．サミー先生は今回のこの教材の制作者ではないのですが分かり難いところを詳しく教えてくれます．
# 
# <font color="Crimson">(9｀･ω･)9 ｡oO(ちゃお！)</font>

# #2. <font color="Crimson">機械学習入門</font>
# 

# ##2-1. <font color="Crimson">機械学習をする際の心構え</font>

# ###2-1-1. <font color="Crimson">データ分割の大切さ</font>

# 機械学習を行う際の最も大切なことであるデータセットの構築方法を紹介します．データセットは以下のように分割します．はじめに，元の全データを「学習セット」と「テストセット」に分けます．さらに，学習セットを人工知能の成長（アルゴリズムが持つパラメータの更新）のためだけに利用する「トレーニングセット」と学習が問題なく進んでいるか（過学習や未学習がおこっていないか）を確認するために利用する「バリデーションセット」に分けます．

# <img src="https://drive.google.com/uc?id=1rL7CPuXUa1MTM8MLWUF4_Xez4d9LSWjJ" width="70%">

# 学習の際には学習セットだけの結果を観察します．学習の際に一瞬でもテストセットにおける予測器の性能を観察するべきではありません．また，「独立であること」もしっかり定義すべきです．データが互いに独立であるとは，機械学習の文脈においては「互いに同じ分布に従うことが予想されるが，その生成過程が異なること」でしょうか．MNIST の学習セットとテストセットには，同一の手書き数字の提供者が同時に含まれている可能性があり，この観点からするとそれらは互いに独立なデータセットではないのかもしれません．

# 新たな人工知能を開発してそれを公開する場合，普通，その人工知能の性能をベンチマークし論文や学会で発表します．その性能を評価するために使うデータセットがテストセットです．もしこのテストセットが学習セットと同じような性質を持っているのであれば，その人工知能は新たなデータを処理するときに期待されるような性能を発揮できないかもしれません．学習セットとテストセットが独立でないのならば，学習セットに過剰適合させた人工知能では，あたかもとても良い性能をしているように見えてしまいます．人工知能の実際の現場での活用方法を鑑みるに，人工知能はある特定のデータにだけではなく，様々なデータに対応できる汎化能力を持たなければなりません．

# 人工知能の開発の研究に関して，その全行程において，データセットの正しい分割は何よりも大切なことです．従来からの科学の分野で研究を続けている人達にとって，機械学習に対する信頼度は未だ低い場合があります．機械学習をやる人がここを適当にやり続ける限り，「機械学習なんて眉唾」みたいな意見はなくならないことをご理解ください．

# ###2-1-2. <font color="Crimson">用語の説明</font>

# 深層学習，機械学習や人工知能に関する用語の整理をします．

# **★予測器と人工知能**
# 
# 予測器と人工知能は同じものを指します．機械学習法によって成長させる対象のことであり，機械学習を行った際の最終産物です．本体は「何らかの入力に対して何らかの出力をするアルゴリズム」と「そのアルゴリズムが必要とするパラメータ（母数）」です．このような特定の問題を解決するための人工知能は特化型人工知能といいます．
# 
# <font color="Crimson">(9｀･ω･)9 ｡oO(アトムやドラえもんみたいな汎用的に問題を解決できる存在のことは汎用人工知能といいます．また，精神を宿したような存在のことをジョン・サールという人は強い人工知能と表現しましたが，アトムとかドラえもんは汎用人工知能であり強い人工知能です．人が思い描く人工知能とはこういった存在であって，このような単なる予測器を人工知能と呼ぶことに抵抗を持つ人は多いですが，特化型人工知能と正確に呼べば納得してもらえるかもしれません．人工知能という言葉は色々なものを再定義しており，ニューラルネットワークを人工知能というのならば，高校数学で習得する最小二乗法を用いて求める線形回帰直線も人工知能です．)</font>
# 
# **★インスタンス**
# 
# データセットにあるひとつのデータのことです．分野によってデータポイントと呼ばれることがありますが，時系列データを扱う際には別の意味でデータポイントという単語が用いられることがあり，インスタンスを使った方がより正確かもしれません．
# 
# **★学習**
# 
# 人工知能にデータを読ませ成長させる過程を学習と言います．
# 
# **★回帰問題と分類問題**
# 
# 人工知能を利用して解決しようとする問題は大きく分けてふたつあります．ひとつは回帰問題です．回帰問題は人工知能に何らかの実数の値を出力させる問題です．一方で，分類問題は人工知能にクラスを出力させる問題です．クラスとは「A，B，C」とか「良い，悪い」のような何らかの分類のことです．各クラスの間に実数と他の実数が持つような大小の関係性は定義されません．これは例えば，「0という手書きの数字が書かれている画像」と「1という手書きの数字が書かれている画像」を分類するというような場合においても当てはまります．このような問題の場合，人工知能の最終的な出力は0または1でありますが，これは人工知能にとって単なるシンボルであって，人工知能は「0という手書きの数字が書かれている画像」が「1という手書きの数字が書かれている画像」より小さいから0という出力をするのではなく，単に「0という手書きの数字が書かれている画像」にある0というパターンが0というシンボルと類似しており，1というシンボルと類似していないため，0を出力するに過ぎません．
# 
# **★入力ベクトル，ターゲットベクトル，出力ベクトル**
# 
# 入力ベクトルとは人工知能を成長させるため，または，開発した人工知能に予測をさせるために入力するデータです．入力ベクトルの各インスタンスは基本的にはベクトルの型式をしているため入力ベクトルと呼ばれます（本当は入力データ，ターゲットデータ，出力データと呼んだ方が良いかもしれないです）．インプットベクトルとも呼ばれます．ターゲットベクトルは教師あり学習の際に入力ベクトルとペアになっているベクトルです．1次元からなるベクトルは，実質スカラですが，ベクトルと呼びます．教師ベクトルとも呼ばれます．出力ベクトルは入力ベクトルを人工知能に処理させたときに出力されるベクトルです．教師あり学習の場合，このベクトルの値がターゲットベクトルに似ているほど良いです（人工知能の性能が）．
# 
# **★エポック**
# 
# 用意したトレーニングデータセット（パラメータ更新にのみ用いるデータセット）の全部を人工知能が処理する時間の単位です．1エポックだと全データを人工知能が舐めたことになります．2エポックだと2回ほど人工知能が全データを舐めたことになります．学習の過程では繰り返しデータを人工知能に読ませるのです．
# 
# **★損失（ロス）と損失関数（ロス関数）とコスト関数**
# 
# 人工知能が出力する値とターゲット（教師）の値がどれだけ似ていないかを表す指標です．これが小さいほど，人工知能はターゲットに近い値を出力できることになります．よって，この損失を小さくすることが学習の目標です．損失を計算するための関数を損失関数と言います．また，コストは損失と似ているものですが，正則化項（人工知能の過剰適合を防ぐために用いる値）を損失に加えた場合の小さくする目標の関数です．それを計算する関数をコスト関数と言います．損失関数とコスト関数はどちらも場合によって学習の目的関数となり得ます．
# 
# **★正確度と精度**
# 
# 正確度とは英語では accuracy と記述されます．略記で ACC とも表現されます．真陽性，偽陽性，偽陰性，真陰性をそれぞれ $a$，$b$，$c$，$d$ とするとき正確度 $u$ は以下の式で定義されます：
# 
# $u=\displaystyle\frac{a+d}{a+b+c+d}$．
# 
# また，精度は英語では precision と記述されます．これは陽性的中率のことであって英語では positive predictive value であり，PPV と略記されます．精度 $v$ は以下の式で定義されます：
# 
# $v=\displaystyle\frac{a}{a+b}$．
# 
# つまり，これは陽性と予想した場合の数に対して本当に陽性であった場合の数の割合です．人工知能の性能を表現するために精度という単語を使う人がたくさんいます．精度は人工知能を評価するひとつの指標に過ぎません．しかも，この指標は正確度や MCC や F1 と比較して頑健な評価指標ではなく，人工知能の開発者が自由に調整できてしまう値です．これだけで人工知能の性能は評価不可能です．科学的な文脈においては，精度という単語を使いすぎないように気を付ける必要があります（多くの場合，性能というべきでしょう）．その他の指標は以下の Wikipedia が詳しいです．
# 
# https://en.wikipedia.org/wiki/Precision_and_recall
# 
# **★パラメータとハイパーパラメータ**
# 
# パラメータは人工知能を構成する一部です．機械学習法で成長させられるアルゴリズムは何らかの入力に対して何らかの出力をしますが，このアルゴリズムがその計算をするために必要とする値を持っている場合があり，それをパラメータと言います．機械学習ではアルゴリズム自体は変化させずに，パラメータを変化させることによって良い人工知能を作ろうとします．深層学習界隈では最急降下法によってパラメータは更新される場合が多いです．また，学習によって更新されるパラメータとは異なり，学習の前に人間が決めなければならないパラメータがありますが，これをハイパーパラメータと言います．ニューラルネットワークでは，活性化関数の種類であったり，各層のニューロンの数であったり，層の数であったりします．このハイパーパラメータすらも自動で最適化しようとする方法もあり，そういう方法を network architecture search（NAS）と言います．
# 
# **★二乗誤差**
# 
# あるベクトルとあるベクトルの距離です．
# 
# **★ソフトマックス**
# 
# 要素を合計すると1になるベクトルタイプのデータです．各要素の最小値は0です．これを出力する関数をソフトマックス関数と言います．入力ベクトルに対して入力ベクトルと同じ要素次元数のベクトル（ソフトマックス）を出力します．
# 
# **★一括学習，逐次学習，ミニバッチ学習**
# 
# 一括学習はバッチ学習とも呼ばれます．人工知能を成長させるときにデータを全部読み込ませた後に初めてパラメータを更新します．逐次学習はオンライン学習とも呼ばれます．人工知能を成長させるときにデータを1個ずつ読み込ませ，その都度パラメータを更新する方法です．ミニバッチ学習は全データからあらかじめ決めた分のデータを読み込ませ，その度にパラメータを更新する方法です．ミニバッチ学習のミニバッチサイズが全データサイズと等しいならそれは一括学習で，ミニバッチサイズが1なら逐次学習です．
# 
# **★早期終了**
# 
# 人工知能を成長させるときに，過学習（データへの過剰適合）が起こる前に学習を停止させることです．最もナイーブには，学習の最中に最も良い性能を示した値を記録しておき，その値を $n$ 回連続で更新しなかった場合に学習を打ち切る方法があります．この $n$ 回のことを patience と言います．
# 
# **★オプティマイザ**
# 
# 最急降下法は最もナイーブな最適化法のひとつです．学習をより良く（学習の速さを良くするとか局所解に陥り難くするとか）進めるために様々な最適化法が考案されています．深層学習で用いられる最適化法は基本的には最急降下法を改善した方法です．そういったものをオプティマイザと言います．ここでは，世界で最も利用されているオプティマイザである Adam を紹介します．最初に，最急降下法の式を紹介します．
# 
# $\theta_{t+1}=\theta_t-\alpha g_t$
# 
# これに対して，世界で最も利用されているオプティマイザである Adam は以下のような更新式で定義されます．
# 
# $m_{t+1}=\beta_{1} m_t+(1-\beta_{1})g_t$
# 
# $v_{t+1}=\beta_{2} v_t+(1-\beta_{2})g_t^2$
# 
# $\hat{m}=\displaystyle\frac{m_{t+1}}{1-\beta_{1}^{t+1}}$
# 
# $\hat{v}=\displaystyle\frac{v_{t+1}}{1-\beta_{2}^{t+1}}$
# 
# $h=\alpha\displaystyle\frac{\hat{m}}{\sqrt{\hat{v}-\epsilon}}$
# 
# $\theta_{t+1}=\theta_t-h$
# 
# ハイパーパラメータは以下の4つです．
# 
# $\alpha=0.001$，$\beta_{1}=0.9$，$\beta_{2}=0.999$，$\epsilon=10^{-6}$
# 
# また，Adam のパラメータの初期値として以下の値を用います．
# 
# $m_0=0$，$v_0=0$
# 
# Adam がやっていることは，勾配の平均値と分散の値を保存して，その値をうまく利用することで学習率を調整することです．例えば，下に凸である二次関数の曲線で言うと，極小値付近では勾配の値は正の値と負の値を行ったり来たりするはずです．そのような際には何ステップかに渡り記録した勾配の分散の値は大きくなるはずです．また，極小値付近においては勾配の値は小さな値となるはずです．よって，分母を勾配の分散にして，分子を勾配の平均値にした値は小さい値になります．これに対して学習率を掛けると得られる値は小さくなります．これが Adam における更新値であり $h$ です．極小値付近ではパラメータ更新が緩やかになります．
# 
# <img src="https://drive.google.com/uc?id=1eiSgnD5Y5HPHrgpIj5R4BCTifuKu77LK">
# 
# これは停留点の中でも厄介な鞍点から，洗練されたオプティマイザがいかに早く抜け出せるかを動画にしたものです．
# 
# 出典：https://rnrahman.com/
# 

# #3. <font color="Crimson">Scikit-learn 入門</font>

# ##3-1. <font color="Crimson">基本操作</font>

# ###3-1-1. <font color="Crimson">インポート</font>

# Scikit-learn は以下のようにインポートします．以下のコードではインポートした Scikit-learn のバージョンを表示させています．

# In[ ]:


#!/usr/bin/env python3
import sklearn
 
def main():
    print(sklearn.__version__)

if __name__ == "__main__":
    main()


# ###3-1-2. <font color="Crimson">データセット</font>

# ここでは，Scikit-learn が備えているデータセットを利用して機械学習アルゴリズムの実装法を紹介します．アイリスというアヤメの咢と花弁のサイズの情報からなる，世界中で利用されてきたとても有名なデータセットを利用します．以下のようにすることでデータセットをダウンロードして中身を表示することができます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris

def main():
    diris = load_iris()
    print(diris)

if __name__ == "__main__":
    main()


# ダウンロードしたデータセットは以下のように表示されているはずです．
# 
# ```
# {'data': array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        ・
#        ・
#        略
#        ・
#        ・
#        [6.3, 2.5, 5. , 1.9],
#        [6.5, 3. , 5.2, 2. ],
#        [6.2, 3.4, 5.4, 2.3],
#        [5.9, 3. , 5.1, 1.8]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'frame': None, 'target_names': array(['setosa', 'versicolor', ．．．略
# ```
# 最初の行に表示されている `[5.1, 3.5, 1.4, 0.2]` が最初のデータです．4個の要素からなるベクトルデータです．このような4個の要素からなるインスタンスが150個あります．すべてのインスタンスの後に表示されている target という項目がありますが，これは，各インスタンスがどのアヤメの種類に属しているかを示しています．アヤメには種類があるらしく，`0` は setosa，`1` は versicolor，`2` は virginica という種類を意味しています．それぞれ均等にデータが取得されており，全部で150個のインスタンスの内，50個が setosa，別の50個が versicolor，残りの50個が virginica です．各インスタンスは4個の要素からなるベクトルデータであることを紹介しましたが，各要素はそのインスタンスの属性（アトリビュート）と言います．このデータの場合，最初の要素は咢の長さです．咢というのは以下の写真の茎ではない緑色の部分を示すものらしいです．単位は cm です．次の要素は咢の幅，次の要素は花弁（花びら）の長さ，最後の要素は花弁の幅です．どれも単位は cm です．

# <img src="https://drive.google.com/uc?id=1RoIncS19BL3lHm1WKnccWf1DhFbQid-1" width="20%">
# 
# 出典：https://ja.wikipedia.org/
# 
# 

# この実習では，アイリスデータセットの各インスタンスのベクトルデータを入力データとして，そのインスタンスがどのアヤメの種類に属するのかを予測する予測器を構築します．分類問題です．また，このデータには分類先のデータ，ターゲットデータとしてアヤメの種類（3個）が与えられていますが，このようなターゲットデータ（教師データ）と入力データを用いて行う学習法を教師あり学習法と言います．

# ###3-1-3. <font color="Crimson">決定木による予測器の構築</font>

# 最初に，データを学習データセットとテストデータセットに分割します．テストデータセットのサイズは全体の2割にします（何割でも良いのですが2割にしてみました）．データセットの分割もとても便利な方法が用意されています．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # このような関数がある
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 0) # このように書くと分割できる．ランダムに並べ替えてくれる．
    print(learnx) # 学習データセットの入力データ．
    print(learnt) # 学習データセットの教師データ．
    print(testx) # テストデータセットの入力データ．
    print(testt) # テストデータセットの教師データ．

if __name__ == "__main__":
    main()


# 以下のように書くと決定木による予測器を構築することができます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 0)
    predictor = DecisionTreeClassifier(random_state=0) # 予測器を生成．ここも乱数の種に注意．
    predictor.fit(learnx, learnt) # 学習．
    print(predictor.predict(testx)) # テストデータセットの入力データを予測器に入れて結果を予測．
    print(testt) # 教師データ．

if __name__ == "__main__":
    main()


# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 0)

    print(testx)
    print(testt)

if __name__ == "__main__":
    main()


# 結果として出力された上段の値とそれに対応する教師データが完全に一致しています．高性能な予測器が作れたということです．

# <font color="Crimson">(9｀･ω･)9 ｡oO(予測器の性能を定量的に示す方法は後で紹介します．)</font>

# 決定木がどのようなものなのかを把握するために，構築された予測器，すなわち，学習済みの決定木を可視化します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image, display_png
from graphviz import Digraph
from six import StringIO
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 0)
    predictor = DecisionTreeClassifier(random_state = 0)
    predictor.fit(learnx, learnt)
    treedata = StringIO()
    export_graphviz(predictor, out_file = treedata, feature_names = diris.feature_names, class_names = diris.target_names)
    graph = pydotplus.graph_from_dot_data(treedata.getvalue())
    display_png(Image(graph.create_png(), width = 760))

if __name__ == "__main__":
    main()


# 決定木はこのような選択肢の分岐を繰り返すことで入力データを分類する手法です．最上段からスタートします．この場合，咢の幅が `0.8` 以下であるなら（`True` の方向へ進む），データは `setosa` と分類されます．`gini` とありますが，これについて気になった場合は「ジニ不純度」のようなキーワードで検索してみてください．

# ###3-1-4. <font color="Crimson">予測器の性能評価</font>

# 上のコードではテストデータセットの入力データを入力したときの予測器の出力とテストデータセットの教師データの比較を並べて目視で行いました．予測器の性能を評価する方法があります．評価を行うために混同行列（confusion matrix）というものを計算させます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2) # 全部正解してしまうから，乱数の種を変えてみた．
    predictor = DecisionTreeClassifier(random_state=0)
    predictor.fit(learnx, learnt)
    dy = predictor.predict(testx)
    cm = confusion_matrix(y_true = testt, y_pred = dy)
    print(cm)

if __name__ == "__main__":
    main()


# 結果は以下のようになりました．
# ```
# [[14  0  0]
#  [ 0  7  1]
#  [ 0  1  7]]
#  ```
#  すべての数字を合計するとテストデータセットのサイズである30となります．1行目は，最初の要素から，本当は setosa であるものを setosa と予測したものの数，versicolor と予測したものの数，virginica と予測したものの数です．2行目は，本当は versicolor であるものに対するそれぞれの予測結果の数，3行目は，本当は virginica であるものに対するそれぞれの予測結果の数が示されています．すなわち，この行列の対角要素の個数は予測器の正解個数です．よってこの予測器の正確度は，28 を 30 で割った値，0.9333 と計算されます．正確度は以下のように計算することも可能です．
# 

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2) # 全部正解してしまうから，乱数の種を変えてみた．
    predictor = DecisionTreeClassifier(random_state=0)
    predictor.fit(learnx, learnt)
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# ###3-1-5. <font color="Crimson">他のアルゴリズムの利用</font>

# 他の様々な機械学習アルゴリズムに同じ計算をさせます．サポートベクトルマシンの分類に利用可能な方法 SVC を利用します（回帰問題には SVR を利用します）．以下のように書きます．この場合の正確度は 0.9667 となりました．決定木を用いた場合と（ほぼ）同じ結果です．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # 決定木ではなくて SVC をインポートする．
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    predictor = SVC(random_state=0) # SVC を利用する．
    predictor.fit(learnx, learnt)
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(たった1行を書き換えるだけで別の機械学習法を利用できました．これが Scikit-learn の強みです．)</font>

# 多層パーセプトロンを利用します．英語では multilayer perceptron（MLP）というものです．深層学習で利用されるニューラルネットワークの最も基礎的な手法です．この場合，正確度は 1.0 と計算されました．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # 変更する．
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    predictor = MLPClassifier(random_state=0) # MLP を利用する．
    predictor.fit(learnx, learnt)
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# ロジスティック回帰法を利用します．ロジスティック回帰は名前に回帰という文字が入っていますが分類問題に用いられる方法です．この場合も大体同じような正確度が得られました．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # 変更する．
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    predictor = LogisticRegression(random_state=0) # MLP を利用する．
    predictor.fit(learnx, learnt)
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# このように Scikit-learn を利用すれば，様々な機械学習アルゴリズムをとても観点に実装することができます．Scikit-learn で利用可能な方法はまだまだたくさんあります．それらの方法は https://scikit-learn.org/stable/supervised_learning.html にまとめられています．
# 
# 
# 

# ##3-2. <font color="Crimson">実際の解析で必要なこと</font>

# ###3-2-1. <font color="Crimson">モデルの保存と呼び出し</font>

# これまでにとても簡単に Scikit-learn を利用した機械学習法の実装法を紹介しました．実際のデータ解析を行う際には，もう少し別の作業が必要です．例えば，構築した予測器は未知の新たなデータに対して利用したいため，どこかに置いて使える状態にしておきたいと思います（上では学習とテストを同時に行いましたが，予測器を利用する度に学習を行うのは良い方法ではありません．というか通常，想像を絶する悪手です．）．このためには，構築した予測器をファイルとして保存できなければなりません．モデル（予測器）の保存は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle # インポートする．
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    predictor = SVC(random_state=0)
    predictor.fit(learnx, learnt)
    fout = open("./predictor.sav", "wb") # wは書き込みを意味します．bはバイナリを意味します．
    pickle.dump(predictor, fout) # 予測器のファイルへの保存．
    fout.close()

if __name__ == "__main__":
    main()


# これで予測器の情報がファイル，predictor.sav としてが保存されました．以下のようにすることでこの計算機上（Google Colaboratory が動いている Google の所有物のどこか遠くにある計算機上）に保存されたファイルを確認することができます．

# In[ ]:


get_ipython().system(' ls')


# <font color="Crimson">(9｀･ω･)9 ｡oO(これは Python のコマンドではありません．Google Colaboratory のコードセルは通常，Python を実行させるためのものです．「!」をコマンドの前に付けて実行すると，この計算機のシェルを動かすことができます．)</font>

# 上のコマンドを実行すると確かに predictor.sav というファイルが保存されていることが確認できました．次はこのファイルを別のプログラムから呼び出して利用します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
 
def main():
    fin = open("./predictor.sav", "rb") # rは読み込みを意味します．
    predictor = pickle.load(fin) # 予測器の読み込み．
    fin.close()
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# このプログラムでは SVC をインポートすらしていないし，学習データセットの学習をするためにコードも書いていないのに，しっかりと予測器が動いていることを確認できます．つまり，学習済み決定木を読み込めたということを意味しています．

# ###3-2-2. <font color="Crimson">ハイパーパラメータの探索</font>

# 機械学習では未学習の決定木やニューラルネットワークやサポートベクトルマシンに学習データセットを学習させます．これらの手法は学習によって何を成長（変化）させているかというと，パラメータと呼ばれる値です．例えば，以下のような線形回帰式のパラメータは $w_1$，$w_2$，$w_3$，$b$ です．この値を学習データセットに適合させることが各機械学習アルゴリズムが行っていることです．
# 
# \begin{eqnarray}
# f(x_1,x_2,x_3)=w_1x_1+w_2x_2+w_3x_3+b
# \end{eqnarray}
# 
# 機械学習アルゴリズムは学習の過程においてパラメータの値を変化させます．これに対して，学習によって決定されないパラメータをハイパーパラメータと言います．例えば，決定木においてはその分岐の深さをあらかじめ決めて分岐の仕方を学習させることができますが，これはハイパーパラメータのひとつです．また，機械学習アルゴリズムに持たせるパラメータのサイズもハイパーパラメータであるし，そのパラメータを学習させる最適化法もハイパーパラメータと言えます．このようなハイパーパラメータは各機械学習アルゴリズムが固有に持っているものです．最適なハイパーパラメータを決定することは，機械学習の利用者の腕の見せ所のひとつです．

# ハイパーパラメータを決定するための探索法には様々なものがあります．例えば，ランダムに決定したハイパーパラメータを利用して学習を完了させ，その性能を擬似的なテストデータセット（バリデーションデータセット）を用いて評価し，最も性能が良くなるハイパーパラメータを最終的なハイパーパラメータとする方法があります．また，ランダム探索より効率的に探索するために，ベイズ最適化法（以前のハイパーパラメータを利用した学習の結果を利用してより良さそうなハイパーパラメータを探索する方法）や進化計算法（離散最適化法の一種）が利用されています．そんな中において，最もナイーブですが強力な探索法としてグリッドサーチがあります．しらみつぶし的にあり得そうなハイパーパラメータの組み合わせを全部計算してしまう方法です．例えば，ハイパーパラメータをふたつ持つ機械学習アルゴリズムについて，最初のハイパーパラメータの値の候補として10点が考えられるとして，また，もうひとつのハイパーパラメータの値の候補として5点が考えられるとした場合，それらを掛け算した組み合わせ分，すなわち50種類のハイパーパラメータの組み合わせすべてで学習を行う方法です．グリッドサーチは以下のように行います．ここではサポートベクトルマシンのハイパーパラメータである「カーネル」と「ガンマ」と「C」の値の組み合わせを探索しています（どのような性質を持つものか興味があったら調べてみてください）．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV # インポートする
from sklearn.svm import SVC
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    diparameter = {
        "kernel" : ["rbf"], # 1点
        "gamma" : [10**i for i in range(-4, 2)], # 5点．
        "C" : [10**i for i in range(-2,4)], # 6点
        "random_state" : [0], # 1点
        } # 探索するハイパーパラメータの候補をディクショナリで指定する．この場合，合計30点探索する．
    licv = GridSearchCV(SVC(), param_grid = diparameter, scoring = "accuracy", cv = 5, n_jobs = 1) # SVCを使うことを指定．上のハイパーパラメータ候補を探索する．
    licv.fit(learnx, learnt) # グリッドサーチ．
    predictor = licv.best_estimator_ # グリッドサーチの結果，最も良い予測器を最終的な予測器とする．
    print(predictor.score(testx, testt))

if __name__ == "__main__":
    main()


# グリッドサーチの際には，クロスバリデーションによる評価を行っています．ここでは，`cv = 5` と指定しています．学習データセットを5分割にし，それらの内の4分割分で学習を行い，残りの1分割分で擬似テストをし，次に別の4分割分で学習を行い，残りの1分割分で擬似テストをする，という行為を合計5回行うという操作です．最終的なハイパーパラメータとしてどのような値が選択されたのかは以下のように書くことでわかります．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
 
def main():
    diris = load_iris()
    learnx, testx, learnt, testt = train_test_split(diris.data, diris.target, test_size = 0.2, random_state = 2)
    diparameter = {
        "kernel" : ["rbf"],
        "gamma" : [10**i for i in range(-4, 2)],
        "C" : [10**i for i in range(-2,4)],
        "random_state" : [0],
        }
    licv = GridSearchCV(SVC(), param_grid = diparameter, scoring = "accuracy", cv = 5, n_jobs = 1)
    licv.fit(learnx, learnt)
    predictor = licv.best_estimator_
    print(sorted(predictor.get_params(True).items())) # 選択されたハイパーパラメータを確認．

if __name__ == "__main__":
    main()


# ###3-2-3. <font color="Crimson">方法の選択</font>

# 機械学習アルゴリズムにはとてもたくさんの種類があります．どのような場合に（どのようなデータセットに対して，また，どのような問題に対して）どの方法を使えば良いかを Scikit-learn がまとめてくれています．以下のチートシートです．これは経験に基づいた選択方法です．実際にはデータセットの性質によって最適な手法は変わるものであるため必ずしも正しいとは限りません．

# <img src="https://drive.google.com/uc?id=1FUTedjoFUz8SU4EqypWF6UfnaXCR7Z6H" width="80%">
# 

# 出典：https://scikit-learn.org/

# <font color="Crimson">(9｀･ω･)9 ｡oO(終わりです．)</font>
