#!/usr/bin/env python
# coding: utf-8

# # 教師あり学習法

# ## scikit-learn の基本操作

# このコンテンツで紹介する scikit-learn とは様々な機械学習アルゴリズムをとても簡単な記述で実現することができる Python のライブラリです．最初に scikit-learn の基本的な操作方法を紹介した後にいくつかの教師あり学習法を紹介します．

# ### インポート

# scikit-learn は以下のようにインポートします．以下のコードではインポートした scikit-learn のバージョンを表示させています．

# In[ ]:


#!/usr/bin/env python3
import sklearn
 
def main():
    print(sklearn.__version__)

if __name__ == "__main__":
    main()


# ### データセット

# ここでは，scikit-learn が備えているデータセットを利用して機械学習アルゴリズムの実装法を紹介します．アイリスというアヤメの咢と花弁のサイズの情報からなる，世界中で利用されてきたとても有名なデータセットを利用します．以下のようにすることでデータセットをダウンロードして中身を表示することができます．

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
# 最初の行に表示されている `[5.1, 3.5, 1.4, 0.2]` が最初のデータです．4個の要素からなるベクトルデータです．このような4個の要素からなるインスタンスが150個あります．すべてのインスタンスの後に表示されている target という項目がありますが，これは，各インスタンスがどのアヤメの種類に属しているかを示しています．アヤメには種類があるらしく，`0` は setosa，`1` は versicolor，`2` は virginica という種類を意味しています．それぞれ均等にデータが取得されており，全部で150個のインスタンスの内，50個が setosa，別の50個が versicolor，残りの50個が virginica です．各インスタンスは4個の要素からなるベクトルデータであることを紹介しましたが，各要素はそのインスタンスの属性（アトリビュート）と言います．このデータの場合，最初の要素は花弁（花びら）の長さです．単位は cm です．次の要素は花弁の幅，次の要素は咢の長さ，最後の要素は咢の幅です．咢というのは以下の写真の茎ではない緑色の部分を示すものらしいです．どれも単位は cm です．

# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/iris.png?raw=1" width="25%"/>
# 
# 出典：https://ja.wikipedia.org/
# 
# 

# この実習では，アイリスデータセットの各インスタンスのベクトルデータを入力データとして，そのインスタンスがどのアヤメの種類に属するのかを予測する予測器を構築します．分類問題です．また，このデータには分類先のデータ，ターゲットデータとしてアヤメの種類（3個）が与えられていますが，このようなターゲットデータ（教師データ）と入力データを用いて行う学習法を教師あり学習法と言います．

# ### 決定木による予測器の構築

# 決定木という機械学習アルゴリズムを用いて人工知能を構築します．最初にデータを学習データセットとテストデータセットに分割します．テストデータセットのサイズは全体の 2 割にします（何割でも良いのですが 2 割にしてみました）．データセットの分割もとても便利な方法が用意されています．以下のように書きます．

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


# 結果として出力された上段の値とそれに対応する教師データが完全に一致しています．高性能な予測器が作れたということです．

# ```{note}
# 出力の 1 行目は予測器がテストデータセットの入力値に対して予測した予測値です．2 行目はテストデータセットの教師データです．
# ```

# ```{note}
# 予測器の性能を定量的に示す方法は後で紹介します．
# ```

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


# 決定木はこのような選択肢の分岐を繰り返すことで入力データを分類する手法です．最上段からスタートします．この場合，咢の幅が `0.8` 以下であるなら（`True` の方向へ進む），データは `setosa` と分類されます．

# ```{hint}
# 画像中に ` gini ` とありますが，これは決定木の目的関数です．決定木はこれを指標にしてデータが分割します．気になった場合は「ジニ不純度」のようなキーワードで検索してみてください．
# ```

# ```{note}
# このように決定木で得られた結果はとても解釈性が高いです．決定木はそれだけで利用すると（集団学習という性能を向上するものがありますが，そういったものを利用せずにという意味）予測性能は良い方ではありませんが，解釈性が高いためビジネスの分野でよく用いられます．
# ```

# ## 予測器の性能評価

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


# ## 他のアルゴリズムの利用

# これまでに決定木を用いて人工知能を構築しましたが，この節では別の機械学習アルゴリズムを利用して人工知能を構築します．

# ### サポートベクトルマシン

# サポートベクトルマシンの分類に利用可能な方法 SVC を利用します（回帰問題には SVR を利用します）．以下のように書きます．この場合の正確度は 0.9667 となりました．決定木を用いた場合と（ほぼ）同じ結果です．

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


# ```{note}
# たった1行を書き換えるだけで別の機械学習法を利用できました．これが scikit-learn の強みです．
# ```

# ### 多層パーセプトロン

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


# ### ロジスティック回帰法

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


# このように scikit-learn を利用すれば，様々な機械学習アルゴリズムをとても簡単に実装することができます．scikit-learn で利用可能な方法はまだまだたくさんあります．それらの方法は https://scikit-learn.org/stable/supervised_learning.html にまとめられています．
# 
# 
# 

# ## 実際の解析で必要なこと

# 上では学習とテストを同時に行いましたが，予測器を利用する度に学習を行うのは良い方法ではありません．というか通常，想像を絶する悪手です．普通はひとつめのコードで与えられたデータを利用して人工知能を構築して，また別の新たなコードでその人工知能を利用する，というのが実際の解析で行うことではないでしょうか．その方法を紹介します．

# ### モデルの保存と呼び出し

# これまでにとても簡単に scikit-learn を利用した機械学習法の実装法を紹介しました．実際のデータ解析を行う際には，もう少し別の作業が必要です．例えば，構築した予測器は未知の新たなデータに対して利用したいため，どこかに置いて使える状態にしておきたいと思います．このためには，構築した予測器をファイルとして保存できなければなりません．モデル（予測器）の保存は以下のように行います．

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


# これで予測器の情報がファイル，predictor.sav として保存されました．以下のようにすることでこの計算機上（Google Colaboratory が動いている Google の所有物のどこか遠くにある計算機上）に保存されたファイルを確認することができます．

# In[ ]:


get_ipython().system(' ls')


# ```{note}
# これは Python のコマンドではありません．Google Colaboratory のコードセルは通常，Python を実行させるためのものです．` ! ` をコマンドの前に付けて実行すると，この計算機のシェルを動かすことができます．
# ```

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


# このプログラムでは SVC をインポートすらしていないし，学習データセットの学習をするためにコードも書いていないのに，しっかりと予測器が動いていることを確認できます．つまり，学習済みサポートベクトルマシンを読み込めたということを意味しています．

# ### ハイパーパラメータの探索

# 機械学習では未学習の決定木やニューラルネットワークやサポートベクトルマシンに学習データセットを学習させます．これらの手法は学習によって何を成長（変化）させているかというと，パラメータと呼ばれる値です．例えば，以下のような線形回帰式のパラメータは $w_1$，$w_2$，$w_3$，$b$ です．この値を学習データセットに適合させることが各機械学習アルゴリズムが行っていることです．
# 
# $
# f(x_1,x_2,x_3)=w_1x_1+w_2x_2+w_3x_3+b
# $
# 
# 機械学習アルゴリズムは学習の過程においてパラメータの値を変化させます．これに対して，学習によって決定されないパラメータをハイパーパラメータと言います．例えば，決定木においてはその分岐の深さをあらかじめ決めて分岐の仕方を学習させることができますが，これはハイパーパラメータのひとつです．また，機械学習アルゴリズムに持たせるパラメータのサイズもハイパーパラメータであるし，そのパラメータを学習させる最適化法もハイパーパラメータと言えます．このようなハイパーパラメータは各機械学習アルゴリズムが固有に持っているものです．

# ```{note}
# 学習によって決定されないパラメータをハイパーパラメータと言い，これをうまく設定することが開発者の腕の見せ所のひとつです．
# ```

# ハイパーパラメータを決定するための探索法には様々なものがあります．例えば，ランダムに決定したハイパーパラメータを利用して学習を完了させ，その性能を擬似的なテストデータセット（バリデーションデータセット）を用いて評価し，最も性能が良くなるハイパーパラメータを最終的なハイパーパラメータとする方法があります．また，ランダム探索より効率的に探索するために，ベイズ最適化法（以前のハイパーパラメータを利用した学習の結果を利用してより良さそうなハイパーパラメータを探索する方法）や進化計算法（離散最適化法の一種）が利用されています．そんな中において，最もナイーブですが強力な探索法としてグリッドサーチがあります．しらみつぶし的にあり得そうなハイパーパラメータの組み合わせを全部計算してしまう方法です．例えば，ハイパーパラメータをふたつ持つ機械学習アルゴリズムについて，最初のハイパーパラメータの値の候補として 10 点が考えられるとして，また，もうひとつのハイパーパラメータの値の候補として 5 点が考えられるとした場合，それらを掛け算した組み合わせ分，すなわち 50 種類のハイパーパラメータの組み合わせすべてで学習を行う方法です．グリッドサーチは以下のように行います．ここではサポートベクトルマシンのハイパーパラメータである「カーネル」と「ガンマ」と「C」の値の組み合わせを探索しています（どのような性質を持つものか興味があったら調べてみてください）．

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


# グリッドサーチの際には，クロスバリデーションによる評価を行っています．ここでは，`cv = 5` と指定しています．学習データセットを 5 分割にし，それらの内の 4 分割分で学習を行い，残りの 1 分割分で擬似テストをし，次に別の 4 分割分で学習を行い，残りの 1 分割分で擬似テストをする，という行為を合計 5 回行うという操作です．最終的なハイパーパラメータとしてどのような値が選択されたのかは以下のように書くことでわかります．

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


# ```{note}
# サポートベクトルマシンとかはデータセットのサイズが大きくなるとかなり計算時間的にきつくなります．
# ```

# ### 方法の選択

# 機械学習アルゴリズムにはとてもたくさんの種類があります．どのような場合に（どのようなデータセットに対して，また，どのような問題に対して）どの方法を使えば良いかを scikit-learn がまとめてくれています．以下のチートシートです．これは経験に基づいた選択方法です．実際にはデータセットの性質によって最適な手法は変わるものであるため必ずしも正しいとは限りません．

# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/cheatSheet.png?raw=1" width="100%" />
# 

# 出典：https://scikit-learn.org/

# ```{note}
# 終わりです．
# ```
