#!/usr/bin/env python
# coding: utf-8

# # 多層パーセプトロン

# ## 扱うデータの紹介

# このコンテンツでは最も基本的な深層学習アルゴリズムである多層パーセプトロンを実装する方法を紹介します．多層パーセプトロンは英語では multilayer perceptron（MLP）と言います．ニューラルネットワークの一種です．層という概念があり，この層を幾重にも重ねることで深層ニューラルネットワークを構築することができます．MLP を実装するためにとても有名なデータセットを利用しますが，この節ではそのデータセットの紹介をします．

# ### MNIST について

# MLP に処理させるデータセットとして，機械学習界隈で最も有名なデータセットである MNIST（Mixed National Institute of Standards and Technology database）を解析対象に用います．「エムニスト」と発音します．MNIST は縦横28ピクセル，合計784ピクセルよりなる画像データです．画像には手書きの一桁の数字（0から9）が含まれています．公式ウェブサイトでは，学習データセット6万個とテストデータセット1万個，全部で7万個の画像からなるデータセットが無償で提供されています．

# ### ダウンロードと可視化

# 公式サイトよりダウンロードしてきても良いのですが，TensorFlow がダウンロードするためのユーティリティを準備してくれているため，それを用います．以下の `tf.keras.datasets.mnist.load_data()` を用いることで可能です．MNIST は合計7万インスタンスからなるデータセットです．5行目でふたつのタプルにデータをダウンロードしていますが，最初のタプルは学習データセット，次のタプルはテストデータセットのためのものです．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    (lilearnx, lilearnt), (litestx, litestt) = tf.keras.datasets.mnist.load_data()
    print("The number of instances in the learning dataset:", len(lilearnx), len(lilearnt))
    print("The number of instances in the test dataset:", len(litestx), len(litestt))
    print("The input vector of the first instance in the learning dataset:", lilearnx[0])
    print("Its shape:", lilearnx[0].shape)
    print("The target vector of the first instance in the learning datast:", lilearnt[0])
    # 2番目のインスタンスのインプットデータとターゲットデータを確認．

if __name__ == "__main__":
	main()


# データを可視化します．可視化のために matplotlib というライブラリをインポートします．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    (lilearnx, lilearnt), (litestx, litestt) = tf.keras.datasets.mnist.load_data()
    plt.imshow(lilearnx[0], cmap="gray")
    plt.text(1, 2.5, int(lilearnt[0]), fontsize=20, color="white")
    # 別のインプットデータを表示．

if __name__ == "__main__":
	main()


# このデータセットがダウンロードされている場所は `~/.keras/datasets` です．以下のような BaSH のコマンドを打つことで確認することができます．

# In[ ]:


get_ipython().system(' ls /root/.keras/datasets')


# MNIST はこのような縦が28ピクセル，横が28ピクセルからなる手書き文字が書かれた（描かれた）画像です（0から9までの値）．それに対して，その手書き文字が0から9のどれなのかという正解データが紐づいています．この画像データを MLP に読み込ませ，それがどの数字なのかを当てるという課題に取り組みます．

# ## MLP の実装

# この節では MLP を実装します．MLP を実装することに加えて，どのように学習を進めるとより良い人工知能を構築できるのかについて紹介します．

# ### 簡単な MLP の実装

# 実際に MNIST を処理する MLP を実装する前に，とても簡単なデータを処理するための MLP を実装します．ここでは，以下のようなデータを利用します．これが学習セットです．ここでは MLP の実装の方法を紹介するだけなのでバリデーションセットもテストセットも使用しません．
# 
# 入力ベクトル | ターゲットベクトル
# :---: | :---:
# [ 1.1, 2.2, 3.0, 4.0 ] | [ 0 ]
# [ 2.0, 3.0, 4.0, 1.0 ] | [ 1 ]
# [ 2.0, 2.0, 3.0, 4.0 ] | [ 2 ]
# 
# すなわち，`[1.1, 2.2, 3.0, 4.0]` が人工知能へ入力されたら，`0` というクラスを返し，`[2.0, 3.0, 4.0, 1.0]` というベクトルが入力されたら `1` というクラスを返し，`[2.0, 2.0, 3.0, 4.0]` というベクトルが入力されたら `2` というクラスを返す人工知能を MLP で構築します．実際には以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データセットの生成
    tx=[[1.1,2.2,3.0,4.0],[2.0,3.0,4.0,1.0],[2.0,2.0,3.0,4.0]]
    tx=np.asarray(tx,dtype=np.float32)
    tt=[0,1,2]
    tt=tf.convert_to_tensor(tt)

    # ネットワークの定義
    model=Network()
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt):
        with tf.GradientTape() as tape:
            ty=model.call(tx)
            costvalue=cce(tt,ty) #正解と出力の順番はこの通りにする必要がある．
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    for epoch in range(1,3000+1): # 学習の回数の上限値
        traincost,trainacc=inference(tx,tt)
        if epoch%100==0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch,traincost,trainacc))
    
    # 学習が本当にうまくいったのか入力ベクトルのひとつを処理させてみる
    tx1=np.asarray([[1.1,2.2,3.0,4.0]],dtype=np.float32)
    ty1=model.call(tx1)
    print(ty1)

    # 未知のデータを読ませてみる
    tu=np.asarray([[999,888,777,666]],dtype=np.float32)
    tp=model.call(tu)
    print(tp)

    # Denseの最初の引数の値やエポックの値や変化させて，何が起こっているか把握する．

class Network(Model):
    def __init__(self):
        super(Network,self).__init__()
        self.d1=Dense(10, activation="relu") # これは全結合層を生成するための記述．
        self.d2=Dense(3, activation="softmax")
    
    def call(self,x):
        y=self.d1(x)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# 上から説明を行います．以下のような記述があります．ここで，上述のデータを生成しています．`tx` は入力ベクトル3つです．`tt` はそれに対応するターゲットベクトル（スカラ）3つです．
# ```python
#     tx=[[1.1,2.2,3.0,4.0],[2.0,3.0,4.0,1.0],[2.0,2.0,3.0,4.0]]
#     tx=np.asarray(tx,dtype=np.float32)
#     tt=[0,1,2]
#     tt=tf.convert_to_tensor(tt)
# ```

# 次に，以下のような記述があります．この記述によって未学習の人工知能を生成します．生成した人工知能は `model` です．
# ```python
#     model=Network()
# ```
# この未学習の人工知能を生成するための記述の本体はプログラムの最下層辺りにある以下の記述です．
# ```python
# class Network(Model):
#     def __init__(self):
#         super(Network,self).__init__()
#         self.d1=Dense(10, activation="relu")
#         self.d2=Dense(3, activation="softmax")
#     
#     def call(self,x):
#         y=self.d1(x)
#         y=self.d2(y)
#         return y
# ```
# ここに `Dense(10, activation="relu")` とありますが，これは10個のニューロンを持つ層を1個生成するための記述です．活性化関数に ReLU を使うようにしています．これによって生成される層の名前は `self.d1()` です．ここでは10個という値を設定していますが，これは100でも1万でも1兆でもなんでも良いです．解きたい課題にあわせて増やしたり減らしたりします．ここをうまく選ぶことでより良い人工知能を構築でき，腕の見せ所です．次に，`Dense(3, activation="softmax")` という記述で3個のニューロンを持つ層を1個生成します．この3個という値は意味を持っています．入力するデータのクラスが0，1または2の3分類（クラス）であるからです．また，活性化関数にはソフトマックス関数を指定しています．ソフトマックス関数の出力ベクトルの要素を合計すると1になります．各要素の最小値は0です．よって出力結果を確率として解釈できます．次の，`def call(self,x):` という記述はこれ（`class Network()`）によって生成した人工知能を呼び出したときにどのような計算をさせるかを定義するものです．入力として `x` というベクトルが与えられたら，それに対して最初の層を適用し，次に，その出力に対して次の層を適用し，その値を出力する，と定義しています．構築した人工知能 `model` に対して `model.call()` のような方法で呼び出すことができます．
# 

# 次の以下の記述は，それぞれ，損失関数，正確度（ACC）を計算する関数，最急降下法の最適化法（パラメータの更新ルール）を定義するものです．これは，TensorFlow ではこのように書くのだと覚えるものです．
# ```python
#     cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
#     acc=tf.keras.metrics.SparseCategoricalAccuracy()
#     optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
# ```

# 次の以下の記述は損失を計算するためのものです．この `tf.GradientTapa()` は上でも出ました．最初に，`model.call()` に入力ベクトルのデータを処理させて出力ベクトル `ty` を得ます．この出力ベクトルとターゲットベクトルを損失関数の入力として損失 `traincost` を得ます．
# ```python
#         with tf.GradientTape() as tape:
#             ty=model.call(tx)
#             costvalue=cce(tt,ty) #正解と出力の順番はこの通りにする必要がある．
# ```
# この損失は人工知能が持つパラメータによって微分可能なので，以下の記述によって勾配を求めます．
# ```python
#         gradient=tape.gradient(costvalue,model.trainable_variables)
# ```
# 以下の記述はパラメータ更新のための最急降下法の定義と損失とは別の性能評価指標である正確度（accuracy（ACC））を計算するための定義です．
# ```python
#         optimizer.apply_gradients(zip(gradient,model.trainable_variables))
#         accvalue=acc(tt,ty)
# ```
# 最後の以下の記述はこの関数の戻り値を定義するものです．
# ```python
#         return costvalue,accvalue
# ```

# 次に記述されている以下の部分は，実際の学習のループに関するものです．このループでデータを何度も何度も予測器（人工知能）に読ませ，そのパラメータを成長させます．この場合，3000回データを学習させます．また，学習100回毎に学習の状況を出力させます．
# ```python
#     for epoch in range(1,3000+1):
#         traincost,trainacc=inference(tx,tt)
#         if epoch%100==0:
#             print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch,traincost,trainacc))
# ```

# 次の記述，以下の部分では学習がうまくいったのかを確認するために学習データのひとつを学習済みの人工知能に読ませて予測をさせています．この場合，最初のデータのターゲットベクトルは0なので0が出力されなければなりません．
# ```python
#     # 学習が本当にうまくいったのか入力ベクトルのひとつを処理させてみる
#     tx1=np.asarray([[1.1,2.2,3.0,4.0]],dtype=np.float32)
#     ty1=model.call(tx1)
#     print(ty1)
# ```
# 出力結果は以下のようになっているはずです．出力はソフトマックス関数なので各クラスの確率が表示されています．これを確認すると，最初のクラス（0）である確率が99%以上であると出力されています．よって，やはり人工知能は意図した通り成長したことが確認できます．
# ```
# tf.Tensor([[9.932116e-01 7.842198e-06 6.780579e-03]], shape=(1, 3), dtype=float32)
# ```
# 

# 次に，全く新たなデータを入力しています．
# ```python
#     # 未知のデータを読ませてみる
#     tu=np.asarray([[999,888,777,666]],dtype=np.float32)
#     tp=model.call(tu)
#     print(tp)
# ```
# `[999,888,777,666]` というベクトルを入力したときにどのような出力がされるかということですが，この場合，以下のような出力がされています．このベクトルを入力したときの予測値は2であるとこの人工知能は予測したということです．
# ```
# tf.Tensor([[0. 0. 1.]], shape=(1, 3), dtype=float32)
# ```
# 

# 以下では `Dense()` の挙動を確認してみます．`Dense()` はもちろんクラスの中でなければ使えない関数ではなく，`main()` の中でも呼び出して利用可能です．これで挙動を確認することでどのようにネットワークが構築されているか把握できるかもしれません．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データセットの生成
    tx=[[1.1,2.2,3.0,4.0],[2.0,3.0,4.0,1.0],[2.0,2.0,3.0,4.0]]
    tx=np.asarray(tx,dtype=np.float32)
    
    # 関数を定義
    d1=Dense(10, activation="relu")

    # データセットの最初の値を入力
    print("1-----------")
    print(d1(tx[0:1]))

    # データセットの全部の値を入力
    print("2-----------")
    print(d1(tx))

    # 活性化関数を変更した関数を定義
    d1=Dense(10, activation="linear")

    # データセットの最初の値を入力
    print("3-----------")
    print(d1(tx[0:1]))

    # データセットの全部の値を入力
    print("4-----------")
    print(d1(tx))

    # 最初の引数の値を変更した関数を定義
    d1=Dense(4, activation="linear")

    # データセットの最初の値を入力
    print("5-----------")
    print(d1(tx[0:1]))

    # データセットの全部の値を入力
    print("6-----------")
    print(d1(tx))

    # 別の関数を定義
    d1=Dense(4, activation="linear")
    d2=Dense(5, activation="relu")

    # データセットの最初の値を入力
    print("7-----------")
    y=d1(tx[0:1])
    print(d2(y))

    # データセットの全部の値を入力
    print("8-----------")
    y=d1(tx)
    print(d2(y))

if __name__ == "__main__":
    main()


# ### MNIST を利用した学習

# 次に，MNIST を処理して「0から9の数字が書かれた（描かれた）手書き文字を入力にして，その手書き文字が0から9のどれなのかを判別する人工知能」を構築します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MAXEPOCH=50
    MINIBATCHSIZE=500
    UNITSIZE=500
    TRAINSIZE=54000
    MINIBATCHNUMBER=TRAINSIZE//MINIBATCHSIZE # ミニバッチのサイズとトレーニングデータのサイズから何個のミニバッチができるか計算

    # データの読み込み
    (lilearnx,lilearnt),(litestx,litestt)=tf.keras.datasets.mnist.load_data()
    outputsize=len(np.unique(lilearnt)) # MNISTにおいて出力ベクトルのサイズは0から9の10
    
    # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
    litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
    livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
    
    # 最大値を1にしておく
    litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
    
    # ネットワークの定義
    model=Network(UNITSIZE,outputsize)
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用するため学習では利用しないが次のコードのために一応定義しておく．
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt,mode): # 「mode」という変数を新たに設定．これでパラメータ更新をするかしないかを制御する（バリデーションではパラメータ更新はしない）．
        with tf.GradientTape() as tape:
            model.trainable=mode
            ty=model.call(tx)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    for epoch in range(1,MAXEPOCH+1):
        # トレーニング
        index=np.random.permutation(TRAINSIZE)
        traincost=0
        for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
            somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
            eomb=somb+MINIBATCHSIZE # 「end of minibatch」
            subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
            traincost+=subtraincost
        traincost=traincost/MINIBATCHNUMBER
        # バリデーション
        validcost,_=inference(livalidx,livalidt,False)
        # 学習過程の出力
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))

    # ユニットサイズやミニバッチサイズを変更したり層を追加したりして挙動を把握する．

class Network(Model):
    def __init__(self,UNITSIZE,OUTPUTSIZE):
        super(Network,self).__init__()
        self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
        self.d1=Dense(UNITSIZE, activation="relu")
        self.d2=Dense(OUTPUTSIZE, activation="softmax")
    
    def call(self,x):
        y=self.d0(x)
        y=self.d1(y)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# プログラムの中身について上から順に説明します．以下の部分はハイパーパラメータを設定する記述です．`MAXEPOCH` は計算させる最大エポックです．このエポックに至るまで繰り返しの学習をさせるということです．`MINIBATCHSIZE` とはミニバッチ処理でサンプリングするデータのサイズです．これが大きいとき実計算時間は短縮されます．この値が `1` のとき，学習法はオンライン学習法であり，この値がトレーニングセットのサイズと等しいとき，学習法は一括更新法です．ミニバッチの大きさは持っているマシンのスペックと相談しつつ，色々な値を試してみて一番良い値をトライアンドエラーで探します．`UNITSIZE` は MLP の層のサイズ，つまり，ニューロンの数です．`TRAINSIZE` はトレーニングセットのインスタンスの大きさです．MNIST の学習セットは60000インスタンスからなるのでその90%をトレーニングセットとして利用することにしています．`MINIBATCHNUMBER` はミニバッチのサイズとデータのサイズから計算されるミニバッチの個数です．オンライン学習法の場合，1エポックでパラメータ更新は，この例の場合，54000回行われます．一括更新法の場合，1エポックでパラメータ更新は1回行われます．このミニバッチのサイズ（500）とデータサイズの場合，1エポックでパラメータ更新は108回行われます．
# 
# ```python
#     # ハイパーパラメータの設定
#     MAXEPOCH=50
#     MINIBATCHSIZE=500
#     UNITSIZE=500
#     TRAINSIZE=54000
#     MINIBATCHNUMBER=TRAINSIZE//MINIBATCHSIZE # ミニバッチのサイズとトレーニングデータのサイズから何個のミニバッチができるか計算
# ```

# データの読み込みは上で説明したため省略し，以下の部分では読み込んだデータをトレーニングセットとバリデーションセットに分割しています．この `:` の利用方法は NumPy の使い方解説のところで行った通りです．
# ```python
#     # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
#     litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
#     livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
# ```

# 以下の記述では，MNIST に含まれる値を0以上1以下の値に変換しています（元々の MNIST は0から255の値で構成されています）．用いるオプティマイザの種類やそのパラメータ更新を大きさを決めるハイパーパラメータ（学習率）の設定によってはこのような操作が良い効果をもたらす場合があります．
# ```python
#     # 最大値を1にしておく
#     litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
# ```

# ネットワークの定義は以下で行います．これは前述の例と同じです．
# ```python
#     # ネットワークの定義
#     model=Network(UNITSIZE,outputsize)
#     cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
#     acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用するため学習では利用しないが次のコードのために一応定義しておく．
#     optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
# ```
# ネットワーク自体は以下の部分で定義されているのですが，前述の例と少し異なります．ここでは，28行28列の行列を784要素のベクトルに変換するための層 `self.d0` を定義しています．
# ```python
# class Network(Model):
#     def __init__(self,UNITSIZE,OUTPUTSIZE):
#         super(Network,self).__init__()
#         self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
#         self.d1=Dense(UNITSIZE, activation="relu")
#         self.d2=Dense(OUTPUTSIZE, activation="softmax")
#     
#     def call(self,x):
#         y=self.d0(x)
#         y=self.d1(y)
#         y=self.d2(y)
#         return y
# ```

# 学習1回分の記述は前述の例と少し異なります．`mode` という変数を利用して，トレーニングの際にはパラメータ更新を行い，バリデーションの際にはパラメータ更新を行わないように制御します．その他は前述の例と同じです．
# ```python
#     # 学習1回の記述
#     @tf.function
#     def inference(tx,tt,mode): # 「mode」という変数を新たに設定．これでパラメータ更新をするかしないかを制御する（バリデーションではパラメータ更新はしない）．
#         with tf.GradientTape() as tape:
#             model.trainable=mode
#             ty=model.call(tx)
#             costvalue=cce(tt,ty)
#         gradient=tape.gradient(costvalue,model.trainable_variables)
#         optimizer.apply_gradients(zip(gradient,model.trainable_variables))
#         accvalue=acc(tt,ty)
#         return costvalue,accvalue
# ```

# 学習ループが開始された最初の `index=np.random.permutation(TRAINSIZE)` ではトレーニングセットのサイズに応じた（この場合，0から53999）整数からなる要素をランダムに並べた配列を生成します．これを利用して，ミニバッチのときにランダムにインスタンスを抽出します．`traincost=0` のところではトレーニングコストを計算するための変数を宣言しています．ミニバッチ処理をするので，トレーニングコストはミニバッチの個数分，この場合108個分計算されるのですが，これを平均するために利用する変数です．この変数にミニバッチ処理1回毎に出力されるコストを足し合わせて，最後にミニバッチ処理の回数で割り平均値を出します．その次の `for subepoch in range(MINIBATCHNUMBER):` がミニバッチの処理です．この場合，最初に `somb` に入る値は `0`，`eomb` に入る値は `500` です．1番目から500番目までのデータを抽出する作業のためです．`index[somb:eomb]` には500個のランダムに抽出された整数が入っていますが，それを `litrainx[index[somb:eomb]]` のように使うことで，トレーニングセットからランダムに500個のインスタンスを抽出します．`traincost+=subtraincost` は1回のミニバッチ処理で計算されたコストを上で準備した変数に足し合わせる記述です．ミニバッチ処理が終了した後は，`traincost=traincost/MINIBATCHNUMBER` によって平均トレーニングコストを計算し，また，`validcost,_=inference(livalidx,livalidt,False)` によってバリデーションコストを計算し，それらの値をエポック毎に出力する記述をしています．
# ```python
#     # 学習ループ
#     for epoch in range(1,MAXEPOCH+1):
#         # トレーニング
#         index=np.random.permutation(TRAINSIZE)
#         traincost=0
#         for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
#             somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
#             eomb=somb+MINIBATCHSIZE # 「end of minibatch」
#             subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
#             traincost+=subtraincost
#         traincost=traincost/MINIBATCHNUMBER
#         # バリデーション
#         validcost,_=inference(livalidx,livalidt,False)
#         # 学習過程の出力
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
# ```

# 次に，出力結果について説明します．このプログラムを実行するとエポックとその時のトレーニングコストとバリデーションコストが出力されます．
# ```
# Epoch    1: Training cost=  0.4437 Validation cost=  0.1900
# Epoch    2: Training cost=  0.1914 Validation cost=  0.1325
# Epoch    3: Training cost=  0.1372 Validation cost=  0.1056
# .
# .
# .
# ```
# これは各エポックのときの人工知能の性能です．エポックが50のとき，トレーニングのコストはとても小さい値です．コストは小さければ小さいほど良いので，学習はしっかりされていることが確認されます．しかし，これはトレーニングデータに対する人工知能の性能です．もしかしたらトレーニングデータに対してのみ性能を発揮できる，トレーニングデータに過剰に適合してしまった人工知能である可能性があります．だから，そうなっていないかどうかを確認する別のデータ，つまり，バリデーションデータセットにおけるコストも確認する必要があります．エポックが50のときのバリデーションのコストはエポック20くらいのときのコストより大きくなっています．すなわち，この人工知能はトレーニングデータに過剰に適合しています．おそらくエポック20くらいの人工知能が最も良い人工知能であって，これを最終的なプロダクトとして選択する必要があります．次の操作ではこれを行います．

# ### 学習曲線の描画

# 学習曲線とは横軸にエポック，縦軸にコストの値をプロットした図です．これを観察することで，どれくらいのエポックで学習が進み始めたか，人工知能の成長が止まったか，どのくらいのエポックで過剰適合が起きたか等を視覚的に理解することができます（慣れたら前述の結果のような数字を読むだけでこの図を想像できるようになるのだと思います）．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MAXEPOCH=50
    MINIBATCHSIZE=500
    UNITSIZE=500
    TRAINSIZE=54000
    MINIBATCHNUMBER=TRAINSIZE//MINIBATCHSIZE # ミニバッチのサイズとトレーニングデータのサイズから何個のミニバッチができるか計算

    # データの読み込み
    (lilearnx,lilearnt),(litestx,litestt)=tf.keras.datasets.mnist.load_data()
    outputsize=len(np.unique(lilearnt)) # MNISTにおいて出力ベクトルのサイズは0から9の10
    
    # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
    litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
    livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
    
    # 最大値を1にしておく
    litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
    
    # ネットワークの定義
    model=Network(UNITSIZE,outputsize)
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用するため学習では利用しないが次のコードのために一応定義しておく．
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt,mode):
        with tf.GradientTape() as tape:
            model.trainable=mode
            ty=model.call(tx)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    liepoch,litraincost,livalidcost=[],[],[]
    for epoch in range(1,MAXEPOCH+1):
        # トレーニング
        index=np.random.permutation(TRAINSIZE)
        traincost=0
        for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
            somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
            eomb=somb+MINIBATCHSIZE # 「end of minibatch」
            subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
            traincost+=subtraincost
        traincost=traincost/MINIBATCHNUMBER
        # バリデーション
        validcost,_=inference(livalidx,livalidt,False)
        # 学習過程の出力
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)

    # 学習曲線の描画    
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

    # 次に進む．

class Network(Model):
    def __init__(self,UNITSIZE,OUTPUTSIZE):
        super(Network,self).__init__()
        self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
        self.d1=Dense(UNITSIZE, activation="relu")
        self.d2=Dense(OUTPUTSIZE, activation="softmax")
    
    def call(self,x):
        y=self.d0(x)
        y=self.d1(y)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# 最初に，コードの変更部位について説明します．以下の部分を追加しました．これは描画に必要なライブラリである `matplotlib` を利用するための記述です．
# ```python
# import matplotlib.pyplot as plt
# ```

# 次に，学習ループの記述ですが，以下のように最初に `liepoch`，`litraincost`，`livalidcost` という3つの空の配列を用意しました．その後ループの最後で，これらの配列に，それぞれ，エポックの値，トレーニングのコストおよびバリデーションのコストをエポックを進めるたびに追加しています．
# ```python
#     # 学習ループ
#     liepoch,litraincost,livalidcost=[],[],[]
#     for epoch in range(1,MAXEPOCH+1):
#         # トレーニング
#         index=np.random.permutation(TRAINSIZE)
#         traincost=0
#         for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
#             somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
#             eomb=somb+MINIBATCHSIZE # 「end of minibatch」
#             subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
#             traincost+=subtraincost
#         traincost=traincost/MINIBATCHNUMBER
#         # バリデーション
#         validcost,_=inference(livalidx,livalidt,False)
#         # 学習過程の出力
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
#         liepoch.append(epoch)
#         litraincost.append(traincost)
#         livalidcost.append(validcost)
# ```

# 最後の以下の部分は学習曲線をプロットするためのコードです．
# ```python
#     # 学習曲線の描画    
#     plt.plot(liepoch,litraincost,label="Training")
#     plt.plot(liepoch,livalidcost,label="Validation")
#     plt.ylim(0,0.2)
#     plt.xlabel("Epoch")
#     plt.ylabel("Cost")
#     plt.legend()
#     plt.show()
# ```

# 結果を観ると，トレーニングセットにおけるコストの値はエポックを経るにつれて小さくなっていることがわかります．これは，人工知能が与えられたデータに適合していることを示しています．一方で，バリデーションセットにおけるコストの値は大体エポックが10と20の間くらいで下げ止まり，その後はコストが増加に転じています．このコストの増加，人工知能がこのデータセットに適合するのとは逆の方向に成長を始めたことを意味しています．この現象が起こった原因は，この人工知能がその成長に利用するデータセット（トレーニングデータセット）に（のみ）過剰に適合し，汎化性能を失ったことにあります．この曲線を観察する限り，エポックは大体10から20の間くらいに留めておいた方が良さそうです．このような画像を観て，大体20で学習を止める，みたいに決めても悪くはありませんが，もっと体系的な方法があるので次にその方法を紹介します．

# ### 早期終了

# 学習の早期終了（early stopping）とは過学習を防ぐための方法です．ここでは，ペイシェンス（patience）を利用した早期終了を紹介します．この方法では最も良い値のバリデーションコストを記録し続けます．そして学習を続け，そのベストなバリデーションコストを $n$ 回連続で更新できなかった場合，そこで学習を打ち切ります．この $n$ がペイシェンスと呼ばれる値です．ペイシェンスには我慢とか忍耐とかそのような意味があります．コードは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MAXEPOCH=50
    MINIBATCHSIZE=500
    UNITSIZE=500
    TRAINSIZE=54000
    MINIBATCHNUMBER=TRAINSIZE//MINIBATCHSIZE # ミニバッチのサイズとトレーニングデータのサイズから何個のミニバッチができるか計算
    PATIENCE=5

    # データの読み込み
    (lilearnx,lilearnt),(litestx,litestt)=tf.keras.datasets.mnist.load_data()
    outputsize=len(np.unique(lilearnt)) # MNISTにおいて出力ベクトルのサイズは0から9の10
    
    # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
    litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
    livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
    
    # 最大値を1にしておく
    litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
    
    # ネットワークの定義
    model=Network(UNITSIZE,outputsize)
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用するため学習では利用しないが次のコードのために一応定義しておく．
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt,mode):
        with tf.GradientTape() as tape:
            model.trainable=mode
            ty=model.call(tx)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    liepoch,litraincost,livalidcost=[],[],[]
    patiencecounter,bestvalue=0,100000
    for epoch in range(1,MAXEPOCH+1):
        # トレーニング
        index=np.random.permutation(TRAINSIZE)
        traincost=0
        for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
            somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
            eomb=somb+MINIBATCHSIZE # 「end of minibatch」
            subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
            traincost+=subtraincost
        traincost=traincost/MINIBATCHNUMBER
        # バリデーション
        validcost,_=inference(livalidx,livalidt,False)
        # 学習過程の出力
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)
        if validcost<bestvalue:
            bestvalue=validcost
            patiencecounter=0
        else:
            patiencecounter+=1
        if patiencecounter==PATIENCE:
            break

    # 学習曲線の描画    
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

    # 次に進む．

class Network(Model):
    def __init__(self,UNITSIZE,OUTPUTSIZE):
        super(Network,self).__init__()
        self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
        self.d1=Dense(UNITSIZE, activation="relu")
        self.d2=Dense(OUTPUTSIZE, activation="softmax")
    
    def call(self,x):
        y=self.d0(x)
        y=self.d1(y)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# プログラムには以下の部分を追加しました．今回は4回までコストが改善しなくても許すが，5回目は許さないということです．
# ```python
#     PATIENCE=5
# ```

# 学習ループを以下のようにコードを追加しました．`patiencecounter` はコストが更新されなかった回数を数えるカウンタです．`bestvalue` は最も良いコストの値を記録する変数です．
# ```python
#     # 学習ループ
#     liepoch,litraincost,livalidcost=[],[],[]
#     patiencecounter,bestvalue=0,100000
#     for epoch in range(1,MAXEPOCH+1):
#         # トレーニング
#         index=np.random.permutation(TRAINSIZE)
#         traincost=0
#         for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
#             somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
#             eomb=somb+MINIBATCHSIZE # 「end of minibatch」
#             subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
#             traincost+=subtraincost
#         traincost=traincost/MINIBATCHNUMBER
#         # バリデーション
#         validcost,_=inference(livalidx,livalidt,False)
#         # 学習過程の出力
#         print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
#         liepoch.append(epoch)
#         litraincost.append(traincost)
#         livalidcost.append(validcost)
#         if validcost<bestvalue:
#             bestvalue=validcost
#             patiencecounter=0
#         else:
#             patiencecounter+=1
#         if patiencecounter==PATIENCE:
#             break
# ```
# 以下の部分で，もし最も良いコストよりさらに良いコストが得られたらベストなコストを更新し，また，ペイシェンスのカウンタを元に（`0`）戻す作業をし，それ以外の場合はペイシェンスのカウンタを1ずつ増やします．もし，カウンタの値があらかじめ設定したペイシェンスの値に達したら学習ループを停止します．
# ```python
#         if validcost<bestvalue:
#             bestvalue=validcost
#             patiencecounter=0
#         else:
#             patiencecounter+=1
#         if patiencecounter==PATIENCE:
#             break
# ```

# 結果を観ると，過学習が起こっていなさそうなところで学習が停止されているのが解ります．

# ### モデルの保存と利用

# これまでに，早期終了を利用して良い人工知能が生成できるエポックが判明しました．機械学習の目的は当然，良い人工知能を開発することです．開発した人工知能は普通，別のサーバーとかトレーニングした時とは別の時間に利用したいはずです．ここで，この学習で発見した人工知能を保存して別のプログラムから，独立した人工知能として利用する方法を紹介します．最後に，テストセットでのその人工知能の性能を評価します．コードは以下のように変更します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MAXEPOCH=50
    MINIBATCHSIZE=500
    UNITSIZE=500
    TRAINSIZE=54000
    MINIBATCHNUMBER=TRAINSIZE//MINIBATCHSIZE # ミニバッチのサイズとトレーニングデータのサイズから何個のミニバッチができるか計算
    PATIENCE=5

    # データの読み込み
    (lilearnx,lilearnt),(litestx,litestt)=tf.keras.datasets.mnist.load_data()
    outputsize=len(np.unique(lilearnt)) # MNISTにおいて出力ベクトルのサイズは0から9の10
    
    # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
    litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
    livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
    
    # 最大値を1にしておく
    litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
    
    # ネットワークの定義
    model=Network(UNITSIZE,outputsize)
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用するため学習では利用しないが次のコードのために一応定義しておく．
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    # モデルを保存するための記述
    checkpoint=tf.train.Checkpoint(model=model)
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt,mode):
        with tf.GradientTape() as tape:
            model.trainable=mode
            ty=model.call(tx)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # 学習ループ
    liepoch,litraincost,livalidcost=[],[],[]
    patiencecounter,bestvalue=0,100000
    for epoch in range(1,MAXEPOCH+1):
        # トレーニング
        index=np.random.permutation(TRAINSIZE)
        traincost=0
        for subepoch in range(MINIBATCHNUMBER): # 「subepoch」は「epoch in epoch」と呼ばれるのを見たことがある．
            somb=subepoch*MINIBATCHSIZE # 「start of minibatch」
            eomb=somb+MINIBATCHSIZE # 「end of minibatch」
            subtraincost,_=inference(litrainx[index[somb:eomb]],litraint[index[somb:eomb]],True)
            traincost+=subtraincost
        traincost=traincost/MINIBATCHNUMBER
        # バリデーション
        validcost,_=inference(livalidx,livalidt,False)
        # 学習過程の出力
        print("Epoch {:4d}: Training cost= {:7.4f} Validation cost= {:7.4f}".format(epoch,traincost,validcost))
        liepoch.append(epoch)
        litraincost.append(traincost)
        livalidcost.append(validcost)
        if validcost<bestvalue:
            bestvalue=validcost
            patiencecounter=0
        else:
            patiencecounter+=1
        if patiencecounter==PATIENCE:
            checkpoint.save("mlp-mnist/model")
            break

    # 学習曲線の描画    
    plt.plot(liepoch,litraincost,label="Training")
    plt.plot(liepoch,livalidcost,label="Validation")
    plt.ylim(0,0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

    # ユニットサイズや層の数やその他のハイパーパラメータを色々変更してより良いバリデーションコストを出力する予測器を作る．

class Network(Model):
    def __init__(self,UNITSIZE,OUTPUTSIZE):
        super(Network,self).__init__()
        self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
        self.d1=Dense(UNITSIZE, activation="relu")
        self.d2=Dense(OUTPUTSIZE, activation="softmax")
    
    def call(self,x):
        y=self.d0(x)
        y=self.d1(y)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# 以下の記述を追加しました．
# ```python
#     # モデルを保存するための記述
#     checkpoint=tf.train.Checkpoint(model=model)
# ```

# また，学習ループの最後に以下のような記述を追加しました．`mlp-mnist` というディレクトリに `model` という名前で学習済みモデルを保存するように，という意味です．
# ```python
#             checkpoint.save("mlp-mnist/model")
# ```

# 以下のシェルのコマンドを打つと，ディレクトリが新規に生成されていることを確認できます．

# In[ ]:


get_ipython().system(' ls mlp-mnist')


# 最後に，以下のコードで保存したモデル（実体はパラメータ）を呼び出して，テストセットにてその性能を評価します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def main():
    # ハイパーパラメータの設定
    UNITSIZE=500
    TRAINSIZE=54000

    # データの読み込み
    (lilearnx,lilearnt),(litestx,litestt)=tf.keras.datasets.mnist.load_data()
    outputsize=len(np.unique(lilearnt)) # MNISTにおいて出力ベクトルのサイズは0から9の10
    
    # 学習セットをトレーニングセットとバリデーションセットに分割（9:1）
    litrainx,litraint=lilearnx[:TRAINSIZE],lilearnt[:TRAINSIZE]
    livalidx,livalidt=lilearnx[TRAINSIZE:],lilearnt[TRAINSIZE:]
    
    # 最大値を1にしておく
    litrainx,livalidx,litestx=litrainx/255,livalidx/255,litestx/255
    
    # ネットワークの定義
    model=Network(UNITSIZE,outputsize)
    cce=tf.keras.losses.SparseCategoricalCrossentropy() #これでロス関数を生成する．
    acc=tf.keras.metrics.SparseCategoricalAccuracy() # これはテストの際に利用する．
    optimizer=tf.keras.optimizers.Adam() #これでオプティマイザを生成する．
    # モデルの読み込み
    checkpoint=tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint("mlp-mnist"))
    
    # 学習1回の記述
    @tf.function
    def inference(tx,tt,mode):
        with tf.GradientTape() as tape:
            model.trainable=mode
            ty=model.call(tx)
            costvalue=cce(tt,ty)
        gradient=tape.gradient(costvalue,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        accvalue=acc(tt,ty)
        return costvalue,accvalue
    
    # テストセットでの性能評価
    testcost,testacc=inference(litestx,litestt,False)
    print("Test cost= {:7.4f} Test ACC= {:7.4f}".format(testcost,testacc))

    # テストセットの最初の画像を入力してみる
    plt.imshow(litestx[0], cmap="gray")
    plt.text(1, 2.5, int(litestt[0]), fontsize=20, color="white")
    ty=model.call(litestx[:1]) # 予測器にデータを入れて予測
    print("Output vector:",ty.numpy()) # 出力ベクトルを表示
    print("Argmax of the output vector:",np.argmax(ty.numpy())) # 出力ベクトルの要素の中で最も大きい値のインデックスを表示

    # 上で構築したハイパーパラメータを変化させたより良い人工知能の性能評価をする．

class Network(Model):
    def __init__(self,UNITSIZE,OUTPUTSIZE):
        super(Network,self).__init__()
        self.d0=Flatten(input_shape=(28,28)) # 行列をベクトルに変換
        self.d1=Dense(UNITSIZE, activation="relu")
        self.d2=Dense(OUTPUTSIZE, activation="softmax")
    
    def call(self,x):
        y=self.d0(x)
        y=self.d1(y)
        y=self.d2(y)
        return y

if __name__ == "__main__":
    main()


# 学習済みモデルは以下のような記述で読み込みます．
# ```python
#     # モデルの読み込み
#     checkpoint=tf.train.Checkpoint(model=model)
#     checkpoint.restore(tf.train.latest_checkpoint("mlp-mnist"))
# ```

# テストセットでの性能評価のための記述です．
# ```python
#     # テストセットでの性能評価
#     testcost,testacc=inference(litestx,litestt,False)
#     print("Test cost= {:7.4f} Test ACC= {:7.4f}".format(testcost,testacc))
# ```

# 最後に，テストセットの最初の画像を予測器に入れてその結果を確認してみます．以下のコードで行います．
# ```python
#     # テストセットの最初の画像を入力してみる
#     plt.imshow(litestx[0], cmap="gray")
#     plt.text(1, 2.5, int(litestt[0]), fontsize=20, color="white")
#     ty=model.call(litestx[:1]) # 予測器にデータを入れて予測
#     print("Output vector:",ty.numpy()) # 出力ベクトルを表示
#     print("Argmax of the output vector:",np.argmax(ty.numpy())) # 出力ベクトルの要素の中で最も大きい値のインデックスを表示
# ```

# 実行すると，テストセットでも高い性能を示すことが確認できました．また，7が答えである画像を入力に，`7` を出力できていることを確認しました．

# ```{note}
# 終わりです．
# ```
