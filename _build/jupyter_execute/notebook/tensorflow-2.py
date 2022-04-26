#!/usr/bin/env python
# coding: utf-8

# # 多層パーセプトロン（編集中）

# ## 基本操作

# この節では TensorFlow の基本的な操作方法を紹介します．

# ### インポート

# NumPy と同じように TensorFlow をインポートします．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    pass
    # TensorFlow のバージョンを出力．
 
if __name__ == "__main__":
    main()


# ### テンソル

# TensorFlow では「テンソル」と呼ばれる NumPy の多次元配列に類似したデータ構造を用います．2行目で TensorFlow をインポートします．5行目のテンソルを生成するためのコマンドは `tf.zeros()` で，これによって，全要素が `0` であるテンソルが生成されます．最初の引数には生成されるテンソルの次元数を指定します．また，データのタイプを指定することができますが以下の場合は32ビットのフロートの値を生成しています．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    tx = tf.zeros([3, 3], dtype=tf.float32)
    print(tx)
    # 1階テンソルを生成．
    # 3階テンソルを生成．

if __name__ == "__main__":
    main()


# 以下のようにすると，整数を生成できます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    tx = tf.zeros([3, 3], dtype=tf.int32) # ここが整数を生成するための記述
    print(tx)
    # 1階テンソルを生成．
    # 3階テンソルを生成．

if __name__ == "__main__":
    main()


# データのタイプを確認したい場合とテンソルのシェイプを確認したい場合は以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    tx = tf.zeros([4, 3], dtype=tf.int32)
    print(tx.dtype)
    print(tx.shape)
    # 浮動小数点数の2行2列の行列を生成して型と形を確認．

if __name__ == "__main__":
    main()


# 一様分布に従う乱数を生成したい場合には以下のようにします．一様分布の母数（パラメータ）は最小値と最大値です．ここでは，最小値が-1で最大値が1の一様分布 $U(-1,1)$ に従う乱数を生成します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    tx = tf.random.uniform([4, 3], minval=-1, maxval=1, dtype=tf.float32)
    print(tx)
    # 何度か実行して値が異なることを確認．

if __name__ == "__main__":
    main()


# 上のコードセルを何度か繰り返し実行すると一様分布に従う4行3列のテンソルの値が生成されますが，1回ごとに異なる値が出力されているはずです．これは計算機実験をする際にとても厄介です．再現性が取れないからです．これを防ぐために「乱数の種」というものを設定します．以下のコードの3行目のような指定を追加します．ここでは，0という値を乱数の種に設定していますが，これはなんでも好きな値を設定して良いです．

# ```{attention}
# 普通，科学的な計算機実験をする際に乱数の種を固定せずに計算を開始することはあり得ません．乱数を使う場合は常に乱数の種を固定しておくことを習慣づける必要があります．
# ```

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)

def main():
    tx = tf.random.uniform([4, 3], minval=-1, maxval=1, dtype=tf.float32)
    print(tx)
    # 何度か繰り返して実行．
    # 全く同じコマンドで別の変数を生成して出力．
    # 何度か繰り返して実行．
    # 乱数のタネを別の値に変更した後に何度か繰り返して実行．

if __name__ == "__main__":
    main()


# Python 配列より変換することもできます．この `tf.constant()` は実際には使う機会は多くありません．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.constant([2, 4], dtype=tf.float32)
    print(tx)
    # 多次元 Python 配列をテンソルに変換．

if __name__ == "__main__":
    main()


# なぜなら，TensorFlow のテンソル（tf.Tensor）と NumPy の多次元配列（ndarray）の変換は以下のふたつのルールによる簡単な変換を TensorFlow が自動で行ってくれるからです．
# 
# 
# 1.   TensorFlowの演算により NumPy の ndarray は自動的に tf.Tensor に変換される．
# 2.   NumPy の演算により tf.Tensor は自動的に ndarray に変換される．
# 
# これに関しては以下の四則計算のところでその挙動を確認します．
# 
# 

# ### 四則計算

# テンソルの四則計算は以下のように行います．最初に足し算を行います．NumPy と同じようにやはり element-wise な計算です．実行結果は `tf.Tensor([3 7], shape=(2,), dtype=int32)` となっており，配列の計算の結果が tf.Tensor に変換されていることが確認できます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.add([2, 4], [1, 3])
    print(tx)
    # 別の計算を実行．

if __name__ == "__main__":
    main()


# 以下では，ふたつの NumPy 多次元配列を生成しそれらを足し合わせます．得られる結果は NumPy の多次元配列でなくて tf.Tensor であることが確認できます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
 
def main():
    na = np.array([[1, 2], [1, 3]])
    nb = np.array([[2, 3], [4, 5]])
    tx = tf.add(na, nb)
    print(tx)
    # 別の計算を実行．

if __name__ == "__main__":
    main()


# その他の四則演算は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
 
def main():
    na = np.array([[1, 2], [1, 3]], dtype=np.float32)
    nb = np.array([[2, 3], [5, 6]], dtype=np.float32)
    print(tf.add(na, nb))
    print(tf.subtract(nb, na))
    print(tf.multiply(na, nb))
    print(tf.divide(nb, na))
    # 別の計算を実行．

if __name__ == "__main__":
    main()


# ```{note}
# 上から足し算，引き算，掛け算，割り算です．
# ```

# 上の `tf.multiply()` はテンソルの要素ごとの積（アダマール積）を計算するための方法です．行列の積は以下のように `tf.matmul()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
 
def main():
    na = np.array([[1, 2], [1, 3]], dtype=np.float32)
    nb = np.array([[2, 3], [5, 6]], dtype=np.float32)
    print(tf.matmul(na, nb))
    # tf.multiply() との違いを確認．

if __name__ == "__main__":
    main()


# テンソルもブロードキャストしてくれます．以下のようなテンソルとスカラの計算も良い感じで解釈して実行してくれます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
 
def main():
    na = np.array([[1, 2], [1, 3]], dtype=np.float32)
    print(tf.add(na, 1))
    # 引き算を実行．

if __name__ == "__main__":
    main()


# 以下のように `+` や `-` を使って記述することも可能です．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    ta = tf.constant([2, 4], dtype=tf.float32)
    tb = tf.constant([5, 6], dtype=tf.float32)
    print(ta + tb)
    print(tb - ta)
    print(ta * tb)
    print(tb / ta)
    # "//" と "%" の挙動を確認．

if __name__ == "__main__":
    main()


# 二乗の計算やテンソルの要素の総和を求めるための便利な方法も用意されています．このような方法は状況に応じてその都度調べて使います．全部覚える必要はありません．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
 
def main():
    nx = np.array([1, 2, 3], dtype=np.float32)
    print(tf.square(nx))
    print(tf.reduce_sum(nx))
    # 多次元配列での挙動を確認．

if __name__ == "__main__":
    main()


# ### 特殊な操作

# 以下のようなスライスの実装も NumPy と同じです．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.constant([[2, 4], [6, 8]], dtype=tf.float32)
    print(tx[:,0])
    # 2行目の値を出力．

if __name__ == "__main__":
    main()


# ```{hint}
# これは2行2列の行列の1列目の値を取り出す操作です．
# ```

# テンソルのサイズの変更には `tf.reshape()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.random.uniform([4, 5], dtype=tf.float32)
    print(tx)
    print(tf.reshape(tx, [20]))
    print(tf.reshape(tx, [1, 20]))
    print(tf.reshape(tx, [5, 4]))
    print(tf.reshape(tx, [-1, 4]))
    # tf.reshape(tx, [20, 1]) の形を確認．

if __name__ == "__main__":
    main()


# 以上のプログラムの6行目では4行5列の行列が生成されています．これを，20要素からなるベクトルに変換するのが7行目の記述です．また，8行目の記述では1行20列の行列を生成できます．また，9行目は5行4列の行列を生成するためのものです．同じく10行目も5行4列の行列を生成します．ここでは，`tf.reshape()` の shape を指定するオプションの最初の引数に `-1` が指定されていますが，これのように書くと自動でその値が推測されます．この場合，`5` であると推測されています．

# ### 変数の変換

# これまでに，NumPyの 多次元配列を TensorFlow のテンソルに変換する方法は確認しました．テンソルを NumPy 配列に変換するには明示的に `numpy()` を指定する方法があります．6行目は NumPy 配列を生成します．8行目はその NumPy 配列をテンソルに変換します．さらに，NumPy 配列に戻すためには10行目のように `.numpy()` を利用します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def main():
    na = np.ones(5)
    print("NumPy:", na)
    ta = tf.constant(na, dtype=tf.float32)
    print("Tensor:", ta)
    na = ta.numpy()
    print("NumPy:", na)
    # さらに32ビット整数型のテンソルに変換．

if __name__ == "__main__":
    main()


# また，テンソルに対して NumPy の演算操作を行うと自動的にテンソルは NumPy 配列に変換されます．以下の8行目と9行目はどちらもベクトルの内積を計算していますが，8行目で得られる結果はテンソル，9行目で得られる結果は NumPy の値です．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def main():
    ta = tf.constant([2, 4], dtype=tf.float32)
    tb = tf.constant([5, 6], dtype=tf.float32)
    print(tf.tensordot(ta, tb, axes=1))
    print(np.dot(ta, tb))
    # NumPy 配列とテンソルの内積をテンソルの演算方法で計算．

if __name__ == "__main__":
    main()


# ## 最急降下法

# 深層ニューラルネットワークのパラメータを更新するためには何らかの最適化法が利用されます．最も簡単な最適化法である最急降下法を実装します．

# ### 単一の変数に対する勾配

# 深層学習法におけるアルゴリズムの中身を分解すると行列の掛け算と微分から構成されていることがわかります．TensorFlow はこの行列の掛け算と微分を行うライブラリです．自動微分機能を提供します．ここでは勾配の計算を紹介するため，以下の式を考えます．
# 
# $y = x^2 + 2$
# 
# これに対して以下の偏微分を計算することができます．
# 
# $\dfrac{\partial y}{\partial x} = 2x$
# 
# よって $x=5$ のときの偏微分係数は以下のように計算できます．
# 
# $\left.\dfrac{\partial y}{\partial x}\right|_{x=5}=10$
# 
# これを TensorFlow で実装すると以下のように書けます．微分は10行目のように `tape.gradient()` によって行います．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.Variable(5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        ty = tx**2 + 2 # ここに勾配を求める対象の計算式を書く．
    grad = tape.gradient(ty, tx)
    print(grad)
    # y=3x^2+x+1をxで偏微分したときの，x=1の値を計算．

if __name__ == "__main__":
    main()


# ### 複数の変数に対する勾配

# 上の程度の微分だとこの自動微分機能はさほど有難くないかもしれませんが，以下のような計算となると，そこそこ有難くなってきます．以下では，(1, 2) の行列 `ts` と (2, 2) の行列 `tt` と (2, 1) の行列 `tu` を順に掛けることで，最終的に (1, 1) の行列の値，スカラー値を得ますが，それを `tt` で微分した値を計算しています（`tt` で偏微分したので得られる行列のシェイプは `tt` と同じ）．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf

def main():
    # Definition
    ts = tf.constant([[2, 1]], dtype=tf.float32)
    tt = tf.Variable([[2, 4], [6, 8]], dtype=tf.float32) # これが変数．
    tu = tf.constant([[4], [1]], dtype=tf.float32)
    # Calculation
    with tf.GradientTape() as tape:
        tz = tf.matmul(tf.matmul(ts, tt), tu)
    grad = tape.gradient(tz,tt)
    print(grad)
    # 2行2列の定数行列taを生成，ts*ta*tt*tuの行列の積を計算し，ttで偏微分．

if __name__ == "__main__":
    main()


# これは以下のような計算をしています．`tf.Variable()` で定義される行列は以下です：
# 
# $
#   t = \left[
#     \begin{array}{cc}
#       v & w \\
#       x & y \\
#     \end{array}
#   \right]
# $．
# 
# また，`tf.constant()` で定義される行列は以下です：
# 
# $s = \left[
#     \begin{array}{cc}
#       2 & 1 \\
#     \end{array}
#   \right]
# $，
# 
# $u = \left[
#     \begin{array}{c}
#       4 \\
#       1
#     \end{array}
#   \right]
# $．
# 
# これに対して11行目の計算で得られる値は以下です：
# 
# $z(v,w,x,y) = 8v+2w+4x+y$．
# 
# よってこれらを偏微分して，それぞれの変数がプログラム中で定義される値のときの値は以下のように計算されます：
# 
# $\left.\dfrac{\partial z}{\partial v}\right|_{(v,w,x,y)=(2,4,6,8)}=8$，
# 
# $\left.\dfrac{\partial z}{\partial w}\right|_{(v,w,x,y)=(2,4,6,8)}=2$，
# 
# $\left.\dfrac{\partial z}{\partial x}\right|_{(v,w,x,y)=(2,4,6,8)}=4$，
# 
# $\left.\dfrac{\partial z}{\partial y}\right|_{(v,w,x,y)=(2,4,6,8)}=1$．

# ```{note}
# これにコスト関数と活性化関数付けて最急降下法やったらニューラルネットワークです．自動微分すごい．
# ```

# ### 最急降下法の実装

# なぜ微分を求めたいかというと，勾配法（深層学習の場合，普通，最急降下法）でパラメータをアップデートしたいからです．以下では最急降下法を実装してみます．最急降下法は関数の最適化法です．ある関数に対して極小値（極大値）を計算するためのものです．以下のような手順で計算が進みます．
# 

# 1.   初期パラメータ（$\theta_0$）をランダムに生成します．
# 2.   もしパラメータ（$\theta_t$）が最適値または，最適値に近いなら計算をやめます．ここで，$t$ は以下の繰り返しにおける $t$ 番目のパラメータです．
# 3.   パラメータを以下の式によって更新し，かつ，$t$ の値を $1$ だけ増やします．ここで，$\alpha$ は学習率と呼ばれる更新の大きさを決める値で，$g_t$ は $t$ のときの目的の関数の勾配です．<br>
#     $\theta_{t+1}=\theta_t-\alpha g_t$
# 4.   ステップ2と3を繰り返します．
# 

# ここでは以下の関数を考えます．
# 
# $y=(x+1)^2+2$
# 
# 初期パラメータを以下のように決めます（実際にはランダムに決める）．
# 
# $x_0=1.6$
# 
# この関数の極小値を見つけたいのです．これは解析的に解くのはとても簡単で，括弧の中が0になる値，すなわち $x$ が $-1$ のとき，極小値 $y=2$ です．

# 最急降下法で解くと，以下の図のようになります．最急降下法は解析的に解くことが難しい問題を正解の方向へ少しずつ反復的に動かしていく方法です．

# <img src="https://drive.google.com/uc?id=1bRRIwEJy-ltIHnzfqOpOVVlZYXrT3PXR" width="70%">

# これを TensorFlow を用いて実装すると以下のようになります．出力中，`Objective` は目的関数の値，`Solution` はその時点での解です．最終的に $x=-0.9968\simeq-1$ のとき，最適値 $y=2$ が出力されています．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.Variable(1.6, dtype=tf.float32) # これが変数．
    epoch, update_value, lr = 1, 5, 0.1 # 更新値はダミー変数．
    while abs(update_value) > 0.001:
        with tf.GradientTape() as tape:
            ty = (tx + 1)**2 + 2
        grad = tape.gradient(ty, tx)
        update_value = lr * grad.numpy()
        tx.assign(tx - update_value)
        print("Epoch {:4d}:\tObjective = {:5.3f}\tSolution = {:7.4f}".format(epoch, ty, tx.numpy()))
        epoch = epoch + 1
        # 下の新たなコードセルで計算．

if __name__ == "__main__":
    main()


# 5行目で最初のパラメータを発生させています．通常は乱数によってこの値を決めますが，ここでは上の図に合わせて1.6とします．次の6行目では，最初のエポック，更新値，学習率を定義します．エポックとは（ここでは）パラメータの更新回数のことを言います．7行目は終了条件です．以上のような凸関数においては勾配の値が0になる点が本当の最適値（正しくは停留点）ではありますが，計算機的にはパラメータを更新する値が大体0になったところで計算を打ち切ります．この場合，「大体0」を「0.001」としました．9行目は目的の関数，10行目で微分をしています．11行目は最急降下法で更新する値を計算しています．12行目の計算で `tx` をアップデートします．この12行目こそが上述の最急降下法の式です．

# ```{note}
# ここで最急降下法について説明しましたが，このような実装は TensorFlow を利用する際にする必要はありません．TensorFlow はこのような計算をしてくれる方法を提供してくれています．よって，ここの部分の意味が解らなかったとしても以降の部分は理解できます．
# ```

# ```{note}
# 終わりです．
# ```

# #3. <font color="Crimson">深層学習入門</font>

# ##3-1. <font color="Crimson">扱うデータの紹介</font>

# ###3-1-1. <font color="Crimson">MNIST について</font>

# このセクションでは深層学習法で用いられるアルゴリズムの中でも最も基本的なものである MLP の実装方法を学びますが，この MLP に処理させるデータセットとして，機械学習界隈で最も有名なデータセットである MNIST（Mixed National Institute of Standards and Technology database）を解析対象に用います．筆者の経験によると，日本人も外国人も今のところ会った人全員がこれのことは「エムニスト」と発音しています．MNIST は縦横28ピクセル，合計784ピクセルよりなる画像データです．画像には手書きの一桁の数字（0から9）が含まれています．公式ウェブサイトでは，学習データセット6万個とテストデータセット1万個，全部で7万個の画像からなるデータセットが無償で提供されています．

# ###3-1-2. <font color="Crimson">ダウンロードと可視化</font>

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

# ##3-2. <font color="Crimson">MLP の実装</font>

# ###3-2-1. <font color="Crimson">簡単な MLP の実装</font>

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
tf.random.set_seed(0)
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np

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
tf.random.set_seed(0)
from tensorflow.keras.layers import Dense
import numpy as np

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


# ###3-2-2. <font color="Crimson">MNIST の基本処理</font>

# 次に，MNIST を処理して「0から9の数字が書かれた（描かれた）手書き文字を入力にして，その手書き文字が0から9のどれなのかを判別する人工知能」を構築します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np

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

# ###3-2-3. <font color="Crimson">学習曲線の描画</font>

# 学習曲線とは横軸にエポック，縦軸にコストの値をプロットした図です．これを観察することで，どれくらいのエポックで学習が進み始めたか，人工知能の成長が止まったか，どのくらいのエポックで過剰適合が起きたか等を視覚的に理解することができます（慣れたら前述の結果のような数字を読むだけでこの図を想像できるようになるのだと思います）．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

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

# ###3-2-4. <font color="Crimson">早期終了</font>

# 学習の早期終了（early stopping）とは過学習を防ぐための方法です．ここでは，ペイシェンス（patience）を利用した早期終了を紹介します．この方法では最も良い値のバリデーションコストを記録し続けます．そして学習を続け，そのベストなバリデーションコストを $n$ 回連続で更新できなかった場合，そこで学習を打ち切ります．この $n$ がペイシェンスと呼ばれる値です．ペイシェンスには我慢とか忍耐とかそのような意味があります．コードは以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

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

# ###3-2-5. <font color="Crimson">モデルの保存と利用</font>

# これまでに，早期終了を利用して良い人工知能が生成できるエポックが判明しました．機械学習の目的は当然，良い人工知能を開発することです．開発した人工知能は普通，別のサーバーとかトレーニングした時とは別の時間に利用したいはずです．ここで，この学習で発見した人工知能を保存して別のプログラムから，独立した人工知能として利用する方法を紹介します．最後に，テストセットでのその人工知能の性能を評価します．コードは以下のように変更します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

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
