#!/usr/bin/env python
# coding: utf-8

# # TensorFlow の基本的な利用方法

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

# ```{note}
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
# $\displaystyle y=f(x)=\frac{1}{2}(x+1)^2+1$
# 
# よって勾配ベクトル場は以下のように計算されます．
# 
# $\nabla f=x+1$
# 
# 初期パラメータを以下のように決めます（実際にはランダムに決める）．
# 
# $x_0=1.6$
# 
# この関数の極小値を見つけたいのです．これは解析的に解くのはとても簡単で，括弧の中が0になる値，すなわち $x$ が $-1$ のとき，極小値 $y=1$ です．

# 最急降下法で解くと，以下の図のようになります．最急降下法は解析的に解くことが難しい問題を正解の方向へ少しずつ反復的に動かしていく方法です．

# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/gradientDescent.svg?raw=1" width="100%" />

# これを TensorFlow を用いて実装すると以下のようになります．出力中，`Objective` は目的関数の値，`Solution` はその時点での解です．最終的に $x=-0.9912\simeq-1$ のとき，最適値 $y=1$ が出力されています．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
 
def main():
    tx = tf.Variable(1.6, dtype=tf.float32) # これが変数．
    epoch, update_value, lr = 1, 5, 0.1 # 更新値はダミー変数．
    while abs(update_value) > 0.001:
        with tf.GradientTape() as tape:
            ty = (1/2) * (tx + 1)**2 + 1
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
