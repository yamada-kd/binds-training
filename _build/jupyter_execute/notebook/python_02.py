#!/usr/bin/env python
# coding: utf-8

# # Python の発展的な使用方法

# ## NumPy の利用方法

# NumPy とは Python で利用可能な数値計算のライブラリです．さまざまな計算をコマンド一発で行うことができます．後の章で紹介する TensorFlow の書き方がこの NumPy の書き方に似ていて，それらを相互に利用できたほうが良いので紹介します．

# ### NumPy のインポート

# NumPy は以下のようにしてインポートします．読み込んだ NumPy には `np` という略称を与えることが普通です．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    pass
 
if __name__ == "__main__":
    main()


# ### ベクトルの基本的な計算

# ベクトルは以下のように生成します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    print(np.array([1, 2, 3]))

if __name__ == "__main__":
    main()


# 要素の参照は普通の Python 配列と同じようにできます．もちろんゼロオリジンです．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3])
    print(na[0])

if __name__ == "__main__":
    main()


# ベクトルの四則計算は以下のようにします．NumPy は基本的に要素ごとに（element-wise）値を計算します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3])
    nb = np.array([5, 7, 9])
    print(na + nb)
    print(na - nb)
    print(na * nb)
    print(nb / na)

if __name__ == "__main__":
    main()


# コピーをする際は気を使わなければならない点があります．あるベクトルから別のベクトル変数を生成，つまり，コピーでベクトルを生成した場合，その生成したベクトルを元のベクトルと別のものとして扱いたい場合は以下のようにしなければなりません．以下では，8行目と9行目で元のベクトルとコピーで生成されたベクトルの要素をそれぞれ別の値で変更していますが，それぞれ別の値にて要素が置換されていることがわかります．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 1])
    nb = na.copy()
    print(na, nb)
    na[0] = 2
    nb[0] = 3
    print(na, nb)

if __name__ == "__main__":
    main()


# 一方で，以下のように `=` を使ってコピー（のようなこと）をすると生成されたベクトルは元のベクトルの参照となってしまい，（この場合の）意図している操作は実現されません．上の挙動とこの挙動は把握していないと結構危険です．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 1])
    nb = na
    print(na, nb)
    na[0] = 2
    nb[0] = 3
    print(na, nb)

if __name__ == "__main__":
    main()


# ### 行列の基本的な計算

# 行列を生成するためにも，`np.array()` を利用します．さらに，行列のサイズは `.shape` によって確認することができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na)
    print(na.shape)
 
if __name__ == "__main__":
    main()


# 行列の要素には以下のようにしてアクセスします．この場合，1行1列目の値にアクセスしています．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na[0,0])
 
if __name__ == "__main__":
    main()


# NumPy 行列は以下のようなアクセスの方法があります．行ごとまたは列ごとのアクセスです．これは多用します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na[0,:]) # all elements of row 1
    print(na[:,1]) # all elements of column 2

if __name__ == "__main__":
    main()


# 以下のようにすると行列に関する様々な統計値を得ることができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na.max())
    print(na.min())
    print(na.sum())
    print(na.mean())
    print(na.var())
    print(na.std())

if __name__ == "__main__":
    main()


# ```{note}
# 全知全能なんですね．
# ```

# 以下の `np.zeros()` や `np.ones()` を用いると引数で指定したサイズの，全要素が0または1の行列を生成することができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    print(np.zeros((3,3)))
    print(np.ones((4,4)))

if __name__ == "__main__":
    main()


# 四則計算は以下のようにします．これもやはり，element-wise な計算です．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    nb = np.array([[5, 6], [7, 8]])
    print(na + nb)
    print(nb - na)
    print(na * nb)
    print(nb / na)

if __name__ == "__main__":
    main()


# 行列の掛け算は以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    nb = np.array([[5, 6], [7, 8]])
    print(np.dot(na, nb))

if __name__ == "__main__":
    main()


# 以下のようにすると一様分布に従う乱数を生成することができます．以下の例は一様分布のものですが，NumPy には一様分布以外にもたくさんの分布が用意されています．引数で指定するのは行列のサイズです．計算機実験をする際にこのようなランダムな値を生成することがあります．そんな中，Python や NumPy に限らず計算機実験をする際に気を付けなければならないことに「乱数のタネを固定する」ということがあります．計算機実験の再現性を得るためにとても重要なので絶対に忘れないようにすべきです．乱数のタネは3行目で行っています．ここでは `0` を設定しています．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
np.random.seed(0)
 
def main():
    print(np.random.rand(3, 3))

if __name__ == "__main__":
    main()


# ```{attention}
# 機械学習の学習前の予測器の初期値は乱数です．再現性を確保するため乱数のタネを指定しないで実験をはじめることは絶対にないように気をつけなければなりません．
# ```

# 以下のようにすると行列式と逆行列を計算することができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[4, 15, 4], [-11, 5, 6], [2, 4, 8]])
    print(np.linalg.det(na)) # determinant of matrix
    print(np.linalg.inv(na)) # inverse matrix

if __name__ == "__main__":
    main()


# 固有値分解は以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    eigenvalue, eigenvector = np.linalg.eig(na)
    print(eigenvalue)
    print(eigenvector)

if __name__ == "__main__":
    main()


# 以下のようにすると「行列の要素の冪乗」を計算できます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(na ** 2)

if __name__ == "__main__":
    main()


# 一方で，「行列の冪乗」は以下のように計算します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(np.linalg.matrix_power(na, 2))

if __name__ == "__main__":
    main()


# 行列の冪乗でも，整数以外の冪指数を用いたい場合は別の方法が必要です．例えば，行列の平方根（2分の1乗）は以下のようにしなければ計算できません．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    eigenvalue, eigenvector = np.linalg.eig(na)
    print(np.dot(np.dot(eigenvector, (np.diag(eigenvalue)) ** (1/2)), np.linalg.inv(eigenvector)))

if __name__ == "__main__":
    main()


# ### ブロードキャスト

# NumPy は「行列にスカラを足す」，このような異様な計算をしても結果を返してくれます．以下の6行目では，行列にスカラを足しています．ここでは，最初に生成した4行4列の行列と同じサイズの，全要素が2からなる行列を自動で生成し，その行列と最初の4行4列の行列の和を計算しています．このような，対象となる行列のサイズに合せて，スカラから行列を生成することを「ブロードキャスト」と言います．この機能は非常に便利で様々な局面で使用することがあります．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(na + 2)

if __name__ == "__main__":
    main()


# ### 特殊な操作

# 以下のようにすると配列の順番を逆向きにして用いることができます．魔法ですね．最初の要素（0）から最後の要素（-1）まで逆向きに（-）連続して（1）いることを意味します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na[::-1])

if __name__ == "__main__":
    main()


# 以下のようにすると指定した条件に対する bool 配列を得ることができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na > 3)

if __name__ == "__main__":
    main()


# これを利用すると条件に合う要素のみに選択的にアクセスすることができます．以下では条件に合う要素を単に出力しているだけですが，例えば，条件に合う値のみを0に置き換えるとか，そのような操作ができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na[na > 3])

if __name__ == "__main__":
    main()


# 以下のように書けば，目的に合う値のインデックスを出力させることもできます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 3])
    print(np.where(na == 3))

if __name__ == "__main__":
    main()


# ## Matplotlib の利用方法

# Matplotlib とは Python で利用可能な描画ライブラリです．奇麗な描画をすることができます．問題に合わせて行いたい描画はその都度変わると思うので，基本的な利用方法だけ習得してそれ以外はその都度利用方法を調べると良いです．

# ### Matplotlib のインポート

# Matplotlib は以下のようにしてインポートします．読み込んだ Matplotlib には `plt` という略称を与えることが普通です．

# In[ ]:


#!/usr/bin/env python3
import matplotlib.pyplot as plt

def main():
    pass
 
if __name__ == "__main__":
    main()


# グーグルコラボラトリーや Jupyter Notebook で利用する場合には以下のようにします．こうすることでノート内にグラフを表示させることができます．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
 
def main():
    pass
 
if __name__ == "__main__":
    main()


# ### 基本的な描画

# 以下のようにすると描画できます．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
 
def main():
    x = np.linspace(0, 1, 100) * 2 * np.pi
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()
 
if __name__ == "__main__":
    main()


# 線の色は以下のようにすると変更できます．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
 
def main():
    x = np.linspace(0, 1, 100) * 2 * np.pi
    y = np.sin(x)
    plt.plot(x, y, color="forestgreen")
    plt.show()
 
if __name__ == "__main__":
    main()


# 同一の領域に別のプロットを追加したい場合は以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
 
def main():
    x = np.linspace(0, 1, 100) * 2 * np.pi
    y = np.sin(x)
    z = np.cos(x)
    plt.plot(x, y, color="forestgreen")
    plt.plot(x, z, color="orange")
    plt.show()
 
if __name__ == "__main__":
    main()


# 図に軸の名前，レジェンド，グリッド線を追加し，軸の最大値と最小値を設定するには以下のようにします．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
 
def main():
    x = np.linspace(0, 1, 100) * 2 * np.pi
    y = np.sin(x)
    z = np.cos(x)
    plt.plot(x, y, color="forestgreen", label="sin")
    plt.plot(x, z, color="orange", label="cos")
    plt.xlim(0, 2*np.pi)
    plt.ylim(-1, 1)
    plt.xlabel("Radian")
    plt.ylabel("Output")
    plt.grid()
    plt.legend()
    plt.show()
 
if __name__ == "__main__":
    main()


# ### 複数個の描画

# 複数の図を並べて描画することができます．`plt.subplot(abc)` という書き方をします．`a` と `b` と `c` にはそれぞれ整数が入ります．プロット領域を縦方向に `a` 個に分割し，横方向に `b` 個に分割したときにおいて，左上の分割領域から右下の分割領域に至るまでの「番号」である `c` を指定します．この `c` は分割領域の ID に相当するものですが，分割領域の上ほど小さく左ほど小さい値です．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
 
def main():
    plt.figure(figsize=(12,6), tight_layout=True)
    x = np.linspace(0, 1, 100) * 2 * np.pi
    y = np.sin(x)
    # 以下は縦に2個横に3個に分割した領域の1番目の図．
    plt.subplot(231)
    plt.plot(x, y, label="231")
    plt.legend()
    # 以下は縦に2個横に3個に分割した領域の2番目の図．
    plt.subplot(232)
    plt.plot(x, y, label="232")
    plt.legend()
    # 以下は縦に2個横に3個に分割した領域の3番目の図．
    plt.subplot(233)
    plt.plot(x, y, label="233")
    plt.legend()
    # 以下は縦に2個横に3個に分割した領域の4番目の図．
    plt.subplot(234)
    plt.plot(x, y, label="234")
    plt.legend()
    # 以下は縦に2個横に3個に分割した領域の5番目の図．
    plt.subplot(235)
    plt.plot(x, y, label="235")
    plt.legend()
    # 以下は縦に2個横に3個に分割した領域の6番目の図．
    plt.subplot(236)
    plt.plot(x, y, label="236")
    plt.legend()
    plt.show()
 
if __name__ == "__main__":
    main()


# ### その他の描画の種類

# 上で紹介した `plt.plot()` 以外には，棒グラフを描くための `plt.bar()`，散布図を描くための `plt.scatter()`，ヒストグラムを描くための `plt.hist()` は研究の分野で利用することがあるかもしれません．

# ```{hint}
# その都度調べましょう．
# ```

# ## オブジェクト指向の記法

# 後の章で紹介する TensorFlow を利用する際にオブジェクト指向の書き方ができたほうが良いので紹介します．

# ### クラスの生成

# Python にはクラスというプログラムのコードを短くすることで可読性をあげたり，チームでの大規模開発に有利になったりするシステムが備わっています．最初にとてもシンプルなクラスを作ってみます．これは後ほどもう少し多機能なものへと改変していきます．クラスの名前を `Dog` とします．最初の辺りで変数名のイニシャルに大文字は使わないことを紹介しましたが，それはクラスの宣言の際にクラス名のイニシャルを大文字にするためです．以下のように書きます．`main()` には `pass` とだけ書いていますが，これは「何も実行しない」というコマンドです．今はクラスを生成することが目的だから `main()` では何も行いません．クラスは以下のように生成します．
# ```
# class クラス名:
# ```
# 以下のクラスには `__init__` という記述があります．これはクラスからインスタンスを生成したときに（<font color="Crimson">クラスからはインスタンスと言われるクラスの実体が生成されます</font>）自動的に実行されるメソッドです（コンストラクタと呼びます）．このコンストラクタの引数は2個です．`self` と `name` と `age` の3個があるように見えますが，`self` は（インスタンスを生成するまでは未知の）インスタンス名をとりあえず `self` にします，という記述です．

# In[ ]:


#!/usr/bin/env python3

def main():
    pass

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# 多分，ここまでで意味がわからなくなったと思うので，まずはインスタンスを生成してみます．以下のように書きます．これによって `mydog` というインスタンスが生成されました．引数には，`Ken` と `6` を与えました．6 歳の Ken という名前の犬の情報です．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# 次に，この生成したインスタンス `mydog` の名前と年齢を出力してみます．以下のように書きます．インスタンスが内包する変数には `.` でアクセスします．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.name)
    print(mydog.age)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# しっかりと出力されました．クラスを使う利点は処理をきれいに整理できることです．もし，クラスを使わずにこの挙動を再現するには `mydog_name = Ken`，`mydog_age = 6` のような変数を生成しなければなりません．クラスを使うと `mydog = Dog("Ken", 6)` だけ，たったひとつの記述だけで複数個の情報を含む変数を生成できるのです．

# ```{hint}
# 以上のように複数のデータをまとめて管理できることがクラスを使うメリットです．クラスを使わない場合，個々の変数を個別に管理する必要が生じ，プログラムが煩雑なものになります．クラスを作る効果ってたったそれだけ？と思われるかもしれませんが，そうです，たったそれだけです．ただ，この後にも説明がある通りクラスは変数だけでなく関数も内部に有することができます．処理が複雑になればなるほど有難くなります．
# ```

# また，上で `self` とはインスタンスの名前そのものであると言及しましたが，それは以下を比較していただくと解ります．出力の際には以下のように書きました．
# ```
# mydog.name
# ```
# クラス内での定義の際には以下のように書きました．
# ```
# self.name
# ```
# インスタンス名（`mydog`）と `self` が同じように使われています．

# ### メソッド

# クラスにメソッドを追加します．以下では `sit()` というメソッドを定義しました．そしてそれを `main()` で呼び出しています．この場合も `.` にてメソッドを呼び出します．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    mydog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{}, which is ({}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# たったひとつの `mydog` という変数（インスタンス）が，複数個の変数を持ち，メソッドも有します．これはかなり整理整頓には良いシステムです．これがクラスを定義しインスタンスを生成する利点です．これは，以下のようにインスタンスを何個も生成するような処理が必要になる場合にさらに有用です．よりシンプルなコードを保てます．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = Dog("Peko", 4)
    mydog.sit()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# ### デフォルト値の設定

# インスタンス生成の際にデフォルトの値を設定することができます．以下の13行目のようにコンストラクタに直接定義します．そのような値も，5行目のような記述でアクセスすることができます．また，6行目のような記述で値を変更することも可能です．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.hometown)
    mydog.hometown = "Yokohama"
    print(mydog.hometown)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
        self.hometown = "Sendai"
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# ### クラス内変数の改変

# また，以下の16および17行目のようなクラス内の変数の値を改変するような関数を定義することも可能です．これを実行すると出力のように，`mydog.age` が `6` から `8` に変化します．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.age)
    mydog.year_pass(2)
    print(mydog.age)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
        self.hometown = "Sendai"
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))
    def year_pass(self, years):
        self.age = self.age + years

if __name__ == "__main__":
    main()


# ### クラス変数

# `インスタンス名.変数名` もしくは `self.変数名` とすると，インスタンスが保持する変数にアクセスすることができましたが，そうではなく，クラスから生成されたすべてのインスタンスで値が共有されるような変数を作ることができます．そのような変数をクラス変数と呼びます．クラス変数を作成するには，単にクラスの定義部で変数に値を代入します．クラス変数にアクセスするには， `クラス名.変数名` とします．別のやり方として，インスタンス変数に同じ名前の変数がない場合には，`インスタンス名.変数名` と書いてもクラス変数にアクセスすることができます（ただし同名のインスタンス変数がある場合には，そちらにアクセスしますので注意が必要です）．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = Dog("Peko", 4)
    print(Dog.type) # クラス名.変数名で参照できる
    print(yourdog.type)

class Dog:
    type = "dog" # クラス変数．
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

if __name__ == "__main__":
    main()


# ### プライベート変数

# オブジェクト指向プログラミングでは，インスタンスへの外部からのアクセスを適切に制限することで，モジュールごとの権限と役割分担を明確にすることができます．例えば，あるモジュールがすべてのデータにアクセス可能だとすると，本来そのモジュールが行うべきではない処理まで，そのモジュールに実装されてしまう恐れがあり，役割分担が崩壊する可能性があります．データにアクセス制限がかかっている場合，そのデータを使う処理は，そのデータにアクセス可能なモジュールに依頼するしかなくなり，適切な役割分担が維持されます．
# 
# クラスにおいて，外部からのアクセスが禁止されている変数をプライベート変数と呼びます．また，外部からのアクセスが禁止されているメソッドをプライベートメソッドと呼びます．
# 
# Pythonでは，プライベート変数やプライベートメソッドを作ることはできません．つまり，どの変数やメソッドも外部からアクセス可能です．
# ただし，変数やメソッドの名前を変えることで，プライベートとして扱われるべき変数やメソッドを区別するという慣習があります．実装を行う人がプライベートとして扱われるべき変数等への外部からのアクセスを行わないようにすることで，本来の目的を達成することができます．Pythonの慣習では，プライベートとして扱われるべき変数やメソッドの先頭文字を以下のようにアンダースコア（_）にします．

# In[ ]:


#!/usr/bin/env python3

def main():
    score_math = PrivateScore(55)
    print(score_math.is_passed())

class PrivateScore:
    def __init__(self, score):
        self._score = score # プライベート変数
    def is_passed(self):
        if self._score >= 60:
            return True
        else:
            return False

if __name__ == "__main__":
    main()


# ### 継承

# あるクラスのインスタンスを複数生成する際，特定のインスタンスのみ処理を変更したり，処理を追加したりする必要が生じることがあります．その場合，そのインスタンス用に新しくクラスを作ることもできますが，元のクラスと大部分が同じである場合，同じコードが大量に複製されることになり，コードが無駄に長くなります．同じ処理を行うコードが複数箇所に存在する場合，不具合を修正する場合にもすべての箇所を修正しないといけなくなるため，正しく修正することが難しくなります．このような場合の別の方法として，元のクラスからの修正部分のみを修正するという方法があります．このような方法を継承と呼び，修正のベースとなる元のクラスを基底クラス（もしくは親クラス），新しく作られるクラスを派生クラス（もしくは子クラス）と呼びます．

# ```{note}
# 基底クラスに複数個のクラスを指定することができます．そのような継承を多重継承と呼びます．
# ```

# 継承を行う際は，派生クラスを定義するときにそのクラス名の次に括弧を書き，その括弧内に基底クラスの名前を書きます．そして，クラス定義の中には，新規に追加するメソッドと基底クラスから変更を行うメソッドについてのみ定義を追加します．以下のように行います．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = TrainedDog("Peko", 4)
    mydog.sit()
    yourdog.trick()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog): # クラス名の次に「(基底クラス名)」を追加する
    def trick(self):
        for i in range(self.age):
            print("bow")

if __name__ == "__main__":
    main()


# ```{note}
# 派生クラスである `TrainedDog` の定義には `sit()` は含まれていません．しかし，基底クラスである `Dog` を継承しているので `TrainedDog` から生成したインスタンスでは `sit()` を利用できています．
# ```

# ### オーバーライド

# これまでに，基底クラスに新たなメソッドを追加する方法である継承を紹介しました．これに対して，派生クラスで基底クラスのメソッドを書き換えることをオーバーライドと言います．オーバーライドは，派生クラスの定義をする際に，基底クラスで定義されているメソッドを再定義することで実現できます．以下のように行います．

# In[ ]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = TrainedDog("Peko", 4)
    mydog.sit()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog):
    def sit(self):
        print("{0} is now sitting and giving its paw.".format(self.name))

if __name__ == "__main__":
    main()


# 派生クラスから生成したインスタンスの `sit()` の挙動が基底クラスのものから変化しているのがわかります．このとき，基底クラスの `sit()` は変化していません．

# ```{note}
# 派生クラスで，派生クラスにおいて利用するメソッドを書き換えるのであって，派生クラスで書き換えたメソッドが基底クラスのメソッドを書き換える（明らかに危険な行為）わけではありません．)
# ```

# 次に，派生クラスにおいてインスタンス変数を増やします．これを実現するためには，コンストラクタ `__init__()` をオーバーライドする必要があります．このとき，既に基底クラスで定義したインスタンス変数をそのまま利用するためにコンストラクタを呼び出します．基底クラスは `super()` を使って呼び出すことができます．以下のように行います．

# In[ ]:


#!/usr/bin/env python3

def main():
    yourdog = TrainedDog("Peko", 4, "Ben")
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog):
    def __init__(self, name, age, trainer):
        super().__init__(name, age) # この super() は基底クラスを意味する
        self.trainer = trainer # 新たなるインスタンス変数
    
    def sit(self):
        print("{0} is now sitting and giving its paw to {1}.".format(self.name, self.trainer))

if __name__ == "__main__":
    main()


# ## ファイルの取り扱い
# 

# 人工知能に学習させるためのデータが存在しているファイルを読み込んだり，構築した人工知能に予測させたいデータが載っているファイルを読み込ませたりしたいので，Python でファイルを取り扱う方法を紹介します．

# ### ファイルの入出力について

# 多くの場合，開発したプログラムで扱いたいものはデータであると思います．データの整理とかデータの解析とかです．そのようなデータはファイルという形式で存在しており，プログラムはそれを読み込む必要があります．ここではプログラムを用いてファイルをどのように操作するのかを学びます．

# ### 繰り返しによる読み込み

# ここでは，以下のような内容の情報からなるファイルを使います．ファイルは全部で3行からなり，各行には文字列が存在します．これのファイル名は `ff.txt` です．
# ```
# Cloud Strife
# Squall Leonhart
# Zidane Tribal
# ```

# このようなファイルを使って練習するために以下の Linux コマンドでファイルをカレントディレクトリに生成します．

# ```{note}
# これはファイルの読み込みをするための準備であって，Python の学習ではありません．
# ```

# In[ ]:


get_ipython().system(' echo -e "Cloud Strife\\nSquall Leonhart\\nZidane Tribal" > ff.txt')


# ```{note}
# Linux システムでは現在ユーザーがいるディレクトリをカレントディレクトリと呼ぶのでした．
# ```

# ファイルが生成されているかどうかは以下のコマンドで確認できます．

# In[ ]:


get_ipython().system(' ls')


# ```{note}
# これはカレントディレクトリに存在しているファイルやフォルダを確認するための Linux コマンドでした．
# ```

# 以上で用いるファイルの準備が完了したので，ファイルを読み込んでみます．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3

def main():
    fin = open("ff.txt", "r")
    for line in fin:
        print(line)
    fin.close()

if __name__ == "__main__":
    main()


# 上のコードにおいて，`fin = open("ff.txt", "r")` は `ff.txt` を読み込みモード（`r`）でファイルを開くための記述です．ファイルを読み込むために生成された変数 `fin` は以下のような形式で用いると，すべての行を繰り返し `line` という変数に代入し続けます．
# ```
# for line in fin:
#     処理
# ```
# 最後に，`fin.close()` にてファイルを閉じる作業をします．

# 次に，読み込んだ各行に存在している改行コードを削除して，削除後の行を出力してみます．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3

def main():
    fin = open("ff.txt", "r")
    for line in fin:
        line = line.rstrip()
        print(line)
    fin.close()

if __name__ == "__main__":
    main()


# 各行をリストに格納し，その後処理をしたい場合があるかもしれません．そのようなときには以下のように書きます．

# In[ ]:


#!/usr/bin/env python3

def main():
    liline = []
    fin = open("ff.txt", "r")
    for line in fin:
        line = line.rstrip()
        liline.append(line)
    fin.close()
    print(liline)

if __name__ == "__main__":
    main()


# 以下のように `with` を使えばファイルを閉じる作業を省くことができます．

# In[ ]:


#!/usr/bin/env python3

def main():
    with open("ff.txt", "r") as fin:
        for line in fin:
            print(line)

if __name__ == "__main__":
    main()


# ### ファイルの一括読み込み

# 以上ではファイルを 1 行ずつ読み込みました．普通，ファイルはそのように読み込む方が良いと思われますが，小さいファイルであれば以下のように一括で読み込むことができます．

# In[ ]:


#!/usr/bin/env python3

def main():
    fin = open("ff.txt", "r")
    contents = fin.read()
    fin.close()
    print(contents)

if __name__ == "__main__":
    main()


# ### ファイルへ出力する意義

# ファイルにプログラム中で生成した情報を書き込めば，プログラムを終了させた後にもその情報を見返して確認することができます．また，途中の状態をファイルに記憶させておけば，ファイルの内容を再度メモリ上に呼び起こしプログラムの続きを計算させることができます．

# ### ファイルへの新規出力

# ファイルへ何かを出力する際には，`open()` のオプションに書き込みモード（`w`）を指定します．以下のように書きます．この場合，`programming.txt` というファイルがカレントディレクトリに生成されます．

# In[ ]:


#!/usr/bin/env python3

def main():
    fout = open("programming.txt", "w")
    print("I like programming.", file=fout)
    fout.close()

if __name__ == "__main__":
    main()


# 当然，複数行を出力することも可能です．以下は，リストに含まれる文をファイルに出力する例です．

# In[ ]:


#!/usr/bin/env python3

def main():
    liline = ["I like programming.", "I love creating new codes."]
    fout = open("programming.txt", "w")
    for line in liline:
        print(line, file=fout)
    fout.close()

if __name__ == "__main__":
    main()


# ### ファイルへの追加出力

# ファイルに上書きする場合には，上書きモード `a` を指定します．上で生成したファイルに追加書きするには以下のようにします．

# In[ ]:


#!/usr/bin/env python3

def main():
    fout = open("programming.txt", "a")
    print("I also love to learn machine learning.", file=fout)
    fout.close()

if __name__ == "__main__":
    main()


# ```{note}
# 終わりです．
# ```
