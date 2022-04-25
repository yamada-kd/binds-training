#!/usr/bin/env python
# coding: utf-8

# ## 数値計算と描画ライブラリ

# ### NumPy のインポート

# NumPy とは Python で利用可能な数値計算のライブラリです．さまざまな計算をコマンド一発で行うことができます．NumPy は以下のようにしてインポートします．読み込んだ NumPy には `np` という略称を与えることが普通です．

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

# #!/usr/bin/env python3
# import numpy as np
#  
# def main():
#     na = np.array([1, 2, 3, 4, 5, 6, 3])
#     print(np.where(na == 3))
# 
# if __name__ == "__main__":
#     main()

# ```{note}
# 終わりです．
# ```
