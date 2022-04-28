#!/usr/bin/env python
# coding: utf-8

# # 教師なし学習法（編集中）

# ## 教師なし学習法の種類

# 教師なし学習法には様々な種類があります．https://scikit-learn.org/stable/unsupervised_learning.html にはそれらがまとめられています．リンク先のページには以下に示すものが列挙されています．
# 
# *   混合ガウスモデル
# *   多様体学習
# *   クラスタリング
# *   バイクラスタリング
# *   行列分解
# *   共分散推定
# *   外れ値検知
# *   密度推定
# *   制約付きボルツマンマシン
# 
# 中でも代表的なものには，階層的クラスタリング法，非階層的クラスタリング法（特に K-means），主成分分析法，t-SNE，カーネル密度推定法，自己組織化マップ，敵対的生成ネットワークがあります．教師なし学習法は主に与えられたデータの性質を理解するために利用されます．与えられたデータの中で類似しているインスタンスを集めるとか，与えられたデータの関係性を人間が理解しやすい方法（次元削減）で可視化するとかです．また，敵対的生成ネットワークはこれらとは異なり特殊な機械学習アルゴリズムで，新たなデータを生成するために利用されます．このコンテンツでは教師なし学習法の中でも scikit-learn を利用して簡単に実装できる最も代表的な手法の使い方を紹介します．

# ```{note}
# 敵対的生成ネットワークは別に紹介します．
# ```

# 次の節では scikit-learn を利用して，階層的クラスタリング法，K-means 法（非階層的クラスタリング法の代表的な手法），主成分分析法，カーネル密度推定法を実装します．

# ## 階層的クラスタリング法

# 与えられたデータをクラスター化する階層的クラスタリング法の利用方法を紹介します．階層的クラスタリング法でクラスター化した各クラスターは階層構造を有します．

# ### 基本的な事柄

# 

# ## K-means 法

# 階層的クラスタリングと異なり K-means 法は非階層的にデータをクラスタ化する手法です．

# ### 基本的な事柄

# K-means 法はあらかじめ決めた $k$ 個のクラスタにデータを分割する方法です．K-means 法は以下のような手順で計算します．
# 
# 1.   ランダムに $k$ 個のクラスタの中心（$\mu_k$）を決定します．
# 2.   各インスタンスからそれぞれのクラスタの中心との距離を計算します．
# 3.   各インスタンスを最も近いクラスタ中心のクラスタ（$C_k$）に所属させます．$C_k$ に所属するインスタンスの数を $n_k$ とします．
# 4.   各クラスタの重心を計算して，その重心を新たなクラスタの中心とします．
# 5.   クラスタの中心が変化しなくなるまで上の2, 3, 4の操作を繰り返します．
# 
# よって，$\mu_k$ は各インスタンスベクトルを $x_i$ としたとき，以下のように計算します．
# 
# $
# \displaystyle \mu_k=\frac{1}{n_k}\sum_{i=1}^{n_k}x_i
# $
# 
# 各クラスタ内に所属するインスタンスとクラスタ中心との二乗距離の合計 $I_k$ は以下のように計算できますが，これをクラスタ内平方和とか慣性とかと呼びます．
# 
# $
# \displaystyle I_k=\sum_{i=1}^{n_k}\|x_i-\mu_k\|_2^2
# $
# 
# K-means ではこの値を最初化するようにクラスタへのインスタンスの割り当てを行います．つまり，K-means では以下の $E$ を最小化します．
# 
# $\displaystyle E=\sum_{i=1}^kI_k$

# ### クラスタリングの実行

# ここでは K-means 法を実行しますが，そのためのデータセットを生成します．scikit-learn にはクラスタリング法用に擬似的なデータセットを生成するためのユーティリティが備わっています．以下のようにすることで，3個のクラスタに分けられるべき150個のインスタンスを生成することができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
plt.style.use("ggplot")
np.random.seed(1000)
 
def main():
    x, t = make_blobs(n_samples=150, centers=3)
    plt.scatter(x[:,0], x[:,1])

if __name__ == "__main__":
    main()


# ```{note}
# 目で見ると3個のクラスタに分かれているように見えますね．
# ```

# このデータを最初に2個のクラスタに分けます．以下のようにします．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use("ggplot")
np.random.seed(1000)
 
def main():
    x, t = make_blobs(n_samples=150, centers=3)

    kmeans = KMeans(n_clusters=2).fit(x)
    cluster = kmeans.labels_

    colors = ["navy", "turquoise"]
    plt.figure()
    for color, i in zip(colors, [0, 1]):
        plt.scatter(x[cluster==i, 0], x[cluster==i, 1], color=color, alpha=0.8, lw=0, label=str(i))
    plt.legend()

if __name__ == "__main__":
    main()


# ```{note}
# 微妙な感じがします．
# ```

# 次に3個のクラスタに分割します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use("ggplot")
np.random.seed(1000)
 
def main():
    x, t = make_blobs(n_samples=150, centers=3)

    kmeans = KMeans(n_clusters=3).fit(x)
    cluster = kmeans.labels_

    colors = ["navy", "turquoise", "darkorange"]
    plt.figure()
    for color, i in zip(colors, [0, 1, 2]):
        plt.scatter(x[cluster==i, 0], x[cluster==i, 1], color=color, alpha=0.8, lw=0, label=str(i))
    plt.legend()

if __name__ == "__main__":
    main()


# ```{note}
# これだって感じがしますね．
# ```

# ```{note}
# 乱数の種は ` make_blobs(n_samples=150, centers=3, random_state=1000) ` のように指定することもできますが，ここで発生する乱数は NumPy の乱数発生機能に依存しているので NumPy を読み込んで ` np.random.seed() ` で行っても良いのです．
# ```

# ### クラスタ数の決定方法

# 上の例だと正解のクラスタ数を知っているので $k$ を上手に設定することができました．実際のデータをクラスタリングしたいときには何個に分割すれば良いか分からない場合が多いと思います．そのときに，$k$ の値を決定するための指標があります．シルエットスコアと言います．

# ```{note}
# シルエットスコアはクラスタに所属するインスタンス間の平均距離（凝集度と言う）をそのクラスタから最も近いクラスタに存在しているインスタンスとの平均距離（乖離度と言う）から引いた値を凝集度または乖離度の内で大きな値で割った値を全 $k$ 個のクラスタについて計算して平均をとった値です．
# ```

# シルエットスコアは以下のように計算します．上と同じデータに対して，$k$ の値を変えて K-means 法を実行した場合のシルエットスコアを出力します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.style.use("ggplot")
np.random.seed(1000)
 
def main():
    x, t = make_blobs(n_samples=150, centers=3)

    for k in range(2,8):
        kmeans = KMeans(n_clusters=k).fit(x)
        cluster = kmeans.labels_
        print(k, ": ", silhouette_score(x, cluster), sep="")

if __name__ == "__main__":
    main()


# シルエットスコアはクラスタの個数が3個のとき最も良い値を示しており，やはり，クラスタ数は3個が良さそうであることが確認できます．

# ```{attention}
# クラスタ数を決定付ける強力な根拠ではありません．こういう指標があるという程度に使ってください．
# ```

# ```{hint}
# 他にはクラスタ数に対するクラスタ内平方和の変化を観察するエルボー法という方法があります．
# ```

# ## 主成分分析法

# 主成分分析法の進め方を紹介します．

# ### 基本的な事柄

# 与えられたデータに対して主成分分析を行う理由は，人が目で見て理解するには複雑なデータを人が理解しやすい形式に整えたいためです．4個以上の要素からなるベクトル形式のデータを2個または3個（または稀に1個）の要素からなるベクトル形式のデータに変換し2次元または3次元（または1次元）平面上にインスタンスをプロットし直すことでデータの関連性を把握することができます．

# ```{note}
# この変換を次元削減と言います．
# ```

# 主成分分析で行っていることはちょうど以下のような作業です．左に示されている2次元平面上にプロットされた点を右に示されているように1次元平面上にプロットしています．
# 
# <img src="../image/pca_01.svg" width="70%" />

# ただし，主成分分析で行いたいことは以下のような次元削減ではありません．以下のようにすると変換後の軸上のインスタンスが互いに重なっており，かなりの情報が失われているように思えます．
# 
# <img src="../image/pca_02.svg" width="70%" />

# ```{note}
# 主成分分析では元の情報をできるだけ維持したままデータの変換をしようとします．この例において新たに生成される軸はこのデータを説明するための情報量が最も大きい方向に設定されます．情報量が最も大きい方向とはデータが最も散らばっている（分散が大きい）方向です．
# ```

# ### 主成分分析法の限界

# 主成分分析は変数を元の変数の線形結合で表される新たな変数へと変換させる方法です．元々何らかの非線形平面で関係を持っていたデータを別の平面へと変換した場合において，元々の非線形な関係性が維持されているとは限りません．非線形な関係を含めて次元削減をしたい場合は他の方法を利用する方法があります．主成分分析法を非線形で行う方法には非線形カーネル関数を利用したカーネル主成分分析法があります．scikit-learn でも利用することが可能です．

# ### 次元の削減

# アヤメのデータセットに対して次元の削減を行います．以下のようにこのデータセットにおける各インスタンスは4個の要素からなるベクトル形式のデータです．よってこれを4次元平面上にプロットしたとしてもその関係性を人は理解できません．

# ```{note}
# そもそも4次元平面なんて描画できませんね．
# ```

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
 
def main():
    diris = load_iris()
    print(diris.data[0])

if __name__ == "__main__":
    main()


# 主成分分析は以下のように行います．主成分分析で得られた全インスタンス（150個）の値を出力させます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
 
def main():
    diris = load_iris()
    x = diris.data
    t = diris.target
    target_names = diris.target_names
    pca = PCA(n_components=2) # n_componentsで縮約後の次元数を指定します．
    xt = pca.fit(x).transform(x)
    print(xt)

if __name__ == "__main__":
    main()


# 次元縮約後の各インスタンスを以下のコードで散布図上にプロットします．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("ggplot")
 
def main():
    diris = load_iris()
    x = diris.data
    t = diris.target
    target_names = diris.target_names
    pca = PCA(n_components=2)
    xt = pca.fit(x).transform(x)
    
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(xt[t==i, 0], xt[t==i, 1], color=color, alpha=0.8, lw=0, label=target_name)
    plt.legend()

if __name__ == "__main__":
    main()


# 各種類のアヤメがそれぞれ集まっていることがわかります．2次元平面上にプロットすることで各インスタンスの関係性を把握することができました．この主成分平面には横軸と縦軸がありますが，これらの軸が何を意味しているのかは解析者がデータの分布の様子を観察する等して決定しなければなりません．

# ```{note}
# 軸の意味の解釈のヒントは主成分負荷量を散布図上にプロットすることである程度は得られます．主成分負荷量とは最終的に得られた固有ベクトル（線形結合の係数）にそれに対応する固有値の正の平方根を掛けた値のことです．
# ```

# ```{note}
# この主成分平面上の任意の点をサンプリングして主成分分析の逆操作をすると新たなデータを生成することも可能です．
# ```

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("ggplot")
 
def main():
    diris = load_iris()
    x = diris.data
    t = diris.target
    target_names = diris.target_names
    pca = PCA(n_components=2)
    xt = pca.fit(x).transform(x)
    
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(xt[t==i, 0], xt[t==i, 1], color=color, alpha=0.8, lw=0, label=target_name)
    plt.legend()

if __name__ == "__main__":
    main()


# ### 次元削減データの説明力

# これまでの計算で4つの要素からなるベクトルデータを2つの要素からなるベクトルデータへと変換しました．そのインスタンスを特徴付ける4つの要素を半分にしたのですから元々インスタンスが持っていた情報は少なくなっているはずです．この主成分分析の操作でどれくらいの情報が失われたのか，どれくらいの情報が維持されているのかは以下のコードで確認できます．元々の情報を1としたときに各軸が持つ説明力の割合を出力することができます．また，それらの値を合計することで元々の情報を2つの軸だけでどれくらい説明できるかを計算できます．

# In[ ]:


#!/usr/bin/env python3
import sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
 
def main():
    diris = load_iris()
    x = diris.data
    t = diris.target
    target_names = diris.target_names
    pca = PCA(n_components=2)
    xt = pca.fit(x).transform(x)
    print(pca.explained_variance_ratio_) # 各軸が持つ説明力の割合．
    print(sum(pca.explained_variance_ratio_)) # 2軸で説明できる割合．

if __name__ == "__main__":
    main()


# 第一主成分のみで全体の大体92%の説明力を持ち，第二主成分で大体5%の説明力を持つようです．ふたつの軸によって元の約98%の説明ができているようです．

# ```{note}
# この説明力の比率は寄与率と言います．それらを（解析者が必要と感じる次元数まで）足したものを累積寄与率と言います．
# ```

# ## カーネル密度推定法

# カーネル密度推定法の利用方法を紹介します．簡単な利用方法に加えて生成モデルとして利用する方法を紹介します．

# ### 基本的な事柄

# カーネル密度推定法は与えられたデータの分布を推定する方法です．与えられたデータ中の疎なインスタンスを入力としてそのデータが従うと思われる分布を推定する方法です．$x_1, x_2, \dots, x_n$ を何らかの確率分布から得られたサンプルとします．このときにカーネル密度推定量 $f$ は以下のように計算されます．
# 
# $
# \displaystyle f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)
# $
# 
# このとき，$K$ はカーネル関数と呼ばれる確率分布を近似するための関数で，$h$ はバンド幅と呼ばれるハイパーパラメータです．カーネル関数として利用される関数には様々なものがありますが，以下の標準正規分布（平均値が0で分散が1である正規分布）の確率密度関数を利用することが多いです．
# 
# $
# \displaystyle K(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}
# $

# ```{note}
# このカーネルのことはガウシアンカーネルとも言いますね．
# ```

# ### 発展的な利用方法

# 昨今の深層学習ブームで専ら利用されている深層ニューラルネットワークは通常たくさんのハイパーパラメータを持ちます．良いハイパーパラメータを同定することは良い人工知能を構築するために重要なことであるため，その探索方法の開発が活発です．ブルートフォース（しらみつぶし）な探索，ランダムな探索，手動による探索，進化戦略法を利用した探索，ベイズ探索等の様々な方法が利用されていますが，その中でも最も有効なもののひとつに代理モデルを利用した逐次最適化法（sequential model-based optimization（SMBO））と呼ばれるベイズ最適化法の範疇に入る方法があります．ハイパーパラメータが従うであろう分布を推定して（この推定した分布を代理モデルと言います），その分布から予想したハイパーパラメータを利用して構築した人工知能の評価を行い，さらにその評価結果から分布の推定を繰り返す，というようなハイパーパラメータの従う分布の推定と人工知能の評価を交互に繰り返すことで最適なハイパーパラメータを持つ人工知能を同定しようとする方法です．この SMBO を行う際の代理モデルの構築にカーネル密度推定法が利用されることがあります．そして，カーネル密度推定法を利用した SMBO は従来の代理モデルの推定法（例えば，ガウス過程回帰法）より良い性能を示すことがあります．

# ```{note}
# SMBO の領域ではカーネル密度推定量はパルツェン推定量と呼ばれています．
# ```

# ```{note}
# ハイパーパラメータの最適化は深層学習の分野で最も重要なトピックのひとつなので紹介してみました．
# ```

# ### 元の確率分布の推定

# ここでは，母数の異なるふたつの正規分布からいくつかのインスタンスをサンプリングして，そのサンプリングしたデータから元の正規分布ふたつからなる二峰性の確率分布を再現できるかということを試します．

# ```{hint}
# 正規分布の母数（パラメータ）は平均値と分散ですね．母数の値が決まればそれに対応する正規分布の形状は一意に決まるのですね．
# ```

# 以下のコードで $N(-3, 1.5)$ と $N(-3, 2)$ の正規分布を描画します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
 
def main():
    x = np.linspace(-8, 8, 100)
    y = (norm.pdf(x, loc=2, scale=1.5) + norm.pdf(x, loc=-3, scale=2)) / 2
    plt.plot(x, y)

if __name__ == "__main__":
    main()


# 次に，以下のコードで $N(-3, 1.5)$ に従う50個のインスタンスと $N(-3, 2)$ に従う50個のインスタンスをサンプリングします．また，そのヒストグラムを描きます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
np.random.seed(0)
 
def main():
    x1 = norm.rvs(loc=2, scale=1.5, size=50)
    plt.hist(x1)
    x2 = norm.rvs(loc=-3, scale=2, size=50)
    plt.hist(x2)

if __name__ == "__main__":
    main()


# ```{attention}
# 計算機実験をする際は乱数の種は固定しなきゃならないのでしたね．
# ```

# カーネル密度推定は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
np.random.seed(0)
 
def main():
    x1 = norm.rvs(loc=2, scale=1.5, size=1000)
    x2 = norm.rvs(loc=-3, scale=2, size=1000)
    x = np.concatenate([x1, x2]) # x1とx2を連結します
    x = x.reshape(-1, 1) # このような入力形式にしないと受け付けてくれないからこうしました．
    kde = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(x) # ハンド幅は適当に選んでみました．
    p = np.linspace(-8, 8, 100)[:, np.newaxis] # プロット用の値を生成しています．
    l = kde.score_samples(p) # これで予測値を計算します．
    plt.plot(p, np.exp(l)) # 予測値は対数値で出力されているのでそれをnp.exp()を利用してプロットします．
    y = (norm.pdf(p, loc=2, scale=1.5) + norm.pdf(p, loc=-3, scale=2)) / 2 # 元の分布です．
    plt.plot(p, y)

if __name__ == "__main__":
    main()


# ```{note}
# サンプリングしたインスタンスを使って予測した分布の形状と元の分布の形状が類似している様子がわかります．
# ```

# ### 生成モデルとしての利用

# カーネル密度推定法は何もないところからデータを生成する生成モデルとして利用することができます．何らかのデータを入力にしてそのデータが出力される確率分布をカーネル密度推定法で推定します．次に，その確率密度分布に基づいてデータを生成する，といった手順です．

# ```{hint}
# 与えられたデータが出力された確率分布を推定できたのなら，その分布から新たなデータは当然出力することができるよねという仕組みです．
# ```

# ここでは機械学習界隈で最も有名なデータセットである MNIST（Mixed National Institute of Standards and Technology database）を解析対象に用います．「エムニスト」と発音します．MNIST は縦横28ピクセル，合計784ピクセルよりなる画像データです．画像には手書きの一桁の数字（0から9）が含まれています．公式ウェブサイトでは，学習データセット6万個とテストデータセット1万個，全部で7万個の画像からなるデータセットが無償で提供されています．そのデータセットを以下のようにダウンロードして最初のデータを可視化します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    plt.imshow(x[0], cmap="gray")
    plt.axis("off")

if __name__ == "__main__":
	main()


# ```{note}
# これは5ですね．
# ```

# 以下のようにすることで新たな画像データを生成することができます．学習済みの生成器に対して利用する `.sample()` というメソッドで新たなデータを生成することができます．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import sklearn
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(0)

def main():
    (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x = x.reshape(-1, 28*28) # 縦横どちらも28ピクセルの画像を784の要素からなるベクトルに変換します．
    kde = KernelDensity().fit(x)
    g = kde.sample(4) # 学習済みの生成器で4個の画像を生成させてみます．
    plt.figure(figsize=(10,10))
    for i in range(len(g)): # 生成データの可視化です．
        s = g[i].reshape(28, 28)
        plt.subplot(1, 4, i+1)
        plt.imshow(s, cmap="gray")
        plt.axis("off")

if __name__ == "__main__":
    main()


# 左から，「2」，「1」，「1」，「6」という画像が生成されているように見えます．

# ```{note}
# 終わりです．
# ```
