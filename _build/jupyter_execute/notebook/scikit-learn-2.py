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

# ほげ

# ## K-means

# ほげ

# ## 主成分分析法

# ほげ

# ## カーネル密度推定法

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

# ### 画像の生成

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


# ```{note}
# 終わりです．
# ```
