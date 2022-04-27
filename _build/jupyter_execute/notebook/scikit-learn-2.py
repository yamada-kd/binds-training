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

# ## scikit-learn による実装

# この節では scikit-learn を利用して，階層的クラスタリング法，K-means 法（非階層的クラスタリング法の代表的な手法），主成分分析法，カーネル密度推定法を実装します．

# ### 階層的クラスタリング法

# ほげ

# ### K-means

# ほげ

# ### 主成分分析法

# ほげ

# ### カーネル密度推定法

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
