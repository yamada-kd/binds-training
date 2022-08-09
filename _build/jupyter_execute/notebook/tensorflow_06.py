#!/usr/bin/env python
# coding: utf-8

# # アテンションネットワーク

# ## 基本的な事柄

# この節ではアテンションがどのようなものなのか，また，アテンションの計算方法について紹介します．

# ### アテンションとは

# アテンション機構とは単にアテンションとも呼ばれる，ニューラルネットワークが入力データのどの部分に注目するかを明示することで予測の性能向上をさせるため技術です．例えば，何らかの風景が写っている画像を人工知能に処理させ，その画像に関する何らかの説明を自然言語によってさせるタスクを考えます．そのような場合において，人工知能が空の色に関する話題に触れる場合には，空が写っている部分の色を重点的に処理するかもしれないし，海の大きさを話題にする場合は海が写っている部分の面積を重点的に処理するかもしれません．少なくとも人は何かの意思決定をする際に，意識を対象に集中させることで複雑な情報から自身が処理すべき情報のみを抽出する能力を有しています．人工知能は人とは全く違う情報処理をしているかもしれないので，このような直接的な対応があるかどうかは定かではないですが，人の情報処理方法にヒントを得て，このような注意を払うべき部分を人工知能に教えるための技術がアテンション機構です．アテンションを有するニューラルネットワークの性能は凄まじく，アテンションネットワークの登場は CNN や RNN 等のアテンション以前に用いられていたニューラルネットワークを過去のものとしました．

# ### アテンションの計算方法

# アテンション機構の処理は非常に単純で簡単な式によって定義されます．アテンションの基本的な計算は以下に示す通りです．アテンション機構とは，$[\boldsymbol{h}_1,\boldsymbol{h}_2,\dots,\boldsymbol{h}_I]$ のようなデータに対して以下のような計算をして，$\boldsymbol{c}$ を得るものです．
# 
# $
# \displaystyle \boldsymbol{c}=\sum_{i=1}^I{\phi}_i\boldsymbol{h}_i
# $
# 
# このとき，スカラである $\phi_i$ は $[0,1]$ の範囲にあり，また，以下の式を満たす値です．
# 
# $
# \displaystyle \sum_{i=1}^I{\phi}_i=1
# $
# 
# すなわち，${\phi}_i$ は何らかの入力に対して得られるソフトマックス関数の出力の $\boldsymbol{\phi}$ の $i$ 番目の要素です．よって，この $\boldsymbol{c}$ は単に $[\boldsymbol{h}_1,\boldsymbol{h}_2,\dots,\boldsymbol{h}_I]$ の加重平均です．人工知能にこのベクトル $\boldsymbol{c}$ を入力値として処理させる場合，$\boldsymbol{h}_i$ に対する ${\phi}_i$ の値を大きな値とすることで $\boldsymbol{h}_i$ が注目すべき要素として認識されるという仕組みです．

# ```{note}
# ソフトマックス関数とはベクトルを入力として，各要素が 0 から 1 の範囲内にあり，各要素の総和が 1 となるベクトルを出力する関数のひとつでした．出力データを確率として解釈できるようになるというものです．入力ベクトルと出力ベクトルの長さは等しいです．
# ```

# ### セルフアテンション

# あるベクトル（ソースベクトル）の各要素が別のベクトル（ターゲットベクトル）のどの要素に関連性が強いかということを明らかにしたい際にはソース・ターゲットアテンションというアテンションを計算します．これに対してこの章ではセルフアテンションを主に扱うのでこれを紹介します．セルフアテンションとは配列データの各トークン間の関連性を計算するためのものです．
# 
# セルフアテンションを視覚的に紹介します．ここでは，$x_1$，$x_2$，$x_3$ という 3 個のトークン（要素）からなるベクトルを入力として $y_1$，$y_2$，$y_3$ という 3 個のトークンからなるベクトルを得ようとします．各々のトークンもベクトルであるとします．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/selfAttention.svg?raw=1" width="100%" />
# 
# アテンションの計算では最初に，$x_1$ に 3 個の別々の重みパラメータを掛け，キー，バリュー，クエリという値を得ます．図ではぞれぞれ，$k_1$，$v_1$，$q_1$ と表されています．同様の計算を $x_2$ と $x_3$ についても行います．この図は既に $y_1$ の計算は終了して $y_2$ の計算が行われているところなので，$x_2$ を例に計算を進めます．$x_2$ に対して計算したクエリの値，$q_2$ をすべてのキーの値との内積を計算します．図中の $\odot$ は内積を計算する記号です．これによって得られた 3 個のスカラ値をその合計値が 1 となるように標準化します．これによって，0.7，0.2，0.1 という値が得られています．次にこれらの値をバリューに掛けます．これによって得られた 3 個のベクトルを足し合わせた値が $y_2$ です．これがセルフアテンションの計算です．クエリとキーの内積を計算することで，元のトークンと別のトークンの関係性を計算し，それを元のトークンに由来するバリューに掛けることで最終的に得られるベクトル（ここでは $y_2$）は別のすべてのトークンの情報を保持しているということになります．文脈情報を下流のネットワークに伝えることができるということです．
# 
# 以下では数式を利用してセルフアテンションを紹介します．入力配列データを $x$ とします．これは，$N$ 行 $m$ 列の行列であるとします．つまり，$N$ 個のトークンからなる配列データです．各トークンの特徴量は $m$ の長さからなるということです．セルフアテンションの計算では最初に，クエリ，キー，バリューの値を計算するのでした．$W_q$，$W_k$，$W_v$ という 3 個の重みパラメータを用いて以下のように計算します．この重みパラメータはそれぞれ $m$ 行 $d$ 列であるとします．
# 
# $
# Q=  xW_q
# $
# 
# $
# K=  xW_k
# $
# 
# $
# V=  xW_v
# $
# 
# これらを用いてアテンション $A$ は以下のように計算します．
# 
# $
# \displaystyle A(x)=\sigma\left(\frac{QK^\mathsf{T}}{\sqrt{d}}\right)V
# $
# 
# この式の $\sigma$ は $QK^\mathsf{T}$ の行方向に対して計算するソフトマックス関数です．これでアテンションの計算は終わりなのですが，アテンションの計算に出てくる項の掛け算の順番を変更するという操作でアテンションの計算量を改良した線形アテンションというものがあるのでそれもあわせて紹介します．線形アテンション $L$ は以下のように計算します．
# 
# $
# L(x)=\tau(Q)(\tau(K)^\mathsf{T}V)
# $
# 
# ここで，$\tau$ は以下の関数です．
# 
# $
# \displaystyle \tau(x)=\begin{cases}
# x+1 & (x > 0) \\
# e^x & (x \leq 0)
# \end{cases}
# $
# 
# アテンションの時間計算量は配列の長さ $N$ に対して $O(N^2)$ なのですが，線形アテンションの時間計算量は線形 $O(N)$ です．とても簡単な工夫なのに恐ろしくスピードアップしています．

# ```{note}
# この項ではベクトルを英小文字の太字ではなくて普通のフォントで記述していることに注意してください．
# ```

# ## トランスフォーマー

# トランスフォーマーはアテンションの最も有用な応用先のひとつです．トランスフォーマーの構造やその構成要素を紹介します．

# ### 基本構造

# トランスフォーマーとはアテンションと位置エンコード（ポジショナルエンコード）といわれる技術を用いて，再帰型ニューラルネットワークとは異なる方法で文字列を処理することができるニューラルネットワークの構造です．機械翻訳や質問応答に利用することができます．
# 
# 例えば，機械翻訳の場合，翻訳したい文字列を入力データ，翻訳結果の文字列を教師データとして利用します．構築した人工知能は翻訳したい文字列を入力値として受け取り，配列を出力します．配列の各要素は文字の個数と同じサイズのベクトル（その要素が何の文字なのかを示す確率ベクトル）です．
# 
# トランスフォーマーはエンコーダーとデコーダーという構造からなります．エンコーダーは配列（機械翻訳の場合，翻訳したい配列）を入力にして，同じ長さの配列を出力します．デコーダーも配列（機械翻訳の場合，翻訳で得たい配列）とエンコーダーが出力した配列を入力にして同じ長さの配列（各要素は確率ベクトル）を出力します．エンコーダーが出力した配列情報をデコーダーで処理する際にアテンションが利用されます．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/transformer.svg?raw=1" width="100%" />
# 
# エンコーダーとデコーダー間のアテンション以外にも，エンコーダーとデコーダーの内部でもそれぞれアテンション（セルフアテンション）が計算されます．アテンションは文字列内における文字の関連性を計算します．
# 
# トランスフォーマーは再帰型ニューラルネットワークで行うような文字の逐次的な処理が不要です．よって，計算機の並列化性能をより引き出せます．扱える文脈の長さも無限です．
# 
# このトランスフォーマーはものすごい性能を発揮しており，これまでに作られてきた様々な構造を過去のものとしました．特に応用の範囲が広いのはトランスフォーマーのエンコーダーの部分です．BERT と呼ばれる方法を筆頭に自然言語からなる配列を入力にして何らかの分散表現を出力する方法として自然言語処理に関わる様々な研究開発に利用されています．

# ```{note}
# 再帰型ニューラルネットワークでも扱える配列の長さは理論上無限です．
# ```

# ```{note}
# 会話でトランスフォーマーという場合は，トランスフォーマーのエンコーダーまたはデコーダーのことを言っている場合があります．エンコーダー・デコーダー，エンコーダー，デコーダー，この3個でそれぞれできることが異なります．
# ```

# ```{hint}
# 実用上，配列を入力にして配列を返す構造とだけ覚えておけば問題はないと思います．
# ```

# ### 構成要素

# トランスフォーマーは大きく分けると 3 個の要素から構成されています．アテンション（セルフアテンション），位置エンコード，ポイントワイズ MLP です．アテンションは上の節で紹介した通り，入力データのトークン間の関係性を下流のネットワークに伝えるための機構です．
# 
# **位置エンコード**は入力データの各トークンの位置関係を下流のネットワークに伝える仕組みです．RNN では入力された配列情報を最初，または，最後から順番に処理します．よって，人工知能は読み込まれた要素の順序，位置情報を明確に認識できています．これに対して，セルフアテンションを導入して文脈情報を扱えるようにした MLP はその認識ができません．アテンションで計算しているものは要素間の関連性であり，位置関係は考慮されていないのです．例えば，「これ　は　私　の　ラーメン　です」という文を考えた場合，アテンション機構では「これ」に対して「は」，「私」，「の」，「ラーメン」，「です」の関連度を計算していますが，「2 番目のタイムステップのは」や「3 番目のタイムステップの私」を認識しているわけではありません．よって，この文章の要素を並び替え，「は　です　の　私　これ　ラーメン」としても同じ計算結果が得られます．この例においては，そのような処理をしたとしても最終的な結果に影響はないかもしれませんが，例えば，「ラーメン　より　牛丼　が　好き　です　か」のような文章を考えた場合には問題が生じます．この文章では「ラーメンより牛丼が好き」かということを問うているのであって，「牛丼よりラーメンが好き」かということは問うていないのです．「より」の前後にある文字列の登場順が重要な意味を持ちます．このような情報を処理する際には各要素の位置情報を人工知能に認識させる必要があります．これは様々な方法で実現できますが，トランスフォーマーで用いている方法を紹介します．ここでは，以下のような配列情報を処理するものとします．
# 
# $
# \boldsymbol{x}=[[x_{11},x_{12},\dots,x_{1d}],[x_{21},x_{22},\dots,x_{2d}],\dots,[x_{l1},x_{l2},\dots,x_{ld}]]
# $
# 
# すなわち，配列 $\boldsymbol{x}$ は $l$ 個の長さからなり，その $l$ 個の各要素は $d$ 個の要素からなるベクトルです．この $[x_{11},x_{12},\dots,x_{1d}]$ のような要素は例えば自然言語においては単語のことを示します．単語が $d$ 個の要素からなる何らかのベクトルとしてエンコードされているとします．このような配列情報に位置情報を持たせるには以下のような位置情報を含んだ配列を用意します．
# 
# $
# \boldsymbol{p}=[[p_{11},p_{12},\dots,p_{1d}],[p_{21},p_{22},\dots,p_{2d}],\dots,[p_{l1},p_{l2},\dots,p_{ld}]]
# $
# 
# この配列は入力配列 $\boldsymbol{x}$ と同じ形をしています．よって $\boldsymbol{x}$ に加算することができます．$\boldsymbol{x}$ に $\boldsymbol{p}$ を加算することで位置情報を保持した配列を生成することができるのです．トランスフォーマーでは位置情報を含んだ配列の要素 $p_{ik}$ を $p_{i(2j)}$ と $p_{i(2j+1)}$ によって，トークンを示すベクトルの要素の偶数番目と奇数番目で場合分けし，それぞれ正弦関数と余弦関数で定義します．偶数番目は以下のように表されます．
# 
# $
# \displaystyle p_{i(2j)}=\sin\left(\frac{i}{10000^{\frac{2j}{d}}}\right)
# $
# 
# 奇数番目は以下のように表されます．
# 
# $
# \displaystyle  p_{i(2j+1)}=\cos\left(\frac{i}{10000^{\frac{2j}{d}}}\right)
# $
# 
# 
# **ポイントワイズ MLP** は可変長の配列データを処理するための MLP です．機械学習モデルを学習させる際には様々なデータをモデルに読み込ませますが，配列データにおいては，それらのデータの長さが不揃いであることがあります．そのような場合において固定長のデータを処理するための方法である MLP を使おうとするとどのような長さのデータに対応させるためにはかなり長い入力層のサイズを用意しなければなりません．かなり長い入力層を利用したとしても，その入力層よりさらに長い入力データを処理しなければならない可能性は排除できません．これに対してポイントワイズ MLP は入力配列の各トークンに対してたったひとつの MLP の計算をする方法です．入力配列のどのトークンに対しても同じパラメータを持った MLP の計算が行われます．10 の長さの入力配列に対しては同じ MLP による計算が 10 回行われ，10 の長さの配列が出力されます．100 の長さの入力配列に対しては同じ MLP による計算が 100 回行われ，100 の長さの配列が出力されます．

# ### 現実世界での利用方法

# トランスフォーマーの現実世界での応用先としては，感情分析，特徴抽出，穴埋め，固有表現抽出（文章中の固有表現を見つける），質問応答，要約，文章生成，翻訳等があります．これらの問題を解決しようとする際には，実際には，事前学習モデルを活用する場合が多いです．事前学習モデルとは解きたい問題ではなくて，あらかじめ別の何らかの問題に対して（大量の）データを利用して学習した学習済みのモデルです．学習済みモデルにはそのドメインにおける知識を獲得していることを期待しています．
# 
# 事前学習モデルとして有名なものには BERT（bidirectional encoder representations from transformers）があります．上述のようにトランスフォーマーはエンコーダー・デコーダー構造を有しますが，BERT はトランスフォーマーのエンコーダー構造を利用して構築されるものです．BERT は自然言語からなる配列データを入力として何らかの配列データを出力する汎用言語表現モデルです．利用方法は多岐にわたりますが，自然言語の分散表現を生成するモデルと言えます．事前学習モデルとして公開されており，自然言語からなる配列を入力として得られる BERT の出力を別の何らかの問題を解くための人工知能アルゴリズムの入力として用いることで，様々な問題を解決するための基礎として利用することができます．
#     

# ```{note}
# 次の章では BERT の利用方法を紹介します．
# ```

# ## エンコーダーの実装

# この節ではトランスフォーマーのエンコーダーに相当する部分の実装をアテンションの計算部分から作ります．

# ### 扱うデータ

# この節では以下のようなデータを利用して，3 個の値の分類問題を解きます．入力データの長さは一定ではありません．このデータには特徴があります．`0` に分類される 3 個の入力データの最初の要素はそれぞれ `1`，`3`，`7` です．また，`1` に分類される 3 個の入力データの最初の要素もそれぞれ `1`，`3`，`7` です．さらに，`2` に分類される 3 個の入力データの最初の要素も同じく `1`，`3`，`7` です．
# 
# 入力ベクトル | ターゲットベクトル
# :---: | :---:
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 1] | [0]
# [3, 9, 3, 4, 7] | [0]
# [7, 5, 8] | [0]
# [1, 5, 8] | [1]
# [3, 9, 3, 4, 6] | [1]
# [7, 3, 4, 1] | [1]
# [1, 3] | [2]
# [3, 9, 3, 4, 1] | [2]
# [7, 5, 5, 7, 7, 5] | [2]
# 
# このような入力データに対してアテンションの計算を行い，その後ポイントワイズ MLP の計算を行った後に，その出力ベクトルの最初の要素（元データの最初のトークンに対応する値）のみに対して MLP の計算を行い，ターゲットベクトルの予測をします．図で示すと以下のようなネットワークです．つまり，この問題はアテンションの出力（の最初の要素）が入力データの全部の文脈情報を保持できていないと解けない問題です．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/contextNetwork.svg?raw=1" width="100%" />
# 

# ### 普通のアテンション

# 普通のアテンションを利用してこの問題を解くには以下のようなコードを書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データの生成．
    trainx = [[1,2,3,4,5,6,7,8,9,1], [3,9,3,4,7], [7,5,8], [1,5,8], [3,9,3,4,6], [7,3,4,1], [1,3], [3,9,3,4,1], [7,5,5,7,7,5]]
    trainx = tf.keras.preprocessing.sequence.pad_sequences(trainx, padding="post", dtype=np.int32, value=0) # 短い配列の後ろにゼロパディングする．
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    
    # ハイパーパラメータの設定．
    maxEpochSize = 2000
    embedSize = 16
    attentionUnitSize = 32
    middleUnitSize = 32
    dropoutRate = 0.5
    minibatchSize = 3
    
    # データのサイズや長さの取得．
    trainSize = trainx.shape[0]
    outputSize = len(np.unique(traint))
    vocabNumber = len(np.unique(trainx))
    minibatchNumber = trainSize // minibatchSize
    
    # モデルの生成．
    model = Network(attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate)
    cceComputer = tf.keras.losses.SparseCategoricalCrossentropy()
    accComputer = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    
    # tf.keras.Modelを継承したクラスからモデルを生成したとき，以下のように入力データの形に等しいデータを読み込ませてsummary()を利用するとモデルの詳細を表示可能．
    model(tf.zeros((minibatchSize, trainx.shape[1])), False)
    model.summary()
    
    # デバッグプリント．入力に対してどのような出力が得られるかを確認する．
#    dp = model.call(trainx, False)
#    print(dp)
#    exit()
    
    # 以下は勾配を計算してコストや正確度を計算するための記述．
    @tf.function
    def run(tx, tt, flag):
        with tf.GradientTape() as tape:
            model.trainable = flag
            ty = model.call(tx, flag)
            costvalue = cceComputer(tt, ty)
        gradient = tape.gradient(costvalue, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        accvalue = accComputer(tt, ty)
        return costvalue, accvalue
    
    # 以下は学習のための記述．
    for epoch in range(1, maxEpochSize+1):
        trainIndex = np.random.permutation(trainSize) # トレーニングデータセットのサイズに相当する整数からなるランダムな配列を生成．
        trainCost, trainAcc = 0, 0
        for subepoch in range(minibatchNumber):
            startIndex = subepoch * minibatchSize
            endIndex = (subepoch+1) * minibatchSize
            miniTrainx = trainx[trainIndex[startIndex:endIndex]] # ランダムに決められた数だけデータを抽出する．
            miniTraint = traint[trainIndex[startIndex:endIndex]]
            miniTrainCost, miniTrainAcc = run(miniTrainx, miniTraint, True) # パラメータの更新をするのでフラッグはTrueにする．
            trainCost += miniTrainCost * minibatchNumber**(-1)
            trainAcc += miniTrainAcc * minibatchNumber**(-1)
        if epoch % 50 == 0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch, trainCost, trainAcc))
            prediction = model.call(trainx, False)
            print("Prediction:", np.argmax(prediction, axis=1)) # 予測値を出力．

class Network(tf.keras.Model):
    def __init__(self, attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate):
        super(Network, self).__init__()
        self.a = Attention(attentionUnitSize, dropoutRate)
        self.embed = tf.keras.layers.Embedding(input_dim=vocabNumber, output_dim=embedSize, mask_zero=True)
        self.pe = PositionalEncoder()
        self.masker = Masker()
        self.w1 = tf.keras.layers.Dense(middleUnitSize)
        self.w2 = tf.keras.layers.Dense(outputSize)
        self.lr = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x, learningFlag):
        maskSeq, minNumber = self.masker(x)
        x = self.embed(x) # エンベッド．質的変数をエンコードする．1を[0.3 0.1 0.7]，2を[0.4 0.9 0.3]のような指定した長さの浮動小数点数からなるベクトルへ変換する行為．
        x = self.pe(x) # 位置エンコード情報の計算．
        y = self.a(x, x, maskSeq, minNumber, learningFlag)
        y = self.lr(y)
        y = self.dropout(y, training=learningFlag)
        y = self.w1(y)
        y = self.lr(y)
        y = y[:, 0, :] # 入力データの最初のトークンに相当する値だけを用いて以降の計算を行う．
        y = self.w2(y)
        y = tf.nn.softmax(y) # 3個の値の分類問題であるため．
        return y

class Attention(tf.keras.Model):
    def __init__(self, attentionUnitSize, dropoutRate):
        super(Attention, self).__init__()
        self.attentionUnitSize = attentionUnitSize
        self.queryLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.keyLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.valueLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.outputLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x1, x2, maskSeq, minNumber, learningFlag):
        q = self.queryLayer(x1) # クエリベクトルの生成．
        k = self.keyLayer(x2) # キーベクトルの生成．
        v = self.valueLayer(x2) # バリューベクトルの生成．
        q = q * self.attentionUnitSize**(-0.5) # 正規化のため．
        a = tf.matmul(q, k, transpose_b=True)
        a = a + tf.cast(maskSeq, dtype=tf.float32) * minNumber # 次のソフトマックス関数の操作でゼロパディングした要素の値をゼロと出力するためにとても小さい値を加える．
        a = tf.nn.softmax(a) # ゼロパティングされたところの値はゼロになる．
        y = tf.matmul(a, v)
        y = tf.keras.activations.relu(y)
        y = self.dropout(y, training=learningFlag)
        y = self.outputLayer(y)
        return y

class Masker(tf.keras.Model):
    # 以下は入力データが質的データのときに利用するもので，ゼロパディングされたトークンの位置（と小さい値）を返すもの．
    def call(self, x):
        x = tf.cast(x, dtype=tf.int32) # 浮動小数点数を整数にする．扱うデータがラベルデータのため．入力データが浮動小数点数であるなら不要．
        inputBatchSize, inputLength = tf.unstack(tf.shape(x)) # バッチサイズと入力データの長さを取得．
        maskSeq = tf.equal(x, 0) # 元データのゼロパディングした要素をTrue，そうでない要素をFalseとしたデータを生成．
        maskSeq = tf.reshape(maskSeq, [inputBatchSize, 1, inputLength]) # データの形の整形．
        return maskSeq, x.dtype.min # 整数の最も小さい値を後で利用するため返す．

class PositionalEncoder(tf.keras.layers.Layer):
    # 以下は位置エンコードをしたデータを返すもの．
    def call(self, x):
        inputBatchSize, inputLength, inputEmbedSize = tf.unstack(tf.shape(x)) # バッチサイズ，入力データの長さ，エンベッドサイズを取得．
        j = tf.range(inputEmbedSize) // 2 * 2 # 2jの生成．2jではなくjという変数名を利用．
        j = tf.tile(tf.expand_dims(j, 0),[inputLength, 1]) # データの形の整形．
        denominator = tf.pow(float(10000), tf.cast(j/inputEmbedSize, x.dtype)) # 10000**(2j/d)の計算．
        phase = tf.cast(tf.range(inputEmbedSize)%2, x.dtype) * np.pi / 2 # 位相の計算．後でsin(90度+x)=cos(x)を利用するため．np.piは3.14ラジアン（180度）．
        phase = tf.tile(tf.expand_dims(phase, 0), [inputLength, 1]) # データの形の整形
        i = tf.range(inputLength) # iの生成．
        i = tf.cast(tf.tile(tf.expand_dims(i, 1), [1, inputEmbedSize]), x.dtype) # データの形の整形．
        encordedPosition = tf.sin(i / denominator + phase) # 位置エンコードの式の計算．
        encordedPosition = tf.tile(tf.expand_dims(encordedPosition, 0), [inputBatchSize, 1, 1]) # データの形の整形．
        return x + encordedPosition

if __name__ == "__main__":
    main()


# プログラムを実行すると以下のような結果が得られたと思います．この `Prediction` が入力データに対する予測結果です．正解を導けたことがわかります．
# 
# ```shell
# .
# .
# .
# Epoch  2000: Training cost= 0.0079, Training ACC= 0.9302
# Prediction: [0 0 0 1 1 1 2 2 2]
# ```

# 以降でプログラムの説明をします．以下の部分はデータを生成するための記述です．入力配列の長さを揃えるためにゼロパディングをしています．
# 
# ```python
#     # データの生成．
#     trainx = [[1,2,3,4,5,6,7,8,9,1], [3,9,3,4,7], [7,5,8], [1,5,8], [3,9,3,4,6], [7,3,4,1], [1,3], [3,9,3,4,1], [7,5,5,7,7,5]]
#     trainx = tf.keras.preprocessing.sequence.pad_sequences(trainx, padding="post", dtype=np.int32, value=0) # 短い配列の後ろにゼロパディングする．
#     traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
# ```

# ```{note}
# ゼロパディングしない世界に生まれたかったです．
# ```

# ハイパーパラメータの設定部分やデータサイズ等の取得の部分の説明は省略して，以下の部分ですが，これはモデルを生成するための記述です．この問題は分類問題なのでクロスエントロピーをコスト関数にします．
# 
# ```python
#     # モデルの生成．
#     model = Network(attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate)
#     cceComputer = tf.keras.losses.SparseCategoricalCrossentropy()
#     accComputer = tf.keras.metrics.SparseCategoricalAccuracy()
#     optimizer = tf.keras.optimizers.Adam()
# ```

# 以下の部分はプログラム実行のために必要なものではありませんが，これを加えておくとネットワーク全体のパラメータサイズやネットワークの構造のまとめを表示することができます．モデル構造を記述したクラスで `tf.keras.Model` を継承することがポイントです．
# 
# ```python
#     # tf.keras.Modelを継承したクラスからモデルを生成したとき，以下のように入力データの形に等しいデータを読み込ませてsummary()を利用するとモデルの詳細を表示可能．
#     model(tf.zeros((minibatchSize, trainx.shape[1])), False)
#     model.summary()
# ```

# ```{note}
# 論文を書くときとかは便利です．
# ```

# Subclassing API でネットワーク構造を作る際，つまり，ネットワーク構造を定義するクラスを作る際には以下のようなデバッグプリントで入力に対してどのような出力が得られるかを確認しながらトライアンドエラーでコーディングすると良いです．
# 
# ```python
#     # デバッグプリント．入力に対してどのような出力が得られるかを確認する．
# #    dp = model.call(trainx, False)
# #    print(dp)
# #    exit()
# ```

# 以下の部分は Subclassing API で勾配を計算したり，最適化法を適用したり，コスト値を計算したりするためのお決まりの書き方です．
# 
# ```python
#     # 以下は勾配を計算してコストや正確度を計算するための記述．
#     @tf.function
#     def run(tx, tt, flag):
#         with tf.GradientTape() as tape:
#             model.trainable = flag
#             ty = model.call(tx, flag)
#             costvalue = cceComputer(tt, ty)
#         gradient = tape.gradient(costvalue, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradient, model.trainable_variables))
#         accvalue = accComputer(tt, ty)
#         return costvalue, accvalue
# ```

# 学習ループの記述は以下の部分です．ミニバッチの計算部分について説明します．最初に，0 からトレーニングデータセットのサイズに相当する整数（この場合 9）までの整数すべてがランダムに並べられたベクトルを生成します．このベクトルからミニバッチのサイズ（`minibatchSize`）分スライスをして抜き出すデータのインデックスを決めます．これをミニバッチの個数（`minibatchNumber`）分行うことで全部のデータをミニバッチ処理することができます．
# 
# ```python
#     # 以下は学習のための記述．
#     for epoch in range(1, maxEpochSize+1):
#         trainIndex = np.random.permutation(trainSize) # トレーニングデータセットのサイズに相当する整数からなるランダムな配列を生成．
#         trainCost, trainAcc = 0, 0
#         for subepoch in range(minibatchNumber):
#             startIndex = subepoch * minibatchSize
#             endIndex = (subepoch+1) * minibatchSize
#             miniTrainx = trainx[trainIndex[startIndex:endIndex]] # ランダムに決められた数だけデータを抽出する．
#             miniTraint = traint[trainIndex[startIndex:endIndex]]
#             miniTrainCost, miniTrainAcc = run(miniTrainx, miniTraint, True) # パラメータの更新をするのでフラッグはTrueにする．
#             trainCost += miniTrainCost * minibatchNumber**(-1)
#             trainAcc += miniTrainAcc * minibatchNumber**(-1)
#         if epoch % 50 == 0:
#             print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch, trainCost, trainAcc))
#             prediction = model.call(trainx, False)
#             print("Prediction:", np.argmax(prediction, axis=1)) # 予測値を出力．
# ```

# 次に，ネットワークの全体構造の説明をします．コンストラクタの部分ですが，アテンションを計算するクラスを最初に呼び出します．これは後で説明します．エンベッド層も呼び出します．エンベッドとは質的変数を指定したサイズのベクトルに包埋するためのものです．計算機で処理させるために行います．ここで行っているのはワンホットエンコーディングではなくて，浮動小数点数からなるベクトルへの包埋です．位置エンコードとマスク（ゼロパディングされた配列の特別な処理）を行うためのクラスの説明は後でします．その他の変数はよくある構成要素です．これらを用いて，このネットワークでは入力データの各トークをエンベッドし，位置エンコードし，アテンションの計算を行い，ポイントワイズ MLP（`self.w1`）の計算をした後に，その出力ベクトルの最初の要素（`y[:, 0, :]`）に MLP（`self.w2`）の計算をします．
# 
# ```python
# class Network(tf.keras.Model):
#     def __init__(self, attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate):
#         super(Network, self).__init__()
#         self.a = Attention(attentionUnitSize, dropoutRate)
#         self.embed = tf.keras.layers.Embedding(input_dim=vocabNumber, output_dim=embedSize, mask_zero=True)
#         self.pe = PositionalEncoder()
#         self.masker = Masker()
#         self.w1 = tf.keras.layers.Dense(middleUnitSize)
#         self.w2 = tf.keras.layers.Dense(outputSize)
#         self.lr = tf.keras.layers.LeakyReLU()
#         self.dropout = tf.keras.layers.Dropout(dropoutRate)
#     def call(self, x, learningFlag):
#         maskSeq, minNumber = self.masker(x)
#         x = self.embed(x) # エンベッド．質的変数をエンコードする．1を[0.3 0.1 0.7]，2を[0.4 0.9 0.3]のような指定した長さの浮動小数点数からなるベクトルへ変換する行為．
#         x = self.pe(x) # 位置エンコード情報の計算．
#         y = self.a(x, x, maskSeq, minNumber, learningFlag)
#         y = self.lr(y)
#         y = self.dropout(y, training=learningFlag)
#         y = self.w1(y)
#         y = self.lr(y)
#         y = y[:, 0, :] # 入力データの最初のトークンに相当する値だけを用いて以降の計算を行う．
#         y = self.w2(y)
#         y = tf.nn.softmax(y) # 3個の値の分類問題であるため．
#         return y
# ```

# マスクした配列を処理するためのクラスは以下に示す通りです．ゼロパディング（マスク）した要素はアテンションの計算をする際に除外をしなければなりません．そのために，入力データのどこがマスクされたのかを知る必要がありますが，それを返す関数のみからなるクラスです．
# 
# ```python
# class Masker(tf.keras.Model):
#     # 以下は入力データが質的データのときに利用するもので，ゼロパディングされたトークンの位置（と小さい値）を返すもの．
#     def call(self, x):
#         x = tf.cast(x, dtype=tf.int32) # 浮動小数点数を整数にする．扱うデータがラベルデータのため．入力データが浮動小数点数であるなら不要．
#         inputBatchSize, inputLength = tf.unstack(tf.shape(x)) # バッチサイズと入力データの長さを取得．
#         maskSeq = tf.equal(x, 0) # 元データのゼロパディングした要素をTrue，そうでない要素をFalseとしたデータを生成．
#         maskSeq = tf.reshape(maskSeq, [inputBatchSize, 1, inputLength]) # データの形の整形．
#         return maskSeq, x.dtype.min # 整数の最も小さい値を後で利用するため返す．
# ```

# 位置エンコードのためのクラスは以下に示す通りです．位置エンコードの式に沿った記述です．最初に $2j$（`j`）を計算し，$10000^{(2j/d)}$（`denominator`）を求めます．位置エンコードの計算ではサインとコサインの計算が登場しますが，ここでは以下の変換公式を利用します．
# 
# $
# \sin(\theta+90^{\circ})=\cos(\theta)
# $
# 
# よって，位相を計算します．この位相を角度ベクトル（実際にはラジアン）に加え，まとめてサインの計算をすることでコサインの計算を同時に行います．最終的に得られた位置エンコードの値を入力データに加えることで位置エンコード情報の付加が終わります．
# 
# ```python
# class PositionalEncoder(tf.keras.layers.Layer):
#     # 以下は位置エンコードをしたデータを返すもの．
#     def call(self, x):
#         inputBatchSize, inputLength, inputEmbedSize = tf.unstack(tf.shape(x)) # バッチサイズ，入力データの長さ，エンベッドサイズを取得．
#         j = tf.range(inputEmbedSize) // 2 * 2 # 2jの生成．2jではなくjという変数名を利用．
#         j = tf.tile(tf.expand_dims(j, 0),[inputLength, 1]) # データの形の整形．
#         denominator = tf.pow(float(10000), tf.cast(j/inputEmbedSize, x.dtype)) # 10000**(2j/d)の計算．
#         phase = tf.cast(tf.range(inputEmbedSize)%2, x.dtype) * np.pi / 2 # 位相の計算．後でsin(90度+x)=cos(x)を利用するため．np.piは3.14ラジアン（180度）．
#         phase = tf.tile(tf.expand_dims(phase, 0), [inputLength, 1]) # データの形の整形
#         i = tf.range(inputLength) # iの生成．
#         i = tf.cast(tf.tile(tf.expand_dims(i, 1), [1, inputEmbedSize]), x.dtype) # データの形の整形．
#         encordedPosition = tf.sin(i / denominator + phase) # 位置エンコードの式の計算．
#         encordedPosition = tf.tile(tf.expand_dims(encordedPosition, 0), [inputBatchSize, 1, 1]) # データの形の整形．
#         return x + encordedPosition
# ```

# これらのパーツを利用した最後のアテンションの計算は以下に示す通りです．コンストラクタの最初で入力ベクトルからクエリ，キー，バリューベクトルを生成するための重みパラメータを生成します．これらを用いてそれぞれ，クエリ，キー，バリューベクトルを生成します．この節で計算するアテンションはセルフアテンションであるため，計算の入力出る `x1` と `x2` を分ける必要はなかったのですが，ソース・ターゲットアテンションへの応用のことも考慮して分けています．その後正規化の計算（式の $\sqrt{d}$ の計算）を行った後に，クエリとキーの内積をとりアテンションを生成します．さらに，マスクした要素にとても小さい値（負の値）を加えます．これを加えた後にソフトマックス関数の計算を行うことで，ゼロパディングされた要素のアテンションの値は 0 になります．得られたアテンションにバリューの値を掛けることでアテンション機構の出力値が得られます．さらに，活性化関数やドロップアウトの計算を行い，出力層であるポイントワイズ MLP の計算を行います．
# 
# ```python
# class Attention(tf.keras.Model):
#     def __init__(self, attentionUnitSize, dropoutRate):
#         super(Attention, self).__init__()
#         self.attentionUnitSize = attentionUnitSize
#         self.queryLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
#         self.keyLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
#         self.valueLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
#         self.outputLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
#         self.dropout = tf.keras.layers.Dropout(dropoutRate)
#     def call(self, x1, x2, maskSeq, minNumber, learningFlag):
#         q = self.queryLayer(x1) # クエリベクトルの生成．
#         k = self.keyLayer(x2) # キーベクトルの生成．
#         v = self.valueLayer(x2) # バリューベクトルの生成．
#         q = q * self.attentionUnitSize**(-0.5) # 正規化のため．
#         a = tf.matmul(q, k, transpose_b=True)
#         a = a + tf.cast(maskSeq, dtype=tf.float32) * minNumber # 次のソフトマックス関数の操作でゼロパディングした要素の値をゼロと出力するためにとても小さい値を加える．
#         a = tf.nn.softmax(a) # ゼロパティングされたところの値はゼロになる．
#         y = tf.matmul(a, v)
#         y = tf.keras.activations.relu(y)
#         y = self.dropout(y, training=learningFlag)
#         y = self.outputLayer(y)
#         return y
# ```

# ```{note}
# とっても簡単なコードでアテンションが実装できますね．
# ```

# アテンションの計算をこの問題に利用することで正解が導き出されたことから，アテンションを利用するとしっかり入力配列データの文脈情報を下流のネットワークに伝えることが出来たと思いますが，もうひとつ比較のための実験を行います．以下ではアテンションを利用せず，つまり，単なるポイントワイズ MLP を利用したときにこの問題を解くことができるかどうかを調べます．できないはずです．以下のようなコードを書いて実行します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データの生成．
    trainx = [[1,2,3,4,5,6,7,8,9,1], [3,9,3,4,7], [7,5,8], [1,5,8], [3,9,3,4,6], [7,3,4,1], [1,3], [3,9,3,4,1], [7,5,5,7,7,5]]
    trainx = tf.keras.preprocessing.sequence.pad_sequences(trainx, padding="post", dtype=np.int32, value=0) # 短い配列の後ろにゼロパディングする．
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    
    # ハイパーパラメータの設定．
    maxEpochSize = 2000
    embedSize = 16
    attentionUnitSize = 32
    middleUnitSize = 32
    dropoutRate = 0.5
    minibatchSize = 3
    
    # データのサイズや長さの取得．
    trainSize = trainx.shape[0]
    outputSize = len(np.unique(traint))
    vocabNumber = len(np.unique(trainx))
    minibatchNumber = trainSize // minibatchSize
    
    # モデルの生成．
    model = Network(attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate)
    cceComputer = tf.keras.losses.SparseCategoricalCrossentropy()
    accComputer = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    
    # tf.keras.Modelを継承したクラスからモデルを生成したとき，以下のように入力データの形に等しいデータを読み込ませてsummary()を利用するとモデルの詳細を表示可能．
    model(tf.zeros((minibatchSize, trainx.shape[1])), False)
    model.summary()
    
    # 以下は勾配を計算してコストや正確度を計算するための記述．
    @tf.function
    def run(tx, tt, flag):
        with tf.GradientTape() as tape:
            model.trainable = flag
            ty = model.call(tx, flag)
            costvalue = cceComputer(tt, ty)
        gradient = tape.gradient(costvalue, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        accvalue = accComputer(tt, ty)
        return costvalue, accvalue
    
    # 以下は学習のための記述．
    for epoch in range(1, maxEpochSize+1):
        trainIndex = np.random.permutation(trainSize) # トレーニングデータセットのサイズに相当する整数からなるランダムな配列を生成．
        trainCost, trainAcc = 0, 0
        for subepoch in range(minibatchNumber):
            startIndex = subepoch * minibatchSize
            endIndex = (subepoch+1) * minibatchSize
            miniTrainx = trainx[trainIndex[startIndex:endIndex]] # ランダムに決められた数だけデータを抽出する．
            miniTraint = traint[trainIndex[startIndex:endIndex]]
            miniTrainCost, miniTrainAcc = run(miniTrainx, miniTraint, True) # パラメータの更新をするのでフラッグはTrueにする．
            trainCost += miniTrainCost * minibatchNumber**(-1)
            trainAcc += miniTrainAcc * minibatchNumber**(-1)
        if epoch % 50 == 0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch, trainCost, trainAcc))
            prediction = model.call(trainx, False)
            print("Prediction:", np.argmax(prediction, axis=1)) # 予測値を出力．

class Network(tf.keras.Model):
    def __init__(self, attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate):
        super(Network, self).__init__()
        self.a = PointwiseMLP(attentionUnitSize, dropoutRate) # ここが変化．
        self.embed = tf.keras.layers.Embedding(input_dim=vocabNumber, output_dim=embedSize, mask_zero=True)
        self.pe = PositionalEncoder()
        self.masker = Masker()
        self.w1 = tf.keras.layers.Dense(middleUnitSize)
        self.w2 = tf.keras.layers.Dense(outputSize)
        self.lr = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x, learningFlag):
        maskSeq, minNumber = self.masker(x)
        x = self.embed(x) # エンベッド．質的変数をエンコードする．1を[0.3 0.1 0.7]，2を[0.4 0.9 0.3]のような指定した長さの浮動小数点数からなるベクトルへ変換する行為．
        x = self.pe(x) # 位置エンコード情報の計算．
        y = self.a(x, x, maskSeq, minNumber, learningFlag)
        y = self.lr(y)
        y = self.dropout(y, training=learningFlag)
        y = self.w1(y)
        y = self.lr(y)
        y = y[:, 0, :] # 入力データの最初のトークンに相当する値だけを用いて以降の計算を行う．
        y = self.w2(y)
        y = tf.nn.softmax(y) # 3個の値の分類問題であるため．
        return y

class Attention(tf.keras.Model):
    def __init__(self, attentionUnitSize, dropoutRate):
        super(Attention, self).__init__()
        self.attentionUnitSize = attentionUnitSize
        self.queryLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.keyLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.valueLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.outputLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x1, x2, maskSeq, minNumber, learningFlag):
        q = self.queryLayer(x1) # クエリベクトルの生成．
        k = self.keyLayer(x2) # キーベクトルの生成．
        v = self.valueLayer(x2) # バリューベクトルの生成．
        q = q * self.attentionUnitSize**(-0.5) # 正規化のため．
        a = tf.matmul(q, k, transpose_b=True)
        a = a + tf.cast(maskSeq, dtype=tf.float32) * minNumber # 次のソフトマックス関数の操作でゼロパディングした要素の値をゼロと出力するためにとても小さい値を加える．
        a = tf.nn.softmax(a) # ゼロパティングされたところの値はゼロになる．
        y = tf.matmul(a, v)
        y = tf.keras.activations.relu(y)
        y = self.dropout(y, training=learningFlag)
        y = self.outputLayer(y)
        return y

class PointwiseMLP(Attention): # Attentionを継承．
    def __init__(self, attentionUnitSize, dropoutRate):
        super().__init__(attentionUnitSize, dropoutRate)
    def call(self, x1, x2, maskSeq, minNumber, learningFlag):
        q = self.queryLayer(x1)
        k = self.keyLayer(x2)
        b = tf.transpose(maskSeq, perm=[0, 2, 1])
        q = q * tf.cast(~b, dtype=tf.float32)
        k = k * tf.cast(~b, dtype=tf.float32)
        y = q * k
        y = self.dropout(y, training=learningFlag)
        y = self.valueLayer(y)
        y = tf.keras.activations.relu(y)
        y = self.dropout(y, training=learningFlag)
        y = self.outputLayer(y)
        return y

class Masker(tf.keras.Model):
    # 以下は入力データが質的データのときに利用するもので，ゼロパディングされたトークンの位置（と小さい値）を返すもの．
    def call(self, x):
        x = tf.cast(x, dtype=tf.int32) # 浮動小数点数を整数にする．扱うデータがラベルデータのため．入力データが浮動小数点数であるなら不要．
        inputBatchSize, inputLength = tf.unstack(tf.shape(x)) # バッチサイズと入力データの長さを取得．
        maskSeq = tf.equal(x, 0) # 元データのゼロパディングした要素をTrue，そうでない要素をFalseとしたデータを生成．
        maskSeq = tf.reshape(maskSeq, [inputBatchSize, 1, inputLength]) # データの形の整形．
        return maskSeq, x.dtype.min # 整数の最も小さい値を後で利用するため返す．

class PositionalEncoder(tf.keras.layers.Layer):
    # 以下は位置エンコードをしたデータを返すもの．
    def call(self, x):
        inputBatchSize, inputLength, inputEmbedSize = tf.unstack(tf.shape(x)) # バッチサイズ，入力データの長さ，エンベッドサイズを取得．
        j = tf.range(inputEmbedSize) // 2 * 2 # 2jの生成．2jではなくjという変数名を利用．
        j = tf.tile(tf.expand_dims(j, 0),[inputLength, 1]) # データの形の整形．
        denominator = tf.pow(float(10000), tf.cast(j/inputEmbedSize, x.dtype)) # 10000**(2j/d)の計算．
        phase = tf.cast(tf.range(inputEmbedSize)%2, x.dtype) * np.pi / 2 # 位相の計算．後でsin(90度+x)=cos(x)を利用するため．np.piは3.14ラジアン（180度）．
        phase = tf.tile(tf.expand_dims(phase, 0), [inputLength, 1]) # データの形の整形
        i = tf.range(inputLength) # iの生成．
        i = tf.cast(tf.tile(tf.expand_dims(i, 1), [1, inputEmbedSize]), x.dtype) # データの形の整形．
        encordedPosition = tf.sin(i / denominator + phase) # 位置エンコードの式の計算．
        encordedPosition = tf.tile(tf.expand_dims(encordedPosition, 0), [inputBatchSize, 1, 1]) # データの形の整形．
        return x + encordedPosition

if __name__ == "__main__":
    main()


# 実行した結果，以下のような出力が得られました．学習がうまく進んでいません．3 個の値の分類問題なのでランダムに予測すると大体正確度は 0.33 になると思いますが，そのようになっています．
# 
# ```shell
# .
# .
# .
# Epoch  2000: Training cost= 1.1102, Training ACC= 0.3295
# Prediction: [0 0 0 0 0 0 0 0 0]
# ```

# ネットワーク構造で変化させた部分は以下の `PointwiseMLP` というクラスを呼び出すところです．
# 
# ```python
# class Network(tf.keras.Model):
#     def __init__(self, attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate):
#         super(Network, self).__init__()
#         self.a = PointwiseMLP(attentionUnitSize, dropoutRate) # ここが変化．
#         self.embed = tf.keras.layers.Embedding(input_dim=vocabNumber, output_dim=embedSize, mask_zero=True)
#         self.pe = PositionalEncoder()
#         self.masker = Masker()
#         self.w1 = tf.keras.layers.Dense(middleUnitSize)
#         self.w2 = tf.keras.layers.Dense(outputSize)
#         self.lr = tf.keras.layers.LeakyReLU()
#         self.dropout = tf.keras.layers.Dropout(dropoutRate)
#     def call(self, x, learningFlag):
#         maskSeq, minNumber = self.masker(x)
#         x = self.embed(x) # エンベッド．質的変数をエンコードする．1を[0.3 0.1 0.7]，2を[0.4 0.9 0.3]のような指定した長さの浮動小数点数からなるベクトルへ変換する行為．
#         x = self.pe(x) # 位置エンコード情報の計算．
#         y = self.a(x, x, maskSeq, minNumber, learningFlag)
#         y = self.lr(y)
#         y = self.dropout(y, training=learningFlag)
#         y = self.w1(y)
#         y = self.lr(y)
#         y = y[:, 0, :] # 入力データの最初のトークンに相当する値だけを用いて以降の計算を行う．
#         y = self.w2(y)
#         y = tf.nn.softmax(y) # 3個の値の分類問題であるため．
#         return y
# ```

# このクラスは以下のように書きました．折角アテンションのクラスを書いたのでそれを継承しました．パラメータのサイズはアテンションのものと同じです．
# 
# ```python
# class PointwiseMLP(Attention): # Attentionを継承．
#     def __init__(self, attentionUnitSize, dropoutRate):
#         super().__init__(attentionUnitSize, dropoutRate)
#     def call(self, x1, x2, maskSeq, minNumber, learningFlag):
#         q = self.queryLayer(x1)
#         k = self.keyLayer(x2)
#         b = tf.transpose(maskSeq, perm=[0, 2, 1])
#         q = q * tf.cast(~b, dtype=tf.float32)
#         k = k * tf.cast(~b, dtype=tf.float32)
#         y = q * k
#         y = self.dropout(y, training=learningFlag)
#         y = self.valueLayer(y)
#         y = tf.keras.activations.relu(y)
#         y = self.dropout(y, training=learningFlag)
#         y = self.outputLayer(y)
#         return y
# ```

# ### 線形計算量のアテンション

# 線形計算量で計算が可能なアテンションの亜種を紹介します．以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # データの生成．
    trainx = [[1,2,3,4,5,6,7,8,9,1], [3,9,3,4,7], [7,5,8], [1,5,8], [3,9,3,4,6], [7,3,4,1], [1,3], [3,9,3,4,1], [7,5,5,7,7,5]]
    trainx = tf.keras.preprocessing.sequence.pad_sequences(trainx, padding="post", dtype=np.int32, value=0) # 短い配列の後ろにゼロパディングする．
    traint = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    
    # ハイパーパラメータの設定．
    maxEpochSize = 2000
    embedSize = 16
    attentionUnitSize = 32
    middleUnitSize = 32
    dropoutRate = 0.5
    minibatchSize = 3
    
    # データのサイズや長さの取得．
    trainSize = trainx.shape[0]
    outputSize = len(np.unique(traint))
    vocabNumber = len(np.unique(trainx))
    minibatchNumber = trainSize // minibatchSize
    
    # モデルの生成．
    model = Network(attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate)
    cceComputer = tf.keras.losses.SparseCategoricalCrossentropy()
    accComputer = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    
    # tf.keras.Modelを継承したクラスからモデルを生成したとき，以下のように入力データの形に等しいデータを読み込ませてsummary()を利用するとモデルの詳細を表示可能．
    model(tf.zeros((minibatchSize, trainx.shape[1])), False)
    model.summary()
    
    # 以下は勾配を計算してコストや正確度を計算するための記述．
    @tf.function
    def run(tx, tt, flag):
        with tf.GradientTape() as tape:
            model.trainable = flag
            ty = model.call(tx, flag)
            costvalue = cceComputer(tt, ty)
        gradient = tape.gradient(costvalue, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        accvalue = accComputer(tt, ty)
        return costvalue, accvalue
    
    # 以下は学習のための記述．
    for epoch in range(1, maxEpochSize+1):
        trainIndex = np.random.permutation(trainSize) # トレーニングデータセットのサイズに相当する整数からなるランダムな配列を生成．
        trainCost, trainAcc = 0, 0
        for subepoch in range(minibatchNumber):
            startIndex = subepoch * minibatchSize
            endIndex = (subepoch+1) * minibatchSize
            miniTrainx = trainx[trainIndex[startIndex:endIndex]] # ランダムに決められた数だけデータを抽出する．
            miniTraint = traint[trainIndex[startIndex:endIndex]]
            miniTrainCost, miniTrainAcc = run(miniTrainx, miniTraint, True) # パラメータの更新をするのでフラッグはTrueにする．
            trainCost += miniTrainCost * minibatchNumber**(-1)
            trainAcc += miniTrainAcc * minibatchNumber**(-1)
        if epoch % 50 == 0:
            print("Epoch {:5d}: Training cost= {:.4f}, Training ACC= {:.4f}".format(epoch, trainCost, trainAcc))
            prediction = model.call(trainx, False)
            print("Prediction:", np.argmax(prediction, axis=1)) # 予測値を出力．

class Network(tf.keras.Model):
    def __init__(self, attentionUnitSize, middleUnitSize, vocabNumber, embedSize, outputSize, dropoutRate):
        super(Network, self).__init__()
        self.a = LinearAttention(attentionUnitSize, dropoutRate)
        self.embed = tf.keras.layers.Embedding(input_dim=vocabNumber, output_dim=embedSize, mask_zero=True)
        self.pe = PositionalEncoder()
        self.masker = Masker()
        self.w1 = tf.keras.layers.Dense(middleUnitSize)
        self.w2 = tf.keras.layers.Dense(outputSize)
        self.lr = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x, learningFlag):
        maskSeq, minNumber = self.masker(x)
        x = self.embed(x) # エンベッド．質的変数をエンコードする．1を[0.3 0.1 0.7]，2を[0.4 0.9 0.3]のような指定した長さの浮動小数点数からなるベクトルへ変換する行為．
        x = self.pe(x) # 位置エンコード情報の計算．
        y = self.a(x, x, maskSeq, minNumber, learningFlag)
        y = self.lr(y)
        y = self.dropout(y, training=learningFlag)
        y = self.w1(y)
        y = self.lr(y)
        y = y[:, 0, :] # 入力データの最初のトークンに相当する値だけを用いて以降の計算を行う．
        y = self.w2(y)
        y = tf.nn.softmax(y) # 3個の値の分類問題であるため．
        return y

class Attention(tf.keras.Model):
    def __init__(self, attentionUnitSize, dropoutRate):
        super(Attention, self).__init__()
        self.attentionUnitSize = attentionUnitSize
        self.queryLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.keyLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.valueLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.outputLayer = tf.keras.layers.Dense(attentionUnitSize, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropoutRate)
    def call(self, x1, x2, maskSeq, minNumber, learningFlag):
        q = self.queryLayer(x1) # クエリベクトルの生成．
        k = self.keyLayer(x2) # キーベクトルの生成．
        v = self.valueLayer(x2) # バリューベクトルの生成．
        q = q * self.attentionUnitSize**(-0.5) # 正規化のため．
        a = tf.matmul(q, k, transpose_b=True)
        a = a + tf.cast(maskSeq, dtype=tf.float32) * minNumber # 次のソフトマックス関数の操作でゼロパディングした要素の値をゼロと出力するためにとても小さい値を加える．
        a = tf.nn.softmax(a) # ゼロパティングされたところの値はゼロになる．
        y = tf.matmul(a, v)
        y = tf.keras.activations.relu(y)
        y = self.dropout(y, training=learningFlag)
        y = self.outputLayer(y)
        return y

class LinearAttention(Attention): # Attentionを継承．
    def __init__(self, attentionUnitSize, dropoutRate):
        super().__init__(attentionUnitSize, dropoutRate)
        self.elu=tf.keras.layers.ELU() # Attentionと異なる非線形関数を導入．
    def call(self, x1, x2, maskSeq, minNumber, learningFlag):
        q = self.queryLayer(x1)
        q = self.elu(q)+1
        k = self.keyLayer(x2)
        v = self.valueLayer(x2)
        q = q * x2.shape[2]**(-0.5)
        maskSeqT = tf.transpose(maskSeq, perm=[0, 2, 1])
        k = self.elu(k) + 1
        m = tf.reduce_mean(k, axis=2)
        m = tf.expand_dims(m, axis=2)
        k = tf.nn.softmax(k)
        k = k * tf.cast(~maskSeqT, dtype=tf.float32)
        v = v * tf.cast(~maskSeqT, dtype=tf.float32)
        a = tf.matmul(k, v, transpose_a=True)
        a = self.dropout(a, training=learningFlag)
        y = tf.matmul(q, a)
        y = y * m**(-1)
        y = tf.keras.activations.relu(y)
        y = self.dropout(y,training=learningFlag)
        y = self.outputLayer(y)
        return y

class Masker(tf.keras.Model):
    # 以下は入力データが質的データのときに利用するもので，ゼロパディングされたトークンの位置（と小さい値）を返すもの．
    def call(self, x):
        x = tf.cast(x, dtype=tf.int32) # 浮動小数点数を整数にする．扱うデータがラベルデータのため．入力データが浮動小数点数であるなら不要．
        inputBatchSize, inputLength = tf.unstack(tf.shape(x)) # バッチサイズと入力データの長さを取得．
        maskSeq = tf.equal(x, 0) # 元データのゼロパディングした要素をTrue，そうでない要素をFalseとしたデータを生成．
        maskSeq = tf.reshape(maskSeq, [inputBatchSize, 1, inputLength]) # データの形の整形．
        return maskSeq, x.dtype.min # 整数の最も小さい値を後で利用するため返す．

class PositionalEncoder(tf.keras.layers.Layer):
    # 以下は位置エンコードをしたデータを返すもの．
    def call(self, x):
        inputBatchSize, inputLength, inputEmbedSize = tf.unstack(tf.shape(x)) # バッチサイズ，入力データの長さ，エンベッドサイズを取得．
        j = tf.range(inputEmbedSize) // 2 * 2 # 2jの生成．2jではなくjという変数名を利用．
        j = tf.tile(tf.expand_dims(j, 0),[inputLength, 1]) # データの形の整形．
        denominator = tf.pow(float(10000), tf.cast(j/inputEmbedSize, x.dtype)) # 10000**(2j/d)の計算．
        phase = tf.cast(tf.range(inputEmbedSize)%2, x.dtype) * np.pi / 2 # 位相の計算．後でsin(90度+x)=cos(x)を利用するため．np.piは3.14ラジアン（180度）．
        phase = tf.tile(tf.expand_dims(phase, 0), [inputLength, 1]) # データの形の整形
        i = tf.range(inputLength) # iの生成．
        i = tf.cast(tf.tile(tf.expand_dims(i, 1), [1, inputEmbedSize]), x.dtype) # データの形の整形．
        encordedPosition = tf.sin(i / denominator + phase) # 位置エンコードの式の計算．
        encordedPosition = tf.tile(tf.expand_dims(encordedPosition, 0), [inputBatchSize, 1, 1]) # データの形の整形．
        return x + encordedPosition

if __name__ == "__main__":
    main()


# アテンションを利用したとき同様，しっかり正解を導き出せています．この線形アテンションが本家アテンションに性能が劣っているかどうかというと，全くそんなことはありません．問題によってはこっちの方が性能が出る場合があります．
# 
# ```shell
# .
# .
# .
# Epoch  2000: Training cost= 0.0287, Training ACC= 0.8869
# Prediction: [0 0 0 1 1 1 2 2 2]
# ```

# ```{note}
# アテンションの亜種は他にもたくさんあります．
# ```

# 線形アテンションのクラスは以下に示す通りです．上で示した式通りの実装なので説明は省略します．
# 
# ```python
# class LinearAttention(Attention): # Attentionを継承．
#     def __init__(self, attentionUnitSize, dropoutRate):
#         super().__init__(attentionUnitSize, dropoutRate)
#         self.elu=tf.keras.layers.ELU() # Attentionと異なる非線形関数を導入．
#     def call(self, x1, x2, maskSeq, minNumber, learningFlag):
#         q = self.queryLayer(x1)
#         q = self.elu(q)+1
#         k = self.keyLayer(x2)
#         v = self.valueLayer(x2)
#         q = q * x2.shape[2]**(-0.5)
#         maskSeqT = tf.transpose(maskSeq, perm=[0, 2, 1])
#         k = self.elu(k) + 1
#         m = tf.reduce_mean(k, axis=2)
#         m = tf.expand_dims(m, axis=2)
#         k = tf.nn.softmax(k)
#         k = k * tf.cast(~maskSeqT, dtype=tf.float32)
#         v = v * tf.cast(~maskSeqT, dtype=tf.float32)
#         a = tf.matmul(k, v, transpose_a=True)
#         a = self.dropout(a, training=learningFlag)
#         y = tf.matmul(q, a)
#         y = y * m**(-1)
#         y = tf.keras.activations.relu(y)
#         y = self.dropout(y,training=learningFlag)
#         y = self.outputLayer(y)
#         return y
# ```

# ```{note}
# 終わりです．
# ```
