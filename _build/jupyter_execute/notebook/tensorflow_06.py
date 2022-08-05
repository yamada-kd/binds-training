#!/usr/bin/env python
# coding: utf-8

# # アテンションネットワーク

# ```{note}
# 作成中．
# ```

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
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/selfAttention.svg?raw=1" width="" />

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

# ### 普通のアテンション

# ### 線形計算量のアテンション

# ```{note}
# 終わりです．
# ```
