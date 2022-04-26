#!/usr/bin/env python
# coding: utf-8

# # 機械学習とその周辺事項

# ## データ科学と機械学習

# ### データ科学

# データ科学という用語が意味するものはとても曖昧で漠然としています．データ科学は第4の科学とかって言われていますがこれまでの科学（理論科学，実験科学，計算機科学）でもデータ使ってましたよね？一方で，半導体技術の向上やインターネットの繁栄により大量のデータを扱えるようになると共にデータ科学が流行を迎えたという背景を考慮すると，従来の科学とデータサイエンスの差異は扱うデータの量にあるという側面はありそうではあります．すなわち，データ科学は大量のデータを処理し，そこに内包される知見を抽出する学問という側面を少なくとも持つのかもしれません．そのような観点において，従来の科学で用いられてきた紙とペン，また，表計算ソフトウェアはデータ科学で用いられる解析手法としては役者不足です．データ科学分野において，大量のデータに内包される知見を抽出する手段として用いられるものは機械学習法に他なりません．

# ### 機械学習

# 機械学習法は人工知能を作り上げる方法です．現在までに人類が開発に成功し活用している人工知能は特化型人工知能と呼ばれる種類の人工知能です．それは，「何らかの入力に対して何らかの出力をするアルゴリズム」と「そのアルゴリズムが有するパラメータ」，このふたつの要素によって構成されます．人工知能を教育する過程では，まっさらな人工知能に対して（通常）何度も何度もデータを繰り返し入力し，このパラメータを最適なものへと変化させます．何らかの入力に対して何らかの出力をするアルゴリズムには様々なものがあります．深層学習で主に利用されるニューラルネットワークやその他にも決定木，サポートベクトルマシン，主成分分析法等の様々な手法があります．次節では機械学習に関する用語の説明をします．

# ## 機械学習の専門用語の説明

# ### 予測器と人工知能

# 予測器と人工知能は同じものを指します．機械学習法によって成長させる対象のことであり，機械学習を行った際の最終産物です．本体は「何らかの入力に対して何らかの出力をするアルゴリズム」と「そのアルゴリズムが必要とするパラメータ（母数）」です．このような特定の問題を解決するための人工知能は特化型人工知能といいます．

# ```{note}
# アトムやドラえもんみたいな汎用的に問題を解決できる存在のことは汎用人工知能といいます．また，精神を宿したような存在のことをジョン・サールという人は強い人工知能と表現しましたが，アトムとかドラえもんは汎用人工知能であり強い人工知能です．人が思い描く人工知能とはこういった存在であって，このような単なる予測器を人工知能と呼ぶことに抵抗を持つ人は多いですが，特化型人工知能と正確に呼べば納得してもらえるかもしれません．
# ```

# ```{note}
# 人工知能という言葉は色々なものを再定義しており，ニューラルネットワークを人工知能というのならば，高校数学で習得する最小二乗法を用いて求める線形回帰直線も人工知能です．
# ```

# ### インスタンス

# データセットにあるひとつのデータのことです．分野によってデータポイントと呼ばれることがありますが，時系列データを扱う際には別の意味でデータポイントという単語が用いられることがあり，インスタンスを使った方がより正確かもしれません．

# ### 学習

# 人工知能にデータを読ませ成長させる過程を学習と言います．

# ### 回帰問題と分類問題

# 人工知能を利用して解決しようとする問題は大きく分けてふたつあります．ひとつは回帰問題です．回帰問題は人工知能に何らかの実数の値を出力させる問題です．一方で，分類問題は人工知能にクラスを出力させる問題です．クラスとは「A，B，C」とか「良い，悪い」のような何らかの分類のことです．各クラスの間に実数と他の実数が持つような大小の関係性は定義されません．これは例えば，「0という手書きの数字が書かれている画像」と「1という手書きの数字が書かれている画像」を分類するというような場合においても当てはまります．このような問題の場合，人工知能の最終的な出力は0または1でありますが，これは人工知能にとって単なるシンボルであって，人工知能は「0という手書きの数字が書かれている画像」が「1という手書きの数字が書かれている画像」より小さいから0という出力をするのではなく，単に「0という手書きの数字が書かれている画像」にある0というパターンが0というシンボルと類似しており，1というシンボルと類似していないため，0を出力するに過ぎません．

# ### 入力ベクトル，ターゲットベクトル，出力ベクトル

# 入力ベクトルとは人工知能を成長させるため，または，開発した人工知能に予測をさせるために入力するデータです．入力ベクトルの各インスタンスは基本的にはベクトルの型式をしているため入力ベクトルと呼ばれます（本当は入力データ，ターゲットデータ，出力データと呼んだ方が良いかもしれないです）．インプットベクトルとも呼ばれます．ターゲットベクトルは教師あり学習の際に入力ベクトルとペアになっているベクトルです．1次元からなるベクトルは，実質スカラですが，ベクトルと呼びます．教師ベクトルとも呼ばれます．出力ベクトルは入力ベクトルを人工知能に処理させたときに出力されるベクトルです．教師あり学習の場合，このベクトルの値がターゲットベクトルに似ているほど良いです（人工知能の性能が）．

# ### エポック

# 用意したトレーニングデータセット（パラメータ更新にのみ用いるデータセット）の全部を人工知能が処理する時間の単位です．1エポックだと全データを人工知能が舐めたことになります．2エポックだと2回ほど人工知能が全データを舐めたことになります．学習の過程では繰り返しデータを人工知能に読ませるのです．

# ### 損失（ロス）と損失関数（ロス関数）とコスト関数

# 人工知能が出力する値とターゲット（教師）の値がどれだけ似ていないかを表す指標です．これが小さいほど，人工知能はターゲットに近い値を出力できることになります．よって，この損失を小さくすることが学習の目標です．損失を計算するための関数を損失関数と言います．また，コストは損失と似ているものですが，正則化項（人工知能の過剰適合を防ぐために用いる値）を損失に加えた場合の小さくする目標の関数です．それを計算する関数をコスト関数と言います．損失関数とコスト関数はどちらも場合によって学習の目的関数となり得ます．

# ### 正確度と精度

# 正確度とは英語では accuracy と記述されます．略記で ACC とも表現されます．真陽性，偽陽性，偽陰性，真陰性をそれぞれ $a$，$b$，$c$，$d$ とするとき正確度 $u$ は以下の式で定義されます：
# 
# $u=\displaystyle\frac{a+d}{a+b+c+d}$．
# 
# また，精度は英語では precision と記述されます．これは陽性的中率のことであって英語では positive predictive value であり，PPV と略記されます．精度 $v$ は以下の式で定義されます：
# 
# $v=\displaystyle\frac{a}{a+b}$．
# 
# つまり，これは陽性と予想した場合の数に対して本当に陽性であった場合の数の割合です．人工知能の性能を表現するために精度という単語を使う人がたくさんいます．精度は人工知能を評価するひとつの指標に過ぎません．しかも，この指標は正確度や MCC や F1 と比較して頑健な評価指標ではなく，人工知能の開発者が自由に調整できてしまう値です．これだけで人工知能の性能は評価不可能です．科学的な文脈においては，精度という単語を使いすぎないように気を付ける必要があります（多くの場合，性能というべきでしょう）．その他の指標は以下の Wikipedia が詳しいです．
# 
# https://en.wikipedia.org/wiki/Precision_and_recall

# 

# パラメータとハイパーパラメータ**
# 
# パラメータは人工知能を構成する一部です．機械学習法で成長させられるアルゴリズムは何らかの入力に対して何らかの出力をしますが，このアルゴリズムがその計算をするために必要とする値を持っている場合があり，それをパラメータと言います．機械学習ではアルゴリズム自体は変化させずに，パラメータを変化させることによって良い人工知能を作ろうとします．深層学習界隈では最急降下法によってパラメータは更新される場合が多いです．また，学習によって更新されるパラメータとは異なり，学習の前に人間が決めなければならないパラメータがありますが，これをハイパーパラメータと言います．ニューラルネットワークでは，活性化関数の種類であったり，各層のニューロンの数であったり，層の数であったりします．このハイパーパラメータすらも自動で最適化しようとする方法もあり，そういう方法を network architecture search（NAS）と言います．
# 
# **★二乗誤差**
# 
# あるベクトルとあるベクトルの距離です．
# 
# **★ソフトマックス**
# 
# 要素を合計すると1になるベクトルタイプのデータです．各要素の最小値は0です．これを出力する関数をソフトマックス関数と言います．入力ベクトルに対して入力ベクトルと同じ要素次元数のベクトル（ソフトマックス）を出力します．
# 
# **★一括学習，逐次学習，ミニバッチ学習**
# 
# 一括学習はバッチ学習とも呼ばれます．人工知能を成長させるときにデータを全部読み込ませた後に初めてパラメータを更新します．逐次学習はオンライン学習とも呼ばれます．人工知能を成長させるときにデータを1個ずつ読み込ませ，その都度パラメータを更新する方法です．ミニバッチ学習は全データからあらかじめ決めた分のデータを読み込ませ，その度にパラメータを更新する方法です．ミニバッチ学習のミニバッチサイズが全データサイズと等しいならそれは一括学習で，ミニバッチサイズが1なら逐次学習です．
# 
# **★早期終了**
# 
# 人工知能を成長させるときに，過学習（データへの過剰適合）が起こる前に学習を停止させることです．最もナイーブには，学習の最中に最も良い性能を示した値を記録しておき，その値を $n$ 回連続で更新しなかった場合に学習を打ち切る方法があります．この $n$ 回のことを patience と言います．
# 
# **★オプティマイザ**
# 
# 最急降下法は最もナイーブな最適化法のひとつです．学習をより良く（学習の速さを良くするとか局所解に陥り難くするとか）進めるために様々な最適化法が考案されています．深層学習で用いられる最適化法は基本的には最急降下法を改善した方法です．そういったものをオプティマイザと言います．ここでは，世界で最も利用されているオプティマイザである Adam を紹介します．最初に，最急降下法の式を紹介します．
# 
# $\theta_{t+1}=\theta_t-\alpha g_t$
# 
# これに対して，世界で最も利用されているオプティマイザである Adam は以下のような更新式で定義されます．
# 
# $m_{t+1}=\beta_{1} m_t+(1-\beta_{1})g_t$
# 
# $v_{t+1}=\beta_{2} v_t+(1-\beta_{2})g_t^2$
# 
# $\hat{m}=\displaystyle\frac{m_{t+1}}{1-\beta_{1}^{t+1}}$
# 
# $\hat{v}=\displaystyle\frac{v_{t+1}}{1-\beta_{2}^{t+1}}$
# 
# $h=\alpha\displaystyle\frac{\hat{m}}{\sqrt{\hat{v}-\epsilon}}$
# 
# $\theta_{t+1}=\theta_t-h$
# 
# ハイパーパラメータは以下の4つです．
# 
# $\alpha=0.001$，$\beta_{1}=0.9$，$\beta_{2}=0.999$，$\epsilon=10^{-6}$
# 
# また，Adam のパラメータの初期値として以下の値を用います．
# 
# $m_0=0$，$v_0=0$
# 
# Adam がやっていることは，勾配の平均値と分散の値を保存して，その値をうまく利用することで学習率を調整することです．例えば，下に凸である二次関数の曲線で言うと，極小値付近では勾配の値は正の値と負の値を行ったり来たりするはずです．そのような際には何ステップかに渡り記録した勾配の分散の値は大きくなるはずです．また，極小値付近においては勾配の値は小さな値となるはずです．よって，分母を勾配の分散にして，分子を勾配の平均値にした値は小さい値になります．これに対して学習率を掛けると得られる値は小さくなります．これが Adam における更新値であり $h$ です．極小値付近ではパラメータ更新が緩やかになります．
# 
# <img src="https://drive.google.com/uc?id=1eiSgnD5Y5HPHrgpIj5R4BCTifuKu77LK">
# 
# これは停留点の中でも厄介な鞍点から，洗練されたオプティマイザがいかに早く抜け出せるかを動画にしたものです．
# 
# 出典：https://rnrahman.com/
# 

# ##2-1. <font color="Crimson">機械学習をする際の心構え</font>

# ###2-1-1. <font color="Crimson">データ分割の大切さ</font>

# 機械学習を行う際の最も大切なことであるデータセットの構築方法を紹介します．データセットは以下のように分割します．はじめに，元の全データを「学習セット」と「テストセット」に分けます．さらに，学習セットを人工知能の成長（アルゴリズムが持つパラメータの更新）のためだけに利用する「トレーニングセット」と学習が問題なく進んでいるか（過学習や未学習がおこっていないか）を確認するために利用する「バリデーションセット」に分けます．

# <img src="https://drive.google.com/uc?id=1rL7CPuXUa1MTM8MLWUF4_Xez4d9LSWjJ" width="70%">

# 学習の際には学習セットだけの結果を観察します．学習の際に一瞬でもテストセットにおける予測器の性能を観察するべきではありません．また，「独立であること」もしっかり定義すべきです．データが互いに独立であるとは，機械学習の文脈においては「互いに同じ分布に従うことが予想されるが，その生成過程が異なること」でしょうか．MNIST の学習セットとテストセットには，同一の手書き数字の提供者が同時に含まれている可能性があり，この観点からするとそれらは互いに独立なデータセットではないのかもしれません．

# 新たな人工知能を開発してそれを公開する場合，普通，その人工知能の性能をベンチマークし論文や学会で発表します．その性能を評価するために使うデータセットがテストセットです．もしこのテストセットが学習セットと同じような性質を持っているのであれば，その人工知能は新たなデータを処理するときに期待されるような性能を発揮できないかもしれません．学習セットとテストセットが独立でないのならば，学習セットに過剰適合させた人工知能では，あたかもとても良い性能をしているように見えてしまいます．人工知能の実際の現場での活用方法を鑑みるに，人工知能はある特定のデータにだけではなく，様々なデータに対応できる汎化能力を持たなければなりません．

# 人工知能の開発の研究に関して，その全行程において，データセットの正しい分割は何よりも大切なことです．従来からの科学の分野で研究を続けている人達にとって，機械学習に対する信頼度は未だ低い場合があります．機械学習をやる人がここを適当にやり続ける限り，「機械学習なんて眉唾」みたいな意見はなくならないことをご理解ください．

# <font color="Crimson">(9｀･ω･)9 ｡oO(終わりです．)</font>
