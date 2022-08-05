#!/usr/bin/env python
# coding: utf-8

# # アテンションネットワーク

# ```{note}
# 作成中．
# ```

# ## 基本的な事柄

# この節ではアテンションがどのようなものなのか，また，アテンションの計算方法について紹介します．

# ### アテンションとは

# ### アテンションの計算方法

# アテンション
# 
# トランスフォーマー
# 
# アテンションの実装

# ## トランスフォーマー

# トランスフォーマーはアテンションの最も有用な応用先のひとつです．トランスフォーマーの構造やその構成要素を紹介します．

# ### 基本構造

# トランスフォーマーとはアテンションとポジショナルエンコードといわれる技術を用いて，再帰型ニューラルネットワークとは異なる方法で文字列を処理することができるニューラルネットワークの構造です．機械翻訳や質問応答に利用することができます．
# 
# 例えば，機械翻訳の場合，翻訳したい文字列を入力データ，翻訳結果の文字列を教師データとして利用します．構築した人工知能は翻訳したい文字列を入力値として受け取り，配列を出力します．配列の各要素は文字の個数と同じサイズのベクトル（その要素が何の文字なのかを示す確率ベクトル）です．
# 
# トランスフォーマーはエンコーダーとデコーダーという構造からなります．エンコーダーは配列（機械翻訳の場合，翻訳したい配列）を入力にして，同じ長さの配列を出力します．デコーダーも配列（機械翻訳の場合，翻訳で得たい配列）とエンコーダーが出力した配列を入力にして同じ長さの配列（各要素は確率ベクトル）を出力します．エンコーダーが出力した配列情報をデコーダーで処理する際にアテンションが利用されます．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/transformer.svg?raw=1" width="100%" />
# 
# エンコーダーとデコーダー間のアテンション以外にも，エンコーダーとデコーダーの内部でもそれぞれアテンション（セルフアテンション）が計算されます．アテンションは文字列内における文字の関連性を計算します．
# 
# トランスフォーマーは再帰型ニューラルネットワークで行うような文字の逐次的な処理が不要です．よって，計算機の並列化性能をより引き出せます．扱える文脈の長さも無限です（再帰型ニューラルネットワークでも理論上無限です．）．
# 
# このトランスフォーマーはものすごい性能を発揮しており，これまでに作られてきた様々な構造を過去のものとしました．特に応用の範囲が広いのはトランスフォーマーのエンコーダーの部分です．BERT と呼ばれる方法を筆頭に自然言語からなる配列を入力にして何らかの分散表現を出力する方法として自然言語処理に関わる様々な研究開発に利用されています．
# 
# （会話でトランスフォーマーという場合は，トランスフォーマーのエンコーダーまたはデコーダーのことを言っている場合があります．エンコーダー・デコーダー，エンコーダー，デコーダー，この3個でそれぞれできることが異なります．）

# ```{hint}
# 実用上，配列を入力にして配列を返す構造とだけ覚えておけば問題はないと思います．
# ```

# ### 構成要素

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
