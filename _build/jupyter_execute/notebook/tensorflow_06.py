#!/usr/bin/env python
# coding: utf-8

# # アテンションネットワーク

# ```{note}
# 作成中．
# ```

# ## 基本的な事柄

# ### アテンション

# アテンション
# 
# トランスフォーマー
# 
# アテンションの実装

# ### トランスフォーマー

# トランスフォーマーとはアテンションとポジショナルエンコードといわれる技術を用いて，再帰型ニューラルネットワークとは異なる方法で文字列を処理することができるニューラルネットワークの構造です．機械翻訳や質問応答に利用することができます．
# 
# 例えば，機械翻訳の場合，翻訳したい文字列を入力データ，翻訳結果の文字列を教師データとして利用します．構築した人工知能は翻訳したい文字列を入力値として受け取り，配列を出力します．配列の各要素は文字の個数と同じサイズのベクトル（その要素が何の文字なのかを示す確率ベクトル）です．
# 
# トランスフォーマーはエンコーダーとデコーダーという構造からなります．エンコーダーは配列（機械翻訳の場合，翻訳したい配列）を入力にして，同じ長さの配列を出力します．デコーダーも配列（機械翻訳の場合，翻訳で得たい配列）とエンコーダーが出力した配列を入力にして同じ長さの配列（各要素は確率ベクトル）を出力します．エンコーダーが出力した配列情報をデコーダーで処理する際にアテンションという技術が利用されます．
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

# ### できること

# Hugging Face で扱うことができるタスクは以下に示すものがあります．これ以外にもありますが自然言語処理に関する代表的なタスクを抽出しました．括弧内の文字列は実際に Hugging Face を利用する際に指定するオプションです（後で利用します．）．

#     
# 
# *   感情分析（`sentiment-analysis`）：入力した文章が有する感情を予測
# *   特徴抽出（`feature-extraction`）：入力した文章をその特徴を示すベクトルに変換
# *   穴埋め（`fill-mask`）：文章中のマスクされた単語を予測
# *   固有表現抽出（`ner`）：入力した文章中の固有表現（名前とか場所とか）にラベルをつける
# *   質問応答（`question-answering`）：質問文とその答えが含まれる何らかの説明文を入力として解答文を生成
# *   要約（`summarization`）：入力した文章を要約
# *   文章生成（`text-generation`）：文章を入力にして，その文章に続く文章を生成
# *   翻訳（`translation`）：文章を他の言語に翻訳
# *   ゼロショット文章分類（`zero-shot-classification`）：文章とそれが属する可能性があるいくつかのカテゴリを入力にしてその文章をひとつのカテゴリに分類
#     
#     

# ### Hugging Face Course

# Hugging Face の利用方法は以下のウェブサイト，Hugging Face Course の Transformer models の部分を読めば大体のことが把握できると思います．
# 
# https://huggingface.co/course/

# ```{note}
# 終わりです．
# ```
