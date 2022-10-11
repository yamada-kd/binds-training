#!/usr/bin/env python
# coding: utf-8

# # 演習問題

# ## プログラミングの基礎
# 
# 

# 「プログラミングの基礎」で触れた事柄に関する問題集です．講義資料に載っていない事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 1-1 行列の掛け算

# 以下の行列の掛け算を NumPy 等のライブラリを用いずに計算するためのプログラムを書いてください．
# 
# $
#   \left[
#     \begin{array}{ccc}
#       1 & 2 & 3 \\
#       3 & 4 & 5\\
#     \end{array}
#   \right]
#   \times
#   \left[
#     \begin{array}{cc}
#       5 & 6 \\
#       6 & 7 \\
#       7 & 8 \\
#     \end{array}
#   \right]
# $

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{note}
# 行列の掛け算なんてライブラリ使わずにやることなんてないからなんて不毛な問題なんだ，と思うかもしれないのですが，プログラミング初心者がこれをできるようになるということは `for` の使い方と動き方をしっかり理解していることの確認となり，その観点からは有意義です．
# ```

# ### 1-2 ファイルの処理

# シェルのコマンドに `wget` というものがありますが，これを使うとインターネット上から情報を取得することができます．Wikipedia の「334」という項目は以下の URL に載っています．
# 
# https://ja.wikipedia.org/wiki/334
# 
# この URL にアクセスして表示されるテキストの中に `334` という値が何個あるか数えるプログラムを作ってください．

# 最初に，`wget` で URL にアクセスしてその情報全部を `wikipedia-334.txt` というファイル名で現在のディレクトリに保存してください．

# In[ ]:


# ここにシェルのコマンドを書く．


# ```{hint}
# オプションコマンド `-O` を利用するとファイル名を指定して保存できます．
# ```

# 次に，このファイルを Python で読んで `334` という文字列が何個あるかカウントして，カウントした結果だけを出力してください．

# In[ ]:


#!/usr/bin/env python3

def main():
    # ここにプログラムを書く．

if __name__ == "__main__":
    main()


# ```{hint}
# ファイルがどこにあるかわからない人は「カレントディレクトリ」という概念を調べてみましょう．
# ```

# ## scikit-learn を利用した機械学習
# 
# 

# 「scikit-learn を利用した機械学習」で触れた事柄に関する問題集です．これも講義資料に載っていない事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 2-1 探索的データ解析

# 探索的データ解析とはこれから利用しようとするデータセットを様々な角度から観察してデータの性質を把握しようとする行為のことです．英語では exploratory data analysis（EDA）と呼ばれます．

# In[ ]:





# ### 2-2 ほげ

# 

# In[ ]:





# ### 2-3 ほげ

# 

# In[ ]:





# ## 深層学習
# 
# 

# 「深層学習」で触れた事柄に関する問題集です．またしても講義資料に載っていない事柄に取り組みますが，現実世界でプログラミングはそういう課題に取り組むことではじめて実力がつきます．

# ### 3-1 ほげ

# 

# In[ ]:





# ### 3-2 ほげ

# 

# In[ ]:





# ### 3-3 ほげ

# 

# In[ ]:





# ### 3-4 ほげ

# 

# In[ ]:





# ### 3-5 ほげ

# 

# In[ ]:





# ```{note}
# 終わりです．
# ```
