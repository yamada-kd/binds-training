#!/usr/bin/env python
# coding: utf-8

# # Python 入門

# ## はじめに
# 

# ### このコンテンツで学ぶこと

# このコンテンツの目的は，ウェブベースの計算環境である Jupyter Notebook（このウェブページを形作っているもの）を利用して，Python の基本的な動作を習得することです．このコンテンツは東北大学大学院情報科学研究のプログラミング初学者向けの授業「ビッグデータスキルアップ演習（Big Data Skillup Training）」の内容の一部を日本語の e-learning コンテンツとして再構築したものです．

# ### この環境について

# #### Jupyter Notebook
# 

# Jupyter Notebook は Python を実行するための環境です．メモを取りながら Python のコードを実行することができます．この環境は，Python プログラムがコマンドライン上で実行される実際の環境とは少し異なるのですが，Python プログラムがどのように動くかということを簡単に確認しながら学習することができます．

# ### 1-2-2. <font color="Crimson">GPU の利用方法</font>
# 

# この環境で GPU を利用するには上のメニューの「ランタイム」から「ランタイムのタイプを変更」と進み，「ハードウェアアクセラレータ」の「GPU」を選択します．

# ##1-3. <font color="Crimson">開始前に行うこと（重要）</font>

# <font color="Crimson">上の（このグーグルコラボラトリー自体の一番上の）「ファイル」をクリックし，さらにポップアップで出てくる項目から「ドライブにコピーを保存」をクリックし，自身のグーグルドライブにこのウェブページ全体のソースを保存します（グーグルのアカウントが必要です）．</font>こうすることによって，自分で書いたプログラムを実行することができるようになります．また，メモ等を自由に以下のスペースに追加することができるようになります．

# ##1-4. <font color="Crimson">コンパニオン</font>

# このコンテンツには，受講者の理解を助けるためのコンパニオンがいます．以下のものは「サミー先生」です．サミー先生はこの教材の英語版の制作者です．分かり難いところを詳しく教えてくれます．
# 
# <font color="Crimson">(9｀･ω･)9 ｡oO(ちゃお！)</font>

# ##1-5. <font color="Crimson">利用方法</font>
# 

# ### 1-5-1. <font color="Crimson">進め方</font>

# 上から順番に読み進めます．Python のコードが書かれている場合は実行ボタンをクリックして実行します．コード内の値を変えたり，関数を変えたりして挙動を確かめてみてください．

# ### 1-5-2. <font color="Crimson">コードセル</font>
# 

# コードセルとは，Python のコードを書き込み実行するためのセルです．以下のような灰色のボックスです．ここにコードを書きます．実行はコードセルの左に表示される「実行ボタン」をクリックするか，コードセルを選択した状態で `Ctrl + Enter` を押します．環境によっては行番号が表示されていると思いますので注意してください（行番号の数字はプログラムの構成要素ではありません）．
# 
# 

# In[1]:


print("This is a code cell.")


# #2. <font color="Crimson">Python の基本的な使用方法</font>

# ##2-1. <font color="Crimson">Python とは</font>

# Python は，科学技術界隈，特にデータ科学に関する分野において昨今，最も利用されていると言って過言ではないプログラミング言語です．これを習得することで今後，プログラミング技術なしでは為し得なかったような大量データの解析ができるようになります．Python は C言語とか Fortran とかの伝統的なプログラミング言語と比べて簡単に学ぶことができるプログラミング言語です．ものすごく人気がある言語であり，たくさんのライブラリ（補助ツール）が存在しているので，色々なことができます．計算の速さもそこそこで，少なくとも悪くはありません．ここでは，Python のバージョン3（Python3）の使い方を学びます．

# ##2-2. <font color="Crimson">Python の実行方法</font>

# ### 2-2-1. <font color="Crimson">実行の手順</font>

# 普通，Python は以下のような手順で使います．
# 
# 1.   プログラムを書きます．プログラムの本体をソースコードと呼びます．
# 2.   書いたソースコードを保存します．拡張子は「.py」とします．
# 3.   そのプログラムを（コマンドライン等から）呼び出します．
# 

# <font color="Crimson">(9｀･ω･)9 ｡oO(このチュートリアルでは，グーグルコラボラトリーを利用するので，2番と3番はやりません（やれません）．)</font>

# ### 2-2-2. <font color="Crimson">Python プログラムの構造</font>

# 以下のようなプログラムを書いて，それを実行するとターミナルウィンドウ等に「Hello world」という表示が出力されます．以下のプログラムにおいて，1～3と5～7行は Python にとって常に必要になる表記です．これは，いかなるときにも取り敢えず書いておくと良いでしょう．なので，以下のプログラムの本体は実質，4行目だけです．これは，「Hello world」という文を画面に表示しなさいという命令です．`print()` という表記の前に存在する空白は，キーボードの `Tab` を押すと入力できる空白文字です．Pythonではこのように空白文字をきちんと入れる必要があります．`def main():` 以下に書いた，タブの右側の表記だけがプログラムとして認識されます．

# In[2]:


#!/usr/bin/env python3

def main():
    print("Hello world")
     
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(実は `print("Hello world")` だけでコードは問題なく動きます．特にこの Google Colaboratory を使う場合，コードセルで実行した結果はメモリ上に保存されるので寧ろコマンドだけ書いた方が良い場合もあります．Google Colaboratory は本来そのようなプログラミング環境として利用することを想定されているのかもしれません．しかし，今回の実習の目的は最終的には Python プログラムを自身の計算機で動かせるようになることとしているので，この書き方を続けます．このようにすることで，コードセルをコピーして別の環境に持って行ったとしても問題なく利用できます．)</font>

# <font color="Crimson">(9｀･ω･)9 ｡oO(Python コードをプログラムファイルとして実行した場合，`__name__` という変数に `"__main__"` という文字列が割り当てられます．一方で，`hoge.py` という名前の Python コードをライブラリとして別のプログラムから `import hoge` で呼び出した場合，`__name__` という変数に `"hoge"` が割り当てられます．つまり，`if __name__ == "__main__":` を書かないコードを別のファイルから呼び出して使おうとすると，望んでもいないのにそのコードの全部がインポート時に勝手に実行されてしまうという弊害が起きてしまうのです．)</font>

# ### 2-2-3. <font color="Crimson">Python コーディング</font>

# Python でコーディングをする際に統一的なルール（エチケット？） として PEP8 が定義されています．PEP とは Python Enhancement Proposal の略です．PEP8 に従って書かれたコードは可読性が高くなります．
# 
# *   https://pep8-ja.readthedocs.io/ja/latest/
# 
# このコンテンツのコードは必ずしも PEP8 に従っていないかもしれません．
# 

# <font color="Crimson">(9｀･ω･)9 ｡oO(Python に愛を持つ人はひたすら読み込みましょう！)</font>

# ##2-3. <font color="Crimson">Python の基本操作</font>

# 以降，実際にプログラムを書いてそれらを実行してみます．

# ###2-3-1. <font color="Crimson">四則計算</font>

# 以下のように書くと足し算ができます．プログラム中の「#」はコメントの宣言のための記号で，これが書いてある部分から右側はプログラムとして認識されません．

# In[3]:


#!/usr/bin/env python3
 
def main():
    print(3 + 15) # addition
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(ハッシュとシャープは実は違うものですよね．)</font>

# その他の四則計算は以下のように書きます．

# In[4]:


#!/usr/bin/env python3
 
def main():
    print(10 - 2) # subtraction
    print(90 / 3) # division
    print(51 * 5) # multiplication
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(コードセルを追加して何か書いてみましょう！)</font>

# ###2-3-2. <font color="Crimson">変数</font>

# 上では，実際の数値を入力してプログラムを書きました．しかし，これではただの電卓です．以下のようにすると，変数を定義することができます．例えば，4行目で「a」という変数を生成し，これに「3」という数値を代入しています．ここで注意しなければならないことは，プログラミング言語において「=」は「その右側と左側が等しい」という意味ではなく，「右側を左側に代入する」という意味で用いられる点です．このように変数というものを使うことでプログラムは色々な処理を一般化することができます．

# In[5]:


#!/usr/bin/env python3
 
def main():
    a = 3 # assign a value "3" to a variable "a"  ==
    b = 18
    print(a + b) # addition
    print(a - b) # subtraction
    print(b / a) # division
    print(a * b) # multiplication
 
if __name__ == "__main__":
    main()


# 変数として使うことができるものは英語アルファベットの中の一文字だけではありません．以下のように文字列を使用することができます．

# In[6]:


#!/usr/bin/env python3
 
def main():
    aaa = 5
 
if __name__ == "__main__":
    main()


# 変数として使用できる文字は以下のものを含みます．
# 
# *   英語アルファベットに含まれる文字
# *   数字
# *   _ (アンダースコア)
# 
# しかし以下のようなことは禁止事項です．
# 
# *   変数の最初の文字が数字であること．
# *   予約語と等しい文字列．例えば，「print」という変数を使おうとしても使えません．これは Python が「print」という関数を備えているためです．
# 
# 変数には数値だけでなく，文字列や文字も代入することができます．以下のようにすると，「greeting」という変数が「print()」という関数に処理されて画面に表示されますが，「greeting」に代入された値は「Hello world」なので，画面に表示される文字列は同じく「Hello world」となります．

# In[7]:


#!/usr/bin/env python3
 
def main():
    greeting = "Hello world"
    print(greeting)
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(Python では数値と文字列は明確に区別されます．数値を変数に代入したときは `"` を利用しませんでした．`"` に囲まれた部分は文字列としてみなされます．)</font>

# ダブルクォーテーション（`"`）に囲まれた部分には何を書いてもそれは文字列とみなされます．よって，上の例で `"greeting"` と書いてその内容（`Hello world`）を出力しようとしても，`greeting` という文字列が出力されるだけで中身を出力できません．目的の動作をさせるには変数の中身をダブルクォーテション内で「展開」しなければならないのです．変数を文字列の中で展開したい場合は `{` と `}` と `format()` を組み合わせて使います．

# In[8]:


#!/usr/bin/env python3
 
def main():
    name = "Ken"
    age = 6
    print("{0}, which is {1} years old is now sitting.".format(name, age))
 
if __name__ == "__main__":
    main()


# ###2-3-3. <font color="Crimson">リストとディクショナリ</font>

# 複数個の値を代入することができる変数があります．これのことをただの変数ではなく，Python ではリスト（list）と呼びます．リスト変数は `[` と `]` と `,` を使って生成します．以下のように生成した変数「list_a」の要素にアクセスするには「list_a[0]」のような感じでリスト変数に添字を付けて書きます．こうすることで「list_a」に格納されている最初の値である「10」にアクセスできます．多くのプログラミング言語ではリストの一番最初を示すインデックスは「0」です（1ではありません．）．こういうシステムのことはゼロオリジンと呼称します．

# In[9]:


#!/usr/bin/env python3
 
def main():
    list_a = [10, 4, "aaa"] # declaration of a variable, list_a, this list variable contains three values
    print(list_a[0]) # access first element of list_a
    print(list_a[1]) # access second element of list_a
    print(list_a[2]) # access third element of list_a
 
if __name__ == "__main__":
    main()


# Python のリストでは上のように数値と文字（列）をひとつのリストに同時に入れることができます．リストを使うことが可能な多くのプログラミング言語ではこのような操作はできない場合が多いです．

# また，以下のような特別な変数も存在します．これをディクショナリ（dictionary）と呼びます．ディクショナリ変数は `{` と `}` と `,` と `:` を使って生成します．ディクショナリ変数には，キー（key）とそれに対応する値（value）を一組にして代入します．その後，キーの値を使って以下のようにしてディクショナリ変数にアクセスすることで，そのキーに対応する値を取得することができます．

# In[10]:


#!/usr/bin/env python3
 
def main():
    dict_a = {"January" : "1", "February" : "2"} # declaration of dictionary, dict_a. this dictionary contains two keys and two values
    print(dict_a["January"]) # access a value corresponding to a key "January"
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(便利ですね．)</font>

# ###2-3-4. <font color="Crimson">繰り返し処理</font>

# Python で繰り返し処理をするには「for」を使います．以下のように書くと，`range(5)` に格納されている「0，1，2，3，4」という値が一行ごとに表示されます．プログラムではこの「for」を様々な局面で利用することによって人間ではできないような同じことの繰り返し作業を実現します．この繰り返し処理の方法と後に出る条件分岐の方法（if）さえ用いればどのようなプログラムでも実現することができます．繰り返し操作が終わった次の行からは，インデントを再びアウトデントするのを忘れないよう注意しましょう．

# In[11]:


#!/usr/bin/env python3
 
def main():
    for i in [0,1,2,3,4]:
        print(i)
    print("hoge")
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(`range(5)` は `[0，1，2，3，4]` の配列を生成するものと考えて問題ないです．)</font>

# 以下のように書くとリストに格納されているデータ全てにひとつずつアクセスすることができます．

# In[12]:


#!/usr/bin/env python3
 
def main():
    list_a = [10, 4, "aaa"] # declaration of variable list_a
    for w in list_a: # repetitive access of list_a. w represents an element.
        print(w) # when you start a "function" you have to insert [tab] (and when you finish the function, also you need outdent)
 
if __name__ == "__main__":
    main()


# ディクショナリに対する繰り返しのアクセスは以下のようにします．

# In[13]:


#!/usr/bin/env python3
 
def main():
    dict_a={"January" : "1", "February" : "2"}
    for k in dict_a.keys(): # dict_a.keys() is a list containing only keys of "dict_a". or you can write simply "dict_a" instead of "dict_a.keys()"
        print(k, dict_a[k]) # k is a key and dict_a[k] is a value corresponding to the key.
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(ディクショナリ変数に `.keys()` をつけることでディクショナリ変数に含まれるキーからなるリストを生成することができます．)</font>

# 以下のように書いても，上と全く同じ挙動を示します．

# In[14]:


#!/usr/bin/env python3
 
def main():
    dict_a={"January" : "1", "February" : "2"}
    for k, v in dict_a.items():
        print(k, v) # k is a key and dict_a[k] is a value corresponding to the key.
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(この場合，`.items()` によってキーと値のペアが同時に抜き出されているのです．変数の内容を変更して，挙動を確認してみましょう！)</font>

# 繰り返し処理をする方法にはその他に，「while」があります．以下のようにして使います．「while」の右側に書いた条件が満たされる場合のみ，それ以下の処理を繰り返すというものです．この場合，i という変数が10より小さいとき処理が繰り返されます．そして，この「i」の値は「while」が1回実行されるたびに1ずつインクリメント（増加）されています．その表記が，「i = i + 1」です．これは，「i」に「iに1を足した値」を代入しろという意味です．なので，この表記の前の i に比べて，表記の後の i は1だけ大きい値です．

# In[15]:


#!/usr/bin/env python3
 
def main():
    i = 0
    while i < 10: # If the condition "i < 10" is satisfied,
        print(i) 
        i = i + 1 # these two lines are processed.
 
if __name__ == "__main__":
    main()


# また，while は以下のようにして用いることもできます．この「!=」は「この記号の右側と左側の値が等しくない」ということを意味します．

# In[16]:


#!/usr/bin/env python3
 
def main():
    i = 0
    while i != 10:
        print(i)
        i = i + 1
 
if __name__ == "__main__":
    main()


# ###2-3-5. <font color="Crimson">条件分岐</font>

# プログラミング言語にとって重要な要素として「if」があります．これは条件分岐のためのシステムです．以下のように書くと，もし `i` の値が2であるときのみ出力がされます．ここで，`==` という記号は「記号の右側と左側の値が等しい」ということを意味し，現実世界の「=」と同義です．

# In[17]:


#!/usr/bin/env python3
 
def main():
    for i in range(5):
        if i == 2:
            print(i)
 
if __name__ == "__main__":
    main()


# 繰り返し処理と条件分岐を組み合わせることで複雑な動作を実現することができます．以下のように書くと，もしキーが「January」であった場合，「1」が表示されます．それ以外でもし，「ディクショナリ変数の値が2」であった場合，`February corresponds to 2 in dict_a.` と表示されます．それら以外の全ての場合おいては，`key is not January or February.` と表示されます．

# In[18]:


#!/usr/bin/env python3
 
def main():
    dict_a={"January" : "1", "February" : "2", "May" : "5"}
    for k in dict_a.keys():
        if k == "January": # a condition means if variable "k" is same as "January"
            print(dict_a[k]) # If the above condition is satisfied, this line is executed. do not forget an indent
        elif dict_a[k] == "2": # If "k" is not "January" and "dict_a[k]" is same as "2",
            print(k,"corresponds to",dict_a[k],"in dict_a.") # this line is executed.
        else: # In all situation except above two condition,
            print(k,"is not January or February.") # this line is executed.
 
if __name__ == "__main__":
    main()


# ###2-3-6. <font color="Crimson">関数</font>

# プログラミングをするにあたり，関数というものはとても大事な概念です．関数はあるプロセスの一群をまとめるためのシステムです．書いたプロセスを関数化することで，ソースコードの可読性が上がります．また，処理に汎用性を与えることができるようになります．同じ処理を変数の値だけを変えて何度も行いたい場合は同じ処理を何度も書くより，関数化して，必要なときにその関数を呼び出す，といったやり方が効率的です．関数は， `def` というものを使って生成します．これまでに書いてきたプログラムにも `def main()` という表記がありました．これは， `main()` という関数を定義するものです．そして，その `main()` という関数をソースコードの一番下，`if __name__ == "__main__":`で呼び出して使っていたということです．この `main()` 以外にも，プログラマーは自由に関数を定義することができます．以下のように `func_1()` という関数を作って，それを `main()` の中で実行することができます．

# In[19]:


#!/usr/bin/env python3
 
def main():
    func_1(); # The function "func_1()" is executed in a function "main()".
     
def func_1(): # Declaration of a novel function "func_1()".
    print("Hello") # substance of "func_1()"
 
if __name__ == "__main__":
    main() # The function "main()" is executed (and "main()" execute "func_1()" and "func_1()" execute "print("Hello")").


# <font color="Crimson">(9｀･ω･)9 ｡oO(`if __name__ == "__main__":` を書かない人結構いますが，では，その書いたプログラム，hoge.py としましょう，別のプログラムでインポート `import hoge` したときの挙動は期待したものと異なると思いますのでので試してみましょう！あと，変数のスコープの管理もしやすいです．)</font>

# 関数を生成する際に `()` が付属していますが，これは，引数（ひきすう，パラメータ）の受け渡しに使うためのものです．以下のように書くと，`Python` という文字列が格納された変数 `str_1` を関数 `func_1()` のパラメータとして関数が実行されます．その後，`func_1()` 内では，パラメータ `str_1 == Python` を `func_1()` の中でのみ定義されるパラメータ変数である `arg_1` で受け取りそれを利用して新たな処理が実行されます．

# In[20]:


#!/usr/bin/env python3
 
def main():
    str_1 = "Python"
    func_1(str_1); # pass the variable "str_1" to the function "func_1"
     
def func_1(arg_1): # To get argument from external call, you need to set a variable to receive the argument. In this case, it is "arg_1".
    print("Hello", arg_1) # print "Hello" and arg_1 (= str_1 in main() = "Python").
 
if __name__ == "__main__":
    main()


# また，関数はその関数内で処理された結果の値を関数の呼び出し元に返す (呼び出し元の関数内で使うことができるように値を渡す) ことができます．例えば，以下の書くと，`summation()` の呼び出し元に対して，`summation()` で計算した結果（リスト内の数値の和）が返ってきます．その返ってきた値を変数 `result_1` に格納してその後の処理に利用することができます．
# 
# 

# In[21]:


#!/usr/bin/env python3
 
def main():
    numbers = [2, 4, 5, -2, 3] # a list to be calculated
    result_1 = summation(numbers) # execution of summation(). Numbers is passed to the function. And the result of the function is assigned to the variable "result_1"
    print(result_1)
     
def summation(arg_1): # arg_1 is a variable to contain the argument (numbers in main()).
    sumvalue = 0 # A variable to contain a result of summation.
    for number in arg_1:
        sumvalue = sumvalue + number # this means "sum is renewed by the value of sum + number"
    return sumvalue # return the result of above calculation to the function caller by the word "return"
 
if __name__ == "__main__":
    main()


# ###2-3-7. <font color="Crimson">ライブラリのインポート</font>

# 上では関数を作ってプログラムを便利にしてきましたが，世界には様々な便利な関数をまとめたライブラリが存在します．これを使うと様々な便利なことをすごく簡単な記述で実現可能です．そういった関数の集まりをライブラリと言いますが，ここではそれらライブラリの使い方を学びます．ライブラリは以下のように，`#!/usr/bin/env python3` の下に `import` という表記によりプログラム中で使うことを宣言します（利用する直前に呼び出しても良い）．以下では，「statistics」というモジュールを使うことを明示しています．以下は実行しても何もしない（ライブラリをインポートすることだけをする）プログラムです．

# In[22]:


#!/usr/bin/env python3
import statistics
 
def main():
    pass
 
if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(この世にどんなライブラリがあるのかは人に聞くとか Google で調べることとかで知ることができます．)</font>

# 複数個のモジュールを呼び込むときは以下のように書くことができます．

# In[23]:


#!/usr/bin/env python3
import statistics
import sys
 
def main():
    pass
 
if __name__ == "__main__":
    main()


# 例えば，モジュール「statistics」を用いれば平均や標準偏差の計算が1行で可能です．モジュールに入っている関数（メソッド）を使うときは「モジュール名.関数名()」という形式で書きます．

# In[24]:


#!/usr/bin/env python3
import statistics
 
def main():
    linumber=[1, 2, 3, 4, 5, 6, 7, 8]
     
    print("mean:", statistics.mean(linumber))
    print("sd:", statistics.stdev(linumber))
 
if __name__ == "__main__":
    main()


# また，モジュール「math」を用いれば数学で用いられる様々な関数 (数学における関数) を簡単に記述することができます．

# In[25]:


#!/usr/bin/env python3
import math
 
def main():
    x=10
    print("log_10:", math.log10(x))
    print("log_2:", math.log2(x))
    print("log_e:", math.log(x))
     
    x=math.radians(180) #This is radian of an angle of 90 degrees
    print(math.sin(x))
    print(math.cos(x))
    print(math.tan(x))
 
if __name__ == "__main__":
    main()


# これまでは Python にデフォルトで組み込まれているモジュールのみを使ってきましたが，外部で用意されているモジュールを使うことも可能です．外部のモジュールはインストールしなければなりません．そのためにはコマンドライン上で「pip3」というコマンドを用います．例えば，数値計算を行うためのモジュール「NumPy」は以下のようにすることでインストールすることができます．現在使っている Python（正確には Python3）と pip3 が紐付いており，pip3 で外部からインストールしたライブラリは Python3 から呼び出すことができるのです．

# In[26]:


get_ipython().system(' pip3 install numpy')


# <font color="Crimson">(9｀･ω･)9 ｡oO(これは Python のコマンドではありません．本来，コマンドラインで打つものです．Python のプログラム中に書くものではありません．「!」が Python のプログラム外でコマンドを実行するための表記です．)</font>

# 現在使っている Python にインストールされているライブラリを確認するには以下のように打ちます．

# In[27]:


get_ipython().system(' pip3 list')


# <font color="Crimson">(9｀･ω･)9 ｡oO(本物のコマンドライン上で実行するときには「!」は不要です．)</font>

# ###2-3-8. <font color="Crimson">文字列処理（正規表現）</font>

# プログラミング言語は「正規表現（regular expression）」と呼ばれる文字列パターン表現方法を持ちます．プログラミングで文字列を操作するための非常に強力なツールです．Python で正規表現を使用する場合は、「re」 ライブラリを利用します．ここでは，以下のような文を用いて正規表現を学びます．

# ```
# 2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade.
# ```

# この文に存在するアラビア数字を抜き出したいとき，以下のようなプログラムを書きます．

# In[28]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    liresult = re.findall("\d+", text)
    print(liresult)

if __name__ == "__main__":
    main()


# このコードにおいて6行目の `\d+` が正規表現です．` \d` が「アラビア数字の任意の1文字」を表し，その後の `+` は「前の表現の1回以上の繰り返すこと」を意味しています．すなわち，`\d+` は「1つ以上のアラビア数字が連続した文字列」を意味します．関数 `re.findall()` は最初の引数に指定されたパターン（正規表現）を2番目ので指定された文字列より検索して抽出するための役割を果たします．

# 次に，以下のプログラムを実行します．これによって得られるリストには，「大文字から始まる文字列」が含まれます．使用する正規表現において，`[A-Z]` は大文字，`[a-z]` は小文字を意味します．

# In[29]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    liresult = re.findall("[A-Z][a-z]+", text)
    print(liresult)

if __name__ == "__main__":
    main()


# 以下のように，関数 `re.search()` を用いると，文の最初（`^`）にアラビア数字が含まれるかどうかを判定できます．

# In[30]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    if re.search("^\d", text):
        print("Yes")

if __name__ == "__main__":
    main()


# 以下のように書くと，「`(` から始まり，何らかの文字が1個以上連続し，`) ` で終了する文字列」を全て削除（何もないものに置換）することができます．関数 `re.sub()` は第3引数の文字列から，第1引数の文字列を同定し，それを第2引数の値で置換するためのものです．この場合，`(` および `)` をそれぞれ `\(` および `\)` のように表記していますが，これは `(` および `)` が正規表現のための記号として用いられるものであるため，正規表現としてそれらを利用したい場合には，そのための特別な表記としなければならないためです．この操作は，「エスケープする」というように表現します．また，`\w` は「何らかの文字」，`\s` は「空白文字（スペース等）」を意味します．

# In[31]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    replaced_text = re.sub("\(\w+\)\s", "", text)
    print(replaced_text)

if __name__ == "__main__":
    main()


# 以下の関数 `re.split()` を利用すると「文字列を空白文字がひとつ以上連続した文字列（`\s+`）」を区切り文字として分割することができます．これは文字列処理をする場合によく用いる関数です．

# In[32]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    liresult = re.split("\s+", text)
    print(liresult)

if __name__ == "__main__":
    main()


# Python の正規表現をまとめたウェブサイトには以下のようなものがあります．
# 
# *   https://www.debuggex.com/cheatsheet/regex/python
# *   https://docs.python.org/3/howto/regex.html
# 
# 

# ##2-4. <font color="Crimson">NumPy の基本的な使用方法</font>

# ###2-4-1. <font color="Crimson">NumPy のインポート</font>

# NumPy とは Python で利用可能な数値計算のライブラリです．さまざまな計算をコマンド一発で行うことができます．NumPy は以下のようにしてインポートします．読み込んだ NumPy には `np` という略称を与えることが普通です．

# In[33]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    pass
 
if __name__ == "__main__":
    main()


# ###2-4-2. <font color="Crimson">ベクトルの基本的な計算</font>

# ベクトルは以下のように生成します．

# In[34]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    print(np.array([1, 2, 3]))
 
if __name__ == "__main__":
    main()


# 要素の参照は普通の Python 配列と同じようにできます．もちろんゼロオリジンです．

# In[35]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3])
    print(na[0])
 
if __name__ == "__main__":
    main()


# ベクトルの四則計算は以下のようにします．NumPy は基本的に要素ごとに（element-wise）値を計算します．

# In[36]:


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

# In[37]:


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

# In[38]:


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


# ###2-4-3. <font color="Crimson">行列の基本的な計算</font>

# 行列を生成するためにも，`np.array()` を利用します．さらに，行列のサイズは `.shape` によって確認することができます．

# In[39]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na)
    print(na.shape)
 
if __name__ == "__main__":
    main()


# 行列の要素には以下のようにしてアクセスします．この場合，1行1列目の値にアクセスしています．

# In[40]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na[0,0])
 
if __name__ == "__main__":
    main()


# NumPy 行列は以下のようなアクセスの方法があります．行ごとまたは列ごとのアクセスです．これは多用します．

# In[41]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    print(na[0,:]) # all elements of row 1
    print(na[:,1]) # all elements of column 2

if __name__ == "__main__":
    main()


# 以下のようにすると行列に関する様々な統計値を得ることができます．

# In[42]:


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


# <font color="Crimson">(9｀･ω･)9 ｡oO(全知全能かよ．)</font>

# 以下の `np.zeros()` や `np.ones()` を用いると引数で指定したサイズの，全要素が0または1の行列を生成することができます．

# In[43]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    print(np.zeros((3,3)))
    print(np.ones((4,4)))

if __name__ == "__main__":
    main()


# 四則計算は以下のようにします．これもやはり，element-wise な計算です．

# In[44]:


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

# In[45]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[1, 2], [3, 4]])
    nb = np.array([[5, 6], [7, 8]])
    print(np.dot(na, nb))

if __name__ == "__main__":
    main()


# 以下のようにすると一様分布に従う乱数を生成することができます．以下の例は一様分布のものですが，NumPy には一様分布以外にもたくさんの分布が用意されています．引数で指定するのは行列のサイズです．計算機実験をする際にこのようなランダムな値を生成することがあります．そんな中，Python や NumPy に限らず計算機実験をする際に気を付けなければならないことに「乱数のタネを固定する」ということがあります．計算機実験の再現性を得るためにとても重要なので絶対に忘れないようにすべきです．乱数のタネは3行目で行っています．ここでは `0` を設定しています．

# In[46]:


#!/usr/bin/env python3
import numpy as np
np.random.seed(0)
 
def main():
    print(np.random.rand(3, 3))

if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(機械学習の学習前の予測器の初期値は乱数です．再現性を確保するため乱数のタネを指定しないで実験をはじめることは絶対にないように気をつけなければなりません．)</font>

# 以下のようにすると行列式と逆行列を計算することができます．

# In[47]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[4, 15, 4], [-11, 5, 6], [2, 4, 8]])
    print(np.linalg.det(na)) # determinant of matrix
    print(np.linalg.inv(na)) # inverse matrix

if __name__ == "__main__":
    main()


# 固有値分解は以下のようにします．

# In[48]:


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

# In[49]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(na ** 2)

if __name__ == "__main__":
    main()


# 一方で，「行列の冪乗」は以下のように計算します．

# In[50]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(np.linalg.matrix_power(na, 2))

if __name__ == "__main__":
    main()


# 行列の冪乗でも，整数以外の冪指数を用いたい場合は別の方法が必要です．例えば，行列の平方根（2分の1乗）は以下のようにしなければ計算できません．

# In[51]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    eigenvalue, eigenvector = np.linalg.eig(na)
    print(np.dot(np.dot(eigenvector, (np.diag(eigenvalue)) ** (1/2)), np.linalg.inv(eigenvector)))

if __name__ == "__main__":
    main()


# ###2-4-4. <font color="Crimson">ブロードキャスト</font>

# NumPy は「行列にスカラを足す」，このような異様な計算をしても結果を返してくれます．以下の6行目では，行列にスカラを足しています．ここでは，最初に生成した4行4列の行列と同じサイズの，全要素が2からなる行列を自動で生成し，その行列と最初の4行4列の行列の和を計算しています．このような，対象となる行列のサイズに合せて，スカラから行列を生成することを「ブロードキャスト」と言います．この機能は非常に便利で様々な局面で使用することがあります．

# In[52]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([[3, 4, 1, 4], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]])
    print(na + 2)

if __name__ == "__main__":
    main()


# ###2-4-5. <font color="Crimson">特殊な操作</font>

# 以下のようにすると配列の順番を逆向きにして用いることができます．魔法ですね．最初の要素（0）から最後の要素（-1）まで逆向きに（-）連続して（1）いることを意味します．

# In[53]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na[::-1])

if __name__ == "__main__":
    main()


# 以下のようにすると指定した条件に対する bool 配列を得ることができます．

# In[54]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na > 3)

if __name__ == "__main__":
    main()


# これを利用すると条件に合う要素のみに選択的にアクセスすることができます．以下では条件に合う要素を単に出力しているだけですが，例えば，条件に合う値のみを0に置き換えるとか，そのような操作ができます．

# In[55]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 7])
    print(na[na > 3])

if __name__ == "__main__":
    main()


# 以下のように書けば，目的に合う値のインデックスを出力させることもできます．

# In[56]:


#!/usr/bin/env python3
import numpy as np
 
def main():
    na = np.array([1, 2, 3, 4, 5, 6, 3])
    print(np.where(na == 3))

if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(NumPy はやばいよね．)</font>

# ##2-5. <font color="Crimson">クラスの生成とその活用</font>

# ###2-5-1. <font color="Crimson">クラスの生成</font>

# Python にはクラスというプログラムのコードを短くすることで可読性をあげたり，チームでの大規模開発に有利になったりするシステムが備わっています．最初にとてもシンプルなクラスを作ってみます．これは後ほどもう少し多機能なものへと改変していきます．クラスの名前を `Dog` とします．最初の辺りで変数名のイニシャルに大文字は使わないことを紹介しましたが，それはクラスの宣言の際にクラス名のイニシャルを大文字にするためです．以下のように書きます．`main()` には `pass` とだけ書いていますが，これは「何も実行しない」というコマンドです．今はクラスを生成することが目的だから `main()` では何も行いません．クラスは以下のように生成します．
# ```
# class クラス名:
# ```
# 以下のクラスには `__init__` という記述があります．これはクラスからインスタンスを生成したときに（<font color="Crimson">クラスからはインスタンスと言われるクラスの実体が生成されます</font>）自動的に実行されるメソッドです（コンストラクタと呼びます）．このコンストラクタの引数は2個です．`self` と `name` と `age` の3個があるように見えますが，`self` は（インスタンスを生成するまでは未知の）インスタンス名をとりあえず `self` にします，という記述です．

# In[57]:


#!/usr/bin/env python3

def main():
    pass

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# 多分，ここまでで意味がわからなくなったと思うので，まずはインスタンスを生成してみます．以下のように書きます．これによって `mydog` というインスタンスが生成されました．引数には，`Ken` と `6` を与えました．6歳の Ken という名前の犬の情報です．

# In[58]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# 次に，この生成したインスタンス `mydog` の名前と年齢を出力してみます．以下のように書きます．インスタンスが内包する変数には `.` でアクセスします．

# In[59]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.name)
    print(mydog.age)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age

if __name__ == "__main__":
    main()


# しっかりと出力されました．クラスを使う利点は処理をきれいに整理できることです．もし，クラスを使わずにこの挙動を再現するには `mydog_name = Ken`，`mydog_age = 6` のような変数を生成しなければなりません．クラスを使うと `mydog = Dog("Ken", 6)` だけ，たったひとつの記述だけで複数個の情報を含む変数を生成できるのです．

# <font color="Crimson">(9｀･ω･)9 ｡oO(以上のように複数のデータをまとめて管理できることがクラスを使うメリットです．クラスを使わない場合，個々の変数を個別に管理する必要が生じ，プログラムが煩雑なものになります．クラスを作る効果ってたったそれだけ？と思われるかもしれませんが，そうです，たったそれだけです．ただ，この後にも説明がある通りクラスは変数だけでなく関数も内部に有することができます．処理が複雑になればなるほど有難くなります．)</font>

# また，上で `self` とはインスタンスの名前そのものであると言及しましたが，それは以下を比較していただくと解ります．出力の際には以下のように書きました．
# ```
# mydog.name
# ```
# クラス内での定義の際には以下のように書きました．
# ```
# self.name
# ```
# インスタンス名（`mydog`）と `self` が同じように使われています．

# ###2-5-2. <font color="Crimson">メソッド</font>

# クラスにメソッドを追加します．以下では `sit()` というメソッドを定義しました．そしてそれを `main()` で呼び出しています．この場合も `.` にてメソッドを呼び出します．

# In[60]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    mydog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# たったひとつの `mydog` という変数（インスタンス）が，複数個の変数を持ち，メソッドも有します．これはかなり整理整頓には良いシステムです．これがクラスを定義しインスタンスを生成する利点です．これは，以下のようにインスタンスを何個も生成するような処理が必要になる場合にさらに有用です．よりシンプルなコードを保てます．

# In[61]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = Dog("Peko", 4)
    mydog.sit()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# ###2-5-3. <font color="Crimson">デフォルト値の設定</font>

# インスタンス生成の際にデフォルトの値を設定することができます．以下の13行目のようにコンストラクタに直接定義します．そのような値も，5行目のような記述でアクセスすることができます．また，6行目のような記述で値を変更することも可能です．

# In[62]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.hometown)
    mydog.hometown = "Yokohama"
    print(mydog.hometown)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
        self.hometown = "Sendai"
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))

if __name__ == "__main__":
    main()


# ###2-5-4. <font color="Crimson">クラス内変数の改変</font>

# また，以下の16および17行目のようなクラス内の変数の値を改変するような関数を定義することも可能です．これを実行すると出力のように，`mydog.age` が `6` から `8` に変化します．

# In[63]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    print(mydog.age)
    mydog.year_pass(2)
    print(mydog.age)

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
        self.hometown = "Sendai"
    def sit(self):
        print("{0}, which is ({1}) years old is now sitting.".format(self.name, self.age))
    def year_pass(self, years):
        self.age = self.age + years

if __name__ == "__main__":
    main()


# ###2-5-5. <font color="Crimson">クラス変数</font>

# `インスタンス名.変数名` もしくは `self.変数名` とすると，インスタンスが保持する変数にアクセスすることができましたが，そうではなく，クラスから生成されたすべてのインスタンスで値が共有されるような変数を作ることができます．そのような変数をクラス変数と呼びます．クラス変数を作成するには，単にクラスの定義部で変数に値を代入します．クラス変数にアクセスするには， `クラス名.変数名` とします．別のやり方として，インスタンス変数に同じ名前の変数がない場合には，`インスタンス名.変数名` と書いてもクラス変数にアクセスすることができます（ただし同名のインスタンス変数がある場合には，そちらにアクセスしますので注意が必要です）．

# In[64]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = Dog("Peko", 4)
    print(Dog.type) # クラス名.変数名で参照できる
    print(yourdog.type)

class Dog:
    type = "dog" # クラス変数．
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

if __name__ == "__main__":
    main()


# ###2-5-6. <font color="Crimson">プライベート変数</font>

# オブジェクト指向プログラミングでは，インスタンスへの外部からのアクセスを適切に制限することで，モジュールごとの権限と役割分担を明確にすることができます．例えば，あるモジュールがすべてのデータにアクセス可能だとすると，本来そのモジュールが行うべきではない処理まで，そのモジュールに実装されてしまう恐れがあり，役割分担が崩壊する可能性があります．データにアクセス制限がかかっている場合，そのデータを使う処理は，そのデータにアクセス可能なモジュールに依頼するしかなくなり，適切な役割分担が維持されます．
# 
# クラスにおいて，外部からのアクセスが禁止されている変数をプライベート変数と呼びます．また，外部からのアクセスが禁止されているメソッドをプライベートメソッドと呼びます．
# 
# Pythonでは，プライベート変数やプライベートメソッドを作ることはできません．つまり，どの変数やメソッドも外部からアクセス可能です．
# ただし，変数やメソッドの名前を変えることで，プライベートとして扱われるべき変数やメソッドを区別するという慣習があります．実装を行う人がプライベートとして扱われるべき変数等への外部からのアクセスを行わないようにすることで，本来の目的を達成することができます．Pythonの慣習では，プライベートとして扱われるべき変数やメソッドの先頭文字を以下のようにアンダースコア（_）にします．

# In[65]:


#!/usr/bin/env python3

def main():
    score_math = PrivateScore(55)
    print(score_math.is_passed())

class PrivateScore:
    def __init__(self, score):
        self._score = score # プライベート変数
    def is_passed(self):
        if self._score >= 60:
            return True
        else:
            return False

if __name__ == "__main__":
    main()


# ###2-5-7. <font color="Crimson">継承</font>

# あるクラスのインスタンスを複数生成する際，特定のインスタンスのみ処理を変更したり，処理を追加したりする必要が生じることがあります．その場合，そのインスタンス用に新しくクラスを作ることもできますが，元のクラスと大部分が同じである場合，同じコードが大量に複製されることになり，コードが無駄に長くなります．同じ処理を行うコードが複数箇所に存在する場合，不具合を修正する場合にもすべての箇所を修正しないといけなくなるため，正しく修正することが難しくなります．このような場合の別の方法として，元のクラスからの修正部分のみを修正するという方法があります．このような方法を継承と呼び，修正のベースとなる元のクラスを基底クラス（もしくは親クラス），新しく作られるクラスを派生クラス（もしくは子クラス）と呼びます．

# <font color="Crimson">(9｀･ω･)9 ｡oO(基底クラスに複数個のクラスを指定することができます．そのような継承を多重継承と呼びます．)</font>

# 継承を行う際は，派生クラスを定義するときにそのクラス名の次に括弧を書き，その括弧内に基底クラスの名前を書きます．そして，クラス定義の中には，新規に追加するメソッドと基底クラスから変更を行うメソッドについてのみ定義を追加します．以下のように行います．

# In[66]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = TrainedDog("Peko", 4)
    mydog.sit()
    yourdog.trick()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog): # クラス名の次に「(基底クラス名)」を追加する
    def trick(self):
        for i in range(self.age):
            print("bow")

if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(派生クラスである `TrainedDog` の定義には `sit()` は含まれていません．しかし，基底クラスである `Dog` を継承しているので `TrainedDog` から生成したインスタンスでは `sit()` を利用できています．)</font>

# ###2-5-8. <font color="Crimson">オーバーライド</font>

# これまでに，基底クラスに新たなメソッドを追加する方法である継承を紹介しました．これに対して，派生クラスで基底クラスのメソッドを書き換えることをオーバーライドと言います．オーバーライドは，派生クラスの定義をする際に，基底クラスで定義されているメソッドを再定義することで実現できます．以下のように行います．

# In[67]:


#!/usr/bin/env python3

def main():
    mydog = Dog("Ken", 6)
    yourdog = TrainedDog("Peko", 4)
    mydog.sit()
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog):
    def sit(self):
        print("{0} is now sitting and giving its paw.".format(self.name))

if __name__ == "__main__":
    main()


# 派生クラスから生成したインスタンスの `sit()` の挙動が基底クラスのものから変化しているのがわかります．このとき，基底クラスの `sit()` は変化していません．

# <font color="Crimson">(9｀･ω･)9 ｡oO(派生クラスで，派生クラスにおいて利用するメソッドを書き換えるのであって，派生クラスで書き換えたメソッドが基底クラスのメソッドを書き換える（明らかに危険な行為）わけではありません．)</font>

# 次に，派生クラスにおいてインスタンス変数を増やします．これを実現するためには，コンストラクタ `__init__()` をオーバーライドする必要があります．このとき，既に基底クラスで定義したインスタンス変数をそのまま利用するためにコンストラクタを呼び出します．基底クラスは `super()` を使って呼び出すことができます．以下のように行います．

# In[68]:


#!/usr/bin/env python3

def main():
    yourdog = TrainedDog("Peko", 4, "Ben")
    yourdog.sit()

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age  = age
    def sit(self):
        print("{0} is now sitting.".format(self.name))

class TrainedDog(Dog):
    def __init__(self, name, age, trainer):
        super().__init__(name, age) # この super() は基底クラスを意味する
        self.trainer = trainer # 新たなるインスタンス変数
    
    def sit(self):
        print("{0} is now sitting and giving its paw to {1}.".format(self.name, self.trainer))

if __name__ == "__main__":
    main()


# <font color="Crimson">(9｀･ω･)9 ｡oO(終わりです．)</font>
