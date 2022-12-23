#!/usr/bin/env python
# coding: utf-8

# # Python の基本的な使用方法

# ## Python とは

# Python は，科学技術界隈，特にデータ科学に関する分野において昨今，最も利用されているプログラミング言語です．これを習得することで今後，プログラミング技術なしでは為し得なかったような大量データの解析ができるようになります．Python は C言語とか Fortran とかの伝統的なプログラミング言語と比べて簡単に学ぶことができるプログラミング言語です．ものすごく人気がある言語であり，たくさんのライブラリ（補助ツール）が存在しているので，色々なことができます．計算の速さもそこそこで，少なくとも悪くはありません．ここでは，Python のバージョン3（Python3）の使い方を学びます．

# ## Python の実行方法

# Python をどうやって実行するかとか，グーグルコラボラトリーでどうやって動かすかとかを紹介します．

# ### 実行の手順

# 普通，Python は以下のような手順で使います．
# 
# 1.   プログラムを書きます．プログラムの本体をソースコードと呼びます．
# 2.   書いたソースコードを保存します．拡張子は「.py」とします．
# 3.   そのプログラムを（コマンドライン等から）呼び出します．
# 

# ### Python プログラムの構造

# 以下のようなプログラムを書いて，それを実行するとターミナルウィンドウ等に「Hello world」という表示が出力されます．以下のプログラムにおいて，1～3と5～7行は Python にとって常に必要になる表記です．これは，いかなるときにも取り敢えず書いておくと良いでしょう．なので，以下のプログラムの本体は実質，4行目だけです．これは，「Hello world」という文を画面に表示しなさいという命令です．`print()` という表記の前に存在する空白は，キーボードの `Tab` を押すと入力できる空白文字です．Pythonではこのように空白文字をきちんと入れる必要があります．`def main():` 以下に書いた，タブの右側の表記だけがプログラムとして認識されます．

# In[ ]:


#!/usr/bin/env python3

def main():
    print("Hello world")

if __name__ == "__main__":
    main()


# ```{note}
# 実は ` print("Hello world") ` だけでコードは問題なく動きます．特にこの Google Colaboratory を使う場合，コードセルで実行した結果はメモリ上に保存されるので寧ろコマンドだけ書いた方が良い場合もあります．Google Colaboratory は本来そのようなプログラミング環境として利用することを想定されているのかもしれません．しかし，今回の実習の目的は最終的には Python プログラムを自身の計算機で動かせるようになることとしているので，この書き方を続けます．このようにすることで，コードセルをコピーして別の環境に持って行ったとしても問題なく利用できます．
# ```

# ```{note}
# Python コードをプログラムファイルとして実行した場合，` __name__ ` という変数に ` "__main__" ` という文字列が割り当てられます．一方で，` hoge.py ` という名前の Python コードをライブラリとして別のプログラムから ` import hoge ` で呼び出した場合，` __name__ ` という変数に ` "hoge" ` が割り当てられます．つまり，` if __name__ == "__main__": ` を書かないコードを別のファイルから呼び出して使おうとすると，望んでもいないのにそのコードの全部がインポート時に勝手に実行されてしまうという弊害が起きてしまうのです．
# ```

# ### Python コーディング

# Python でコーディングをする際に統一的なルール（エチケット？） として PEP8 が定義されています．PEP とは Python Enhancement Proposal の略です．PEP8 に従って書かれたコードは可読性が高くなります．
# 
# https://pep8-ja.readthedocs.io/ja/latest/
# 
# このコンテンツのコードは必ずしも PEP8 に従っていないかもしれません．
# 

# ## Python の基本操作

# 以降，実際にプログラムを書いてそれらを実行してみます．何かを画面に表示させたり，変数というものを使ったり，リストとかディクショナリとかいうものを使ったりします．

# ### 四則計算

# 以下のように書くと足し算ができます．プログラム中の「#」はコメントの宣言のための記号で，これが書いてある部分から右側はプログラムとして認識されません．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    print(3 + 15) # addition
 
if __name__ == "__main__":
    main()


# ```{attention}
# ハッシュ（#）とシャープ（♯）は実は違うものですよね．
# ```

# その他の四則計算は以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    print(10 - 2) # subtraction
    print(90 / 3) # division
    print(51 * 5) # multiplication

if __name__ == "__main__":
    main()


# ```{hint}
# コードセルを追加して何か書いてみましょう！
# ```

# ### 変数

# 上では，実際の数値を入力してプログラムを書きました．しかし，これではただの電卓です．以下のようにすると，変数を定義することができます．例えば，4行目で「a」という変数を生成し，これに「3」という数値を代入しています．ここで注意しなければならないことは，プログラミング言語において「=」は「その右側と左側が等しい」という意味ではなく，「右側を左側に代入する」という意味で用いられる点です．このように変数というものを使うことでプログラムは色々な処理を一般化することができます．

# In[ ]:


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

# In[ ]:


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

# In[ ]:


#!/usr/bin/env python3
 
def main():
    greeting = "Hello world"
    print(greeting)

if __name__ == "__main__":
    main()


# ```{note}
# Python では数値と文字列は明確に区別されます．数値を変数に代入したときは ` " ` を利用しませんでした．` " ` に囲まれた部分は文字列としてみなされます．
# ```

# ダブルクォーテーション（`"`）に囲まれた部分には何を書いてもそれは文字列とみなされます．よって，上の例で `"greeting"` と書いてその内容（`Hello world`）を出力しようとしても，`greeting` という文字列が出力されるだけで中身を出力できません．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    greeting = "Hello world"
    print("greeting")

if __name__ == "__main__":
    main()


# よって，以下のような計算もできません．文字列と文字の割り算をさせようとしているからです．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    a = "3"
    b = "18"
    print(a)
    print(b / a)

if __name__ == "__main__":
    main()


# 文字列を扱う際に，その文字列の途中で変数を利用したい場合があるかもしれません．そのような際には，変数の中身をダブルクォーテション内で「展開」する必要があります．変数を文字列の中で展開したい場合は `{` と `}` と `format()` を組み合わせて使います．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    name = "Ken"
    age = 6
    print("{}, which is {} years old is now sitting.".format(name, age))

if __name__ == "__main__":
    main()


# ### リストとディクショナリ

# 複数個の値を代入することができる変数があります．これのことをただの変数ではなく，Python ではリスト（list）と呼びます．リスト変数は `[` と `]` と `,` を使って生成します．以下のように生成した変数「list_a」の要素にアクセスするには「list_a[0]」のような感じでリスト変数に添字を付けて書きます．こうすることで「list_a」に格納されている最初の値である「10」にアクセスできます．多くのプログラミング言語ではリストの一番最初を示すインデックスは「0」です（1ではありません．）．こういうシステムのことはゼロオリジンと呼称します．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    list_a = [10, 4, "aaa"] # declaration of a variable, list_a, this list variable contains three values
    print(list_a[0]) # access first element of list_a
    print(list_a[1]) # access second element of list_a
    print(list_a[2]) # access third element of list_a

if __name__ == "__main__":
    main()


# Python のリストでは上のように数値と文字（列）をひとつのリストに同時に入れることができます．リストを使うことが可能な多くのプログラミング言語ではこのような操作はできない場合が多いです．

# また，以下のような特別な変数も存在します．これをディクショナリ（dictionary）と呼びます．ディクショナリ変数は `{` と `}` と `,` と `:` を使って生成します．ディクショナリ変数には，キー（key）とそれに対応するバリュー（value）を一組にして代入します．その後，キーの値を使って以下のようにしてディクショナリ変数にアクセスすることで，そのキーに対応するバリューを取得することができます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    dict_a = {"January" : "1", "February" : "2"} # declaration of dictionary, dict_a. this dictionary contains two keys and two values
    print(dict_a["January"]) # access a value corresponding to a key "January"

if __name__ == "__main__":
    main()


# ### 繰り返し処理

# Python で繰り返し処理をするには「for」を使います．以下のように書くと，配列 `[0，1，2，3，4]` に含まれる各要素が一行ごとに表示されます．プログラムではこの「for」を様々な局面で利用することによって人間ではできないような同じことの繰り返し作業を実現します．この繰り返し処理の方法と後に出る条件分岐の方法（if）さえ用いればどのようなプログラムでも実現することができます．繰り返し操作が終わった次の行からは，インデントを再びアウトデントするのを忘れないよう注意しましょう．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    for i in [0, 1, 2, 3, 4]:
        print(i)
    print("hoge")

if __name__ == "__main__":
    main()


# ```{note}
# アウトデントした部分は繰り返されていませんよね？
# ```

# この処理は以下のようにも書けます．以下のように書くと，`range(5)` に格納されている「0，1，2，3，4」という値が一行ごとに表示されます．`range()` に指定した回数の繰り返しが行われるのですね．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    for i in range(5):
        print(i)
    print("hoge")

if __name__ == "__main__":
    main()


# ```{hint}
# ` range(5) ` は ` [0，1，2，3，4] ` の配列を生成するものと考えて問題ないです．
# ```

# 以下のように書くとリストに格納されているデータ全てにひとつずつアクセスすることができます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    list_a = [10, 4, "aaa"] # declaration of variable list_a
    for w in list_a: # repetitive access of list_a. w represents an element.
        print(w) # when you start a "function" you have to insert [tab] (and when you finish the function, also you need outdent)
 
if __name__ == "__main__":
    main()


# ディクショナリに対する繰り返しのアクセスは以下のようにします．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    dict_a = {"January" : "1", "February" : "2"}
    for k in dict_a.keys(): # dict_a.keys() is a list containing only keys of "dict_a". or you can write simply "dict_a" instead of "dict_a.keys()"
        print(k, dict_a[k]) # k is a key and dict_a[k] is a value corresponding to the key.
 
if __name__ == "__main__":
    main()


# ```{hint}
# ディクショナリ変数に `.keys()` をつけることでディクショナリ変数に含まれるキーからなるリストを生成することができます．
# ```

# 以下のように書いても，上と全く同じ挙動を示します．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    dict_a = {"January" : "1", "February" : "2"}
    for k, v in dict_a.items():
        print(k, v) # k is a key and dict_a[k] is a value corresponding to the key.
 
if __name__ == "__main__":
    main()


# ```{note}
# この場合，`.items()` によってキーと値のペアが同時に抜き出されているのです．変数の内容を変更して，挙動を確認してみましょう！
# ```

# 繰り返し処理をする方法にはその他に，`while` があります．以下のようにして使います．`while` の右側に書いた条件が満たされる場合のみ，それ以下の処理を繰り返すというものです．この場合，`i` という変数が `10` より小さいとき処理が繰り返されます．そして，この `i` の値は `while` が1回実行されるたびに `1` ずつインクリメント（増加）されています．その表記が，`i = i + 1` です．これは，`i` に「`i` に `1` を足した値」を代入しろという意味です．なので，この表記の前の `i` に比べて，表記の後の `i` は `1` だけ大きい値です．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    i = 0
    while i < 10: # If the condition "i < 10" is satisfied,
        print(i)
        i = i + 1 # these two lines are processed.
 
if __name__ == "__main__":
    main()


# また，`while` は以下のようにして用いることもできます．この `!=` は「この記号の右側と左側の値が等しくない」ということを意味します．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    i = 0
    while i != 10:
        print(i)
        i = i + 1

if __name__ == "__main__":
    main()


# ### 条件分岐

# プログラミング言語にとって重要な要素として「if」があります．これは条件分岐のためのシステムです．以下のように書くと，もし `i` の値が2であるときのみ出力がされます．ここで，`==` という記号は「記号の右側と左側の値が等しい」ということを意味し，現実世界の「=」と同義です．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    for i in range(5):
        if i == 2:
            print(i)

if __name__ == "__main__":
    main()


# 繰り返し処理と条件分岐を組み合わせることで複雑な動作を実現することができます．以下のように書くと，もしキーが「January」であった場合，「1」が表示されます．それ以外でもし，「ディクショナリ変数のバリューが 2」であった場合，`February corresponds to 2 in dict_a.` と表示されます．それら以外の全ての場合おいては，`key is not January or February.` と表示されます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    dict_a = {"January" : "1", "February" : "2", "May" : "5"}
    for k in dict_a.keys():
        if k == "January": # a condition means if variable "k" is same as "January"
            print(dict_a[k]) # If the above condition is satisfied, this line is executed. do not forget an indent
        elif dict_a[k] == "2": # If "k" is not "January" and "dict_a[k]" is same as "2",
            print(k, "corresponds to", dict_a[k], "in dict_a.") # this line is executed.
        else: # In all situation except above two condition,
            print(k,"is not January or February.") # this line is executed.
 
if __name__ == "__main__":
    main()


# ### 関数

# プログラミングをするにあたり，関数というものはとても大事な概念です．関数はあるプロセスの一群をまとめるためのシステムです．書いたプロセスを関数化することで，ソースコードの可読性が上がります．また，処理に汎用性を与えることができるようになります．同じ処理を変数の値だけを変えて何度も行いたい場合は同じ処理を何度も書くより，関数化して，必要なときにその関数を呼び出す，といったやり方が効率的です．関数は， `def` というものを使って生成します．これまでに書いてきたプログラムにも `def main()` という表記がありました．これは， `main()` という関数を定義するものです．そして，その `main()` という関数をソースコードの一番下，`if __name__ == "__main__":`で呼び出して使っていたということです．この `main()` 以外にも，プログラマーは自由に関数を定義することができます．以下のように `func_1()` という関数を作って，それを `main()` の中で実行することができます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    func_1() # The function "func_1()" is executed in a function "main()".
     
def func_1(): # Declaration of a novel function "func_1()".
    print("Hello") # substance of "func_1()"

if __name__ == "__main__":
    main() # The function "main()" is executed (and "main()" execute "func_1()" and "func_1()" execute "print("Hello")").


# ```{note}
# ` if __name__ == "__main__": ` を書かない人結構いますが，では，その書いたプログラム，hoge.py としましょう，別のプログラムでインポート ` import hoge ` したときの挙動は期待したものと異なると思いますのでので試してみましょう！あと，変数のスコープの管理もしやすいです．
# ```

# 関数を生成する際に `()` が付属していますが，これは，引数（ひきすう，パラメータ）の受け渡しに使うためのものです．以下のように書くと，`Python` という文字列が格納された変数 `str_1` を関数 `func_1()` のパラメータとして関数が実行されます．その後，`func_1()` 内では，パラメータ `str_1`（`== Python`）を `func_1()` の中でのみ定義されるパラメータ変数である `arg_1` で受け取りそれを利用して新たな処理が実行されます．

# In[ ]:


#!/usr/bin/env python3
 
def main():
    str_1 = "Python"
    func_1(str_1) # pass the variable "str_1" to the function "func_1"
     
def func_1(arg_1): # To get argument from external call, you need to set a variable to receive the argument. In this case, it is "arg_1".
    print("Hello", arg_1) # print "Hello" and arg_1 (= str_1 in main() = "Python").

if __name__ == "__main__":
    main()


# また，関数はその関数内で処理された結果の値を関数の呼び出し元に返す (呼び出し元の関数内で使うことができるように値を渡す) ことができます．例えば，以下のように書くと，`summation()` の呼び出し元に対して，`summation()` で計算した結果（リスト内の数値の和）が返ってきます．その返ってきた値を変数 `result_1` に格納してその後の処理に利用することができます．
# 
# 

# In[ ]:


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


# ### ライブラリのインポート

# 上では関数を作ってプログラムを便利にしてきましたが，世界には様々な便利な関数をまとめたライブラリが存在します．これを使うと様々な便利なことをすごく簡単な記述で実現可能です．そういった関数の集まりをライブラリと言いますが，ここではそれらライブラリの使い方を学びます．ライブラリは以下のように，`#!/usr/bin/env python3` の下に `import` という表記によりプログラム中で使うことを宣言します（利用する直前に呼び出しても良い）．以下では，「statistics」というモジュールを使うことを明示しています．以下は実行しても何もしない（ライブラリをインポートすることだけをする）プログラムです．

# In[ ]:


#!/usr/bin/env python3
import statistics
 
def main():
    pass

if __name__ == "__main__":
    main()


# ```{hint}
# この世にどんなライブラリがあるのかは人に聞くとか Google で調べることとかで知ることができます．
# ```

# 複数個のモジュールを呼び込むときは以下のように書くことができます．

# In[ ]:


#!/usr/bin/env python3
import statistics
import sys
 
def main():
    pass

if __name__ == "__main__":
    main()


# 例えば，モジュール「statistics」を用いれば平均や標準偏差の計算が 1 行で可能です．モジュールに入っている関数（メソッド）を使うときは「モジュール名.関数名()」という形式で書きます．

# In[ ]:


#!/usr/bin/env python3
import statistics
 
def main():
    linumber = [1, 2, 3, 4, 5, 6, 7, 8]
     
    print("mean:", statistics.mean(linumber))
    print("sd:", statistics.stdev(linumber))
 
if __name__ == "__main__":
    main()


# また，モジュール「math」を用いれば数学で用いられる様々な関数 (数学における関数) を簡単に記述することができます．

# In[ ]:


#!/usr/bin/env python3
import math
 
def main():
    x = 10
    print("log_10:", math.log10(x))
    print("log_2:", math.log2(x))
    print("log_e:", math.log(x))
    
    x = math.radians(180) #This is radian of an angle of 180 degrees
    print(math.sin(x))
    print(math.cos(x))
    print(math.tan(x))

if __name__ == "__main__":
    main()


# これまでは Python にデフォルトで組み込まれているモジュールのみを使ってきましたが，外部で用意されているモジュールを使うことも可能です．外部のモジュールはインストールしなければなりません．そのためにはコマンドライン上で「pip3」というコマンドを用います．例えば，数値計算を行うためのモジュール「NumPy」は以下のようにすることでインストールすることができます．現在使っている Python（正確には Python3）と pip3 が紐付いており，pip3 で外部からインストールしたライブラリは Python3 から呼び出すことができるのです．

# In[ ]:


get_ipython().system(' pip3 install numpy')


# ```{note}
# これは Python のコマンドではありません．本来，コマンドラインで打つものです．Python のプログラム中に書くものではありません．「!」が Python のプログラム外でコマンドを実行するための表記です．
# ```

# 現在使っている Python にインストールされているライブラリを確認するには以下のように打ちます．

# In[ ]:


get_ipython().system(' pip3 list')


# ```{note}
# 本物のコマンドライン上で実行するときには ` ! ` は不要です．
# ```

# ### 文字列処理

# プログラミング言語は「正規表現（regular expression）」と呼ばれる文字列パターン表現方法を持ちます．プログラミングで文字列を操作するための非常に強力なツールです．Python で正規表現を使用する場合は、「re」 ライブラリを利用します．ここでは，以下のような文を用いて正規表現を学びます．

# ```
# 2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade.
# ```

# この文に存在するアラビア数字を抜き出したいとき，以下のようなプログラムを書きます．

# In[ ]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    liresult = re.findall("\d+", text)
    print(liresult)

if __name__ == "__main__":
    main()


# このコードにおいて 6 行目の `\d+` が正規表現です．` \d` が「アラビア数字の任意の 1 文字」を表し，その後の `+` は「前の表現の 1 回以上の繰り返し」を意味しています．すなわち，`\d+` は「1 つ以上のアラビア数字が連続した文字列」を意味します．関数 `re.findall()` は最初の引数に指定されたパターン（正規表現）を2番目で指定された文字列より検索して抽出するための役割を果たします．

# 次に，以下のプログラムを実行します．これによって得られるリストには，「大文字から始まる文字列」が含まれます．使用する正規表現において，`[A-Z]` は大文字，`[a-z]` は小文字を意味します．

# In[ ]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    liresult = re.findall("[A-Z][a-z]+", text)
    print(liresult)

if __name__ == "__main__":
    main()


# 以下のように，関数 `re.search()` を用いると，文の最初（`^`）にアラビア数字が含まれるかどうかを判定できます．

# In[ ]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    if re.search("^\d", text):
        print("Yes")

if __name__ == "__main__":
    main()


# 以下のように書くと，「`(` から始まり，何らかの文字が1個以上連続し，`) ` で終了する文字列」を全て削除（何もないものに置換）することができます．関数 `re.sub()` は第3引数の文字列から，第1引数の文字列を同定し，それを第2引数の値で置換するためのものです．この場合，`(` および `)` をそれぞれ `\(` および `\)` のように表記していますが，これは `(` および `)` が正規表現のための記号として用いられるものであるため，正規表現としてそれらを利用したい場合には，そのための特別な表記としなければならないためです．この操作は，「エスケープする」というように表現します．また，`\w` は「何らかの文字」，`\s` は「空白文字（スペース等）」を意味します．

# In[ ]:


#!/usr/bin/env python3
import re

def main():
    text = "2018 (MMXVIII) was a common year starting on Monday of the Gregorian calendar, the 2018th year of the Common Era (CE) and Anno Domini (AD) designations, the 18th year of the 3rd millennium, the 18th year of the 21st century, and the 9th year of the 2010s decade."
    replaced_text = re.sub("\(\w+\)\s", "", text)
    print(replaced_text)

if __name__ == "__main__":
    main()


# 以下の関数 `re.split()` を利用すると「文字列を空白文字がひとつ以上連続した文字列（`\s+`）」を区切り文字として分割することができます．これは文字列処理をする場合によく用いる関数です．

# In[ ]:


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

# ちょっと以下のものは行儀が良くないので，説明は省きますが以下のようにすると何かのプログラムを実行した結果を処理することができます．

# In[ ]:


#!/usr/bin/env python3
import subprocess

def main():
    text = subprocess.check_output("ls sample_data", shell=True).decode("utf-8").rstrip().split("\n")
    print(text)

if __name__ == "__main__":
    main()


# ```{note}
# これのどこが行儀が良くないのかはインジェクションで調べてみるとわかります．
# ```

# ```{note}
# 終わりです．
# ```
