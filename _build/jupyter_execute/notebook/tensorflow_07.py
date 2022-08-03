#!/usr/bin/env python
# coding: utf-8

# # Hugging Face の利用方法

# ## Hugging Face とは

# Hugging Face とは主にトランスフォーマーに関連する深層学習モデルやそれに付随する技術の提供を行っているサービスです．トランスフォーマー（のエンコーダーとデコーダー）はニューラルネットワーク等からなる，様々なタイプのデータを処理できる機械学習モデルです．トランスフォーマーは色々な問題を解決するために少しずつ違う構造を持ちますが，そのようなものをまとめてコマンド一発で利用可能にしてくれているのが Hugging Face です．また，自然言語処理の分野で扱うデータのサイズはとても大きい場合があるのですが，どこかの誰かがあらかじめ何かのデータを利用して学習済みのデータを提供してくれていることがあり，そのような学習済みモデルも簡単に利用できるように整備してくれています．Hugging Face を利用すれば，特に自然言語処理に関する問題を簡単に解けるようになるため紹介します．

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

# ### インストール

# Hugging Face（のトランスフォーマー）は TensorFlow または PyTorch とあわせて利用可能なライブラリです．本来は TensorFlow か PyTorch をあらかじめインストールする必要があります．グーグルコラボラトリーには既に TensorFlow がインストールされているため不要です．以下のように `transformers` のみをインストールすれば利用可能です．

# In[ ]:


get_ipython().system(' pip3 install transformers')


# ```{attention}
# このコンテンツの実行にグーグルコラボラトリーを使っておらず，自身の環境で Anaconda 等を利用している場合は気をつけてください．例えば，Anaconda を利用しているのであれば ` conda install -c huggingface transformers ` のようなコマンドでインストールした方が良いです．
# ```

# ## 基本的な使い方

# とても簡単に自然言語処理を実現することができる利用方法を紹介します．世界最高性能を求めたいとかでないなら，ここで紹介する方法を利用して様々なことを達成できると思います．

# ### 感情分析

# 最も簡単な `tranformers` の利用方法は以下のようになると思います．`pipeline` を読み込んで，そこに取り組みたいタスク（`sentiment-analysis`）を指定します．初回の起動の際には事前学習済みモデルがダウンロードされるため時間がかかります．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    classifier = pipeline("sentiment-analysis")
    text = "I have a pen."
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    main()


# 入力した文章がポジティブな文章なのかネガティブな文章なのかを分類できます．ここでは1個の文章を入力しましたが，以下のように2個以上の文章も入力可能です．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    classifier = pipeline("sentiment-analysis")
    litext = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
    result = classifier(litext)
    print(result)

if __name__ == "__main__":
    main()


# ### 特徴抽出

# 文字列の特徴を抽出して何らかの分散表現にしたい場合は `feature-extraction` を指定します．文字列の長さに依存した配列が出力されますが，同じ長さのベクトルにしたい場合は配列長に渡って要素の値を足し算すること等で特徴量を得ることができます．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    converter = pipeline("feature-extraction")
    text = "We are very happy to introduce pipeline to the transformers repository."
    result = converter(text)
    print(result)

if __name__ == "__main__":
    main()


# ```{hint}
# 例えば，BERT を使うと各トークンが768次元のベクトルでトークン数個からなる要素のベクトルが出力されます．
# ```

# ### ゼロショット文章分類

# ゼロショット文章分類は以下のように利用します．ゼロショットとは訓練中に一度も出現しなかったクラスの分類タスクです．ラベルが未定義の問題に利用することができます．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    classifier = pipeline("zero-shot-classification")
    text = "This is a course about the Transformers library",
    lilabel = ["education", "politics", "business"]
    result = classifier(text, candidate_labels = lilabel)
    print(result)

if __name__ == "__main__":
    main()


# ```{note}
# ゼロショット分類では，例えば，馬を入力にして猫とか犬とかのラベルそのものを予測されるのではなくて猫および犬ベクトルを予測させどちらに近いかを予測させることができます．そろそろマシンパワー的にきついかもしれませんね．
# ```

# ### 文章生成

# 文章の生成は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    generator = pipeline("text-generation")
    text = "In this course, we will teach you how to"
    result = generator(text)
    print(result)

if __name__ == "__main__":
    main()


# ### 穴埋め

# 穴埋めは以下のように行います．以下のコードでは可能性が高いものふたつを表示させます．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    unmasker = pipeline("fill-mask")
    text = "This course will teach you all about <mask> models."
    result = unmasker(text, top_k=2)
    print(result)

if __name__ == "__main__":
    main()


# ### 固有表現抽出

# 固有表現抽出は以下のように行います．結果の `PER` は人名，`ORG` は組織名，`LOC` は地名です．それらの固有表現の文書中における位置も `start` と `end` で示されています．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    ner = pipeline("ner", grouped_entities=True)
    text = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    result = ner(text)
    print(result)

if __name__ == "__main__":
    main()


# ### 質問応答

# SQuAD のような機械学習コンテストで行われる質問応答は質問だけを入力にして何かを出力する問題ではありません．質問文と何らかの説明文を入力にして解答を出力される問題です．以下のように利用します．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    question_answerer = pipeline("question-answering")
    question_text = "Where do I work?"
    explanation = "My name is Sylvain and I work at Hugging Face in Brooklyn"
    result = question_answerer(question = question_text, context = explanation)
    print(result)

if __name__ == "__main__":
    main()


# ### 要約

# 文章の要約は以下のように行います．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    summarizer = pipeline("summarization")
    text = """
        America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
    """
    result = summarizer(text)
    print(result)

if __name__ == "__main__":
    main()


# ### 翻訳

# 翻訳はこれまでと少しだけ指定方法が異なります．以下のように `translation_XX_to_YY` としなければなりません．ここでは，英語からフランス語への翻訳を行います．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    translator = pipeline("translation_en_to_fr")
    text = "This course is produced by Hugging Face."
    result = translator(text)
    print(result)

if __name__ == "__main__":
    main()


# ## 応用的な使い方

# これまでに利用したものとは異なるモデルを利用したいとか，自身が持っているデータセットにより適合させたいとかの応用的な利用方法を紹介します．

# ### 他のモデルの利用

# これまでに，Hugging Face が自動でダウンロードしてくれたデフォルトの事前学習済みモデルを利用した予測を行いましたが，そうでなくて，モデルを指定することもできます．以下のページをご覧ください．Model Hub と言います．
# 
# https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
# 
# 
# この Model Hub の Tasks というところでタグを選択できます．例えば，Text Generation の `distilgpt2` を利用するには以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline
 
def main():
    generator = pipeline("text-generation", model="distilgpt2")
    text = "In this course, we will teach you how to"
    result = generator(text)
    print(result)

if __name__ == "__main__":
    main()


# ### 日本語の解析

# 日本語も扱うことができます．ここでは，日本語で書かれた文章の感情分析を行います．最初に，必要なライブラリをダウンロードしてインストールします．

# In[ ]:


get_ipython().system(' pip install fugashi')
get_ipython().system(' pip install ipadic')


# ```{note}
# これをインストールしないで使ったらインストールしろってメッセージが出たからインストールしました．
# ```

# ```{attention}
# グーグルコラボラトリーでのインストール方法です．
# ```

# Model Hub で調べたら以下のような感情分析のモデルが公開されていたので，それを使います．トークナイザーとは文章をトークン化（単語化して数字を割り当てます）してくれるものです．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline

def main():
    classifier = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment", tokenizer="daigo/bert-base-japanese-sentiment")
    text = "みんながマリオのチョコエッグツイートをしていく中，未だにひとつも買えなくて焦る…．"
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    main()


# 以下のようにモデルやトークナイザーは明示的に書くこともできます．ここでは，東北大学の乾研が公開している日本語版BERTを利用してみました．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline, AutoTokenizer

def main():
    mytokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    unmasker = pipeline("fill-mask", model="cl-tohoku/bert-base-japanese-whole-word-masking", tokenizer=mytokenizer)
    text = "みんなが[MASK]のチョコエッグツイートをしていく中，未だにひとつも買えなくて焦る…．"
    result = unmasker(text)
    print(result)

if __name__ == "__main__":
    main()


# ### モデルの設定変更

# 上で紹介したのと同様に，モデルとトークナイザーをそれぞれ，`TFAutoModelForSequenceClassification` と `AutoTokenizer` で読み込むことができます．以下の通りです．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer

def main():
    mymodel = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    mytokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("sentiment-analysis", model=mymodel, tokenizer=mytokenizer)
    text = "I do not have a pen but I am happy."
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    main()


# この際に `Auto` を使わずにあらかじめ用意されていいるクラスを明示的に指定することもできます．この場合，以下のように書きます．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline, TFDistilBertForSequenceClassification, DistilBertTokenizer

def main():
    mymodel = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    mytokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("sentiment-analysis", model=mymodel, tokenizer=mytokenizer)
    text = "I do not have a pen but I am happy."
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    main()


# さらにモデルをカスタマイズすることもできます．ここでは，`DistilBert` の構造を変えてしまっているので，全く性能が出ていません．学習をし直す必要がありそうです．

# In[ ]:


#!/usr/bin/env python3
from transformers import pipeline, TFDistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

def main():
    myconfig = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
    mymodel = TFDistilBertForSequenceClassification(myconfig)
    mytokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("sentiment-analysis", model=mymodel, tokenizer=mytokenizer)
    text = "I do not have a pen but I am happy."
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    main()


# ### ファインチューニング

# 事前学習モデルを自分の解きたい問題に合わせてファインチューニングすることができます．ここでは，インターネット上の映画レビューのデータセット IMDb に対してモデルのファインチューニングを行います．最初に Hugging Face が提供してくれているデータセットを利用するためのモジュールをインストールします．

# In[ ]:


get_ipython().system(' pip3 install datasets')


# 中身を確認します．映画に関するレビューが含まれています．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset

def main():
    imdb = load_dataset("imdb")
    print(imdb["train"][0])

if __name__ == "__main__":
    main()


# このデータセットをトークナイズします．以下のようなコードを追加します．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    print(tokenized_imdb)

if __name__ == "__main__":
    main()


# データセット中のデータをパディングします．データセット中の最大の長さのデータに合わせてパディングしても良いのですが，それだと非効率的なので，バッチ毎にパディングする方法を行います．ダイナミックパディングと呼ばれる方法です．以下のようにします．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if __name__ == "__main__":
    main()


# ファインチューニング前のモデルを呼び出して，テストデータセットの最初の5個について感情分析をしてみます．ポジティブとネガティブの判定はどれも 0.5 くらいであり，判別しきれていません．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, pipeline

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(classifier(imdb["test"][0:5]["text"]))

if __name__ == "__main__":
    main()


# データセットを TensorFlow で処理できるように変換します．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(classifier(imdb["test"][0:5]["text"]))

    tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

if __name__ == "__main__":
    main()


# 学習の条件を設定し，学習を行います．ファインチューニング済みモデルを用いてテストデータセットの最初の5個の予測をしていますが，どれも判別の確率が上がっています．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(classifier(imdb["test"][0:5]["text"]))

    tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(
        init_lr=2e-5, 
        num_warmup_steps=0, 
        num_train_steps=total_train_steps
    )

    model.compile(optimizer=optimizer)
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
    )

    print(classifier(imdb["test"][0:5]["text"]))

if __name__ == "__main__":
    main()


# ファインチューニング済みのモデルやトークナイザーは以下のように保存します．

# In[ ]:


#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

def main():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(classifier(imdb["test"][0:5]["text"]))

    tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'label'],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(
        init_lr=2e-5, 
        num_warmup_steps=0, 
        num_train_steps=total_train_steps
    )
    
    model.compile(optimizer=optimizer)
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
    )

    print(classifier(imdb["test"][0:5]["text"]))

    save_directory = './pretrained'
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

if __name__ == "__main__":
    main()


# ```{note}
# 終わりです．
# ```
