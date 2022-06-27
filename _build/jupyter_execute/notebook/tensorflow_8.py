#!/usr/bin/env python
# coding: utf-8

# # 敵対的生成ネットワーク

# ```{attention}
# 作成途中！！！！！！！
# ```

# ## 基本的な事柄
# 
# 

# 敵対的生成ネットワークとは英語では generative adversarial network（GAN）と言う（主に）ニューラルネットワークを用いてデータを生成する方法です．入力データを変換することもできますが，やっていることは変更したデータを「生成」することであるため「生成する方法です」と書きました．GAN は現実世界の色々なところで既に応用されている技術です．とても重要な技術なのでここで紹介します．最も基本的な GAN を紹介した後に，その GAN の問題点を解決する GAN を紹介して，最後により便利な利用が可能な GAN を紹介します．

# ```{note}
# GAN の構造には必ずしもニューラルネットワークが含まれる必要はありません．
# ```

# ### 原理と利用方法

# 生成器，識別器，生成器のコスト関数，識別器のコスト関数

# GAN の構成要素はふたつです．生成器（generator）と識別器（discriminator）です．識別器は以下で紹介するワッサースタイン GAN を利用する場合は critic と名前を変えます．これの和訳は多分ありません．ここではクリティックと表記します．GAN はデータを生成（その派生として変換）する人工知能を作る方法です．最終的に得たいのは性能良く育った生成器です．識別器の役割は生成器をより良く育てることです．
# 
# GAN の学習は，より良い質の偽札を作ろうとする紙幣の偽造者とその偽札を見破る警察によるイタチごっこに例えられます．GAN においては偽造者が生成器で，警察が識別器です．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/gan.svg?raw=1" width="100%" />
# 
# GAN の学習において，
# 
# 

# ```{hint}
# 実用上，配列を入力にして配列を返す構造とだけ覚えておけば問題はないと思います．
# ```

# ### 色々な GAN

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

# ## 基本的な GAN の実装

# 最も基本的な GAN の構造は以下のプログラムに含まれるものです．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MiniBatchSize = 300
    NoiseSize = 100 # GANはランダムなノイズベクトルから何かを生成する方法なので，そのノイズベクトルのサイズを設定する．
    MaxEpoch = 500
    
    # データセットの読み込み
    (learnX, learnT), (_, _) = tf.keras.datasets.mnist.load_data()
    learnX = learnX.reshape([60000, 784])
    learnX = (learnX - 127.5) / 127.5
    
    # 生成器と識別器の構築
    generator = Generator() # 下のクラスを参照．
    discriminator = Discriminator() # 下のクラスを参照．
    
    # コスト関数と正確度を計算する関数の生成
    costComputer = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracyComputer = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # オプティマイザは生成器と識別器で別々のものを利用
    optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizerDiscriminator = tf.keras.optimizers.Adam(learning_rate=0.00004)
    
    # 生成器を成長させるためのコストを計算する関数
    def generatorCostFunction(discriminatorOutputFromGenerated):
        cost = costComputer(tf.ones(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated) # 生成器のコストの引数は生成器の出力を識別器に入れた結果．生成器としては全部正解を出しているはずなので教師はすべて1となるはず．そのように学習すべき．
        accuracy = accuracyComputer(tf.ones(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated)
        return cost, accuracy
    
    # 識別器を成長させるためのコストを計算する関数
    def discriminatorCostFunction(discriminatorOutputFromReal,discriminatorOutputFromGenerated): # 識別器のコストの引数は本物の情報を識別器に入れた結果と生成器の出力を識別器に入れた結果．
        realCost = costComputer(tf.ones(discriminatorOutputFromReal.shape[0]), discriminatorOutputFromReal) # 本物の情報の場合はすべて正例（1）と判断すべき．これが例のコスト関数の左の項に相当．
        fakeCost = costComputer(tf.zeros(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated) # 偽物の情報の場合はすべて負例（0）と判断すべき．これが例のコスト関数の右の項に相当．
        cost = realCost + fakeCost
        realAccuracy = accuracyComputer(tf.ones(discriminatorOutputFromReal.shape[0]), discriminatorOutputFromReal)
        fakeAccuracy = accuracyComputer(tf.zeros(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated)
        accuracy=(realAccuracy + fakeAccuracy) / 2
        return cost, accuracy
    
    @tf.function()
    def run(generator, discriminator, noiseVector, realVector):
        with tf.GradientTape() as generatorTape, tf.GradientTape() as discriminatorTape:
            generatedVector = generator(noiseVector) # 生成器によるデータの生成．
            discriminatorOutputFromGenerated = discriminator(generatedVector) # その生成データを識別器に入れる．
            discriminatorOutputFromReal = discriminator(realVector) # 本物データを識別器に入れる．
            # 識別器の成長
            discriminatorCost, discriminatorAccuracy = discriminatorCostFunction(discriminatorOutputFromReal, discriminatorOutputFromGenerated)
            gradientDiscriminator = discriminatorTape.gradient(discriminatorCost, discriminator.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
            optimizerDiscriminator.apply_gradients(zip(gradientDiscriminator, discriminator.trainable_variables))
            # 生成器の成長
            generatorCost, generatorAccuracy = generatorCostFunction(discriminatorOutputFromGenerated)
            gradientGenerator = generatorTape.gradient(generatorCost,generator.trainable_variables) # 生成器のパラメータで勾配を計算．
            optimizerGenerator.apply_gradients(zip(gradientGenerator,generator.trainable_variables))
            return discriminatorCost, discriminatorAccuracy, generatorCost, generatorAccuracy
    
    # ミニバッチセットの生成
    learnX = tf.data.Dataset.from_tensor_slices(learnX) # このような方法を使うと簡単にミニバッチを実装することが可能．
    learnT = tf.data.Dataset.from_tensor_slices(learnT)
    learnA = tf.data.Dataset.zip((learnX, learnT)).shuffle(60000).batch(MiniBatchSize) # 今回はインプットデータしか使わないけど後にターゲットデータを使う場合があるため．
    miniBatchNumber = len(list(learnA.as_numpy_iterator()))
    # 学習ループ
    for epoch in range(1,MaxEpoch+1):
        discriminatorCost, discriminatorAccuracy, generatorCost, generatorAccuracy = 0, 0, 0, 0
        for learnx, _ in learnA:
            noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
            discriminatorCostTmp, discriminatorAccuracyTmp, generatorCostTmp, generatorAccuracyTmp = run(generator, discriminator, noiseVector, learnx)
            discriminatorCost += discriminatorCostTmp / miniBatchNumber
            discriminatorAccuracy += discriminatorAccuracyTmp / miniBatchNumber
            generatorCost += generatorCostTmp / miniBatchNumber
            generatorAccuracy += generatorAccuracyTmp / miniBatchNumber
        # 疑似的なテスト
        if epoch%10 == 0:
            print("Epoch {:10d} D-cost {:6.4f} D-acc {:6.4f} G-cost {:6.4f} G-acc {:6.4f} ".format(epoch,float(discriminatorCost),float(discriminatorAccuracy),float(generatorCost),float(generatorAccuracy)))
            validationNoiseVector = generateNoise(1, NoiseSize)
            validationOutput = generator(validationNoiseVector)
            validationOutput = np.asarray(validationOutput).reshape([1, 28, 28])
            plt.imshow(validationOutput[0], cmap = "gray")
            plt.pause(1)

# 入力されたデータを0か1に分類するネットワーク
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.d1 = tf.keras.layers.Dense(units=128)
        self.d2 = tf.keras.layers.Dense(units=128)
        self.d3 = tf.keras.layers.Dense(units=128)
        self.d4 = tf.keras.layers.Dense(units=2, activation="softmax")
        self.a = tf.keras.layers.LeakyReLU()
        self.d = tf.keras.layers.Dropout(0.5)
    def call(self,x):
        y = self.d1(x)
        y = self.a(y)
        y = self.d(y)
        y = self.d2(y)
        y = self.a(y)
        y = self.d(y)
        y = self.d3(y)
        y = self.a(y)
        y = self.d(y)
        y = self.d4(y)
        return y

# 入力されたベクトルから別のベクトルを生成するネットワーク
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.d1=tf.keras.layers.Dense(units=256)
        self.d2=tf.keras.layers.Dense(units=256)
        self.d3=tf.keras.layers.Dense(units=784)
        self.a=tf.keras.layers.LeakyReLU()
        self.b1=tf.keras.layers.BatchNormalization()
        self.b2=tf.keras.layers.BatchNormalization()
    def call(self,x):
        y = self.d1(x)
        y = self.a(y)
        y = self.b1(y)
        y = self.d2(y)
        y = self.a(y)
        y = self.b2(y)
        y = self.d3(y)
        return y

def generateNoise(miniBatchSize, randomNoiseSize):
    return np.random.uniform(-1,1,size=(miniBatchSize,randomNoiseSize)).astype("float32")

if __name__ == "__main__":
    main()


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


# ## WGAN-gp の実装

# これまでに利用したものとは異なるモデルを利用したいとか，自身が持っているデータセットにより適合させたいとかの応用的な利用方法を紹介します．

# In[ ]:


#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)

def main():
    # ハイパーパラメータの設定
    MiniBatchSize = 300
    NoiseSize = 100 # GANはランダムなノイズベクトルから何かを生成する方法なので，そのノイズベクトルのサイズを設定する．
    MaxEpoch = 300
    CriticLearningNumber = 5
    GradientPenaltyCoefficient = 10
    
    # データセットの読み込み
    (learnX, learnT), (_, _) = tf.keras.datasets.mnist.load_data()
    learnX = np.asarray(learnX.reshape([60000, 784]), dtype="float32")
    learnX = (learnX - 127.5) / 127.5
    
    # 生成器と識別器の構築
    generator = Generator() # 下のクラスを参照．
    critic = Critic() # 下のクラスを参照．
    
    # オプティマイザは生成器と識別器で同じで良い．が，ハイパーパラメータを変えたくなるかもしれないからふたつ用意．
    optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
    optimizerCritic = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
    
    @tf.function()
    def runCritic(generator, critic, noiseVector, realVector):
        with tf.GradientTape() as criticTape:
            generatedVector = generator(noiseVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedVector) # その生成データを識別器に入れる．
            criticOutputFromReal = critic(realVector) # 本物データを識別器に入れる．
            epsilon = tf.random.uniform(generatedVector.shape, minval=0, maxval=1)
            intermediateVector = generatedVector + epsilon * (realVector - generatedVector)
            # 勾配ペナルティ
            with tf.GradientTape() as gradientPenaltyTape:
                gradientPenaltyTape.watch(intermediateVector)
                criticOutputFromIntermediate = critic(intermediateVector)
                gradientVector = gradientPenaltyTape.gradient(criticOutputFromIntermediate, intermediateVector)
                gradientNorm = tf.norm(gradientVector, ord="euclidean", axis=1) # gradientNorm = tf.sqrt(tf.reduce_sum(tf.square(gradientVector), axis=1)) と書いても良い．
                gradientPenalty = (gradientNorm - 1)**2
            # 識別器の成長
            criticCost = tf.reduce_mean(criticOutputFromGenerated - criticOutputFromReal + GradientPenaltyCoefficient * gradientPenalty) # 識別器を成長させるためのコストを計算．WGANの元論文の式そのまま．
            gradientCritic = criticTape.gradient(criticCost, critic.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
            optimizerCritic.apply_gradients(zip(gradientCritic, critic.trainable_variables))
            return criticCost
    
    @tf.function()
    def runGenerator(generator, critic, noiseVector):
        with tf.GradientTape() as generatorTape:
            generatedVector = generator(noiseVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedVector) # その生成データを識別器に入れる．
            # 生成器の成長
            generatorCost = -tf.reduce_mean(criticOutputFromGenerated) # 生成器を成長させるためのコストを計算．
            gradientGenerator = generatorTape.gradient(generatorCost,generator.trainable_variables) # 生成器のパラメータで勾配を計算．
            optimizerGenerator.apply_gradients(zip(gradientGenerator,generator.trainable_variables))
            return generatorCost
    
    # ミニバッチセットの生成
    learnX = tf.data.Dataset.from_tensor_slices(learnX) # このような方法を使うと簡単にミニバッチを実装することが可能．
    learnT = tf.data.Dataset.from_tensor_slices(learnT)
    learnA = tf.data.Dataset.zip((learnX, learnT)).shuffle(60000).batch(MiniBatchSize) # 今回はインプットデータしか使わないけど後にターゲットデータを使う場合があるため．
    miniBatchNumber = len(list(learnA.as_numpy_iterator()))
    # 学習ループ
    for epoch in range(1,MaxEpoch+1):
        criticCost, generatorCost = 0, 0
        for learnx, _ in learnA:
            # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
            for _ in range(CriticLearningNumber):
                noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
                criticCostPiece = runCritic(generator, critic, noiseVector, learnx)
                criticCost += criticCostPiece / (CriticLearningNumber * miniBatchNumber)
            # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
            for _ in range(1):
                noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
                generatorCostPiece = runGenerator(generator, critic, noiseVector)
                generatorCost += generatorCostPiece / miniBatchNumber
        # 疑似的なテスト
        if epoch%10 == 0:
            print("Epoch {:10d} D-cost {:6.4f} G-cost {:6.4f} ".format(epoch,float(criticCost),float(generatorCost)))
            validationNoiseVector = generateNoise(1, NoiseSize)
            validationOutput = generator(validationNoiseVector)
            validationOutput = np.asarray(validationOutput).reshape([1, 28, 28])
            plt.imshow(validationOutput[0], cmap = "gray")
            plt.pause(1)

# 入力されたデータを0か1に分類するネットワーク
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.d1 = tf.keras.layers.Dense(units=128)
        self.d2 = tf.keras.layers.Dense(units=128)
        self.d3 = tf.keras.layers.Dense(units=128)
        self.d4 = tf.keras.layers.Dense(units=1)
        self.a = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(0.5)
    def call(self,x):
        y = self.d1(x)
        y = self.a(y)
        y = self.dropout(y)
        y = self.d2(y)
        y = self.a(y)
        y = self.dropout(y)
        y = self.d3(y)
        y = self.a(y)
        y = self.dropout(y)
        y = self.d4(y)
        return y

# 入力されたベクトルから別のベクトルを生成するネットワーク
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.d1=tf.keras.layers.Dense(units=128)
        self.d2=tf.keras.layers.Dense(units=128)
        self.d3=tf.keras.layers.Dense(units=128)
        self.d4=tf.keras.layers.Dense(units=784)
        self.a=tf.keras.layers.LeakyReLU()
        self.b1=tf.keras.layers.BatchNormalization()
        self.b2=tf.keras.layers.BatchNormalization()
        self.b3=tf.keras.layers.BatchNormalization()
    def call(self,x):
        y = self.d1(x)
        y = self.a(y)
        y = self.b1(y)
        y = self.d2(y)
        y = self.a(y)
        y = self.b2(y)
        y = self.d3(y)
        y = self.a(y)
        y = self.b3(y)
        y = self.d4(y)
        y = tf.keras.activations.tanh(y)
        return y

def generateNoise(miniBatchSize, randomNoiseSize):
    return np.random.uniform(-1,1,size=(miniBatchSize,randomNoiseSize)).astype("float32")

if __name__ == "__main__":
    main()


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


# ## Conditional GAN の実装

# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/cgan.svg?raw=1" width="100%" />
# 

# ```{note}
# 終わりです．
# ```
