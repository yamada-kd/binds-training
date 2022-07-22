#!/usr/bin/env python
# coding: utf-8

# # 強化学習法

# ```{note}
# 作成途中
# ```

# ## 基本的な事柄
# 
# 

# ほげ

# ### 強化学習法とは

# ほげ
# 
# ほげ
# 
# $
# \displaystyle P(\boldsymbol{\theta})=\frac{1}{N}\sum_{i=1}^{N}\log(1-D(\boldsymbol{\phi},G(\boldsymbol{\theta},\boldsymbol{z}_i)))
# $
# 
# 
# 
# 

# ```{note}
# 元論文の式，これ厳密でしょうか？これだけ見せられたら意味わからないです．
# ```

# ### 強化学習法の種類

# ほげ
# 

# ## Q 学習

# この節ではフィールド上を動いてゴールを目指すオブジェクトの動きをコントロールするというタスクを通じて Q 学習の利用方法を理解します．

# ### 解きたい問題

# 以下のような 5 行 4 列の行列を考えます．これはフィールドです．このフィールド上で何らかのオブジェクト（人でも犬でも何でも良いです）を動かしてゴール（G）を目指すこととします．オブジェクトはフィールド上のグリッド線で囲まれたどこかの位置に存在できることとします．このグリッド線で囲まれた位置のことをこれ以降マスと呼びます．オブジェクトは上下左右に動かせるものとします．ただし，フィールドは壁に囲まれているとします．つまり，オブジェクトはこの 5 行 4 列のフィールドの外には出られません．また，フィールドには障害物（X）があり，障害物が存在する位置にオブジェクトは行けないこととします．オブジェクトは最初ゴールでも障害物でもないオブジェクトが移動可能な普通マス（S），座標にして `(4, 0)` に位置しているものとします．このオブジェクトをうまく移動させてゴールまで導くエージェントを作ることをここでの問題とします．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/field.svg?raw=1" width="100%" />

# ```{hint}
# 左上のマスを `(0, 0)` として右下のマスを `(4, 3)` とします．つまり行列の要素の呼び方と同じですね．
# ```

# ```{note}
# この動かす対象であるオブジェクトのことをエージェントとして記載している記事がインターネット上にたくさんあります．例えば，囲碁における碁に相当するような存在だと思いますが，それをエージェントと呼ぶことはないと思います．エージェントとはプレイヤーであって，ゲームで言うところの環境の一部であるキャラクターや碁等のオブジェクトを移動させたりオブジェクトに何かをさせたりする存在であると思います．よってここではオブジェクトとエージェントを明確に使い分けます．
# ```

# ### Q 学習で行うこと

# 強化学習はエージェントがもらえる報酬を最大化するように学習を行います．エージェントがオブジェクトを動かした結果ゴールまでオブジェクトを導けたのであれば多くの報酬を獲得できます．一方で，障害物があるマスにはオブジェクトを移動できませんがその際には報酬が減ります．その他のマスにオブジェクトを移動させる場合は報酬は得られません．
# 
# このような状況において，Q 学習では報酬を最大化するために参照する Q テーブルなるものを構築します．学習の過程でこの Q テーブルは更新されます．エージェントは良い Q テーブルを参照することによってより良い性能でオブジェクトを的確に動かせるようになります．つまり，Q 学習で成長させられるものは Q テーブルです．Q テーブルには Q 値が格納されます．というよりは Q 値の集合が Q テーブルです．Q 値は環境がある状態 $s$ にあるときに $a$ というアクションをエージェントがとることの良さを示す値です．$Q(s,a)$ と表します．

# ```{hint}
# 例えば，あるマス「`(4, 0)`」という「状態」でエージェントがオブジェクトを「上に移動させる」という「アクション」をとったときの良さを表すものが Q 値です．
# ```

# ```{note}
# Q 学習の Q は quality のイニシャルです．
# ```

# Q 値の更新式を紹介します．ここでは更新された Q 値を $Q'$ と書きます．Q 値はエージェントがある行動をとって環境が遷移した後に更新されます．この環境遷移前の Q 値を $Q$，状態を $s$，行動を $a$，獲得した報酬を $r$ と書きます．また，環境遷移後の状態を $s'$，環境遷移後にとり得る行動を $a'$ と書きます．このとき，Q 値の更新式は以下のように表されます．
# 
# $
# \displaystyle Q'(s,a)=Q(s,a)+\alpha(r+\gamma\max Q(s',a')-Q(s,a))
# $
# 
# このとき，$\alpha$ は学習率と呼ばれ，0 から 1 の値をとります．また，$\gamma$ は割引率であり，0 から 1 の値をとります．この割引率は直後の括弧内の値（TD（temporal difference）誤差という値）をどれくらい更新の際に考慮するかという値です．
# 
# この式において，$\max Q(s',a')$ は現在の行動によって遷移した状態において取り得るすべての行動（この場合上下左右の 4 個）に対する Q 値の中で最も大きな値のことです．

# ```{hint}
# TD 誤差の部分を確認していただければわかるように，Q 値の更新式はある状態から次の状態に遷移したとき，最初の状態の Q 値を次の状態の最も大きな Q 値と報酬の和に近づけることを意味しています．例えば，ゴール直前の状態とそのひとつ前の状態を考えたとき，ひとつ前の状態の Q 値はゴール直前の状態に遷移しようとするように値が大きくなります．
# ```

# ### Q 学習の実装

# 実際の Q 学習は以下の手順で行われます．以下の項目の括弧内の記述はこの節で扱う問題に関する記述です．
# 
# 1.   環境の初期化（オブジェクトをスタートのマスに置く）．
# 2.   エージェントによる行動選択（オブジェクトを動かす方向である上下左右を選択する）．
# 3.   環境の更新（オブジェクトを動かそうとする）．
# 4.   環境の更新がそれ以上され得るかの終了条件の判定（ゴールに到達したかどうかを確認）．
# 5.   Q テーブルの更新．
# 6.   上記 1 から 5 の繰り返し．
# 
# 繰り返し作業の単位をエピソードと言います．上の 1 から 5 で 1 エピソードの計算です．学習の最中に epsilon-greedy 法という方法を利用して行動選択を行っています．epsilon-greedy 法とは $\epsilon$ の確率でランダムに行動選択をし，それ以外の $(1-\epsilon)$ の確率では最も Q 値の高い行動を選択する方法です．
# 
# 以上の Q 学習を実行するためのコードは以下のものです．
# 

# In[ ]:


#!/usr/bin/env python3
import numpy as np
np.random.seed(0)

def main():
    env = Environment()
    observation = env.reset()
    agent = Agent(alpha=0.1, epsilon=0.3, gamma=0.9, actions=np.arange(4), observation=observation)
    
    for episode in range(1, 50+1):
        rewards = []
        observation = env.reset() # 環境の初期化．
        while True:
            action = agent.act(observation) # エージェントによってオブジェクトにさせるアクションを選択する．
            observation, reward, done = env.step(action) # 環境を進める．
            agent.update(observation, action, reward) # Qテーブルの更新．
            rewards.append(reward)
            if done: break
        print("Episode: {:3d}, number of steps: {:3d}, mean reward: {:6.3f}".format(episode, len(rewards), np.mean(rewards)))

class Environment:
    def __init__(self):
        self.actions = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.field = [["X", "X", "O", "G"],
                      ["O", "O", "O", "O"],
                      ["X", "O", "O", "O"],
                      ["O", "O", "X", "O"],
                      ["O", "O", "O", "O"]]
        self.done = False
        self.reward = None
        self.iteration = None
    
    # 以下は環境を初期化する関数．
    def reset(self):
        self.objectPosition = 4, 0
        self.done = False
        self.reward = None
        self.iteration = 0
        return self.objectPosition
    
    # 以下は環境を進める関数．
    def step(self, action):
        self.iteration += 1
        y, x = self.objectPosition
        if self.checkMovable(x, y, action) == False: # オブジェクトの移動が可能かどうかを判定．
            return self.objectPosition, -1, False # 移動できないときの報酬は-1．
        else:
            if action == self.actions["up"]:
                y += -1 # フィールドと座標の都合上，上への移動の場合は-1をする．
            elif action == self.actions["down"]:
                y += 1
            elif action == self.actions["left"]:
                x += -1
            elif action == self.actions["right"]:
                x += 1
            # 以下のifは報酬の計算とオブジェクトがゴールに到達してゲーム終了となるかどうかの判定のため．
            if self.field[y][x] == "O":
                self.reward = 0
            elif self.field[y][x] == "G":
                self.done = True
                self.reward = 100
            self.objectPosition = y, x
            return self.objectPosition, self.reward, self.done
    
    # 以下は移動が可能かどうかを判定する関数．
    def checkMovable(self, x, y, action):
        if action == self.actions["up"]:
            y += -1
        elif action == self.actions["down"]:
            y += 1
        elif action == self.actions["left"]:
            x += -1
        elif action == self.actions["right"]:
            x += 1
        if y < 0 or y >= len(self.field):
            return False
        elif x < 0 or x >= len(self.field[0]):
            return False
        elif self.field[y][x] == "X":
            return False
        else:
            return True

class Agent:
    def __init__(self, alpha=0.1, epsilon=0.3, gamma=0.9, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.observation = str(observation)
        self.previous_action = None
        self.qValues = {} # Qテーブル
        self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
    
    # 以下の関数は行動を選択する関数．
    def act(self, observation):
        self.observation = str(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
        else:
            action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
        self.previous_action = action
        return action
    
    # 以下はQテーブルを更新する関数．
    def update(self, objectNewPosition, action, reward):
        objectNewPosition = str(objectNewPosition)
        if objectNewPosition not in self.qValues:  # 始めて訪れる状態であれば
            self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
        q = self.qValues[self.observation][action]  # Q(s,a)の計算．
        maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
        self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．

if __name__ == "__main__":
    main()


# 実行した結果，エポックを経るに従って数字が含まれたような画像が生成されたはずです．GAN の学習の過程では，生成器と識別器の性能は拮抗するはずなので，生成器の正確度と識別器の正確度はどちらも 0.5 に収束すると良いと考えられます．

# 以下の部分ではハイパーパラメータを設定します．ミニバッチに含まれるデータのサイズは 300 個にしました．GAN はランダムに発生させたノイズから何らかのデータを生成するものですが，このノイズのサイズ `NoiseSize` を 100 に設定しました．変更しても良いです．学習は今回の場合，500 回行うことにしました．実際はコストの値を観察しながら過学習が起こっていないエポックで学習を止めると良いと思います．
# 
# ```python
#     # ハイパーパラメータの設定
#     MiniBatchSize = 300
#     NoiseSize = 100 # GANはランダムなノイズベクトルから何かを生成する方法なので，そのノイズベクトルのサイズを設定する．
#     MaxEpoch = 500
# ```

# データセットの読み込みの部分ですが，TensorFlow が提要してくれる MNIST は 0 から 255 の値からなるデータなので，これを -1 から 1 の範囲からなるデータに変換しています．
# 
# ```python
#     # データセットの読み込み
#     (learnX, learnT), (_, _) = tf.keras.datasets.mnist.load_data()
#     learnX = np.asarray(learnX.reshape([60000, 784]), dtype="float32")
#     learnX = (learnX - 127.5) / 127.5
# ```

# 以下の部分では生成器と識別器を生成しています．
# 
# ```python
#     # 生成器と識別器の構築
#     generator = Generator() # 下のクラスを参照．
#     discriminator = Discriminator() # 下のクラスを参照．
# ```
# 
# 生成器を生成するためのクラスは以下に示す通りです．入力されたノイズデータに対して 3 層の全結合層の計算がされます．各層の活性化関数は Leaky ReLU です．また，バッチノーマライゼーションも利用しています．最終層のユニットサイズは 784 ですが，これは MNIST 画像のピクセル数に相当します．
# 
# ```python
# # 入力されたベクトルから別のベクトルを生成するネットワーク
# class Generator(tf.keras.Model):
#     def __init__(self):
#         super(Generator,self).__init__()
#         self.d1=tf.keras.layers.Dense(units=256)
#         self.d2=tf.keras.layers.Dense(units=256)
#         self.d3=tf.keras.layers.Dense(units=784)
#         self.a=tf.keras.layers.LeakyReLU()
#         self.b1=tf.keras.layers.BatchNormalization()
#         self.b2=tf.keras.layers.BatchNormalization()
#     def call(self,x):
#         y = self.d1(x)
#         y = self.a(y)
#         y = self.b1(y)
#         y = self.d2(y)
#         y = self.a(y)
#         y = self.b2(y)
#         y = self.d3(y)
#         y = tf.keras.activations.tanh(y)
#         return y
# ```
# 
# 識別器を生成するためのクラスは以下に示す通りです．今回の場合，識別器の入力データは 784 ピクセルの画像データです．これを入力にして，このデータが真か偽かを識別します．よって，最終層のユニットサイズは 2 です．
# 
# ```python
# # 入力されたデータを0か1に分類するネットワーク
# class Discriminator(tf.keras.Model):
#     def __init__(self):
#         super(Discriminator,self).__init__()
#         self.d1 = tf.keras.layers.Dense(units=128)
#         self.d2 = tf.keras.layers.Dense(units=128)
#         self.d3 = tf.keras.layers.Dense(units=128)
#         self.d4 = tf.keras.layers.Dense(units=2, activation="softmax")
#         self.a = tf.keras.layers.LeakyReLU()
#         self.d = tf.keras.layers.Dropout(0.5)
#     def call(self,x):
#         y = self.d1(x)
#         y = self.a(y)
#         y = self.d(y)
#         y = self.d2(y)
#         y = self.a(y)
#         y = self.d(y)
#         y = self.d3(y)
#         y = self.a(y)
#         y = self.d(y)
#         y = self.d4(y)
#         return y
# ```

# 最適化法は生成器と識別器で異なるものを利用した方が良い場合があります．経験的に識別器の方が性能が出やすいので，ここでは学習率を少し小さく設定しています．
# 
# ```python
#     # オプティマイザは生成器と識別器で別々のものを利用
#     optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=0.0001)
#     optimizerDiscriminator = tf.keras.optimizers.Adam(learning_rate=0.00004)
# ```

# 以下の部分は生成器を成長させるためのコストを計算するコスト関数を含む記述です．これの引数は `discriminatorOutputFromGenerated` と書いていますが，これは生成器がノイズから生成したデータを識別器に入力したときに識別器が出力した値です．つまり，`0` または `1` です．`cost` からはじまる行では `tf.ones` で `1` という値からなるデータをミニバッチのサイズ分生成しています．それを `discriminatorOutputFromGenerated` と比較しています．この差を小さくしたいわけなので，つまり，生成器から出力されたデータはすべて真であると学習させるということを意味しています．
# 
# ```python
#     # 生成器を成長させるためのコストを計算する関数
#     def generatorCostFunction(discriminatorOutputFromGenerated):
#         cost = costComputer(tf.ones(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated) # 生成器のコストの引数は生成器の出力を識別器に入れた結果．生成器としては全部正解を出しているはずなので教師はすべて1となるはず．そのように学習すべき．
#         accuracy = accuracyComputer(tf.ones(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated)
#         return cost, accuracy
# ```

# 以下は識別器を成長させるためのコストを計算するコスト関数を含む記述です．識別器の引数は `discriminatorOutputFromReal` と `discriminatorOutputFromGenerated` で，それぞれ，本物のデータを入力されたときの識別器の出力値と偽物のデータを入力されたときの識別器の出力値です．上の生成器の場合と同じ書き方をしているので確認してほしいのですが，識別器は本物のデータが入力されたら本物と識別するように，また，偽物のデータが入力されたら偽物と識別するように成長させられます．
# 
# ```python
#     # 識別器を成長させるためのコストを計算する関数
#     def discriminatorCostFunction(discriminatorOutputFromReal,discriminatorOutputFromGenerated): # 識別器のコストの引数は本物の情報を識別器に入れた結果と生成器の出力を識別器に入れた結果．
#         realCost = costComputer(tf.ones(discriminatorOutputFromReal.shape[0]), discriminatorOutputFromReal) # 本物の情報の場合はすべて正例（1）と判断すべき．これが例のコスト関数の左の項に相当．
#         fakeCost = costComputer(tf.zeros(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated) # 偽物の情報の場合はすべて負例（0）と判断すべき．これが例のコスト関数の右の項に相当．
#         cost = realCost + fakeCost
#         realAccuracy = accuracyComputer(tf.ones(discriminatorOutputFromReal.shape[0]), discriminatorOutputFromReal)
#         fakeAccuracy = accuracyComputer(tf.zeros(discriminatorOutputFromGenerated.shape[0]), discriminatorOutputFromGenerated)
#         accuracy=(realAccuracy + fakeAccuracy) / 2
#         return cost, accuracy
# ```

# 以下は 1 回あたりの学習のための記述です．`tf.GradientTape()` を 2 個利用している点がちょっと珍しい書き方かもしれません．`generatedInputVector` からはじまる行はノイズデータからデータを生成する記述です．そしてそのデータを識別器に入力して得られる値が次の行の `discriminatorOutputFromGenerated` で，また，本物のデータを識別器に入力して得られる値が次の行の `discriminatorOutputFromReal` です．
# 
# ```python
#     @tf.function()
#     def run(generator, discriminator, noiseVector, realInputVector):
#         with tf.GradientTape() as generatorTape, tf.GradientTape() as discriminatorTape:
#             generatedInputVector = generator(noiseVector) # 生成器によるデータの生成．
#             discriminatorOutputFromGenerated = discriminator(generatedInputVector) # その生成データを識別器に入れる．
#             discriminatorOutputFromReal = discriminator(realInputVector) # 本物データを識別器に入れる．
#             # 識別器の成長
#             discriminatorCost, discriminatorAccuracy = discriminatorCostFunction(discriminatorOutputFromReal, discriminatorOutputFromGenerated)
#             gradientDiscriminator = discriminatorTape.gradient(discriminatorCost, discriminator.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
#             optimizerDiscriminator.apply_gradients(zip(gradientDiscriminator, discriminator.trainable_variables))
#             # 生成器の成長
#             generatorCost, generatorAccuracy = generatorCostFunction(discriminatorOutputFromGenerated)
#             gradientGenerator = generatorTape.gradient(generatorCost,generator.trainable_variables) # 生成器のパラメータで勾配を計算．
#             optimizerGenerator.apply_gradients(zip(gradientGenerator,generator.trainable_variables))
#             return discriminatorCost, discriminatorAccuracy, generatorCost, generatorAccuracy
# ```

# 以下の記述はミニバッチセットを生成するために TensorFlow が用意してくれたユーティリティを使うためのものです．
# 
# ```python
#     # ミニバッチセットの生成
#     learnX = tf.data.Dataset.from_tensor_slices(learnX) # このような方法を使うと簡単にミニバッチを実装することが可能．
#     learnT = tf.data.Dataset.from_tensor_slices(learnT)
#     learnA = tf.data.Dataset.zip((learnX, learnT)).shuffle(60000).batch(MiniBatchSize) # 今回はインプットデータしか使わないけど後にターゲットデータを使う場合があるため．
#     miniBatchNumber = len(list(learnA.as_numpy_iterator()))
# ```

# ```{note}
# 便利ですねこれ．
# ```

# 学習ループに関しては特に説明しませんが，ループ内で用いられている `generateNoise` という名の関数はランダムなノイズを発生させるためのものです．
# 
# ```python
# def generateNoise(miniBatchSize, randomNoiseSize):
#     return np.random.uniform(-1, 1, size=(miniBatchSize,randomNoiseSize)).astype("float32")
# ```

# ### 環境の可視化

# 上のプログラムを実行した際には観測できなかったかもしれませんが，GAN の学習を行った結果得られる人工知能（学習済み生成器）が同じような出力しかしなくなる現象があります．MNIST を例にすると 1 が含まれる画像しか生成しなくなるような現象です．この現象のことをモード崩壊と言います．英語では mode collpase と書きます．データを生成するために利用する GAN なので，利用目的にも依りますが，この現象は普通好ましくないものであると考えられます．これを解決しようとした WGAN-gp を以下で紹介します．
# 
# また，生成器の学習がとても進みにくいという問題もあります．最適化法を変更するとか，ハイパーパラメータの設定を慎重に行うとか，学習データサイズを増やすとかの色々な対策をとる必要があります．GAN の学習をうまく進めるために様々なテクニックがインターネット上でまとめられていますが，それらを自分の問題に合わせて取捨選択すると良いと思います．
# 
# その他の GAN の問題点としては，上のプログラムを実行した結果から確認できたと思いますが，生成したいデータの条件を指定できないことです．例えば，MNIST のような数字が含まれた画像の中でも 2 が含まれる画像のみを生成したい場合，上のプログラムではかなり蕪雑な方法を使わないとできません．これを解決しようとした方法を以下で紹介します．

# ### Q テーブルの出力

# ## OpenAI Gym

# 基本的な GAN の改良版である WGAN-gp の実装方法を紹介します．

# ### WGAN-gp とは

# WGAN-gp は基本的な GAN で起こりやすく問題となるモード崩壊を解決しようとした方法です．GAN では何らかの分布に従ってデータが生成していると考えますが，この分布と本物のデータが生成される分布を近づけようとします．基本的な GAN の学習で行っている行為はそのふたつの分布間のヤンセン・シャノンダイバージェンスと言う値を小さくしようとすることに相当します．これに対して WGAN-gp ではより収束性能に優れたワッサースタイン距離を小さくしようとします．詳しくは元の論文を参照してください．
# 
# WGAN-gp の学習における生成器のコスト関数は以下の式で表されます．
# 
# $
# \displaystyle P(\boldsymbol{\theta})=-\frac{1}{N}\sum_{i=1}^{N}D(\boldsymbol{\phi},G(\boldsymbol{\theta},\boldsymbol{z}_i))
# $
# 
# また，WGAN-gp において識別器は真か偽の二値を識別して出力するものではなくて，実数を出力するものへと変わるため，これを識別器と呼ばず，クリティックと呼ぶ場合があるため，ここでもそのように呼びます．クリティックのコスト関数は以下の式で表されます．
# 
# $
# \displaystyle Q(\boldsymbol{\phi})=\frac{1}{N}\sum_{i=1}^{N}(D(\boldsymbol{\phi},G(\boldsymbol{\theta},\boldsymbol{z}_i))-D(\boldsymbol{\phi},\boldsymbol{x}_i)+\lambda(\|\boldsymbol{g}(\boldsymbol{\phi},\boldsymbol{\hat{x}}_i)\|_2-1)^2)
# $
# 
# ここでも，生成器を $G$，クリティックを $D$ で表します．それぞれのニューラルネットワークのパラメータは $\boldsymbol{\theta}$ と $\boldsymbol{\phi}$ で，また，生成器とクリティックへ入力されたデータの個数を $N$ とします．生成器への入力データである $N$ 個のノイズの $i$ 番目のデータを $\boldsymbol{z}_i$ と表し，クリティックへの $i$ 番目の入力データを $\boldsymbol{x}_i$ と表します．最後に，生成器のコスト関数を $P$，クリティックのコスト関数を $Q$ で表します．このクリティックのコスト関数にあるラムダを含む項は勾配ペナルティです．英語では gradient penalty（gp）です．このラムダはハイパーパラメータであり元の論文では10に設定されています．また，$\boldsymbol{g}$ は勾配ベクトル場であり，以下のように定義されます．
# 
# $
# \displaystyle \boldsymbol{g}(\boldsymbol{\phi},\boldsymbol{x})=\frac{\partial D(\boldsymbol{\phi},\boldsymbol{x})}{\partial\boldsymbol{x}}
# $
# 
# また，$\boldsymbol{\hat{x}}_i$ は以下の式で計算される値です．
# 
# $
# \boldsymbol{\hat{x}}_i=\epsilon\boldsymbol{x}_i+(1-\epsilon)G(\boldsymbol{\theta},\boldsymbol{z}_i)
# $
# 
# 式中で用いられている $\epsilon$ は，最小値が 0 で最大値が 1 の一様分布からサンプリングしたランダムな値です．

# ### WGAN-gp の実装

# WGAN-gp を実装します．このプログラムでも MNIST の学習データセットを読み込んで，類似した数字画像を出力する人工知能を構築します．以下のように書きます．

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
    def runCritic(generator, critic, noiseVector, realInputVector):
        with tf.GradientTape() as criticTape:
            generatedInputVector = generator(noiseVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedInputVector) # その生成データを識別器に入れる．
            criticOutputFromReal = critic(realInputVector) # 本物データを識別器に入れる．
            epsilon = tf.random.uniform(generatedInputVector.shape, minval=0, maxval=1)
            intermediateVector = generatedInputVector + epsilon * (realInputVector - generatedInputVector)
            # 勾配ペナルティ
            with tf.GradientTape() as gradientPenaltyTape:
                gradientPenaltyTape.watch(intermediateVector)
                criticOutputFromIntermediate = critic(intermediateVector)
                gradientVector = gradientPenaltyTape.gradient(criticOutputFromIntermediate, intermediateVector)
                gradientNorm = tf.norm(gradientVector, ord="euclidean", axis=1) # gradientNorm = tf.sqrt(tf.reduce_sum(tf.square(gradientVector), axis=1)) と書いても良い．
                gradientPenalty = GradientPenaltyCoefficient * (gradientNorm - 1)**2
            # 識別器の成長
            criticCost = tf.reduce_mean(criticOutputFromGenerated - criticOutputFromReal + gradientPenalty) # 識別器を成長させるためのコストを計算．WGANの元論文の式そのまま．
            gradientCritic = criticTape.gradient(criticCost, critic.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
            optimizerCritic.apply_gradients(zip(gradientCritic, critic.trainable_variables))
            return criticCost
    
    @tf.function()
    def runGenerator(generator, critic, noiseVector):
        with tf.GradientTape() as generatorTape:
            generatedInputVector = generator(noiseVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedInputVector) # その生成データを識別器に入れる．
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
            print("Epoch {:10d} D-cost {:6.4f} G-cost {:6.4f}".format(epoch,float(criticCost),float(generatorCost)))
            validationNoiseVector = generateNoise(1, NoiseSize)
            validationOutput = generator(validationNoiseVector)
            validationOutput = np.asarray(validationOutput).reshape([1, 28, 28])
            plt.imshow(validationOutput[0], cmap = "gray")
            plt.pause(1)

# 入力されたデータを評価するネットワーク
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
    return np.random.uniform(-1, 1, size=(miniBatchSize,randomNoiseSize)).astype("float32")

if __name__ == "__main__":
    main()


# このプログラムを実行すると，上の基本的な GAN を用いたときよりも奇麗な画像が出力されたのではないかと思います．また，生成器が綺麗な画像を出力しだす実時間も基本的な GAN を用いた場合よりも早かったのではないでしょうか．

# ```{note}
# WGAN-gp が優れた方法であることがわかりますね．
# ```

# 以下の部分はハイパーパラメータの設定部分ですが，基本的な GAN と比べて，`CriticLearningNumber` が新たに加わっています．これは，生成器のパラメータ更新 1 回に対してクリティックのパラメータ更新をさせる回数です．基本的な GAN の学習をうまく進めるために生成器の学習回数を増やすことがあるのですが，WGAN-gp ではクリティックの方の学習回数を増やします．また，`GradientPenaltyCoefficient` は勾配ペナルティに欠ける係数です．これはハイパーパラメータなのですが，元の論文では 10 に設定されていたため，ここでも 10 にしました．
# 
# ```python
#     # ハイパーパラメータの設定
#     MiniBatchSize = 300
#     NoiseSize = 100 # GANはランダムなノイズベクトルから何かを生成する方法なので，そのノイズベクトルのサイズを設定する．
#     MaxEpoch = 300
#     CriticLearningNumber = 5
#     GradientPenaltyCoefficient = 10
# ```

# WGAN-gp では生成器とクリティックの学習回数を変えるため，パラメータ更新のための関数は別々に用意する必要があります．以下はクリティックのパラメータ更新を行うための記述です．
# 
# ```python
#     @tf.function()
#     def runCritic(generator, critic, noiseVector, realInputVector):
#         with tf.GradientTape() as criticTape:
#             generatedInputVector = generator(noiseVector) # 生成器によるデータの生成．
#             criticOutputFromGenerated = critic(generatedInputVector) # その生成データを識別器に入れる．
#             criticOutputFromReal = critic(realInputVector) # 本物データを識別器に入れる．
#             epsilon = tf.random.uniform(generatedInputVector.shape, minval=0, maxval=1)
#             intermediateVector = generatedInputVector + epsilon * (realInputVector - generatedInputVector)
#             # 勾配ペナルティ
#             with tf.GradientTape() as gradientPenaltyTape:
#                 gradientPenaltyTape.watch(intermediateVector)
#                 criticOutputFromIntermediate = critic(intermediateVector)
#                 gradientVector = gradientPenaltyTape.gradient(criticOutputFromIntermediate, intermediateVector)
#                 gradientNorm = tf.norm(gradientVector, ord="euclidean", axis=1) # gradientNorm = tf.sqrt(tf.reduce_sum(tf.square(gradientVector), axis=1)) と書いても良い．
#                 gradientPenalty = GradientPenaltyCoefficient * (gradientNorm - 1)**2
#             # 識別器の成長
#             criticCost = tf.reduce_mean(criticOutputFromGenerated - criticOutputFromReal + gradientPenalty) # 識別器を成長させるためのコストを計算．WGANの元論文の式そのまま．
#             gradientCritic = criticTape.gradient(criticCost, critic.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
#             optimizerCritic.apply_gradients(zip(gradientCritic, critic.trainable_variables))
#             return criticCost
# ```
# 
# 上で紹介したクリティックのコストを計算するための記述が含まれています．上の式の $\epsilon$ は `epsilon` からはじまる行で生成されます．単に一様分布からのサンプリングです．`intermediateVector` は $\hat{x}$ です．さらに，クリティックの出力に対してこの $\hat{x}$ に対する勾配を計算する必要がありますが，それを行っているのが `gradientVector` の行の記述です．引き続き `gradientPenalty` を行っています．その下の `criticCost` からはじまる行はクリティックのコストを求める上の式そのものです．

# 以下の記述は生成器のコストを求めるためのものです．`generatorCost` からはじまる行が生成器を求めるための上の式そのものなので理解しやすいのではないでしょうか．
# 
# ```python
#     @tf.function()
#     def runGenerator(generator, critic, noiseVector):
#         with tf.GradientTape() as generatorTape:
#             generatedInputVector = generator(noiseVector) # 生成器によるデータの生成．
#             criticOutputFromGenerated = critic(generatedInputVector) # その生成データを識別器に入れる．
#             # 生成器の成長
#             generatorCost = -tf.reduce_mean(criticOutputFromGenerated) # 生成器を成長させるためのコストを計算．
#             gradientGenerator = generatorTape.gradient(generatorCost,generator.trainable_variables) # 生成器のパラメータで勾配を計算．
#             optimizerGenerator.apply_gradients(zip(gradientGenerator,generator.trainable_variables))
#             return generatorCost
# ```

# 学習ループの部分は基本的な GAN のものとほぼ同じなのですが，識別器のパラメータ更新の回数を生成器のそれと変えるため，`for _ in range(CriticLearningNumber):` の部分でハイパーパラメータとして設定した分だけパラメータ更新のループを設定しています．
# 
# ```python
#     # 学習ループ
#     for epoch in range(1,MaxEpoch+1):
#         criticCost, generatorCost = 0, 0
#         for learnx, _ in learnA:
#             # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
#             for _ in range(CriticLearningNumber):
#                 noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
#                 criticCostPiece = runCritic(generator, critic, noiseVector, learnx)
#                 criticCost += criticCostPiece / (CriticLearningNumber * miniBatchNumber)
#             # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
#             for _ in range(1):
#                 noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
#                 generatorCostPiece = runGenerator(generator, critic, noiseVector)
#                 generatorCost += generatorCostPiece / miniBatchNumber
#         # 疑似的なテスト
#         if epoch%10 == 0:
#             print("Epoch {:10d} D-cost {:6.4f} G-cost {:6.4f}".format(epoch,float(criticCost),float(generatorCost)))
#             validationNoiseVector = generateNoise(1, NoiseSize)
#             validationOutput = generator(validationNoiseVector)
#             validationOutput = np.asarray(validationOutput).reshape([1, 28, 28])
#             plt.imshow(validationOutput[0], cmap = "gray")
#             plt.pause(1)
# ```

# ## 深層 Q 学習

# この節では条件を指定することで条件に合ったデータを生成することができる GAN の改良版である CGAN の実装方法を紹介します．

# ### CGAN とは

# GAN のプログラムで確認したように，0から9までの手書き数字の文字が含まれた画像を入力データとして学習させた学習済みのGANの生成器は，ランダムに生成されたノイズを入力データとして，ランダムに0から9までの数字が描かれた画像を生成することができます．生成される数字はノイズに応じたランダムなものであり，特定の数字が描かれた画像を意図的に出力させることはできません．これに対して，CGAN は生成器への入力データとしてノイズに加えて，何らかの条件を入力情報として与えることができる方法です．この条件を例えば0から9までの数字として設定して学習すれば，学習済みの生成器に特定の数字を条件として入力することで特定の数字を含む画像だけを生成させることができます．CGAN の全体像は以下の図に示す通りです．ほぼ GAN と同様なのですが，条件データを生成器と識別器に入れることができます．
# 
# <img src="https://github.com/yamada-kd/binds-training/blob/main/image/cgan.svg?raw=1" width="100%" />
# 
# 基本的な GAN は新たなデータを生成することが可能でしたが，CGANを利用すれば，データの変換を行うことができます．例えば，人の顔画像を出力すように学習させた生成器に「笑う」，「怒る」，「悲しむ」のような条件を同時に与えることで画像中の人の表情を変化させることができます．また，風景画を出力するように学習させた生成器に「歌川広重」，「フェルメール」，「レンブラント」等のような画家（転じて画風）の条件を与えることで，指定した画家が描いたような風景画を出力させることができます．
# 

# ```{note}
# 上手に使えば色々なことを実現できます．
# ```

# ### CGAN の実装

# CGAN を実装します．ただし，この CGAN では基本的な GAN の学習法ではなくて WGAN-gp の方法を使っています．WGAN-gp が非常に強力な方法だからです．よって，これは元々の CGAN でなくて CWGAN-gp とでも呼ぶべきものです．このプログラムでも MNIST の学習データセットを読み込んで，類似した数字画像を出力する人工知能を構築します．以下のように書きます．

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
    def runCritic(generator, critic, noiseVector, realInputVector, realConditionVector):
        with tf.GradientTape() as criticTape:
            generatedInputVector = generator(noiseVector, realConditionVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedInputVector, realConditionVector) # その生成データを識別器に入れる．
            criticOutputFromReal = critic(realInputVector, realConditionVector) # 本物データを識別器に入れる．
            epsilon = tf.random.uniform(generatedInputVector.shape, minval=0, maxval=1)
            intermediateVector = generatedInputVector + epsilon * (realInputVector - generatedInputVector)
            # 勾配ペナルティ
            with tf.GradientTape() as gradientPenaltyTape:
                gradientPenaltyTape.watch(intermediateVector)
                criticOutputFromIntermediate = critic(intermediateVector, realConditionVector)
                gradientVector = gradientPenaltyTape.gradient(criticOutputFromIntermediate, intermediateVector)
                gradientNorm = tf.norm(gradientVector, ord="euclidean", axis=1) # gradientNorm = tf.sqrt(tf.reduce_sum(tf.square(gradientVector), axis=1)) と書いても良い．
                gradientPenalty = GradientPenaltyCoefficient * (gradientNorm - 1)**2
            # 識別器の成長
            criticCost = tf.reduce_mean(criticOutputFromGenerated - criticOutputFromReal + gradientPenalty) # 識別器を成長させるためのコストを計算．WGANの元論文の式そのまま．
            gradientCritic = criticTape.gradient(criticCost, critic.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
            optimizerCritic.apply_gradients(zip(gradientCritic, critic.trainable_variables))
            return criticCost
    
    @tf.function()
    def runGenerator(generator, critic, noiseVector, generatedConditionVector):
        with tf.GradientTape() as generatorTape:
            generatedInputVector = generator(noiseVector, generatedConditionVector) # 生成器によるデータの生成．
            criticOutputFromGenerated = critic(generatedInputVector, generatedConditionVector) # その生成データを識別器に入れる．
            # 生成器の成長
            generatorCost = -tf.reduce_mean(criticOutputFromGenerated) # 生成器を成長させるためのコストを計算．
            gradientGenerator = generatorTape.gradient(generatorCost,generator.trainable_variables) # 生成器のパラメータで勾配を計算．
            optimizerGenerator.apply_gradients(zip(gradientGenerator,generator.trainable_variables))
            return generatorCost
    
    # ミニバッチセットの生成
    learnX = tf.data.Dataset.from_tensor_slices(learnX) # このような方法を使うと簡単にミニバッチを実装することが可能．
    learnT = tf.data.Dataset.from_tensor_slices(learnT)
    learnA = tf.data.Dataset.zip((learnX, learnT)).shuffle(60000).batch(MiniBatchSize) # インプットデータもターゲットデータも両方使うため．
    miniBatchNumber = len(list(learnA.as_numpy_iterator()))
    # 学習ループ
    for epoch in range(1,MaxEpoch+1):
        criticCost, generatorCost = 0, 0
        for learnx, learnt in learnA:
            # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
            for _ in range(CriticLearningNumber):
                noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
                criticCostPiece = runCritic(generator, critic, noiseVector, learnx, learnt)
                criticCost += criticCostPiece / (CriticLearningNumber * miniBatchNumber)
            # WGAN-gpでは識別器1回に対して生成器を複数回学習させるのでそのためのループ．
            for _ in range(1):
                noiseVector = generateNoise(MiniBatchSize, NoiseSize) # ミニバッチサイズで100個の要素からなるノイズベクトルを生成．
                generatedConditionVector = generateConditionVector(MiniBatchSize)
                generatorCostPiece = runGenerator(generator, critic, noiseVector, generatedConditionVector)
                generatorCost += generatorCostPiece / miniBatchNumber
        # 疑似的なテスト
        if epoch%10 == 0:
            print("Epoch {:10d} D-cost {:6.4f} G-cost {:6.4f}".format(epoch,float(criticCost),float(generatorCost)))
            validationNoiseVector = generateNoise(1, NoiseSize)
            validationConditionVector = generateConditionVector(1)
            validationOutput = generator(validationNoiseVector, validationConditionVector)
            validationOutput = np.asarray(validationOutput).reshape([1, 28, 28])
            plt.imshow(validationOutput[0], cmap = "gray")
            plt.text(1, 2.5, int(validationConditionVector[0]), fontsize=20, color="white")
            plt.pause(1)

# 入力されたデータを評価するネットワーク
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.d1 = tf.keras.layers.Dense(units=128)
        self.d2 = tf.keras.layers.Dense(units=128)
        self.d3 = tf.keras.layers.Dense(units=128)
        self.d4 = tf.keras.layers.Dense(units=1)
        self.a = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.embed = tf.keras.layers.Embedding(input_dim=10, output_dim=64, mask_zero=False)
        self.concatenate = tf.keras.layers.Concatenate()
    def call(self,x,c):
        y = self.d1(x)
        c = self.embed(c)
        y = self.concatenate([y, c])
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
        self.d1 = tf.keras.layers.Dense(units=128)
        self.d2 = tf.keras.layers.Dense(units=128)
        self.d3 = tf.keras.layers.Dense(units=128)
        self.d4 = tf.keras.layers.Dense(units=784)
        self.a = tf.keras.layers.LeakyReLU()
        self.b1 = tf.keras.layers.BatchNormalization()
        self.b2 = tf.keras.layers.BatchNormalization()
        self.b3 = tf.keras.layers.BatchNormalization()
        self.embed = tf.keras.layers.Embedding(input_dim=10, output_dim=64, mask_zero=False)
        self.concatenate = tf.keras.layers.Concatenate()
    def call(self,x,c):
        y = self.d1(x)
        c = self.embed(c)
        y = self.concatenate([y, c])
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
    return np.random.uniform(-1, 1, size=(miniBatchSize,randomNoiseSize)).astype("float32")

def generateConditionVector(miniBatchSize):
    return np.random.choice(range(10), size=(miniBatchSize))

if __name__ == "__main__":
    main()


# 実行した結果として生成される画像の中心から左上の辺りに数字が表示されていると思います．これはランダムに発生させた条件です．ランダムに 0 から 9 の範囲内にある整数が選択されます．この整数で指定した条件と同じような手書き数字（のようなもの）を出力してほしいのですが，結果を確認すると意図通りにできていますよね．

# CGAN の説明はこれまでのプログラムが理解できている人には不要かもしれません．WGAN-gp の実装変わっている点は 1 点だけです．以下はクリティックのコストを求めるための記述ですが，引数がひとつ増えています．`realConditionVector` が増えているのですが，これは条件を指定するためのベクトルです．生成器とクリティックの入力ベクトルとして条件データを入力する必要があるため，これが新たに加わっただけです．その他の計算は WGAN-gp のものと全く同じです．
# 
# ```python
#     @tf.function()
#     def runCritic(generator, critic, noiseVector, realInputVector, realConditionVector):
#         with tf.GradientTape() as criticTape:
#             generatedInputVector = generator(noiseVector, realConditionVector) # 生成器によるデータの生成．
#             criticOutputFromGenerated = critic(generatedInputVector, realConditionVector) # その生成データを識別器に入れる．
#             criticOutputFromReal = critic(realInputVector, realConditionVector) # 本物データを識別器に入れる．
#             epsilon = tf.random.uniform(generatedInputVector.shape, minval=0, maxval=1)
#             intermediateVector = generatedInputVector + epsilon * (realInputVector - generatedInputVector)
#             # 勾配ペナルティ
#             with tf.GradientTape() as gradientPenaltyTape:
#                 gradientPenaltyTape.watch(intermediateVector)
#                 criticOutputFromIntermediate = critic(intermediateVector, realConditionVector)
#                 gradientVector = gradientPenaltyTape.gradient(criticOutputFromIntermediate, intermediateVector)
#                 gradientNorm = tf.norm(gradientVector, ord="euclidean", axis=1) # gradientNorm = tf.sqrt(tf.reduce_sum(tf.square(gradientVector), axis=1)) と書いても良い．
#                 gradientPenalty = GradientPenaltyCoefficient * (gradientNorm - 1)**2
#             # 識別器の成長
#             criticCost = tf.reduce_mean(criticOutputFromGenerated - criticOutputFromReal + gradientPenalty) # 識別器を成長させるためのコストを計算．WGANの元論文の式そのまま．
#             gradientCritic = criticTape.gradient(criticCost, critic.trainable_variables) # 識別器のパラメータだけで勾配を計算．つまり生成器のパラメータは行わない．
#             optimizerCritic.apply_gradients(zip(gradientCritic, critic.trainable_variables))
#             return criticCost
# ```

# 以下のミニバッチを構築するための記述の `learnA` からはじまる行のコメントに注目してください．これまではここに，「今回はインプットデータしか使わないけど後にターゲットデータを使う場合があるため．」と書いていましたが，ここでは，「インプットデータもターゲットデータも両方使うため．」と書きました．CGAN では MNIST の教師データを学習ループ内で使うためです．
# 
# ```python
#     # ミニバッチセットの生成
#     learnX = tf.data.Dataset.from_tensor_slices(learnX) # このような方法を使うと簡単にミニバッチを実装することが可能．
#     learnT = tf.data.Dataset.from_tensor_slices(learnT)
#     learnA = tf.data.Dataset.zip((learnX, learnT)).shuffle(60000).batch(MiniBatchSize) # インプットデータもターゲットデータも両方使うため．
#     miniBatchNumber = len(list(learnA.as_numpy_iterator()))
# ```

# ```{note}
# 終わりです．
# ```
