#!/usr/bin/env python
# coding: utf-8

# # 強化学習法

# ```{note}
# 作成途中
# ```

# ## 基本的な事柄
# 
# 

# この章では強化学習法を紹介します．中でも Q 学習と呼ばれる方法を紹介します．その後，深層学習法を利用して Q 学習を実行する深層 Q 学習の紹介をします．この節ではそれらを理解するために必要な基礎知識の説明をします．

# ### 強化学習法とは

# 機械学習法は大きく分けて，教師あり学習法，教師なし学習法，強化学習法の 3 個に分類されます．教師あり学習法は入力データとそれに紐づいた教師データからなるデータを利用して，入力データに対して教師データに応じた出力をするように人工知能（機械学習モデル）を成長させるものです．強化学習法で成長させるものはエージェントと呼ばれる人工知能です．強化学習法をする際の登場人物は環境とエージェントのふたつです．エージェントが何らかの根拠に基づいて環境に作用します．例えば，ゲームで言うところの環境とはマップであったり空間であったり，そこに存在するキャラクターであったりしますが，それに作用するとは，プレイヤーの分身であるキャラクターを空間上で動かすこと等に相当します．このエージェントによる行動決定の根拠を方策と言います．強化学習法ではエージェントを成長させますが，これに与えるものは入力データと教師データの組み合わせではありません．エージェントは行動を選択した際のその行動の良さの指標である報酬という値を獲得します．エージェントは自身の行動に基づいてこの報酬を得ますが，環境の遷移が終わったとき（ゲームをクリアしたときや対戦の決着がついたとき）の報酬を最大化するように成長させられます．
# 
# 
# 

# ```{note}
# 環境は遷移します．例えば，囲碁というゲームでは碁をプレイヤーが動かす度に盤面が変化しますが，この盤面が別の盤面に変化することを環境の遷移と言います．
# ```

# ### 用語の整理

# 強化学習法ではこれまでの機械学習法では用いなかった用語を利用するため，それについての整理を行います．
# 
# **エージェント**とは強化学習法で成長させられる対象です．以下で紹介する Q 学習において成長させられるものは Q テーブルというものであるため厳密には違うのですが，エージェントはそれを参照して行動を決定するため，このように書きました．
# 
# **環境**とはエージェントが作用を及ぼす対象です．囲碁でいうところの盤面と碁が環境です．任天堂のマリオを動かすゲームでいうところのフィールドやマリオ等が環境です．この環境はエージェントが選択した行動によって遷移をします．
# 
# **報酬**とはエージェントを成長させるために与える得点です．何かの行動を選択した際に，その行動が良かったなら報酬をエージェントに与えます．エージェントは獲得できる報酬を最大化するように成長します．ただし，この行動に対して得られた報酬を即時的に処理してエージェントが成長するわけではありません．以下の価値関数のところで記述しますが，この報酬を長期的に獲得し続けた結果の価値という値を指標にしてエージェントは成長させられます．
# 
# **価値関数**とはその他の機械学習法でいうところの評価関数です．価値関数はさらに状態価値関数と行動価値関数というものに細分化されます．状態価値関数とはある方策のもとにおいて，ある状態であった場合に将来的にもらえる報酬の和を出力する関数です．一方で，行動価値関数とは，ある方策のものである状態であった場合にある行動を選択したときにもらえる報酬の和を出力する関数です．これは以下の節で紹介する Q 学習において Q 値と呼ばれる値です．報酬と価値は似て非なるものです．報酬はエージェントが何らかの行動を選択したときに即時的に得られる値ですが，価値は得られる報酬の期待値です．ここで報酬の和と表現した値です．
# 
# **方策**とはエージェントが行動を選択するために利用するルールです．意思決定則です．これに基づいてエージェントは行動を決定します．以下の項で紹介する方策ベース法では方策関数というものを利用しますが，これは現在の環境の状態を入力として行動（または行動の確率）を出力する関数です．
# 
# **エピソード**とはエージェントが環境に作用し環境の遷移が何らかの停止条件に到達し終了するまでの時間の単位です．エピソードの中において環境が遷移する時間の単位はステップとかサイクルと言います．全ステップが 1 エピソードということです．
# 
# **割引率**とは $\gamma$ で表される値です．エピソードの最初から最後までの報酬の和を価値と言いますが，現在の状態から未来の報酬を計算する場合，不確定要素が入ってくることを避けれません．それを吸収するために未来の報酬に 0 から 1 の間の値である $\gamma$ を掛けたもので和を計算します．この場合の報酬を割引報酬と言います．時刻 $t$ における割引報酬 $R_t$ は以下のように表されます．
# 
# $
# R_t=r_{t+1}+\gamma R_{t+1}
# $
# 
# **グリーディ（greedy）**な方法とは長期的に考えればより良い解が見つかることを考慮せずに，短期的に最も良いと考えられる解を見つけようとする方法のことです．これは強化学習の分野における専門的な言葉ではありません．この教材でははじめて出したような気がしたのでここで一応まとめてみました．

# ### 強化学習法の種類

# 強化学習法を実行する方法には様々な分類の方法があります．モデルに基づく方法か価値または方策に基づく方法か等の分類方法です．強化学習法の種類は現在開発が活発であり，どの方法がどのような分類となるかを整理して言及することをこの教材では避けます．
# 
# 最初に，モデルに基づく方法かそうでないかの観点から分類をすると，**動的計画法**はモデルに基づく方法です．動的計画法は正確な環境モデルを持っている場合に最適な方策を決定することが可能な方法です．エピソードを終了せずとも方策を最適化しようとすることが可能です．これ以降で紹介する方法はモデルに基づく方法ではありません．
# 
# モデルフリーの方法をさらに分類すると，価値ベース法，方策ベース法，アクター・クリティック法の 3 個に大別されます．
# 
# モデルフリーの方法であり，価値ベース法としては**モンテカルロ法**があります．モンテカルロ法はモデルに基づく方法である動的計画法とは異なり，完全な環境モデルを持たない場合，行動の結果として報酬が得られるかどうかが不明な際に用いられる方法です．エージェントの行動に応じて学習を進めます．エピソードを終了させることではじめて報酬を計算します．これを実績ベースであると言います．
# 
# モデルフリーの方法であり，価値ベースの方法であり，実績ベースでない方法としては**時間的差分法**（Temporal Difference（TD）法）があります．時間的差分法は上述のふたつを組み合わせたような方法です．行動に対して得られる実際の報酬から環境遷移の度に評価関数を改善することで最適な方策を見つけようとする方法です．この 3 個の方法の中では時間的差分法が最も現実的な学習法であると思います．時間的差分法はさらに，アルゴリズムの実装の違いにより細分化されます．エージェントの行動を決定する方策と評価関数の更新に用いる方策が異なる方策オフ型の方法である **Q 学習法**と，エージェントの行動を決定する方策と評価関数の更新に用いる方策が同じ方策オン型の方法である **SARSA** です．この章で紹介する方法は Q 学習です．
# 
# それ以外にモデルフリーの方法として，方策ベース法があります．また，モデルフリーの方法としてアクター・クリティック法という方法もありますが，これは時間的差分法の一種であるものの価値ベースの方法ではないもののことを言います．時間的差分法で価値ベースのものは Q 学習法と SARSA でしたね．アクター・クリティック法ではアクターという存在が行動を選択しますが，それを評価するものとしてクリティックという存在があり，アクターの行動をクリティックが評価することで方策を決定する方法です．
# 

# ```{note}
# ある状態とそのときに選択する行動から別の状態に遷移する確率と，ある状態から別の状態に遷移した際の報酬が明らかになっている方法がモデルベースの方法です．
# ```

# ```{note}
# 方策ベース法とは，エージェントが何らかの方策（ニューラルネットワークによって記述されるルールとか）を持っており，その方法を改善する方法です．価値ベース法とは，行動から価値を得て（価値関数の値を推定して），その価値を高める方策を選択する方法です．エージェントの詳細な制御方法が解らなくてもエピソードが成功裏に終わらせたという価値をもとにエージェントの成長をさせる方法です．
# ```

# ```{hint}
# より詳しく学びたい人は Bellman 方程式について調べてみてください．
# ```

# ```{note}
# 深層学習の時代において開発が進められている方法は，時間的差分法に分類される Q 学習（価値ベース）と同じくアクター・クリティック法です．
# ```

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
# このとき，$\alpha$ は学習率と呼ばれ，0 から 1 の値をとります．また，$\gamma$ は割引率であり，0 から 1 の値をとります．この割引率は直後の括弧内の値（TD 誤差という値）をどれくらい更新の際に考慮するかという値です．TD 誤差は目標の価値と現在の価値のずれを示しています．時間的差分法ではこの TD 誤差を小さくすることでより良い行動価値関数を推定する方法と言えます．
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
        self.qValues = {} # Qテーブル
        self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
    
    # 以下の関数は行動を選択する関数．
    def act(self, observation):
        self.observation = str(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
        else:
            action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
        return action
    
    # 以下はQテーブルを更新する関数．
    def update(self, objectNewPosition, action, reward):
        objectNewPosition = str(objectNewPosition)
        if objectNewPosition not in self.qValues: # Qテーブルのキーを新たに作る．
            self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
        q = self.qValues[self.observation][action]  # Q(s,a)の計算．
        maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
        self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．

if __name__ == "__main__":
    main()


# 実行した結果，エピソードを経るに従ってゴールに到達するまでに要した環境の更新数（`number of steps`）が小さくなり，その更新回数に渡って平均した報酬値（`mean reward`）が大きくなったことがわかります．つまり，学習（Q テーブルの更新）がうまく進みエージェントが成長したことがわかります．

# ```{note}
# この出力だけ見るとこのプログラムを実行して何が起こったのかわかりませんね．次の項でオブジェクトがフィールドをどのように動いたかを可視化します．
# ```

# プログラム最初から順に説明しますが，以下の部分では環境（マスとか移動するオブジェクトとかそのオブジェクトの位置とか）のインスタンスを生成します．その後，生成した環境を初期化します．次に環境中のオブジェクトを操作するエージェントのインスタンスを生成します．
# 
# ```python
#     env = Environment()
#     observation = env.reset()
#     agent = Agent(alpha=0.1, epsilon=0.3, gamma=0.9, actions=np.arange(4), observation=observation)
# ```

# 環境のクラス `Environment` は以下に示す通りです．以降で中身の要素の説明をします．
# 
# ```python
# class Environment:
#     def __init__(self):
#         self.actions = {"up": 0, "down": 1, "left": 2, "right": 3}
#         self.field = [["X", "X", "O", "G"],
#                       ["O", "O", "O", "O"],
#                       ["X", "O", "O", "O"],
#                       ["O", "O", "X", "O"],
#                       ["O", "O", "O", "O"]]
#         self.done = False
#         self.reward = None
#         self.iteration = None
#     
#     # 以下は環境を初期化する関数．
#     def reset(self):
#         self.objectPosition = 4, 0
#         self.done = False
#         self.reward = None
#         self.iteration = 0
#         return self.objectPosition
#     
#     # 以下は環境を進める関数．
#     def step(self, action):
#         self.iteration += 1
#         y, x = self.objectPosition
#         if self.checkMovable(x, y, action) == False: # オブジェクトの移動が可能かどうかを判定．
#             return self.objectPosition, -1, False # 移動できないときの報酬は-1．
#         else:
#             if action == self.actions["up"]:
#                 y += -1 # フィールドと座標の都合上，上への移動の場合は-1をする．
#             elif action == self.actions["down"]:
#                 y += 1
#             elif action == self.actions["left"]:
#                 x += -1
#             elif action == self.actions["right"]:
#                 x += 1
#             # 以下のifは報酬の計算とオブジェクトがゴールに到達してゲーム終了となるかどうかの判定のため．
#             if self.field[y][x] == "O":
#                 self.reward = 0
#             elif self.field[y][x] == "G":
#                 self.done = True
#                 self.reward = 100
#             self.objectPosition = y, x
#             return self.objectPosition, self.reward, self.done
#     
#     # 以下は移動が可能かどうかを判定する関数．
#     def checkMovable(self, x, y, action):
#         if action == self.actions["up"]:
#             y += -1
#         elif action == self.actions["down"]:
#             y += 1
#         elif action == self.actions["left"]:
#             x += -1
#         elif action == self.actions["right"]:
#             x += 1
#         if y < 0 or y >= len(self.field):
#             return False
#         elif x < 0 or x >= len(self.field[0]):
#             return False
#         elif self.field[y][x] == "X":
#             return False
#         else:
#             return True
# ```

# 以下の部分は環境の初期化をする記述です．最初に `self.actions` ですが，この環境が取りするコマンドはオブジェクトを上下左右に動かすためのもので，その記述です．次の `self.field` ですが，これは上に画像で示したフィールドを生成するためのものです．`X` が障害物，`G` がゴール，それ以外の `O` がオブジェクトが自由に移動できるマスです．`self.done` はオブジェクトがゴールに到達することでこのゲーム（オブジェクトをゴールまで動かすゲーム）が終了したかどうかを判定するための変数です．`self.reward` は報酬を格納する変数です．実はこの `self.iteration` はこのプログラムでは使わないのですが，後のレンダリングの際に必要なので加えています．
# 
# ```python
#     def __init__(self):
#         self.actions = {"up": 0, "down": 1, "left": 2, "right": 3}
#         self.field = [["X", "X", "O", "G"],
#                       ["O", "O", "O", "O"],
#                       ["X", "O", "O", "O"],
#                       ["O", "O", "X", "O"],
#                       ["O", "O", "O", "O"]]
#         self.done = False
#         self.reward = None
#         self.iteration = None
# ```

# ```{note}
# `self.reward` はインスタンス変数にしなくても良かったかもしれません．
# ```

# 環境はエピソード毎にリセットする必要がありますが，そのための記述です．オブジェクトは `(4, 0)` のマスに置きます．このメソッドは戻り値としてオブジェクトの位置を返します．
# 
# ```python
#     # 以下は環境を初期化する関数．
#     def reset(self):
#         self.objectPosition = 4, 0
#         self.done = False
#         self.reward = None
#         self.iteration = 0
#         return self.objectPosition
# ```

# 以下は環境を進めるための記述です．最初に `self.checkMovable` でオブジェクトを移動させることができるかを判定します．オブジェクトは壁の外に移動させることができないし，また，障害物のあるマスには移動させることができません．そのような場合は，報酬としては `-1` の値を与えて，また，オブジェクトの存在するマスを変化させません．それ以外の場合は，入力された上下左右のコマンド（`action`）に従ってオブジェクトの位置を変化させます．さらに，報酬は障害物やゴール以外のマスにオブジェクトが位置する場合は `0` でゴールの場合は `100` を与えるようにします．ゴールにオブジェクトが到達している場合は `self.done` に `True` を入れます．
# 
# ```python
#     # 以下は環境を進める関数．
#     def step(self, action):
#         self.iteration += 1
#         y, x = self.objectPosition
#         if self.checkMovable(x, y, action) == False: # オブジェクトの移動が可能かどうかを判定．
#             return self.objectPosition, -1, False # 移動できないときの報酬は-1．
#         else:
#             if action == self.actions["up"]:
#                 y += -1 # フィールドと座標の都合上，上への移動の場合は-1をする．
#             elif action == self.actions["down"]:
#                 y += 1
#             elif action == self.actions["left"]:
#                 x += -1
#             elif action == self.actions["right"]:
#                 x += 1
#             # 以下のifは報酬の計算とオブジェクトがゴールに到達してゲーム終了となるかどうかの判定のため．
#             if self.field[y][x] == "O":
#                 self.reward = 0
#             elif self.field[y][x] == "G":
#                 self.done = True
#                 self.reward = 100
#             self.objectPosition = y, x
#             return self.objectPosition, self.reward, self.done
#     
#     # 以下は移動が可能かどうかを判定する関数．
#     def checkMovable(self, x, y, action):
#         if action == self.actions["up"]:
#             y += -1
#         elif action == self.actions["down"]:
#             y += 1
#         elif action == self.actions["left"]:
#             x += -1
#         elif action == self.actions["right"]:
#             x += 1
#         if y < 0 or y >= len(self.field):
#             return False
#         elif x < 0 or x >= len(self.field[0]):
#             return False
#         elif self.field[y][x] == "X":
#             return False
#         else:
#             return True
# ```

# 次にエージェントのクラス `Agent` は以下に示す通りです．以降で中身の説明をします．
# 
# ```python
# class Agent:
#     def __init__(self, alpha=0.1, epsilon=0.3, gamma=0.9, actions=None, observation=None):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.actions = actions
#         self.observation = str(observation)
#         self.qValues = {} # Qテーブル
#         self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
#     
#     # 以下の関数は行動を選択する関数．
#     def act(self, observation):
#         self.observation = str(observation)
#         if np.random.uniform() < self.epsilon:
#             action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
#         else:
#             action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
#         return action
#     
#     # 以下はQテーブルを更新する関数．
#     def update(self, objectNewPosition, action, reward):
#         objectNewPosition = str(objectNewPosition)
#         if objectNewPosition not in self.qValues: # Qテーブルのキーを新たに作る．
#             self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
#         q = self.qValues[self.observation][action]  # Q(s,a)の計算．
#         maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
#         self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．
# ```

# 最初の記述は，エージェントが持つ変数を生成するためのものです．`self.alpha` や `self.gamma` は Q 値の更新式で利用するものです．`self.epsilon` は epsilon-greedy 法に利用する値です．`self.observation` は上の Q 値の更新式における `s` に相当するものです．マスの座標です．その後の Q テーブル構築の際のディクショナリのキーとなる値です．これを文字列化します．その次のディクショナリ `self.qValues` が Q テーブルです．ここに，ある状態におけるそれぞれの行動の Q 値を格納し，それを学習の過程で更新します．
# 
# ```python
#     def __init__(self, alpha=0.1, epsilon=0.3, gamma=0.9, actions=None, observation=None):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.actions = actions
#         self.observation = str(observation)
#         self.qValues = {} # Qテーブル
#         self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
# ```

# 以下の部分はエージェントが行動を選択するための記述です．epsilon-greedy 法を利用しています．確率的にランダムな行動を選択するか，または，これまでの Q テーブルを参照して最も Q 値が高い行動をとります．
# 
# ```python
#     # 以下の関数は行動を選択する関数．
#     def act(self, observation):
#         self.observation = str(observation)
#         if np.random.uniform() < self.epsilon:
#             action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
#         else:
#             action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
#         return action
# ```

# 最後の以下の部分は Q テーブルを更新するための記述です．行動選択によって新たなマスへの移動が行われる場合は新たに Q テーブルのキーを生成します．その後，現在の Q 値を計算（参照）し，また，環境遷移後の最も高い Q 値を示す行動の Q 値の値を計算します．これらは，上述の Q 値の更新式で利用する値です．これらを利用して Q 値を更新します．
# 
# ```python
#     # 以下はQテーブルを更新する関数．
#     def update(self, objectNewPosition, action, reward):
#         objectNewPosition = str(objectNewPosition)
#         if objectNewPosition not in self.qValues: # Qテーブルのキーを新たに作る．
#             self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
#         q = self.qValues[self.observation][action]  # Q(s,a)の計算．
#         maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
#         self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．
# ```

# 最後に，`main()` に戻って以下の部分の説明をします．エピソードは 50 回分計算します．環境の初期化をした後に，`agent.act()` によってオブジェクトにさせる行動を選択します．その行動を基に `env.step()` で環境を勧めます．引き続いて Q テーブルの更新を行います．もし，オブジェクトがゴールに達している場合はそれ以上のオブジェクトの移動や Q テーブルの更新を停止します．最後に，ゴールに到達するまでに要した環境遷移の回数と各エピソード毎に得られた報酬の平均値を出力します．
# 
# ```python
#     for episode in range(1, 50+1):
#         rewards = []
#         observation = env.reset() # 環境の初期化．
#         while True:
#             action = agent.act(observation) # エージェントによってオブジェクトにさせるアクションを選択する．
#             observation, reward, done = env.step(action) # 環境を進める．
#             agent.update(observation, action, reward) # Qテーブルの更新．
#             rewards.append(reward)
#             if done: break
#         print("Episode: {:3d}, number of steps: {:3d}, mean reward: {:6.3f}".format(episode, len(rewards), np.mean(rewards)))
# ```

# ### 環境の可視化

# 上のプログラムを実行しただけではエージェントによる行動の選択や環境の遷移によってどのようなことが起こっているのかがよくわかりませんでした．以下のプログラムを動かすとどのように環境が遷移したのかを観察することができます．フィールドの様子を可視化している点以外は上のブログラムと同じものです．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
np.random.seed(0)

# 環境をステップ毎に描画するようにしたもの．

def main():
    env = Environment()
    observation = env.reset()
    agent = Agent(alpha=0.1, epsilon=0.3, gamma=0.9, actions=np.arange(4), observation=observation)
    
    for episode in range(1, 50+1):
        rewards = []
        observation = env.reset() # 環境の初期化．
        env.render()
        while True:
            action = agent.act(observation) # エージェントによってオブジェクトにさせるアクションを選択する．
            observation, reward, done = env.step(action) # 環境を進める．
            env.render()
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
    
    # 以下はフィールドとオブジェクト（8）の様子を可視化する関数．
    def render(self):
        y, x = self.objectPosition
        field = [["X", "X", "O", "G"],
                 ["O", "O", "O", "O"],
                 ["X", "O", "O", "O"],
                 ["O", "O", "X", "O"],
                 ["O", "O", "O", "O"]]
        field[y][x] = "8"
        print("Iteration = {:3d}".format(self.iteration))
        for i in range(5):
            for j in range(4):
                print(field[i][j], end=" ")
            print()

class Agent:
    def __init__(self, alpha=0.1, epsilon=0.3, gamma=0.9, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.observation = str(observation)
        self.qValues = {} # Qテーブル
        self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
    
    # 以下の関数は行動を選択する関数．
    def act(self, observation):
        self.observation = str(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
        else:
            action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
        return action
    
    # 以下はQテーブルを更新する関数．
    def update(self, objectNewPosition, action, reward):
        objectNewPosition = str(objectNewPosition)
        if objectNewPosition not in self.qValues: # Qテーブルのキーを新たに作る．
            self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
        q = self.qValues[self.observation][action]  # Q(s,a)の計算．
        maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
        self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．

if __name__ == "__main__":
    main()


# プログラムを実行した結果，フィールドが表示され，環境遷移のイタレーションの度に `8` で表示されるオブジェクトがスタート位置からゴール位置へと移動している様子が可視化されました．

# ```{note}
# エピソードの数値が小さいとき，つまり，Q テーブルの性能が低い場合は `8` はフラフラしていますね．
# ```

# このプログラムが上のプログラムと変わっている点は環境のレンダリングをしているところです．プログラムでは以下の部分に `env.render()` という記述が加わっています．
# 
# ```python
#     for episode in range(1, 50+1):
#         rewards = []
#         observation = env.reset() # 環境の初期化．
#         env.render()
#         while True:
#             action = agent.act(observation) # エージェントによってオブジェクトにさせるアクションを選択する．
#             observation, reward, done = env.step(action) # 環境を進める．
#             env.render()
#             agent.update(observation, action, reward) # Qテーブルの更新．
#             rewards.append(reward)
#             if done: break
#         print("Episode: {:3d}, number of steps: {:3d}, mean reward: {:6.3f}".format(episode, len(rewards), np.mean(rewards)))
# ```

# クラス `Environment` の部分に，`env.render()` がしていることの記述があります．フィールドにオブジェクトの位置を代入してそれを表示するだけですね．
# 
# ```
#     # 以下はフィールドとオブジェクト（8）の様子を可視化する関数．
#     def render(self):
#         y, x = self.objectPosition
#         field = [["X", "X", "O", "G"],
#                  ["O", "O", "O", "O"],
#                  ["X", "O", "O", "O"],
#                  ["O", "O", "X", "O"],
#                  ["O", "O", "O", "O"]]
#         field[y][x] = "8"
#         print("Iteration = {:3d}".format(self.iteration))
#         for i in range(5):
#             for j in range(4):
#                 print(field[i][j], end=" ")
#             print()
# ```

# ### Q テーブルの出力

# 最後に Q テーブルがどんなように成長したのかを以下のプログラムで確認します．

# In[ ]:


#!/usr/bin/env python3
import numpy as np
np.random.seed(0)

# Qテーブルを出力可能にしたもの．

def main():
    env = Environment()
    observation = env.reset()
    agent = Agent(alpha=0.1, epsilon=0.3, gamma=0.9, actions=np.arange(4), observation=observation)
    
    for episode in range(1, 50+1):
        rewards = []
        observation = env.reset() # 環境の初期化．
        env.render()
        while True:
            action = agent.act(observation) # エージェントによってオブジェクトにさせるアクションを選択する．
            observation, reward, done = env.step(action) # 環境を進める．
            env.render()
            agent.update(observation, action, reward) # Qテーブルの更新．
            rewards.append(reward)
            if done: break
        print("Episode: {:3d}, number of steps: {:3d}, mean reward: {:6.3f}".format(episode, len(rewards), np.mean(rewards)))
        agent.outputQTable()

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
    
    # 以下はフィールドとオブジェクト（8）の様子を可視化する関数．
    def render(self):
        y, x = self.objectPosition
        field = [["X", "X", "O", "G"],
                 ["O", "O", "O", "O"],
                 ["X", "O", "O", "O"],
                 ["O", "O", "X", "O"],
                 ["O", "O", "O", "O"]]
        field[y][x] = "8"
        print("Iteration = {:3d}".format(self.iteration))
        for i in range(5):
            for j in range(4):
                print(field[i][j], end=" ")
            print()

class Agent:
    def __init__(self, alpha=0.1, epsilon=0.3, gamma=0.9, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.observation = str(observation)
        self.qValues = {} # Qテーブル
        self.qValues[self.observation] = np.repeat(0.0, len(self.actions))
    
    # 以下の関数は行動を選択する関数．
    def act(self, observation):
        self.observation = str(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.actions)) # イプシロンの確率でランダムに行動する．
        else:
            action = np.argmax(self.qValues[self.observation]) # 最もQ値が高い行動を選択．
        return action
    
    # 以下はQテーブルを更新する関数．
    def update(self, objectNewPosition, action, reward):
        objectNewPosition = str(objectNewPosition)
        if objectNewPosition not in self.qValues: # Qテーブルのキーを新たに作る．
            self.qValues[objectNewPosition] = np.repeat(0.0, len(self.actions))
        q = self.qValues[self.observation][action]  # Q(s,a)の計算．
        maxQ = max(self.qValues[objectNewPosition])  # max(Q(s',a'))の計算．
        self.qValues[self.observation][action] = q + (self.alpha * (reward + (self.gamma * maxQ) - q)) # Q'(s, a) = Q(s, a) + alpha * (reward + gamma * maxQ(s',a') - Q(s, a))の計算．
    
    # 以下はQテーブルを出力する関数．
    def outputQTable(self):
        print("Q-table:    Up   Down   Left  Right")
        for key in sorted(self.qValues.keys()):
            print(key, end=" ")
            for j in range(4):
                print("{:7.3f}".format(self.qValues[key][j]), end="")
            print()
        print()

if __name__ == "__main__":
    main()


# これは Q テーブルを出力する記述を加えただけなので説明は不要かもしれませんが，Q テーブルを出力するための記述はクラス `Agent` の以下の部分に追加しました．
# 
# ```python
#     # 以下はQテーブルを出力する関数．
#     def outputQTable(self):
#         print("Q-table:    Up   Down   Left  Right")
#         for key in sorted(self.qValues.keys()):
#             print(key, end=" ")
#             for j in range(4):
#                 print("{:7.3f}".format(self.qValues[key][j]), end="")
#             print()
#         print()
# ```

# 全てのエピソードが終了した後の Q テーブルは以下のようなものとなりました．例えば，ゴール付近のマスである，`(0, 2)` や `(1, 3)` ではどのような行動が高い Q 値を持つかと確認してみると，オブジェクトが `(0, 2)` のときは「右」に移動させることが，`(1, 3)` のときは「上」に移動させることが最も高い Q 値を示していました．良い Q テーブルへと成長したことが確認できます．
# 
# ```
# Q-table:    Up   Down   Left  Right
# (0, 2)  10.674  1.824 -0.190 99.030
# (0, 3)   0.000  0.000  0.000  0.000
# (1, 0)  -0.100 -0.100 -0.100  7.221
# (1, 1)  27.443  0.433  0.238 68.646
# (1, 2)  84.938  7.338 18.591 12.410
# (1, 3)  46.856  0.000  6.858  1.610
# (2, 1)  48.270  0.997  9.978 10.193
# (2, 2)  44.444  0.453  0.000  0.000
# (3, 0)   1.532  1.509  2.382 20.779
# (3, 1)  32.544  0.346  0.864  3.671
# (4, 0)  12.940 -0.254  1.262  0.341
# (4, 1)   0.457 -0.171  3.350  0.000
# (4, 2)  -0.271 -0.100  0.041  0.000
# ```

# ## OpenAI Gym

# OpenAI Gym を利用すると様々なゲームを Python からコントロールできるようになります．スーパーマリオブラザーズ等の有名なゲームも Python コード上で構築したエージェントが動かすことができるようになります．たくさんの Atari のゲームも動かすことができます．

# ```{hint}
# ただし，現在の OpenAI Gym には ROM ファイルは同梱されていません．著作権の問題なのかもしれませんが，詳しい事情は知りません．よって，マリオのゲームや Atari のゲームを利用したい場合はどこかからか ROM ファイルを入手する必要があります．
# ```

# ### インストール

# 以下のようなコマンドを打つことで OpenAI Gym をインストールすることができます．グーグルコラボラトリーにはあらかじめインストールされているためこのコマンドは打つ必要はありません．打っても良いです．

# In[ ]:


get_ipython().system('pip install gym')


# ### 環境の生成

# どのような環境（ゲーム）が利用可能であるかは以下のようなコードによって確認できます．

# In[ ]:


from gym import envs

def main():
    envids = [spec.id for spec in envs.registry.all()]
    print("The number of environments:", len(envs.registry.all())) # 全ての環境の個数を出力．
    for envid in envids:
        print(envid)

if __name__ == "__main__":
    main()


# この中の `CartPole-v1` というものを呼び出します．また，環境を初期化し，ランダムな入力から行動選択し，その行動によって環境を遷移させます．以下のように書きます．

# In[ ]:


import gym

def main():
    env = gym.make("CartPole-v1")
    observation = env.reset() # 環境の初期化，リセット．
    print(1,observation)
    action = env.action_space.sample() # ランダムな行動選択．強化学習をする際はここはエージェントによる行動選択となる．
    observation, reward, done, info = env.step(action) # 環境の遷移．現在の状況に対して行動を行って，次の状態，報酬，停止条件の真偽，ゲーム状態を出力．
    print(2,observation)

if __name__ == "__main__":
    main()


# 観測値として出力されている値は 4 個の要素からなるリストですが，最初から「カートの位置」，「カートの速さ」，「棒の角度」，「棒の角速度」です．何のことを言っているのかわからないと思いますが，次の項で可視化すると意味がわかると思います．

# ```{note}
# この gym を利用した場合に，`.reset()` とか `.step()` のような書き方があります．上の節で紹介した Q 学習の部分で環境のクラス `Enviroment` にも同様の方法がありましたが，あれは gym の挙動に似せて作ったものです．
# ```

# ### 可視化と画像化

# グーグルコラボラトリー可視化を行うためには特殊なコマンドを打って準備をする必要があります．以下のコマンドを打ちます．

# In[ ]:


get_ipython().system(' apt update')
get_ipython().system(' apt install xvfb')
get_ipython().system(' pip install pyvirtualdisplay')


# 以下のプログラムを実行して描画に必要なディスプレイの起動をします．

# In[2]:


from pyvirtualdisplay import Display

def main():
    gymDisplay = Display()
    gymDisplay.start()

if __name__ == "__main__":
    main()


# ```{hint}
# **描画をする前にこのコマンドを打たなければなりません．これを打つ前に以下にあるプログラムを実行するとエラーが出ます．気をつけてください．**やってしまった場合はグーグルコラボラトリーのランタイムを削除して再起動してやり直してください．
# ```

# 以下のようなプログラムでゲームの実行画面を描画することができます．ここでは描画の方法を 2 個紹介しますが，その 1 個目です．

# In[ ]:


import gym
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import animation

def main():
    env = gym.make('CartPole-v1')

    images = [] # 描画のための記述
    figure = plt.figure() # 描画のための記述
    ax = plt.gca() # 描画をきれいにするための記述
    ax.axes.xaxis.set_visible(False) # 描画をきれいにするための記述
    ax.axes.yaxis.set_visible(False) # 描画をきれいにするための記述
    observation = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        image = plt.imshow(env.render(mode="rgb_array")) # 描画のための記述
        images.append([image]) # 描画のための記述

        if done: env.reset()

    generatedAnimation = animation.ArtistAnimation(figure, images, interval=15, blit=True) # 描画のための記述
    display.display(display.HTML(generatedAnimation.to_jshtml())) # 描画のための記述

if __name__ == "__main__":
    main()


# 上のプログラムでは動画をグーグルコラボラトリー上で表示しましたが，次のプログラムを実行すると動画を GIF ファイルとして保存することができます．

# In[ ]:


import gym
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import animation

def main():
    env = gym.make('CartPole-v1')

    images = []
    figure = plt.figure()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    observation = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        image = plt.imshow(env.render(mode="rgb_array"))
        images.append([image])

        if done: env.reset()

    generatedAnimation = animation.ArtistAnimation(figure, images, interval=15, blit=True)
    generatedAnimation.save("cartpole-01.gif", writer="pillow", fps=50) # ここだけ変化．

if __name__ == "__main__":
    main()


# グーグルコラボラトリーの左にあるサイドバー上のフォルダのアイコンをクリックすると保存されたファイルを確認することができます．そのファイル名をダブルクリックすると画面右に動画が表示されます．

# ### 環境のインポート

# ```{note}
# この項の記述はこの教材を実行する際に必要なものではありません．もし他の色々なゲームを導入したい場合にやってみてください．
# ```

# この gym で実際に呼び出せる環境はデフォルトのままだととても少ないです．Atari の ROM を入手してそれを呼び出したい場合は以下のようにします．
# 
# 最初に以下のコマンドで Atari Learning Environment をインストールします．
# 
# ```
# ! pip install ale-py
# ```
# 
# 次に ROM ファイルを入れたフォルダをグーグルコラボラトリーを利用しているユーザーのグーグルドライブに置きます．グーグルドライブのフォルダをグーグルコラボラトリーから参照できるように以下のコードを実行します．
# 
# ```python
# #!/usr/bin/env python3
# from google.colab import drive
# 
# def main():
#     drive.mount("/content/drive", force_remount=True)
# 
# if __name__ == "__main__":
#     main()
# ```
# 
# そのフォルダの名前が `atari2600` であったとします．その場合，以下のようなコマンドで ROM ファイルをインポートします．
# 
# ```
# ! ale-import-roms drive/MyDrive/atari2600/
# ```
# 
# この結果，以下のような Atari に由来する環境を呼び出せるようになります．
# 
# ```python
# import gym
# 
# def main():
#     env = gym.make("MontezumaRevengeDeterministic-v4")
# 
# if __name__ == "__main__":
#     main()
# ```

# ## 深層 Q 学習

# Q 学習に深層学習法を利用した深層 Q 学習という方法を紹介します．

# ### 深層学習と Q 学習

# ```{note}
# 終わりです．
# ```
