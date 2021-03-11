# Reinforcement Learning project
### movielens 데이터셋을 활용한 추천환경에의 강화학습 적용 실습


### 01. 강화학습(Reinforcement Learning)
환경과의 상호작용을 통해 어떤 상황에서 취해야 할 행동을 점차 학습해 나가는 것.

### 02. dataset
- [movielens small](https://grouplens.org/datasets/movielens/)
- ratings.csv(rating data), movies.csv(item data)

### 03. Actor-Critic
두 개의 네트워크를 사용함
> Actor-Critic 알고리즘에 영향을 받은 가장 유명한 예 : 딥마인드 AlphaGo
- Actor : 상태가 주어졌을 때 행동을 결정함(행동 확률의 계산)
    - 주어진 상황에서 action을 반환
    - logits(크기 = num_actions)
    - distribution : Categorical(주어진 logits을 사용하여 액션 하나를 추출)
- Critic : 상태의 가치를 평가(가치함수)
    - observation에 대한 가치를 평가(value)
    - value(크기 = 1)
    

#### 03-1) A2C(Advantage Actor Critic)
Actor-Critic의 기대출력으로 Advantage를 사용하는 경우
> Advantage : 예상했던 V(s)보다 얼마나 더 좋은가 -> (Q(s,a)-V(s)=A(s,a))

+) 여러개의 에이전트를 통시에 실행하는 A3C도 있다(Asynchronous)

### 04. Applying Gym Environment to MovieLens dataset
- actions : user의 1~5 사이의 rating 예측(np.eyes(5))
- observation : 유저별, 영화별 평점의 평균으로 초기화

**[Gym methods Override]**
- step : action을 취하면 그에 따른 환경에 대한 정보를 return
- reset : 상태 초기화
- render : 모드를 설정함('human', 'logger')
