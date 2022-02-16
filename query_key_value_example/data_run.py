import torch
from data import x, w_key, w_query, w_value

# query, key, value(쿼리, 키, 밸류) 만들기
keys = torch.matmul(x, w_key)               # torch.matmul은 행렬곱을 수행하는 함수
querys = torch.matmul(x, w_query)
values = torch.matmul(x, w_value)


# attention score 만들기
attn_scores = torch.matmul(querys, keys.T)  # keys.T는 키 벡터들을 전치한 행렬ㅠ
print(attn_scores)                          # 어텐션 스코어 체크


# softmax 확률값 만들기
import numpy as np
from torch.nn.functional import softmax
key_dim_sqrt = np.sqrt(keys.shape[-1])      # 앞 코드 결과에 키 벡터의 차원수 제곱근으로 나눈 후 소프트맥스를 취하는 과정
attn_probs = softmax(attn_scores / key_dim_sqrt, dim = 1)
print(attn_probs)                           # 위 과정을 수행한 결과 값


# 소프트맥스 확률과 밸류를 가중합하기
weighted_values = torch.matmul(attn_probs, values)
print(weighted_values)


## 셀프 어텐션 학습 대상은 쿼리, 키 밸류를 만드는 가중치 행렬(W_Q, W_K, W_V)이다.
## 코드에서는 w_query, w_key, w_value                                    
## 이들은 태스크(예: 기계번역)를 가장 잘 수행하는 방향으로 학습 과정에서 업데이트 됨
