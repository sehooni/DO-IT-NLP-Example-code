# DO-IT_NLP_Example-code
교재 'Do It NLP' 내의 교재들을 직접 코드화 하여 진행해보는 예제이다.

본래 colab을 이용하여 진행된 예제들로서,

컴퓨터 내, cuda와 cuDNN을 제대로 설치된 환경에서,
경로만 바르게 설정하여 주면 코드 실행이 가능하다.


## GPT_model & tokenizer
어디까지나 자연어처리에서의 토큰화과정을 연습해보는 코드

anaconda3 가상 환경으로, python 3.7, cuda 11.2, cuDNN 8.1 환경입니다.

노트북 사양: NVIDIA GeForce GTX 1650 Ti로 gpu를 지원.

python file: 주피터노트북으로 돌리던 환경을 컴퓨터 자체로 가동시키는 파일.


## checking_GPU
정상적으로 작동할 경우, 일시적으로 GPU를 가속시켜 작업환경에서 cuda가 작동됨을 확인할 수 있다.


## query_key_value_example
self-attention : 쿼리, 키, 밸류 3개 요소 사이의 문맥적 관계성을 추출하는 과정

self-attention 과정에서 쿼리, 키, 밸류를 어떻게 학습하는가를 보여주는 코드
