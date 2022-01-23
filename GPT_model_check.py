# 2-4 토큰화하기

# 1. GPT 입력값 만들기
# GPT 토크나이저 선언
from transformers import GPT2Tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained("C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/bbpe")
tokenizer_gpt.pad_token = "[PAD]"

# GPT 토크나이저로 토큰화하기
sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠... 포스터보고 초딩영화인줄...오버연기조차 가볍지 않구나",
    "별루 였다..",
]
tokenized_sentences = [tokenizer_gpt.tokenize(sentence) for sentence in sentences]
print("GPT 토크나이저 토큰화 결과 : ")
print(tokenized_sentences) # GPT 토크나이저 토큰화 결과

# GPT 모델 입력 만들기
batch_inputs = tokenizer_gpt(
    sentences,
    padding="max_length",# 문장의 최대 길이에 맞춰 패딩
    max_length=12, # 문장의 토큰 기준 최대 길이
    truncation=True, # 문장 잘림 허용 옵션
)

    # 위 코드의 실행 결과로 2가지 입력값이 생성됨
    # input_ids와 attention_mask
    # input_ids는 토큰 인덱스 시퀀스를 나타냄
    # attention_mask는 일반토큰이 자리한 곳(1)과 패딩토큰이 자리한 곳(0)을 구분해 알려주는 장치
print("input_ids 체크 : ")
print(batch_inputs["input_ids"]) #input_ids 체크
print("attention_mask 체크 : ")
print(batch_inputs["attention_mask"]) # attention_mask 체크

# 2. BERT 입력값 만들기
# BERT 토크나이저 선언
from transformers import BertTokenizer
tokenizer_bert = BertTokenizer.from_pretrained(
    "C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/wordpiece",
    do_lower_case=False,
)

# BERT 토크나이저로 토큰화하기
sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠... 포스터보고 초딩영화인줄...오버연기조차 가볍지 않구나",
    "별루 였다..",
]
tokenized_sentences = [tokenizer_bert.tokenize(sentence) for sentence in sentences]
print("BERT 토크나이저 토큰화 결과 : ")
print(tokenized_sentences) # BERT 토크나이저 토큰화 결과

# BERT 모델 입력 만들기
batch_inputs = tokenizer_bert(
    sentences,
    padding="max_length",
    max_length=12,
    truncation=True,
)

    # 위 코드의 실행 결과로 3가지 입력값이 생성됨
    # GPT 모델과 마찬가지로 input_ids, attention_mask와 더불어 token_type_ids를 생성
    # token_type_ids 는 segment에 해당하는 것으로 모두 0.
    # 세그먼트 정보를 입력하는 것 BERT 모델의 특징
print("input_ids 체크 : ")
print(batch_inputs["input_ids"]) #input_ids 체크
print("attention_mask 체크 : ")
print(batch_inputs["attention_mask"]) # attention_mask 체크
print("token_type_ids 체크 : ")
print(batch_inputs["token_type_ids"]) # token_type_ids 체크
