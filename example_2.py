# 2-3 어휘 집합 구축하기
# 1. 말뭉치 내려받기 및 전처리
# NSMC 다운로드
from Korpora import Korpora
nsmc = Korpora.load("nsmc", force_download=True)

# NSMC 전처리
import os
def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')
write_lines("C:/Users/74seh/example_of_NLP/Do it NLP/train.txt", nsmc.train.get_all_texts())
write_lines("C:/Users/74seh/example_of_NLP/Do it NLP/test.txt", nsmc.test.get_all_texts())

# 2. GPT 토크나이저 구축
# 디렉터리 만들기
os.makedirs("C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/bbpe",exist_ok=True)

# 바이트 수준 BPE 어휘집합 구축
from tokenizers import ByteLevelBPETokenizer
bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files = ["C:/Users/74seh/example_of_NLP/Do it NLP/train.txt", "C:/Users/74seh/example_of_NLP/Do it NLP/test.txt"], # 학습 말뭉치를 리스트 형태로 넣기
    vocab_size = 10000, # 어휘 집합 크기 조절
    special_tokens = ["[PAD]"] # 특수 토큰 추가
)
bytebpe_tokenizer.save_model("C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/bbpe")

# 3. BERT 토크나이저 구축
# 디렉터리 만들기
os.makedirs("C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/wordpiece",exist_ok=True)

# 워드피스 어휘집합 구축
from tokenizers import BertWordPieceTokenizer
wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files = ["C:/Users/74seh/example_of_NLP/Do it NLP/train.txt", "C:/Users/74seh/example_of_NLP/Do it NLP/test.txt"],
    vocab_size=10000,
)
wordpiece_tokenizer.save_model("C:/Users/74seh/example_of_NLP/Do it NLP/nlpbook/wordpiece")

