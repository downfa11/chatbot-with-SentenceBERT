import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests


openai.api_key = "KEY"
model = SentenceTransformer('jhgan/ko-sroberta-multitask')


df = pd.read_csv('drive/MyDrive/Colab Notebooks/NLP/SentenceBERT/wellness_dataset_original.csv')
df = df.drop(columns=['Unnamed: 3']) # Blank Columns 제거
df = df[~df['챗봇'].isna()] # isna : NaN 제거

qa_dict = {}


df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

def calculate_distance(text, model):
    embedding = model.encode(text)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    return df

# 거리가 가장 먼 질문
def get_max_distance_answer(df):
    answer = qa_dict[df.loc[df['distance'].idxmax(), '유저']]
    return answer

df_new = pd.DataFrame(columns=['유저', '챗봇', 'embedding', 'distance'])

# 유저 입력 받아 거리가 일정 이상이면 저장, 이하면 질문 리스트에서 가장 유사한 질문에 해당하는 답변을 가져옴
while True:
    text = input('유저: ')
    df = calculate_distance(text, model)
    max_distance = df['distance'].max()
    if max_distance < 0.5:
        sim_dict = {}
        for q, a in qa_dict.items():
          sim = cosine_similarity([model.encode(q)], [model.encode(text)]).squeeze()
          sim_dict[q] = sim
          max_sim = max(sim_dict.values())
          if max_sim >= 0.8:
            similar_question = list(sim_dict.keys())[list(sim_dict.values()).index(max_sim)]
            answer = qa_dict[similar_question]
            print(f"챗봇: {answer}")
          else:
            response = openai.Completion.create(
                engine="davinci", prompt=text, max_tokens=60, n=1, stop=None, temperature=0.5
            )
            answer = response.choices[0].text.strip()
            qa_dict[text] = answer
            df_new = df_new.append({'유저': text, '챗봇': answer, 'embedding': model.encode(text), 'distance': max_distance}, ignore_index=True)
            print(f"챗봇: {answer}")
    else:
      qa_dict[text] = df.loc[df['distance'].idxmax(), '챗봇']
     