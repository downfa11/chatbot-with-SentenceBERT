# chatbot-with-SentenceBERT
Implement the chatbot with SentenceBERT. if answer's accurancy is low, stack the data using chatGPT. 

공공데이터 중에서 Wellness 환자 진료 발화데이터를 토대로 진찰의 답변을 예측하는 모델로 계획했었는데

발화 데이터에 관련도가 떨어지는 내용을 질문하면 엉뚱한 답변을 하길래 chatGPT를 때려박았음.

답변의 Accurancy가 낮을 경우, chatGPT에게 물어본 답변을 가져오고 새롭게 데이터를 등록합니다. 

그렇게 점점 GPT의 의존도를 낮춰가면서 발화 데이터를 풍족하게 쌓으면 심심이처럼 쓸 수 있지 않을까요? 아님랄로