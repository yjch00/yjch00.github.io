---
layout: post
title:  "PROJECT#1 AVISIONS(1)"
sidebar:
   nav: "test"
---

# 텍스트 유사도 구하기

### 0. import


```python
import pandas as pd
from konlpy.tag import Komoran
```

### 1. 구글링 코드 예시


```python
from sklearn.feature_extraction.text import TfidfVectorizer

sent = ("휴일 인 오늘 도 서쪽 을 중심 으로 폭염 이 이어졌는데요, 내일 은 반가운 비 소식 이 있습니다.", 
        "폭염 을 피해서 휴일 에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니다.") 
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sent) #문장 벡터화 진행

idf = tfidf_vectorizer.idf_
print(dict(zip(tfidf_vectorizer.get_feature_names(), idf)))
```

    {'갑작스런': 1.4054651081081644, '내일': 1.4054651081081644, '놀러왔다가': 1.4054651081081644, '망연자실': 1.4054651081081644, '반가운': 1.4054651081081644, '서쪽': 1.4054651081081644, '소식': 1.4054651081081644, '오늘': 1.4054651081081644, '으로': 1.4054651081081644, '이어졌는데요': 1.4054651081081644, '인해': 1.4054651081081644, '있습니다': 1.0, '중심': 1.4054651081081644, '폭염': 1.0, '피해서': 1.4054651081081644, '하고': 1.4054651081081644, '휴일': 1.0}
    

    C:\anaconda\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    


```python
from sklearn.metrics.pairwise import cosine_similarity

# 코사인 유사도를 구해보자
cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
```




    array([[0.17952266]])



### 2. 강의계획서(lec)와 직군 설명(job) 불러오기  (크롤링 x)


```python
# 1.데이터마이닝 강좌
datamining = '산업공학 데이터마이닝 의사결정 사용 Insight Foresight 데이터 추출 수리 계산적 작업 연관분석, 군집화, 회귀분석, 의사결정나무, k-NN, 베이지언, 신경회로망 핵심적 기법 공부 Python 활용 실습 다양한 실제 비즈니스 상황 어떠한 기법으로 어떠한 분석을 실시하여 문제를 해결하는지 공부 실제 비즈니스 문제 데이터마이닝 문제 변환 데이터 전처리 모델링 수행 인사이트 포사이트 추출 평가 기말 프로젝트 수행 비즈니스 밸류 문제  도출 도출 능력 함양 최근 많은 인사이트 관심 받고 있는 Artificial Intelligence, Big Data, Internet of Things, Smart Factory, Social Media, Fintech 관련성 공부 비즈니스애널리틱스 Data Mining for Business Analytics R edition 프로그래밍 python 기초 확률통계 기초 선형대수 전처리 인사이트 프로젝트 밸류 데이터 확보 시험 시험 분석 계획 전터리 시각화 baseline 모델링 인사이트 의미 해석 밸류 제시 데이터마이닝 비즈니스 데이터소스 데이터마이닝 문제 모델 구출평가 kaggle 비즈니스 임팩트 데이터의 양과 질 Business Background, Business Problem, Data Mining Problem, Visualization of Obtained Data, Models to use, Expected Data Mining Results, Expected Business Implications or Business Impact K means 시각화 연관분석 전처리 클러스터링 컴퓨팅 알고리즘 인공지능 머신러닝 예측 분류 모델링 분류 앙상블 크롤링 텍스트 분석 term project proposal 분류 모델 평가 k-nn nb div사례 반응 모델링 neural network 구조 machine learning'
```


```python
lec1 = datamining.lower()
lec1
```




    '산업공학 데이터마이닝 의사결정 사용 insight foresight 데이터 추출 수리 계산적 작업 연관분석, 군집화, 회귀분석, 의사결정나무, k-nn, 베이지언, 신경회로망 핵심적 기법 공부 python 활용 실습 다양한 실제 비즈니스 상황 어떠한 기법으로 어떠한 분석을 실시하여 문제를 해결하는지 공부 실제 비즈니스 문제 데이터마이닝 문제 변환 데이터 전처리 모델링 수행 인사이트 포사이트 추출 평가 기말 프로젝트 수행 비즈니스 밸류 문제  도출 도출 능력 함양 최근 많은 인사이트 관심 받고 있는 artificial intelligence, big data, internet of things, smart factory, social media, fintech 관련성 공부 비즈니스애널리틱스 data mining for business analytics r edition 프로그래밍 python 기초 확률통계 기초 선형대수 전처리 인사이트 프로젝트 밸류 데이터 확보 시험 시험 분석 계획 전터리 시각화 baseline 모델링 인사이트 의미 해석 밸류 제시 데이터마이닝 비즈니스 데이터소스 데이터마이닝 문제 모델 구출평가 kaggle 비즈니스 임팩트 데이터의 양과 질 business background, business problem, data mining problem, visualization of obtained data, models to use, expected data mining results, expected business implications or business impact k means 시각화 연관분석 전처리 클러스터링 컴퓨팅 알고리즘 인공지능 머신러닝 예측 분류 모델링 분류 앙상블 크롤링 텍스트 분석 term project proposal 분류 모델 평가 k-nn nb div사례 반응 모델링 neural network 구조 machine learning'




```python
#하이닉스 datascience
hynix_datascience = 'Data Science 세계 최고 반도체 개발 을 위한 끊임 없는 연구 데이터 분석 기술 ( Machine / Deep Learning ) 을 활용 정형 / 비정형 데이터 분석 을 수행, 분석결과 기반 Business Insight 도출 문제 해결 업무 수행 Data Analytics 데이터 를 분석 문제 정의, 과제 발굴, 데이터 수집 가공, 통계 선형대수 ML 방법론 을 이용한 문제 해결 방법 을 제시 실험 평가 구현 Deep Learning 데이터 전처리, 적합한 딥러닝 모델 설계 및 구현, 학습과 추론에 대한 시스템 구축, 최신 딥러닝 기술 F/U 업무 를 수행AI Engineering Data 를 수집 활용 End to End Data Pipeline 을 설계 및 구현 필요한 분석 서비스 를 SW 기술 을 활용 구축'
```


```python
job1 = hynix_datascience.lower()
job1
```




    'data science 세계 최고 반도체 개발 을 위한 끊임 없는 연구 데이터 분석 기술 ( machine / deep learning ) 을 활용 정형 / 비정형 데이터 분석 을 수행, 분석결과 기반 business insight 도출 문제 해결 업무 수행 data analytics 데이터 를 분석 문제 정의, 과제 발굴, 데이터 수집 가공, 통계 선형대수 ml 방법론 을 이용한 문제 해결 방법 을 제시 실험 평가 구현 deep learning 데이터 전처리, 적합한 딥러닝 모델 설계 및 구현, 학습과 추론에 대한 시스템 구축, 최신 딥러닝 기술 f/u 업무 를 수행ai engineering data 를 수집 활용 end to end data pipeline 을 설계 및 구현 필요한 분석 서비스 를 sw 기술 을 활용 구축'




```python
#생산관리 강좌
lec2 = '산업공학 생산시스템 의 운영 과 관련 된 제반문제 해결 을 위한 계량적 접근방법 을 소개 이용 한 생산시스템 의 효율적인 관리 및 통제기법 을 소개 글로벌 시대에 요구 고객 만족 설계 물류관리 혁신, CALS EC, ERP 기법 을 소개 주요 내용 생산시스템 기본 개념, 고객 만족, 생산 기획, 물류관리, 생산일정계획, 생산성 향상 공장자동화와 생산전략 등을 포함하고 있다.'
lec2
```




    '산업공학 생산시스템 의 운영 과 관련 된 제반문제 해결 을 위한 계량적 접근방법 을 소개 이용 한 생산시스템 의 효율적인 관리 및 통제기법 을 소개 글로벌 시대에 요구 고객 만족 설계 물류관리 혁신, CALS EC, ERP 기법 을 소개 주요 내용 생산시스템 기본 개념, 고객 만족, 생산 기획, 물류관리, 생산일정계획, 생산성 향상 공장자동화와 생산전략 등을 포함하고 있다.'




```python
#삼성전자 생산관리 메모리사업부
job2 = '제품 생산 계획, 생산성 관리, 시스템 기반 SCM 구축을 통해 생산성 을 관리 직무 제품 생산 관리 생산계획 수립, 자재 수급 관리, 원가관리 통한 생산성 향상 제품 별 생산 기획, 진도 관리 생산 인프라 활용 효율 을 높여 생산 설비 최적화 생산설비, Wafer Cost 변동 추이 분석 통한 원가 절감 시스템 기반 생산체계 구축 생산 및 정체 Scheduler 관리 를 통한 생산성 향상 반도체 생산라인 에 최적화된 SCM 구축 및 개선 Smart Factory 구축 및 혁신 완전 자동화 생산 시스템 구축 및 물류 운영방식 개선 제조라인 의 비효율 업무 개선 및 인프라 시스템 최적화 로직 수립 Big Data / AI를 접목한 인프라 구축 을 통해 생산성 향상'
job2 = job2.lower()
job2
```




    '제품 생산 계획, 생산성 관리, 시스템 기반 scm 구축을 통해 생산성 을 관리 직무 제품 생산 관리 생산계획 수립, 자재 수급 관리, 원가관리 통한 생산성 향상 제품 별 생산 기획, 진도 관리 생산 인프라 활용 효율 을 높여 생산 설비 최적화 생산설비, wafer cost 변동 추이 분석 통한 원가 절감 시스템 기반 생산체계 구축 생산 및 정체 scheduler 관리 를 통한 생산성 향상 반도체 생산라인 에 최적화된 scm 구축 및 개선 smart factory 구축 및 혁신 완전 자동화 생산 시스템 구축 및 물류 운영방식 개선 제조라인 의 비효율 업무 개선 및 인프라 시스템 최적화 로직 수립 big data / ai를 접목한 인프라 구축 을 통해 생산성 향상'



### 3. 코사인유사도 측정 및 함수 형성


```python
sent=(lec1, job1)
sent
```




    ('산업공학 데이터마이닝 의사결정 사용 insight foresight 데이터 추출 수리 계산적 작업 연관분석, 군집화, 회귀분석, 의사결정나무, k-nn, 베이지언, 신경회로망 핵심적 기법 공부 python 활용 실습 다양한 실제 비즈니스 상황 어떠한 기법으로 어떠한 분석을 실시하여 문제를 해결하는지 공부 실제 비즈니스 문제 데이터마이닝 문제 변환 데이터 전처리 모델링 수행 인사이트 포사이트 추출 평가 기말 프로젝트 수행 비즈니스 밸류 문제  도출 도출 능력 함양 최근 많은 인사이트 관심 받고 있는 artificial intelligence, big data, internet of things, smart factory, social media, fintech 관련성 공부 비즈니스애널리틱스 data mining for business analytics r edition 프로그래밍 python 기초 확률통계 기초 선형대수 전처리 인사이트 프로젝트 밸류 데이터 확보 시험 시험 분석 계획 전터리 시각화 baseline 모델링 인사이트 의미 해석 밸류 제시 데이터마이닝 비즈니스 데이터소스 데이터마이닝 문제 모델 구출평가 kaggle 비즈니스 임팩트 데이터의 양과 질 business background, business problem, data mining problem, visualization of obtained data, models to use, expected data mining results, expected business implications or business impact k means 시각화 연관분석 전처리 클러스터링 컴퓨팅 알고리즘 인공지능 머신러닝 예측 분류 모델링 분류 앙상블 크롤링 텍스트 분석 term project proposal 분류 모델 평가 k-nn nb div사례 반응 모델링 neural network 구조 machine learning',
     'data science 세계 최고 반도체 개발 을 위한 끊임 없는 연구 데이터 분석 기술 ( machine / deep learning ) 을 활용 정형 / 비정형 데이터 분석 을 수행, 분석결과 기반 business insight 도출 문제 해결 업무 수행 data analytics 데이터 를 분석 문제 정의, 과제 발굴, 데이터 수집 가공, 통계 선형대수 ml 방법론 을 이용한 문제 해결 방법 을 제시 실험 평가 구현 deep learning 데이터 전처리, 적합한 딥러닝 모델 설계 및 구현, 학습과 추론에 대한 시스템 구축, 최신 딥러닝 기술 f/u 업무 를 수행ai engineering data 를 수집 활용 end to end data pipeline 을 설계 및 구현 필요한 분석 서비스 를 sw 기술 을 활용 구축')




```python
tfidf_matrix = tfidf_vectorizer.fit_transform(sent)
idf = tfidf_vectorizer.idf_
cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
```




    array([[0.21122317]])




```python
def cos_sim(lec, job):
    sent=(lec, job)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sent)
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
```

## 4. 결론


```python
#데이터마이닝 강좌와 하이닉스 data science 직무
print(cos_sim(lec1, job1))
```

    [[0.21122317]]
    


```python
#생산관리 강좌와 삼성전자 생산관리 직무
print(cos_sim(lec2, job2))
```

    [[0.10499002]]
    


```python
#데이터마이닝 강좌와 삼성전자 생산관리 직무
print(cos_sim(lec1, job2))
#생산관리 강좌와 하이닉스 data science 직무
print(cos_sim(lec2, job1))
```

    [[0.02052328]]
    [[0.02403872]]
    

## 5. 추가 자연어 처리


```python
from konlpy.tag import Okt 
okt = Okt()

```


```python
print(okt.nouns(job1))
```

    ['세계', '최고', '반도체', '개발', '위', '임', '연구', '데이터', '분석', '기술', '활용', '정형', '비정', '데이터', '분석', '수행', '분석', '결과', '기반', '도출', '문제', '해결', '업무', '수행', '데이터', '를', '분석', '문제', '정의', '과제', '발굴', '데이터', '수집', '가공', '통계', '선형대수', '방법론', '이용', '문제', '해결', '방법', '제시', '실험', '평가', '구현', '데이터', '처리', '딥', '러닝', '모델', '설계', '및', '구현', '학습', '추론', '대한', '시스템', '구축', '최신', '딥', '러닝', '기술', '업무', '를', '수행', '를', '수집', '활용', '설계', '및', '구현', '분석', '서비스', '를', '기술', '활용', '구축']
    


```python

```
