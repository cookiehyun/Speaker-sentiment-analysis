# Speaker-sentiment-analysis

<br>

# 🎙️ 발화자 감정 판별 프로젝트

**고령자 구술 대화 데이터 기반 감정 분석 모델**, **SENTI-ORAL**입니다.  
질문–응답 형태의 대화를 입력받아, **발화자의 정서 상태**를 분석합니다.  
특히 고령자의 생활사 구술 데이터 특성에 맞춰, **불안(2단계) & 우울(2단계)** 총 4개의 점수를 예측합니다.

---


## 📑 목차

- [📌 프로젝트 소개](#-프로젝트-소개)
- [🎯 프로젝트 목표](#-프로젝트-목표)
- [📂 학습 데이터](#-학습-데이터)
- [✨ 주요 기능](#-주요-기능)
- [🏛️ 시스템 아키텍처](#-시스템-아키텍처)
- [🧠 모델 구성](#-모델-구성)
- [📂 프로젝트 구조](#-프로젝트-구조)
- [⚙️ 설치 및 환경 설정](#-설치-및-환경-설정)
- [🚀 실행 방법](#-실행-방법)
- [🧪 평가 및 분석](#-평가-및-분석)
- [💡 권장 환경](#-권장-환경)
---

## 📌 프로젝트 소개

**SENTI-ORAL**은 노년층 발화에서 감정 신호를 포착하고,  
대화 내 정서 변화를 정량적으로 분석하는 **감정 판별 AI 모델**입니다.

- 멀티턴 대화에서 **맥락 기반 감정 변화 추론**  
- **불안·우울 지표**를 각각 2단계로 판별  
- 고령자 근현대 경험 서사 데이터를 기반으로 학습  

> 일상 회상·전쟁 경험·가족사·사회 변화 경험 등  
> **감정이 미묘하게 드러나는 대화 패턴 분석**에 최적화

---

## 🎯 프로젝트 목표

- 💬 **정확하고 맥락을 이해한 응답 제공**  
- 🧠 **다중 턴 대화에서 문맥 유지 및 자연스러운 흐름 구현**  
- 📚 **실제 금융 민원 시나리오 기반 데이터 학습**  
- 🪶 **Instruction Tuning으로 고품질 응답 생성**

---

## 📂 학습 데이터

본 프로젝트는 **AI Hub 고령층 근현대 생활사 구술 데이터**를 기반으로 학습되었습니다.

> 📘 [AI Hub 고령자 근현대 구술 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=2&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=71703)

- 질문–응답 기반 구술 대화
- 감정 라벨링(불안 2단계 + 우울 2단계)  
- 세대·시대 경험 기반 감정 발화 특성 반영  

---

## ✨ 주요 기능

### 고령자 대화 기반 감정 인식
- 질문–응답 형태의 구술 발화 입력
- 고령자 생활사·과거 경험 맥락 반영
- 발화자 정서 상태에 기반한 **불안·우울 4-score 동시 예측**

### KoBERT 기반 텍스트 정서 표현 학습
- `skt/kobert-base-v1` 기반 한국어 발화 임베딩
- 실제 구술 감정 데이터로 **Multi-task Fine-tuning**
- CLS + Hidden sequence 활용하여 세밀한 감정 신호 추출
- Fine-tuned 이후 **Freeze → Feature Extractor**로 사용

### FT-Transformer 기반 메타데이터 심리 케어 모델링
- 고령자 특성 반영 데이터 입력  
  (연령대, 성별, 혼인 여부, 지역, 교육 연수, 동거 인원 등)
- **수치형 + 범주형 토크나이징 및 포지션 학습**
- FT-Transformer Encoder로 인구사회적 요인 특징화

### Cross-Attention 멀티모달 감정 Fusion
- FT-Transformer 출력과 KoBERT hidden state 간 **Cross-Attention**
- 텍스트 감정 표현 + 인구통계적 맥락의 상호 정보 결합
- 단순 concat이 아닌 **Attention 기반 융합 구조**
- 최종 Shared MLP + **4-Head Multi-task Classification**

### 효율적 & 안정적 학습 전략
- Tokenization 사전 캐싱 → 학습 속도 향상
- KoBERT gradient freeze로 자원 효율화
- Warmup + Linear Decay 스케줄러
- Gradient clipping + 평균 Multi-task Loss 적용
- 불균형 감정 점수 데이터 대응 (balanced fine-tuning 데이터)

### 데이터 전처리 및 처리 파이프라인
- O/X → Binary, Label Encoding, Standard Scaling
- Fine-tuning 세트와 FT-Transformer 세트 분리
- Tabular + Text 병렬 데이터 로딩 및 캐싱 Dataset 구조

### 학습/추론 지원
- KoBERT Multi-task Finetuning 스크립트 제공
- FT-Transformer + Cross-Attention 통합 학습 루프
- 모델·스케일러·인코더 저장 및 로드 지원
- Evaluation: Accuracy & Weighted-F1 계산

---

## 🏛️ 시스템 아키텍처

```
[고령자 대화 텍스트 입력]
        ↓
[텍스트 정제 및 토크나이징]
        ↓
[KoBERT Fine-Tuning (Multi-task)]
  └─ 불안/우울 4-score 학습
        ↓
[KoBERT Freeze]
  └─ CLS + Hidden Sequence Feature 추출
        ↓
[고령자 메타데이터 입력]
  └─ 나이/가족/지역/교육/동거 등
        ↓
[FT-Transformer Encoding]
  └─ 수치형 + 범주형 → Tabular Embedding
        ↓
[Cross-Attention Fusion]
  └─ Tabular CLS ↔ KoBERT Hidden Attention
        ↓
[Shared MLP]
        ↓
[4-Head Multi-task Output]
  ├─ Anxiety Score 1 (0~4)
  ├─ Anxiety Score 2 (0~4)
  ├─ Depression Score 1 (0~4)
  └─ Depression Score 2 (0~4)
        ↓
[감정 스코어 반환]
```
---

## 🧠 모델 구성

**텍스트 정제 및 토크나이징**
- 구술 발화 전처리 (특수기호 및 불필요 텍스트 정리)
- `skt/kobert-base-v1` 토크나이저 적용
- 발화 문장 → 토큰 시퀀스 변환

↓

**KoBERT Multi-Task Fine-Tuning**
- 모델: `skt/kobert-base-v1`
- 목표: 불안/우울 4-score 동시 예측
- Fine-tuning 후 **Freeze** → 정서 feature extractor로 사용
- 특징: CLS + Hidden embedding 활용

↓

**고령자 메타데이터 인코딩 (FT-Transformer)**
- 수치형 + 범주형 feature → Tokenization & Embedding
- 학습 요소: 나이/가족관계/거주/교육/지역/동거 인원 등
- FT-Transformer Encoder로 심리·환경적 요인 학습
- CLS-기반 tabular representation 생성

↓

**Cross-Attention Fusion**
- Tabular-CLS ↔ KoBERT Hidden representation 상호 주의(attention)
- 단순 concat이 아닌 **Cross-Attention 정보 통합**
- 세대·환경·언어 기반 감정 신호 결합

↓

**Shared MLP + Multi-Task Heads**
- 공통 감정 표현(shared feature) → MLP
- 4개의 분류 Head 수행  
  - Anxiety Score 1  
  - Anxiety Score 2  
  - Depression Score 1  
  - Depression Score 2  
- 각 task: 0~4 등급 분류

↓

**최종 출력**
- 발화자의 불안·우울 상태 4-dim 점수 예측
- 고령자 감정 평가지표 기반 점수 출력

## 📂 디렉토리 구조 (Directory Structure)
```
├── artifacts/               # 학습된 모델, Scaler, Encoder 저장소
│   ├── scaler.pkl
│   ├── cat_encoders.pkl
│   └── final_kobert_fttransformer_mlp.pt
├── data/                    # 데이터셋 폴더
│   ├── processed_features_cleaned.json
│   └── finetune_sqrt_balanced_processed.json
├── src/
│   ├── data_preprocess.py   # 데이터 전처리 및 파생변수 생성
│   ├── kobert_finetune.py   # Stage 1: KoBERT 단독 Fine-tuning
│   ├── fusion_model.py      # FT-Transformer + CrossAttention 모델 정의
│   └── train_fusion.py      # Stage 2: 결합 모델 학습 (Main)
├── README.md
└── requirements.txt
```

## ⚙️ 설치 및 환경 설정 (Installation)

### 1. 환경 요구사항
* Python 3.8 이상
* PyTorch 1.12 이상 (CUDA 환경 권장)

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```
---

## 🚀 실행 방법 (Usage)

본 프로젝트는 **데이터 전처리 → KoBERT 파인튜닝 → Fusion 모델 학습**의 3단계로 진행됩니다.

### Step 1. 데이터 전처리 (Preprocessing)
Raw 데이터에서 학습/테스트 셋을 분리하고, 수치형 변수(StandardScaler) 및 범주형 변수(LabelEncoder) 변환을 수행합니다.
```bash
python src/data_preprocess.py
```
* **결과물:** `train_df.json`, `test_df.json`, `scaler.pkl`, `cat_encoders.pkl`

### Step 2. KoBERT 단독 Fine-tuning (Stage 1)
텍스트 인코더의 이해도를 높이기 위해 KoBERT를 먼저 학습시킵니다. (Multi-task Learning)
```bash
python src/kobert_finetune.py --epochs 5 --lr 3e-5
```
* **결과물:** `artifacts/fine_tuned_kobert_cls.pt`

### Step 3. Fusion 모델 학습 (Stage 2 - Main)
Fine-tuned KoBERT(Freeze)와 FT-Transformer를 결합하여 최종 학습을 진행합니다.
```bash
python src/train_fusion.py --epochs 5 --lr 1e-4
```
* **Hyperparameters:**
    * Batch Size: 16
    * Learning Rate: 1e-4
    * Fusion Dimension: 192

---

## 🧪 평가 및 분석 (Results)

Validation Set 기준, 단순 Concat 방식 대비 **Cross-Attention** 적용 시 모든 Target에서 성능 향상을 확인했습니다.

| Target | Model Type | Accuracy | F1-Score |
|:---:|:---:|:---:|:---:|
| **Anxiety Score 1** | **Cross-Attention** | **0.882** | **0.871** |
| **Anxiety Score 2** | **Cross-Attention** | **0.926** | **0.917** |
| **Depression Score 1** | **Cross-Attention** | **0.871** | **0.861** |
| **Depression Score 2** | **Cross-Attention** | **0.857** | **0.847** |
---
