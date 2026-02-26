# Tokenizer 파일 요약 (app/token/)

## 1. tokenizer.json

> **HuggingFace Tokenizers 라이브러리 형식**의 토크나이저 정의 파일 (약 33MB, 2,379,611줄)

### 기본 정보

| 항목 | 값 |
|------|-----|
| **버전** | 1.0 |
| **모델 타입** | BPE (Byte Pair Encoding) |
| **어휘(vocab) 크기** | **262,144** 토큰 |
| **병합(merges) 수** | **514,906** 개 |
| **truncation** | 없음 (null) |
| **padding** | 없음 (null) |

### 특수 토큰 (Special Tokens)

| ID | 토큰 | 용도 |
|----|-------|------|
| 0 | `<pad>` | 패딩 |
| 1 | `<eos>` | 문장 끝 (End of Sequence) |
| 2 | `<bos>` | 문장 시작 (Beginning of Sequence) |
| 3 | `<unk>` | 알 수 없는 토큰 |
| 105 | `<start_of_turn>` | 대화 턴 시작 |
| 106 | `<end_of_turn>` | 대화 턴 종료 |
| 255999 | `<start_of_image>` | 이미지 시작 |
| 256000 | `<end_of_image>` | 이미지 종료 |
| 262144 | `<image_soft_token>` | 이미지 소프트 토큰 |

### 추가 토큰 (Added Tokens)

- 총 **6,415개**의 추가 토큰 정의
- `<unused0>` ~ `<unused99>`: 100개의 미사용 예약 토큰 (ID 6~105)
- `<mask>` (ID 4): 마스킹용 토큰
- `[multimodal]` (ID 5): 멀티모달 표시 토큰
- ID 107~137: 줄바꿈(newline) 시퀀스 토큰
- ID 138~167: 공백(indent) 시퀀스 토큰 (▁ 문자 반복)
- ID 168~237: **HTML 태그 토큰** (`<table>`, `<tr>`, `<td>`, `<h1>`~`<h6>`, `<code>`, `<img>`, `<div>` 등)
- ID 255968~255998: 추가 예약 토큰

### 전처리 파이프라인

| 단계 | 타입 | 설명 |
|------|------|------|
| **Normalizer** | Replace | 공백(` `)을 `▁` (U+2581 lower one eighth block)로 치환 |
| **Pre-tokenizer** | Split | 공백 기준으로 분리, `MergedWithPrevious` 동작 |
| **Post-processor** | TemplateProcessing | 입력 시작에 `<bos>` 토큰 자동 추가 |
| **Decoder** | Sequence | `▁` → 공백 복원 → ByteFallback → Fuse 순서로 디코딩 |

---

## 2. tokenizer_config.json

> **HuggingFace Transformers 라이브러리**용 토크나이저 설정 파일 (약 1.1MB, 51,347줄)

### 기본 설정

| 항목 | 값 |
|------|-----|
| **tokenizer_class** | `GemmaTokenizer` |
| **processor_class** | `Gemma3Processor` |
| **model_max_length** | 1,000,000,000,000,000,019,884,624,838,656 (사실상 무제한) |
| **clean_up_tokenization_spaces** | `False` |
| **spaces_between_special_tokens** | `False` |

### 토큰 설정

| 항목 | 토큰 |
|------|-------|
| **bos_token** (시작) | `<bos>` |
| **eos_token** (끝) | `<eos>` |
| **pad_token** (패딩) | `<pad>` |
| **unk_token** (알 수 없음) | `<unk>` |
| **boi_token** (이미지 시작) | `<start_of_image>` |
| **eoi_token** (이미지 끝) | `<end_of_image>` |
| **image_token** | `<image_soft_token>` |
| **add_bos_token** | `True` (자동 추가) |
| **add_eos_token** | `False` (자동 추가하지 않음) |

### Chat Template (Jinja2)

- **길이**: 1,532자의 Jinja2 템플릿
- **기능**: 대화형 프롬프트 포맷 자동 생성
- **동작 방식**:
  1. `{{ bos_token }}`으로 시작
  2. 첫 번째 메시지가 `system` 역할이면 시스템 프롬프트를 user prefix로 설정
  3. 각 메시지를 `<start_of_turn>` / `<end_of_turn>` 태그로 감쌈
  4. `user` → `model` 역할 순으로 대화를 포맷팅

### 멀티모달 지원

- **이미지 관련 토큰**이 별도로 정의되어 있어 **멀티모달(텍스트+이미지) 입력**을 지원
- `extra_special_tokens`에 `boi_token`, `eoi_token`, `image_token`이 등록됨

### Added Tokens Decoder

- 총 **6,415개** 토큰에 대한 디코더 매핑 정의 (`tokenizer.json`의 `added_tokens`와 대응)

---

## 종합 요약

이 두 파일은 **Google Gemma 3** 모델의 토크나이저 설정 파일입니다.

- **모델**: Google Gemma 3 (멀티모달 지원)
- **토크나이저 알고리즘**: BPE (Byte Pair Encoding)
- **어휘 크기**: 262,144 토큰
- **주요 특징**:
  - 대화형 턴 기반 포맷 (`<start_of_turn>`, `<end_of_turn>`)
  - 멀티모달 이미지 입력 지원 (`<start_of_image>`, `<end_of_image>`, `<image_soft_token>`)
  - HTML 구조 태그를 위한 전용 토큰
  - Jinja2 기반 Chat Template으로 대화 포맷 자동화
  - SentencePiece 스타일의 공백 처리 (`▁` 문자 사용)
