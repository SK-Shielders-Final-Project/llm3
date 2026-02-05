# Oracle -> MongoDB 동기화 설계

이 문서는 현재 프로젝트 구조를 기준으로 다음 3가지를 제공합니다.

1. CDC 기반 파이프라인 설계
2. 최소한의 배치 동기화 스크립트 설계/사용법
3. Mongo 컬렉션 스키마 설계

---

## 1) CDC 기반 파이프라인 설계(권장)

### 목표

- Oracle의 INSERT/UPDATE/DELETE를 거의 실시간으로 MongoDB에 반영
- MongoDB를 조회 전용 소스로 활용하더라도 최신성 문제를 최소화

### 구성 요소

- Oracle: redo log 기반 변경 추출
- CDC: Debezium Oracle connector(또는 GoldenGate)
- 메시지 브로커: Kafka
- Sink: Kafka Connect MongoDB Sink(Upsert/삭제 반영)
- MongoDB: 조회/검색 데이터 저장소

### 데이터 흐름

1. Oracle에서 변경 발생
2. Debezium이 변경 이벤트를 Kafka 토픽으로 발행
3. Sink가 이벤트를 MongoDB 컬렉션에 upsert/delete 반영

### 토픽/컬렉션 매핑 예시

- `oracle.users` -> `users`
- `oracle.rentals` -> `rentals`
- `oracle.payments` -> `payments`
- `oracle.bikes` -> `bikes`
- `oracle.inquiries` -> `inquiries`

### Upsert/삭제 정책

- Upsert 키: 각 테이블의 PK(예: `user_id`, `rental_id`)
- 삭제 이벤트: Mongo 문서를 `delete` 또는 `is_deleted=true` 소프트 삭제

### 권장 메타데이터 필드

- `_source`: `"oracle"`
- `_table`: `"users"` 등
- `_synced_at`: CDC 반영 시각

### 장단점

- 장점: 최신성 우수, 삭제 반영 가능
- 단점: 구축/운영 복잡도 증가

---

## 2) 최소한의 배치 동기화 스크립트

### 사용 목적

- CDC 구축 전 임시 운영
- 하루 1~N회 배치 동기화

### 스크립트 위치

- `app/sync_oracle_to_mongo.py`

### 동기화 방식

- 기본: `updated_at`(또는 `created_at`) 워터마크 기반 증분 sync
- 워터마크가 없는 테이블: `--full` 옵션에서 전체 재적재
- 삭제는 반영되지 않음(삭제 반영이 필요하면 CDC 권장)

### 환경변수

- `ORACLE_DSN`: 예) `localhost:1521/XEPDB1`
- `ORACLE_USER`
- `ORACLE_PASSWORD`
- `MONGODB_URI`
- `MONGODB_DB_NAME`

### 실행 예시

- 전체 테이블 증분 sync
  - `python app/sync_oracle_to_mongo.py`
- 특정 테이블만 sync
  - `python app/sync_oracle_to_mongo.py --tables users,rentals`
- 강제 전체 재적재
  - `python app/sync_oracle_to_mongo.py --full --tables users`

---

## 3) Mongo 컬렉션 스키마 설계

### 공통 필드

- `_id`: Mongo ObjectId
- `_source`: `"oracle"`
- `_table`: 테이블명
- `_synced_at`: 동기화 시각
- `oracle_id`: Oracle PK 복사

### 컬렉션별 기본 구조

#### users

```
{
  "oracle_id": 123,
  "user_id": 123,
  "username": "...",
  "name": "...",
  "email": "...",
  "phone": "...",
  "total_point": 0,
  "admin_level": 0,
  "created_at": "...",
  "updated_at": "...",
  "_source": "oracle",
  "_table": "users",
  "_synced_at": "..."
}
```

#### rentals

```
{
  "oracle_id": 991,
  "rental_id": 991,
  "user_id": 123,
  "bike_id": 50,
  "start_time": "...",
  "end_time": "...",
  "total_distance": 12.4,
  "created_at": "...",
  "_source": "oracle",
  "_table": "rentals",
  "_synced_at": "..."
}
```

#### payments

```
{
  "oracle_id": 88,
  "payment_id": 88,
  "user_id": 123,
  "amount": 3500,
  "payment_status": "COMPLETED",
  "payment_method": "CARD",
  "transaction_id": "...",
  "created_at": "...",
  "_source": "oracle",
  "_table": "payments",
  "_synced_at": "..."
}
```

#### bikes

```
{
  "oracle_id": 50,
  "bike_id": 50,
  "serial_number": "...",
  "model_name": "...",
  "status": "AVAILABLE",
  "latitude": 37.123,
  "longitude": 127.123,
  "created_at": "...",
  "updated_at": "...",
  "_source": "oracle",
  "_table": "bikes",
  "_synced_at": "..."
}
```

#### inquiries

```
{
  "oracle_id": 7,
  "inquiry_id": 7,
  "user_id": 123,
  "title": "...",
  "content": "...",
  "image_url": "...",
  "file_id": 42,
  "admin_reply": "...",
  "created_at": "...",
  "updated_at": "...",
  "_source": "oracle",
  "_table": "inquiries",
  "_synced_at": "..."
}
```

### RAG/Vector 검색용 컬렉션

정형 데이터와 분리 권장:

- `knowledge` 또는 `manual` 컬렉션에 `content` + `embedding`
- 사용자 데이터는 벡터화하지 않고 일반 조회

```
{
  "content": "...",
  "embedding": [ ... ],
  "metadata": {
    "doc_type": "policy",
    "category": "manual",
    "access_level": "public"
  }
}
```

### 인덱스 권장

- `users.user_id`, `rentals.user_id`, `payments.user_id`
- `bikes.status`, `bikes.latitude`, `bikes.longitude`(위치 검색용)
- `knowledge.embedding`(Vector index)

---

## 운영 가이드 요약

- 정확성/삭제 반영이 중요하면 CDC
- 배치만 사용할 경우 삭제 누락/최신성 지연 감수
- Mongo는 조회/검색 최적화, Oracle은 원본 진실
