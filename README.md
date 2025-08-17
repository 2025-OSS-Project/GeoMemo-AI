# GeoMemo-AI

GeoMemo의 **AI 모듈** 레포지토리입니다. (감정 분석 · 개인화 장소 추천 · 주간 인사이트)

> **한눈에 보기 (TL;DR)**
>
> * **개인화 장소 추천**: **MQ 워커**로 동작 (요청 큐 ▶ 결과 큐)
> * **주간 인사이트**: **MQ 워커**로 생성/저장
> * **감정 분석**: **MQ 워커**로 메모 텍스트의 감정 라벨/점수 산출

---

## ✨ 주요 기능

### 1) 감정 분석 (Emotion Worker)

* 한국어 **6감정**(기쁨·놀람·분노·불안·상처·슬픔) 분류
* 입력: `{ memo_id, content }`  → 저장: DB `EmotionEntity (memo_id, emotion_label, emotion_score)`
* 모델은 로컬 HF 디렉터리(`EMO_MODEL_DIR`)에서 로드

### 2) 주간 인사이트 (Weekly Insights)

* `ai/infra/mq_emotion.py`의 `run_insight_worker()`가 \*\*`insight.req`\*\*를 소비하여 DB `InsightEntity`에 저장 (간단 요약)
* ※ HTTP/LLM 기반 구현은 대회용 공개 문서에서 제외합니다.

### 3) 개인화 장소 추천 (Recommender Worker)

* 백엔드가 MQ **요청 큐**로 필터된 후보/컨텍스트를 보내면 워커가 **재랭킹**하여 **결과 큐**로 응답
* 사용 신호(예시): 장소 **긍정비율**, 최근 **감정 컨텍스트**, **카테고리 선호**, **스크랩 보너스**, **팔로우 긍정 작성자 수**
* `debug: true`면 각 후보에 score 구성요소(reason) 포함

### 4) 이벤트 캐시 컨슈머 (선택)

* `ai/infra/mq_consumer.py`가 \*\*토픽 익스체인지 `geomemo.events`\*\*를 구독해 인메모리 캐시 유지
* 캐시는 **프로세스 내부**에서만 유효 — 다른 프로세스와 공유되지 않음

---

## 🧱 아키텍처

```mermaid
flowchart LR
  subgraph Backend
    B1[API 서버]
  end

  subgraph MQ[(Amazon MQ / RabbitMQ)]
    QR[(reco.req)]:::q --- QRES[(reco.res)]:::q
    QE[(emotion.req)]:::q
    QI[(insight.req)]:::q
    EX{{geomemo.events (topic exchange)}}:::ex
  end

  subgraph AI[GeoMemo-AI]
    RW[Recommender Worker
(ai/recommender/mq_recommender_worker.py)]:::svc
    EW[Emotion Worker
(ai/infra/mq_emotion.py)]:::svc
    IW[Insight Worker
(ai/infra/mq_emotion.py)]:::svc
    EC[Events Consumer (캐시)
(ai/infra/mq_consumer.py)]:::svc
  end

  B1 -- 후보/컨텍스트 publish --> QR --> RW --> QRES --> B1
  B1 -- 메모 텍스트 publish --> QE --> EW
  B1 -- 주간 로그 publish --> QI --> IW
  B1 -. topic bind .-> EX
  EX --> EC

  classDef q fill:#fff8;
  classDef svc fill:#f6ff;
  classDef ex fill:#eef;
```

---

## 📂 디렉터리 구조

```
GeoMemo-AI/
├─ ai/
│  ├─ __init__.py
│  ├─ infra/
│  │  ├─ __init__.py
│  │  ├─ mq_common.py                # 추천 워커용 공통 AMQP 유틸 (큐 선언, prefetch 등)
│  │  ├─ mq_consumer.py              # events 캐시 컨슈머 (geomemo.events 토픽 바인딩)
│  │  └─ mq_emotion.py               # 감정·인사이트 통합 워커 (emotion.req, insight.req)
│  ├─ recommender/
│  │  ├─ __init__.py
│  │  ├─ mq_recommender_worker.py    # 추천 MQ 워커 (reco.req → reco.res)
│  │  ├─ profile_builder.py          # 사용자 프로필 빌드(보조)
│  │  ├─ recommender.py              # 재랭킹·점수 계산 (가중치/정규화/디버그 리즌)
│  │  ├─ schema.py                   # Place, Memo, RecentEmotion, PlaceSignal 등
│  │  └─ scorer.py                   # 대안/실험용 점수 함수(보조)
├─ kc_saved_model/                   # HF 로컬 모델 디렉터리 (EMO_MODEL_DIR 기본 경로)
│  ├─ config.json
│  └─ id2label.json
├─ requirements.txt
├─ .env.sample
└─ README.md
```

> `ai/emotion/worker.py`와 같이 과거 버전 파일이 있다면 **현재 infra 구조와 일치하는지** 확인 후 사용하세요.

---

## 🚀 빠른 시작 (Quickstart)

### 0) 준비물

* Python **3.11**
* RabbitMQ/Amazon MQ **AMQP(S)** 엔드포인트
* DB 연결 문자열 (`DATABASE_URL`) — 인사이트/감정 저장용

### 1) 설치

```bash
conda create -n geomemo python=3.11 -y && conda activate geomemo
pip install -U pip && pip install -r requirements.txt
cp .env.sample .env   # 값 채우기
```

### 2) 실행

#### 2.1 개별 워커 실행

* **추천 워커**: `python -m ai.recommender.mq_recommender_worker`
* **감정·인사이트 통합 워커**: `python -m ai.infra.mq_emotion`
* (선택) **이벤트 캐시 컨슈머**: `python -m ai.infra.mq_consumer`

#### 2.2 전체 워커 실행

* **모두 한 번에**: `(venv) python run_all_workers.py`

### 3) 빠른 테스트 (MQ 메시지)

* `reco.req`로 publish → `reco.res`로 응답 수신
* `emotion.req`로 `{ "memo_id": 1, "content": "문장" }` publish → DB `EmotionEntity`에 저장 확인
* `insight.req`로 주간 로그 publish → DB `InsightEntity`에 저장 확인

> 큐 publish는 백엔드 서비스 또는 RabbitMQ 관리 콘솔/간단한 퍼블리셔 스크립트로 수행하세요.

---

## 🔐 환경변수 (`.env`)

아래는 **최종 ********`.env`******** 형태**를 기준으로 정리한 샘플입니다. 실제 키/비밀번호는 절대 커밋하지 마세요.

```ini
# ── OpenAI ───────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...            # 실제 키 입력 금지(예시만)
GPT_MODEL=gpt-4o-mini            # 최종 기본값
OPENAI_TEMPERATURE=0.65
OPENAI_TIMEOUT=30

# ── Database ─────────────────────────────────────────────────────────────
DATABASE_URL=mysql+aiomysql://<user>:<pass>@<host>:3306/geomemo?charset=utf8mb4

# ── AMQP (Amazon MQ / RabbitMQ) ─────────────────────────────────────────
AMQP_URL=amqps://<user>:<pass>@<host>:5671/   # vhost가 "/"면 끝 슬래시 필수
MQ_QUEUE_TYPE=quorum                           # 기본 큐 타입
RECO_REQ_QUEUE_TYPE=quorum                     # 추천 요청 큐 타입
RECO_RES_QUEUE_TYPE=quorum                     # 추천 응답 큐 타입

# ── 큐 이름(백엔드 ↔ AI) ────────────────────────────────────────────────
# 1) 감정분석: Backend → AI  (응답은 DB 저장)
EMOTION_REQ_QUEUE=emotion.req
# 2) 인사이트: Backend → AI  (응답은 DB 저장)
INSIGHT_REQ_QUEUE=insight.req
# 3) 장소추천: Backend → AI → Backend (응답도 큐)
RECO_REQ_QUEUE=reco.req
RECO_RES_QUEUE=reco.res

# ── prefetch(동시처리) ─────────────────────────────────────────────────
EMOTION_PREFETCH=16
INSIGHT_PREFETCH=16
RECO_PREFETCH=8

# ── 테이블/상태 ─────────────────────────────────────────────────────────
EMOTION_TABLE=EmotionEntity
INSIGHT_TABLE=InsightEntity
INSIGHT_STATUS=DONE

# ── 모델 경로 ───────────────────────────────────────────────────────────
EMO_MODEL_DIR=/home/ubuntu/GeoMemo-AI/kc_saved_model/
```

**선택(옵션)** — 이벤트 캐시 컨슈머를 쓸 경우에만:

```ini
EVENTS_EXCHANGE=geomemo.events
EVENTS_QUEUE=geomemo.events.cache
EVENTS_KEYS=location.upsert,memo.upsert,memo.delete,scrap.event,follow.event
```

**모델/비용 메모**

* 기본값은 `gpt-4o-mini`. 서비스 추론비용만 보면 대체로 `o4-mini`가 저렴하며, **캐시 입력이 매우 많은 경우** `gpt-5` 전환을 고려할 수 있습니다.
* vhost가 루트(`/`)면 `AMQP_URL` **끝에 슬래시**가 있어야 연결됩니다.
* TTL은 현재 기본 설정에 **없음**. 필요 시 `*_TTL_MS`를 추가로 정의해 오래된 메시지를 자동 만료시키세요.
* Windows 경로는 공백이 있으면 따옴표로 감싸거나 슬래시(`/`) 표기를 쓰세요. 한 환경에서 **하나의 경로만** 유지하세요.

## 🔒 공개 배포·보안 안내

* **비밀 키 금지**: `OPENAI_API_KEY`, DB 계정 등 **절대 커밋 금지**. 공개 전 커밋 이력에 노출됐으면 **즉시 키 교체**.
* **샘플만 공개**: `.env.sample`만 커밋하고 실제 `.env`는 배포 환경에만 보관.
* **개인정보/좌표**: 실제 사용자 텍스트·정확 좌표·계정 식별자 등 **PII를 레포/이슈/로그에 올리지 마세요.** 데모는 **가명/합성 데이터** 사용 권장.
* **로그 안전**: 운영 로그 레벨을 `INFO`로 유지하고, 디버그 모드에서 **원문 텍스트/키**가 출력되지 않도록 주의.
* **브로커 보안**: `amqps://` 사용, 보안 그룹에서 포트 제한(5671), **vhost 슬래시(********`/`****\*\*\*\*)** 확인.

## 📡 인터페이스 명세

### A) 감정 분석 – MQ (`emotion.req`)

요청: `{ "memo_id": 123, "content": "문장 텍스트" }`  → 저장: `EmotionEntity.memo_id, emotion_label, emotion_score`

### B) 주간 인사이트 – MQ 입력 (`insight.req`)

```json
{
  "userId": 7,
  "logs": [
    {"timestamp":"2025-08-04T09:00:00Z","label":"기쁨","score":0.88,"category":"공원"},
    {"timestamp":"2025-08-05T14:12:00Z","label":"분노","score":0.71,"category":"사무실"}
  ]
}
```

### C) 추천 워커 (MQ)

**요청 (예시)**

```json
{
  "requestId": "req-backend-001",
  "userId": 12,
  "top": 5,
  "debug": true,
  "candidates": [
    {"placeId": 101, "name": "을지로 감성카페", "category": "카페", "latitude": 37.5665, "longitude": 126.9781},
    {"placeId": 202, "name": "숲속공원",       "category": "공원", "latitude": 37.5700, "longitude": 126.9800}
  ],
  "context": {
    "recentEmotion": {"label": "불안", "score": 0.71},
    "favCategories": {"카페": 12, "공원": 4},
    "scrapPlaceIds": [101, 303],
    "placeSignals": [
      {"placeId": 101, "posRatio": 0.72, "followedPositiveCount": 2},
      {"placeId": 202, "posRatio": 0.55, "followedPositiveCount": 0}
    ]
  }
}
```

**응답 (예시)**

```json
{
  "requestId": "req-backend-001",
  "userId": 12,
  "status": "ok",
  "items": [
    {
      "placeId": 101,
      "name": "을지로 감성카페",
      "category": "카페",
      "latitude": 37.5665,
      "longitude": 126.9781,
      "score": 0.892,
      "reason": {"pos_ratio": 0.72, "emo_component": 0.77, "cat_pref": 1.0, "scrap": 1.0, "social": 0.667, "raw": 0.812}
    }
  ],
  "meta": {"model": "reco-v1.1", "elapsedMs": 23}
}
```

> 오류 시 `{ status: "error", error: "Type: message" }` 반환

---

## 😶‍🌫️ Emotion Worker (MQ) — 사용 가이드

**감정분석 워커**는 `emotion.req` 큐의 메모 본문을 받아 로컬 HF 분류 모델로 감정을 추론하고, 결과를 `EmotionEntity`에 저장합니다.

### 🔌 큐 & 동작 개요

* **수신 큐 이름**: `EMOTION_REQ_QUEUE` (기본 `emotion.req`)
* **큐 타입**: `MQ_QUEUE_TYPE=quorum`
* **prefetch**: `EMOTION_PREFETCH` (기본 16)
* **처리 흐름**: (1) 파싱(`memo_id`,`content`) → (2) 토크나이즈 & 모델 추론 → (3) DB **업서트**

### 🧾 메시지 스키마 (요청 본문)

```json
{ "memo_id": 22, "content": "메모 본문" }
```

* 동의어 허용: `memoId|id`, `text|body`

### 🧠 모델 & 라벨

* **모델 경로**: `EMO_MODEL_DIR` (HF 체크포인트; `config.json`, `id2label.json`, `tokenizer.*` 등)
* **라벨 매핑**: `config.id2label` 우선 (없으면 0..N‑1)

  * 예: `{ "0":"기쁨","1":"놀람","2":"분노","3":"불안","4":"상처","5":"슬픔" }`
* **추론**: 입력 최대 256 토큰, softmax 최댓값을 `emotion_score`로 저장

### 🗄️ DB 쓰기 계약

* 테이블: `EmotionEntity`
* **업서트 전략**: `memo_id` 기준 UPDATE(영향 0 → INSERT)
* 사용 컬럼: `memo_id, emotion_label, emotion_score`

### 🚨 실패/로깅

* **유효성 실패**(필수 필드 누락): 에러 로그 + *reject(requeue=false)*
* **모델/추론 실패**: 스택 트레이스 + *reject(requeue=false)*
* **DB 오류**: 에러 로그 + *reject(requeue=false)*
* 로그 레벨: `INFO / WARNING / ERROR`

---

## 📬 Insight Worker (MQ) — 사용 가이드

GeoMemo‑AI의 **인사이트 워커**는 백엔드가 Amazon MQ(RabbitMQ)의 `insight.req` 큐로 보낸 **주간 로그**를 수신하여, 간단 통계와 **150자 이내 요약**을 생성하고 DB(`InsightEntity`)에 저장합니다.

### 🔌 큐 & 동작 개요

* **수신 큐 이름**: `INSIGHT_REQ_QUEUE` (기본 `insight.req`)
* **큐 타입**: `MQ_QUEUE_TYPE=quorum`
* **prefetch**: `INSIGHT_PREFETCH` (기본 16)
* **처리 흐름**: (1) 메시지 파싱 → (2) 감정/장소 통계 → (3) 요약(150자, GPT 사용) → (4) DB에 **최종 1건만 INSERT** (`status = DONE`)
* **Processing 레코드 없음**: 실패 시 **중간 레코드 저장 없음**

### 🧾 메시지 스키마 (요청 본문)

워커는 아래 형태의 JSON을 기대합니다(동의어 허용).

```json
{
  "userId": 7,
  "logs": [
    {"label":"기쁨|놀람|분노|불안|상처|슬픔", "category":"장소카테고리", "placeName":"장소이름(선택)", "timestamp":"ISO8601(선택)", "score":0.0}
  ]
}
```

* **label**: 반드시 6개 라벨 중 하나 (`기쁨, 놀람, 분노, 불안, 상처, 슬픔`)
* **동의어**: `category↔placeCat`, `placeName↔name`
* **timestamp/score**: 선택
* **최소 샘플 수**: 1

#### ✅ 예시

* **예시 1 (권장)**

```json
{"userId":7,"logs":[{"timestamp":"2025-08-15T09:10:00Z","label":"기쁨","category":"공원","placeName":"잠실한강공원"},{"timestamp":"2025-08-15T14:20:00Z","label":"분노","category":"사무실","placeName":"HQ 10F"},{"timestamp":"2025-08-16T19:30:00Z","label":"슬픔","category":"집"}]}
```

* **예시 2 (alias 사용)**

```json
{"user_id":7,"logs":[{"label":"기쁨","placeCat":"카페","name":"블루보틀 성수"},{"label":"불안","placeCat":"사무실"}]}
```

* **예시 3 (최소 입력)**

```json
{"userId":7,"logs":[{"label":"기쁨","placeCat":"공원"}]}
```

### 🧠 요약 생성 규칙

* 상위(최대 3개) **카테고리 평균 감정지수**와 **감정 분포**로 프롬프트 생성
* 톤/형식: *예상치 못한 패턴 1개 + 따뜻한 설명 + 작은 행동 제안 1개*, **150자 이내**, `~해요/해보세요` 어미
* **OpenAI 실패/미설정** 시 내부 **fallback 한 줄 문구** 저장
* 결과는 `InsightEntity.content`에 **한 문장 요약만** 저장(집계 표는 저장하지 않음)

### 🗄️ DB 쓰기 계약

* 테이블: `InsightEntity`
* 컬럼: `user_id, content, status, createdAt`
* INSERT **1회만** 수행 (`status=DONE`, `createdAt=CURRENT_TIMESTAMP`)

### 🚨 실패/로깅

* **유효성 실패**(userId 없음, logs 비리스트): *reject(requeue=false)* + 에러 로그
* **요약 실패/빈 응답**: 경고/에러 로그 + fallback 문구 저장
* **DB 오류**: 에러 로그 + *reject(requeue=false)*
* 로그 레벨: `INFO/ WARNING/ ERROR` (프롬프트 프리뷰는 `DEBUG`)

### 🧪 트러블슈팅

* **content가 항상 비슷**: `OPENAI_API_KEY` 누락/오류 여부, 유효한 `category/placeCat` 분포 확인
* **라벨 오류**: 6개 라벨 외 값은 무시되어 통계 왜곡 가능
* **중복 저장 우려**: 본 워커는 1회 INSERT 계약. **동일 메시지 중복 발행** 여부 확인

---

## 🧭 Recommender Worker (MQ) — 사용 가이드

**추천 워커**는 백엔드가 보낸 후보 장소와 사용자 컨텍스트를 받아 **상위 N개**를 재랭킹하여 MQ로 응답합니다. 실행 모듈: `ai/recommender/mq_recommender_worker.py`.

### 🔌 큐 & 동작 개요

* **요청 큐**: `RECO_REQ_QUEUE` (기본 `reco.req`) — 타입 `RECO_REQ_QUEUE_TYPE`(기본 `quorum`)
* **응답 큐**: `RECO_RES_QUEUE` (기본 `reco.res`) — 타입 `RECO_RES_QUEUE_TYPE`(기본 `quorum`)
* **prefetch**: `RECO_PREFETCH` (기본 8)
* **reply\_to 지원**: MQ 메시지에 `reply_to`가 있으면 **그 큐로 응답**, 없으면 `RECO_RES_QUEUE`로 응답
* **mandatory publish**: 응답 라우팅 실패 시 즉시 에러 로그

### 🧾 메시지 스키마

* **입력/출력 예시**는 위 **인터페이스 명세(C)** 참조
* 필수: 각 후보의 `placeId, name, category, latitude, longitude`
* 옵션: `top`(기본 5), `debug`(기본 false)

### ⚙️ 처리 흐름 & 로깅

1. 수신 → 2) 파싱/검증 → 3) 점수 계산(`ai/recommender/recommender.py`) → 4) 상위 N 추출 → 5) 응답 발행 → 6) ACK

* 로그 예: `[recv] corr=… reply_to=… bytes=…` → `[publish] ok → queue='…'` → `[ack] corr=…`

### 🧠 점수 & 디버그

* 가중치/정규화/지터 규칙은 아래 **“추천 스코어 구성”** 참조
* `debug: true`일 때 항목별 기여치(`reason`)와 원시 점수 포함

### 🚨 실패/트러블슈팅

* **PRECONDITION\_FAILED (x-queue-type)**: 브로커 큐 타입과 `.env`의 `MQ_QUEUE_TYPE/RECO_*_QUEUE_TYPE`이 불일치 → 타입 통일 후 재시도
* **응답 미도착**: `reply_to` 큐 존재/권한 확인, 또는 `RECO_RES_QUEUE`를 미리 선언
* **ValidationError**: 후보 필드 누락/오타 → 페이로드 키 확인

---

---

## 🧮 추천 스코어 구성 (`ai/recommender/recommender.py`) (`ai/recommender/recommender.py`)

가중치(기본):

```
w_pos=0.35  # 장소 긍정비율
w_emo=0.25  # 감정 컨텍스트 매칭
w_cat=0.20  # 카테고리 선호
w_scrap=0.10 # 스크랩 보너스
w_social=0.10 # 팔로우 긍정 작성자 보너스
```

* **정규화**: 배치 내 min‑max → `[0.30, 0.99]` 구간 스케일, **타이브레이커 지터** 적용
* **감정 컴포넌트**: 최근 감정이 \*\*부정(분노/불안/상처/슬픔)\*\*이면 `pos_ratio` 가중 강화, \*\*긍정(기쁨/놀람)\*\*이면 카테고리 선호 비중 강화
* **팔로우 보너스**: `followedPositiveCount/3.0` (최대 1.0)
* `debug: true`일 때 `reason`에 구성요소/원시 점수가 포함됨

> `utils/scorer.py`는 대안·실험용이며 실제 워커는 `recommender.py`를 사용

---

## 🛠️ 개발 워크플로우

* 브랜치 전략: `main`(보호) · `dev`(기본)
* 작업 흐름: **Issue** → `feature/<topic>` → **PR** → `dev` 병합 → 안정화 후 `main` 릴리스
* 커밋 규칙: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:` 등)

---

## ❗ 트러블슈팅

* **AMQPS 연결 실패**: 포트/보안그룹, 인증, vhost 사용 시 **URL 끝 슬래시** 확인. TLS 미사용 브로커는 `amqp://...:5672/` 사용
* **큐 타입/TTL**: 운영 브로커가 **quorum**이면 `MQ_QUEUE_TYPE=quorum` 유지. TTL은 `*_TTL_MS`로 개별 적용 가능
* **인사이트 요약이 빈 문자열**: 데이터 0건이면 기본 안내문 저장
* **점수 타이 동점**: 의도된 지터로 소폭 섞이므로 정상
* **캐시 컨슈머 효과 없음**: 캐시는 **프로세스 로컬**입니다. 같은 프로세스에서 추천을 띄우지 않으면 영향이 없습니다.

---

## 📄 라이선스

MIT

---

## 🙌 크레딧

* Team GeoMemo (Backend/Frontend)
* OpenAI API, FastAPI, Pydantic, SQLAlchemy, aio‑pika

