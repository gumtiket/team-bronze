"""서비스와 어댑터가 주고받는 데이터.

이 모듈이 구조로 강제하는 규칙:
  * 환경에는 서비스 DB에 저장해도 되는 값(역할 ARN, External ID, 서버 주소)만
    담는다. 개인 키와 토큰은 절대 들어가지 않는다.
  * 결과는 성공이거나, 오류를 담은 실패 중 하나다. 둘 다이거나 둘 다 아닌 경우는 없다.
  * 로그 이벤트는 구조화된 데이터를 담을 수 있지만, 비밀을 걸러내는 일은
    어댑터의 몫이다(redact.py 참고).
"""
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

ENV_ID_PATTERN = r"^[a-z][a-z0-9-]{1,20}$"
APP_NAME_PATTERN = r"^[a-z][a-z0-9-]{2,62}$"  # DNS 라벨, Compose 프로젝트 이름, Lambda 이름으로 쓰인다
IMAGE_TAG_PATTERN = r"^[0-9a-f]{7,40}$"  # 커밋 SHA


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- 환경 -----------------------------------------------------------------------------
class AwsEnvironment(AdapterModel):
    """사용자의 AWS 계정. 교차 계정 배포 역할로 접근한다."""

    kind: Literal["aws"] = "aws"
    env_id: str = Field(pattern=ENV_ID_PATTERN)
    role_arn: str = Field(pattern=r"^arn:aws:iam::[0-9]{12}:role/[\w+=,.@/-]+$")
    external_id: str = Field(min_length=16, max_length=128, pattern=r"^[\w+=,.@:/-]+$")
    region: str = Field(default="ap-northeast-2", pattern=r"^[a-z]{2}(-[a-z]+)+-[0-9]$")

    # 사용자 계정 공용 기반(infra/user-account)의 출력값. 서비스가 기반을 만든 뒤 저장해 두었다가
    # 넘겨 준다. 기반이 아직 없으면 비어 있고, 이 경우 check와 deploy는 `foundation_missing`으로 실패한다.
    # 비밀이 아니다(주소와 비밀의 ARN뿐이며, 비밀번호 자체는 Secrets Manager에만 있다).
    host: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
    ssh_user: str = Field(default="deploy", pattern=r"^[a-z_][a-z0-9_-]{0,31}$")
    ssh_port: int = Field(default=22, ge=1, le=65535)
    # RDS 주소는 psql 접속 문자열에 들어가므로, 임의의 호스트가 끼어들지 못하게 RDS 도메인만 허용한다.
    db_address: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,62}(\.[a-z0-9-]{1,63})*\.rds\.amazonaws\.com$")
    db_port: int = Field(default=5432, ge=1024, le=65535)
    db_secret_arn: str | None = Field(
        default=None, pattern=r"^arn:aws:secretsmanager:[a-z0-9-]+:[0-9]{12}:secret:[\w+=,.@/-]+$")


class OnpremEnvironment(AdapterModel):
    """사용자가 한 번 준비해 둔 서버(Docker, 배포 계정, 우리 공개 키)."""

    kind: Literal["onprem"] = "onprem"
    env_id: str = Field(pattern=ENV_ID_PATTERN)
    host: str = Field(pattern=r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
    ssh_user: str = Field(default="deploy", pattern=r"^[a-z_][a-z0-9_-]{0,31}$")
    ssh_port: int = Field(default=22, ge=1, le=65535)


Environment = Annotated[Union[AwsEnvironment, OnpremEnvironment], Field(discriminator="kind")]

# AI 쪽이 만든 배포 명세. 어댑터는 이 값을 믿지 않는다. 자신이 사용하는 모든
# 필드(이름, 명령, 크기)를 직접 다시 검증한다.
Spec = Mapping[str, Any]

# 한 번의 배포에 쓰는 비밀 값(이름 -> 값). 메모리에만 있고 서비스 DB, 로그,
# 결과에 남기지 않는다.
Secrets = Mapping[str, str]


# --- 진행 로그 ---------------------------------------------------------------------------
class LogEvent(AdapterModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: Literal["info", "warn", "error"] = "info"
    step: int | None = Field(default=None, ge=1)
    total: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=80)
    message: str = Field(max_length=2000)
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def step_within_total(self):
        if self.step is not None and self.total is not None and self.step > self.total:
            raise ValueError("step must not exceed total")
        return self


# --- 결과 ---------------------------------------------------------------------------------
class AdapterError(AdapterModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")  # 기계가 읽는 코드(예: ssh_unreachable)
    message: str = Field(max_length=2000)  # 사용자에게 보여 줄 메시지
    hint: str | None = Field(default=None, max_length=2000)  # 사용자가 할 수 있는 조치
    retryable: bool = False


class Result(AdapterModel):
    ok: bool
    error: AdapterError | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ok_xor_error(self):
        if self.ok and self.error is not None:
            raise ValueError("a successful result must not carry an error")
        if not self.ok and self.error is None:
            raise ValueError("a failed result must carry an error")
        return self


class CheckResult(Result):
    pass


class DeployResult(Result):
    url: str | None = None
    image_tag: str | None = None


class StatusResult(Result):
    state: Literal["running", "unhealthy", "stopped", "not_deployed", "unknown"] = "unknown"
    url: str | None = None
    image_tag: str | None = None


class DestroyResult(Result):
    pass
