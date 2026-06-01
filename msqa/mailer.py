"""SMTP 기반 메일 발송 — 실험 결과·진행 이슈를 자동으로 메일로 전달한다.

설계 의도
---------
백그라운드(detached) 실험 프로세스가 **자체적으로** 메일을 보내야 한다(채팅 세션이
끊겨도 동작해야 하므로 MCP/에이전트에 의존하지 않는다). 따라서 표준 라이브러리
`smtplib` 만으로 동작하며, 자격증명은 `.env`(common 에서 자동 load_dotenv)에서 읽는다.

필요한 환경변수(.env)
---------------------
- SMTP_HOST       기본 smtp.gmail.com
- SMTP_PORT       기본 587 (STARTTLS). 465 면 자동으로 SSL 사용.
- SMTP_USER       SMTP 로그인 계정(=기본 발신자). 필수.
- SMTP_PASSWORD   앱 비밀번호(Gmail 2단계인증 시 16자리 앱 비밀번호). 필수.
- MAIL_FROM       발신 표시 주소. 미설정 시 SMTP_USER.
- MAIL_TO         기본 수신자. 미설정 시 thomas@itengineers.net.

사용
----
    # 단독 발송 테스트(자격증명 점검)
    python -m msqa.mailer --test

    # 코드에서
    from msqa.mailer import send_mail, notify_issue, notify_result
    send_mail("제목", "본문")
    notify_issue("retriever 초기화", err)      # 실패 시 즉시 알림(예외 안전)
    notify_result("V0 topk50 완료", summary)    # 결과 알림
"""
from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import socket
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from .common import get_logger

logger = get_logger("mailer")

DEFAULT_TO = "thomas@itengineers.net"


class MailConfigError(RuntimeError):
    """SMTP 자격증명/설정 누락."""


def _cfg() -> dict:
    """환경변수에서 SMTP 설정을 읽어 검증한다. 누락 시 MailConfigError."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.environ.get("SMTP_PORT", "587").strip() or "587")
    user = (os.environ.get("SMTP_USER") or "").strip()
    password = (os.environ.get("SMTP_PASSWORD") or "").strip()
    mail_from = (os.environ.get("MAIL_FROM") or user).strip()
    mail_to = (os.environ.get("MAIL_TO") or DEFAULT_TO).strip()

    missing = [k for k, v in (("SMTP_USER", user), ("SMTP_PASSWORD", password)) if not v]
    if missing:
        raise MailConfigError(
            f"SMTP 자격증명 누락: {', '.join(missing)}. .env 에 추가하세요 "
            f"(예: SMTP_USER=you@gmail.com / SMTP_PASSWORD=앱비밀번호16자리)."
        )
    return {
        "host": host, "port": port, "user": user, "password": password,
        "from": mail_from or user, "to": mail_to,
    }


def send_mail(
    subject: str,
    body: str,
    *,
    to: str | list[str] | None = None,
    html: str | None = None,
    timeout: float = 30.0,
) -> None:
    """메일 1통 발송. 실패 시 예외를 올린다(테스트/명시 호출용).

    실험 루프에서 '절대 죽으면 안 되는' 알림은 send_mail 을 직접 쓰지 말고
    notify_issue/notify_result(예외 안전 래퍼)를 사용한다.
    """
    cfg = _cfg()
    recipients = to or cfg["to"]
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(",") if r.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="msqa.local")
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")

    context = ssl.create_default_context()
    if cfg["port"] == 465:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=timeout, context=context) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg, from_addr=cfg["from"], to_addrs=recipients)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=timeout) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg, from_addr=cfg["from"], to_addrs=recipients)

    logger.info("메일 발송 완료 → %s | subject=%r", ", ".join(recipients), subject)


def _safe_send(subject: str, body: str, **kw) -> bool:
    """예외를 삼키고 로깅만 하는 안전 발송. 실험 루프 중단 방지용. 성공 시 True."""
    try:
        send_mail(subject, body, **kw)
        return True
    except Exception as e:  # noqa: BLE001 — 알림 실패가 실험을 멈추면 안 된다
        logger.warning("메일 발송 실패(무시하고 진행): %s: %s", type(e).__name__, e)
        return False


def _host_line() -> str:
    try:
        return f"{socket.gethostname()}"
    except Exception:  # noqa: BLE001
        return "unknown-host"


def notify_issue(context: str, error: object, *, run_label: str = "") -> bool:
    """진행 중 이슈를 즉시 메일로 알린다(예외 안전). 성공 시 True.

    context: 어디서 난 문제인지(짧은 설명). error: 예외/메시지 객체.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"[{run_label}] " if run_label else ""
    subject = f"⚠️ [msqa] {tag}실험 이슈: {context}"
    body = (
        f"실험 진행 중 이슈가 발생했습니다.\n\n"
        f"- 시각: {ts}\n"
        f"- 호스트: {_host_line()}\n"
        f"- 실행: {run_label or '(미지정)'}\n"
        f"- 위치: {context}\n"
        f"- 내용: {type(error).__name__ if isinstance(error, BaseException) else 'message'}: {error}\n\n"
        f"— msqa 자동 알림\n"
    )
    return _safe_send(subject, body)


def notify_result(title: str, summary: str, *, run_label: str = "") -> bool:
    """결과/완료 요약을 메일로 알린다(예외 안전). 성공 시 True."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"[{run_label}] " if run_label else ""
    subject = f"✅ [msqa] {tag}{title}"
    body = (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"- 시각: {ts}\n"
        f"- 호스트: {_host_line()}\n"
        f"- 실행: {run_label or '(미지정)'}\n\n"
        f"{summary}\n\n"
        f"— msqa 자동 알림\n"
    )
    return _safe_send(subject, body)


def _main() -> int:
    ap = argparse.ArgumentParser(description="SMTP 메일 발송 테스트/단발 발송")
    ap.add_argument("--test", action="store_true", help="설정 점검용 테스트 메일 발송")
    ap.add_argument("--to", default=None, help="수신자(기본=MAIL_TO 또는 thomas@itengineers.net)")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--body", default=None)
    args = ap.parse_args()

    if args.test or not (args.subject and args.body):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = args.subject or "[msqa] 메일 발송 테스트 (SMTP)"
        body = args.body or (
            "이 메일이 보이면 SMTP 자동 발송이 정상 동작하는 것입니다.\n\n"
            f"- 시각: {ts}\n"
            f"- 호스트: {_host_line()}\n"
            "- 경로: 백그라운드 프로세스 → smtplib → SMTP 서버\n\n"
            "실험 결과 및 진행 중 이슈가 이 경로로 자동 전달됩니다.\n\n"
            "— msqa 자동 알림\n"
        )
    else:
        subject, body = args.subject, args.body

    try:
        send_mail(subject, body, to=args.to)
    except MailConfigError as e:
        print(f"[mailer] 설정 오류: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"[mailer] 발송 실패: {type(e).__name__}: {e}")
        return 1
    print(f"[mailer] 발송 성공 → {args.to or os.environ.get('MAIL_TO') or DEFAULT_TO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
