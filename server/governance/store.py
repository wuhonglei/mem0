"""Persist Dream pass reports into the app Postgres database."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import DreamAction, DreamPass

logger = logging.getLogger(__name__)


def persist_report(db: Session, report: Dict[str, Any]) -> None:
    if not report or not report.get("pass_id"):
        return
    row = DreamPass(
        pass_id=report["pass_id"],
        user_id=report.get("user_id"),
        agent_id=report.get("agent_id"),
        run_id=report.get("run_id"),
        source=report.get("source") or "manual",
        stats=report.get("stats") or {},
        summary=report.get("summary"),
        duration_ms=report.get("duration_ms"),
    )
    db.add(row)
    db.flush()
    for action in report.get("actions") or []:
        db.add(
            DreamAction(
                pass_pk=row.id,
                pass_id=row.pass_id,
                type=action.get("type") or "unknown",
                source_ids=action.get("source_ids") or action.get("evidence_ids"),
                canonical_id=action.get("canonical_id"),
                old_id=action.get("old_id"),
                new_id=action.get("new_id") or action.get("id"),
                reason=action.get("reason"),
                old_content=action.get("old_content"),
                new_content=action.get("new_content"),
            )
        )
    db.commit()


def list_passes(db: Session, *, user_id: Optional[str] = None, limit: int = 50) -> List[DreamPass]:
    stmt = select(DreamPass).order_by(DreamPass.created_at.desc()).limit(limit)
    if user_id:
        stmt = select(DreamPass).where(DreamPass.user_id == user_id).order_by(DreamPass.created_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars().all())


def get_pass(db: Session, pass_id: str) -> Optional[Dict[str, Any]]:
    row = db.execute(select(DreamPass).where(DreamPass.pass_id == pass_id)).scalar_one_or_none()
    if row is None:
        return None
    actions = db.execute(select(DreamAction).where(DreamAction.pass_id == pass_id)).scalars().all()
    return {
        "pass_id": row.pass_id,
        "user_id": row.user_id,
        "agent_id": row.agent_id,
        "run_id": row.run_id,
        "source": row.source,
        "stats": row.stats,
        "summary": row.summary,
        "duration_ms": row.duration_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "actions": [
            {
                "type": action.type,
                "source_ids": action.source_ids,
                "canonical_id": action.canonical_id,
                "old_id": action.old_id,
                "new_id": action.new_id,
                "reason": action.reason,
                "old_content": action.old_content,
                "new_content": action.new_content,
            }
            for action in actions
        ],
    }
