"""Dream governance REST endpoints."""

import logging
from typing import List, Optional

from auth import verify_auth
from db import get_db
from errors import upstream_error
from fastapi import APIRouter, Depends, HTTPException, Query
from governance.engine import run_dream
from governance.store import get_pass, list_passes, persist_report
from pydantic import BaseModel, Field
from server_state import get_memory_instance
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dream", tags=["dream"])


class DreamRunRequest(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    consolidate: bool = Field(True, description="Run the consolidate (merge/supersede) pass.")
    synthesize: bool = Field(True, description="Run the synthesis pass after consolidate.")
    force: bool = Field(False, description="Bypass the minimum-memory threshold for synthesis.")
    source: Optional[str] = Field(
        None,
        description=(
            "Label recorded on the pass (default 'manual'). Use it to tell callers apart, "
            "e.g. source=scheduler for a cron job, then read them back with "
            "GET /dream?source=manual,scheduler."
        ),
    )


@router.post("")
def run_dream_pass(body: DreamRunRequest, _auth=Depends(verify_auth), db: Session = Depends(get_db)):
    if not any([body.user_id, body.agent_id, body.run_id]):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required.",
        )
    try:
        report = run_dream(
            get_memory_instance(),
            user_id=body.user_id,
            agent_id=body.agent_id,
            run_id=body.run_id,
            consolidate=body.consolidate,
            synthesize=body.synthesize,
            force=body.force,
            source=body.source,
        )
        try:
            persist_report(db, report)
        except Exception:
            logger.exception("Failed to persist Dream pass %s", report.get("pass_id"))
        return report
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise upstream_error()


@router.get("")
def list_dream_passes(
    user_id: Optional[str] = None,
    source: Optional[List[str]] = Query(
        None,
        description=(
            "Only return passes whose source matches one of these values "
            "(repeatable or comma-separated). Use source=manual to get full "
            "passes only and skip the on_add light consolidations."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    _auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    # 允许 ?source=manual 与 ?source=manual,api 两种写法
    sources = [
        item.strip() for value in (source or []) for item in value.split(",") if item.strip()
    ]
    rows = list_passes(db, user_id=user_id, sources=sources or None, limit=limit)
    return {
        "results": [
            {
                "pass_id": row.pass_id,
                "user_id": row.user_id,
                "agent_id": row.agent_id,
                "run_id": row.run_id,
                "source": row.source,
                "stats": row.stats,
                "summary": row.summary,
                "duration_ms": row.duration_ms,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@router.get("/{pass_id}")
def get_dream_pass(pass_id: str, _auth=Depends(verify_auth), db: Session = Depends(get_db)):
    report = get_pass(db, pass_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Dream pass not found")
    return report
