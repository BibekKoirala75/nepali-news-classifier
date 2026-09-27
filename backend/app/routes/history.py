from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import NewsPrediction
from app.schemas import HistoryItem

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=list[HistoryItem])
def get_history(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    stmt = (
        select(NewsPrediction)
        .order_by(NewsPrediction.created_at.desc())
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()


@router.delete("/history/{item_id}", status_code=204)
def delete_history_item(item_id: int, db: Session = Depends(get_db)):
    record = db.get(NewsPrediction, item_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"History item {item_id} not found.")
    db.execute(delete(NewsPrediction).where(NewsPrediction.id == item_id))
    db.commit()
    return None
