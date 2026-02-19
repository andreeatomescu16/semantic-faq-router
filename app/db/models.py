from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Float

try:
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:
    Vector = None  # type: ignore[assignment]


class Base(DeclarativeBase):
    pass


class KBItem(Base):
    __tablename__ = "kb_items"
    __table_args__ = (UniqueConstraint("question_norm", "category", name="uq_qnorm_category"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    question_norm: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    answer_norm: Mapped[str] = mapped_column(Text, nullable=False)
    raw_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    embeddings: Mapped[list["KBEmbedding"]] = relationship(
        "KBEmbedding",
        back_populates="kb_item",
        cascade="all, delete-orphan",
    )


class KBEmbedding(Base):
    __tablename__ = "kb_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "content_hash",
            "model_name",
            "embedding_version",
            name="uq_hash_model_version",
        ),
        Index("ix_kb_embeddings_kb_item_id", "kb_item_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    kb_item_id: Mapped[int] = mapped_column(ForeignKey("kb_items.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(1536) if Vector else ARRAY(Float),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(50), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    kb_item: Mapped[KBItem] = relationship("KBItem", back_populates="embeddings")
