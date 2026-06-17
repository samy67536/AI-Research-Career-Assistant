# backend/database/connection.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
import uuid
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings


DATABASE_URL = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username   = Column(String(50), unique=True, nullable=False)
    email      = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    filename     = Column(String(255), nullable=False)
    file_path    = Column(Text, nullable=False)
    file_type    = Column(String(10))
    doc_type     = Column(String(20))   # 'research' | 'resume'
    page_count   = Column(Integer)
    word_count   = Column(Integer)
    vector_path  = Column(Text)
    uploaded_at  = Column(DateTime, server_default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    doc_id      = Column(String, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    module_type = Column(String(20))   # 'research' | 'resume'
    created_at  = Column(DateTime, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id  = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role        = Column(String(10))   # 'user' | 'assistant'
    content     = Column(Text, nullable=False)
    sources     = Column(JSON, nullable=True)
    created_at  = Column(DateTime, server_default=func.now())


class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id          = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    doc_id           = Column(String, ForeignKey("documents.id"))
    ats_score        = Column(Float)
    skills_found     = Column(JSON)
    skills_missing   = Column(JSON)
    jd_text          = Column(Text)
    similarity_score = Column(Float)
    suggestions      = Column(JSON)
    cover_letter     = Column(Text)
    analyzed_at      = Column(DateTime, server_default=func.now())


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
