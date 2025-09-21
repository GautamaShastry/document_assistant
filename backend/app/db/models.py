from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean 
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import uuid
from torch.ao.quantization.quantize import is_activation_post_process

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    phone = Column(String, unique=True, index=True, nullable=True)
    is_verified = Column(Boolean, default=False)
    profile_picture = Column(String, nullable=True)  # URL or file path
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    otp_codes = relationship("OTPVerification", back_populates="user", cascade="all, delete-orphan")
    
class OTPVerification(Base):
    __tablename__ = "otp_codes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    email = Column(String, nullable=False)
    otp_code = Column(String, nullable=False)
    purpose = Column(String, nullable=False)  # e.g., "email_verification", "password_reset"
    is_used = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="otp_codes")
    
    @classmethod
    def create_otp(cls, user_id: str, email: str, purpose: str, expires_in_minutes: int = 10 * 60):
        import random
        otp_code = f"{random.randint(100000, 999999)}"
        expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
        
        return cls(
            user_id=user_id,
            email=email,
            otp_code=otp_code,
            purpose=purpose,
            expires_at=expires_at
        )
        
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String, nullable=False)
    vector_store_name = Column(String, nullable=False)
    index_id = Column(String, unique=True, nullable=False)
    chunk_count = Column(Integer, default=0)
    upload_status = Column(String, default="pending")  # e.g., "pending", "processing", "completed", "failed"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="documents")
    chats = relationship("Chat", back_populates="document")
    
class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="chats")
    document = relationship("Document", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.id"), nullable=False)
    role = Column(String, nullable=False)  # e.g., "user", "assistant"
    content = Column(Text, nullable=False)
    msg_metadata = Column(JSON, nullable=True)  # e.g., {"source": "document.pdf", "page": 2}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    chat = relationship("Chat", back_populates="messages")