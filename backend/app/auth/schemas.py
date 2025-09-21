from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
from pydantic.functional_validators import field_validator
from transformers.models import roberta

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    phone: Optional[str] = None
    
class UserCreate(UserBase):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if(len(v) < 8):
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in r"!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`~" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    profile_picture: Optional[str] = None
    
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        if(len(v) < 8):
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in r"!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`~" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v
    
class PasswordReset(BaseModel):
    email: EmailStr
    
class PasswordResetConfirm(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str
    
    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        if(len(v) < 8):
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in r"!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`~" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v
    
class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    phone: Optional[str] = None
    profile_picture: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True
        
class EmailRequest(BaseModel):
    email: EmailStr
    
class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp_code: str
    purpose: str = "login"  # e.g., "email_verification", "password_reset"
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # in minutes
    user: UserResponse
    
class ChatMessageCreate(BaseModel):
    message: str
    
class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
        
class ChatCreate(BaseModel):
    title: str
    
class ChatCreateWithDocumentRequest(BaseModel):
    title: str
    
class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    status: str
    upload_date: datetime
    
class ChatWithDocumentResponse(BaseModel):
    chat_id: str
    title: str
    document: DocumentUploadResponse
    message: str
    
class ChatResponse(BaseModel):
    id: str
    user_id: str
    documents: List[dict] = []
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []
    
    class Config:
        from_attributes = True
        
        
class ChatQuery(BaseModel):
    message: str
    k: int = 4
    
class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size: Optional[int] = None
    file_type: str
    store_name: str
    chunk_count: int
    upload_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True