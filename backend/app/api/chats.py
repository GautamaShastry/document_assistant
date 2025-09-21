# backend/app/api/chats.py
import os
import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from backend.app.db.session import get_db
from backend.app.db.models import User, Chat, Message, Document, ChatDocument
from backend.app.auth.dependencies import get_current_verified_user
from backend.app.auth.schemas import (
    ChatCreate, ChatResponse, ChatListResponse, ChatWithDocumentResponse,
    ChatQuery, MessageResponse, ChatMessageResponse, DocumentUploadResponse,
    DocumentListResponse, AddDocumentToChatResponse
)

# Import your existing AI services directly
from ai.services.document import load_document, split_documents
from ai.services.vector_store import create_vector_store
from ai.services.rag_pipeline import conversational_answer, stream_conversational_answer, format_chat_messages_for_context
from backend.app.utils.file import save_upload_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chats", tags=["chats"])

# Helper function for document processing
async def process_document_upload(file: UploadFile, user_id: str, chunk_size: int = 900, chunk_overlap: int = 100) -> Document:
    """Process uploaded file and create document record"""
    # Save file
    file_path = await save_upload_file(file, dest_dir="backend/app/data/uploads")
    
    try:
        # Use your existing document processing
        docs = load_document(file_path)
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create vector store with user-scoped name
        import re
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)
        store_name = f"user_{user_id[:8]}_{safe_name}_{uuid.uuid4().hex[:8]}"
        
        create_vector_store(chunks, store_name)
        
        # Create database record
        document = Document(
            id=f"doc_{uuid.uuid4().hex}",
            user_id=user_id,
            filename=os.path.basename(file_path),
            original_name=file.filename,
            file_path=file_path,
            file_size=getattr(file, 'size', None),
            file_type=os.path.splitext(file.filename)[-1].lower(),
            store_name=store_name,
            chunk_count=len(chunks),
            upload_settings={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            upload_status="completed"
        )
        
        return document
        
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise e

# Chat Management Endpoints
@router.post("/create-with-document", response_model=ChatWithDocumentResponse)
async def create_chat_with_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    chunk_size: int = Form(int(os.getenv("CHUNK_SIZE", 900))),
    chunk_overlap: int = Form(int(os.getenv("CHUNK_OVERLAP", 100))),
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Create new chat with initial document upload"""
    try:
        # Process document
        document = await process_document_upload(file, current_user.id, chunk_size, chunk_overlap)
        
        # Create chat
        chat = Chat(
            id=f"chat_{uuid.uuid4().hex}",
            user_id=current_user.id,
            title=title
        )
        db.add(chat)
        db.add(document)
        db.flush()
        
        # Link document to chat
        chat_document = ChatDocument(
            id=f"chatdoc_{uuid.uuid4().hex}",
            chat_id=chat.id,
            document_id=document.id
        )
        db.add(chat_document)
        db.commit()
        
        return ChatWithDocumentResponse(
            chat_id=chat.id,
            title=chat.title,
            document=DocumentUploadResponse(
                id=document.id,
                filename=document.original_name,
                chunk_count=document.chunk_count,
                status="completed"
            ),
            message="Chat created successfully with document!"
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating chat with document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")

@router.get("/", response_model=List[ChatListResponse])
async def get_user_chats(
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get all chats for the current user"""
    try:
        chats = db.query(Chat).filter(
            and_(Chat.user_id == current_user.id, Chat.is_active == True)
        ).order_by(desc(Chat.updated_at)).all()
        
        chat_list = []
        for chat in chats:
            doc_count = db.query(ChatDocument).filter(ChatDocument.chat_id == chat.id).count()
            last_message = db.query(Message).filter(Message.chat_id == chat.id).order_by(desc(Message.created_at)).first()
            
            chat_list.append(ChatListResponse(
                id=chat.id,
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                document_count=doc_count,
                last_message=last_message.content[:100] + "..." if last_message and len(last_message.content) > 100 else last_message.content if last_message else None
            ))
            
        return chat_list
        
    except Exception as e:
        logger.error(f"Error fetching user chats: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch chats")

@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get a specific chat with messages and documents"""
    try:
        chat = db.query(Chat).filter(
            and_(Chat.id == chat_id, Chat.user_id == current_user.id, Chat.is_active == True)
        ).first()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Get messages and documents
        messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
        documents = db.query(Document).join(ChatDocument).filter(
            and_(ChatDocument.chat_id == chat_id, Document.is_active == True)
        ).all()
        
        return ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            documents=[
                DocumentListResponse(
                    id=doc.id,
                    filename=doc.original_name,
                    chunk_count=doc.chunk_count,
                    upload_date=doc.created_at.isoformat(),
                    status=doc.upload_status
                ) for doc in documents
            ],
            messages=[
                ChatMessageResponse(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    sources=msg.sources,
                    created_at=msg.created_at
                ) for msg in messages
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch chat")

@router.post("/{chat_id}/upload-document", response_model=AddDocumentToChatResponse)
async def add_document_to_chat(
    chat_id: str,
    file: UploadFile = File(...),
    chunk_size: int = Form(int(os.getenv("CHUNK_SIZE", 900))),
    chunk_overlap: int = Form(int(os.getenv("CHUNK_OVERLAP", 100))),
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Add additional document to existing chat"""
    try:
        # Verify chat ownership
        chat = db.query(Chat).filter(
            and_(Chat.id == chat_id, Chat.user_id == current_user.id)
        ).first()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Process document
        document = await process_document_upload(file, current_user.id, chunk_size, chunk_overlap)
        
        # Link to chat
        chat_document = ChatDocument(
            id=f"chatdoc_{uuid.uuid4().hex}",
            chat_id=chat_id,
            document_id=document.id
        )
        db.add(document)
        db.add(chat_document)
        db.commit()
        
        return AddDocumentToChatResponse(
            id=document.id,
            filename=document.original_name,
            chunk_count=document.chunk_count,
            status=document.upload_status,
            message="Document added to chat successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding document to chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to add document to chat")

@router.post("/{chat_id}/messages", response_model=MessageResponse)
async def send_message(
    chat_id: str,
    query: ChatQuery,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response"""
    try:
        # Verify chat ownership
        chat = db.query(Chat).filter(
            and_(Chat.id == chat_id, Chat.user_id == current_user.id)
        ).first()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Save user message
        user_message = Message(
            id=f"msg_{uuid.uuid4().hex}",
            chat_id=chat_id,
            content=query.message,
            role="user"
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # Get chat documents
        documents = db.query(Document).join(ChatDocument).filter(
            and_(ChatDocument.chat_id == chat_id, Document.is_active == True)
        ).all()
        
        # Get conversation history (exclude current message)
        chat_history = db.query(Message).filter(
            and_(Message.chat_id == chat_id, Message.id != user_message.id)
        ).order_by(Message.created_at.asc()).all()
        
        # Generate AI response
        if not documents:
            ai_response = "I don't have any documents to reference. Please upload documents to this chat first."
            sources = []
        else:
            try:
                # Use your existing conversational_answer function
                if len(documents) == 1:
                    # Single document - use your function directly
                    conversation_history = format_chat_messages_for_context(chat_history)
                    result = conversational_answer(
                        query.message,
                        documents[0].store_name,
                        conversation_history=conversation_history,
                        k=query.k,
                        include_sources=True
                    )
                    ai_response = result["answer"]
                    sources = result.get("sources", [])
                else:
                    # Multi-document: get answers from each document and combine
                    conversation_history = format_chat_messages_for_context(chat_history)
                    all_results = []
                    all_sources = []
                    
                    for doc in documents:
                        try:
                            result = conversational_answer(
                                query.message,
                                doc.store_name,
                                conversation_history=conversation_history,
                                k=query.k // len(documents) + 1,
                                include_sources=True
                            )
                            all_results.append(f"From {doc.original_name}:\n{result['answer']}")
                            all_sources.extend(result.get("sources", []))
                        except Exception as e:
                            logger.error(f"Error with document {doc.id}: {e}")
                            continue
                    
                    ai_response = "\n\n".join(all_results) if all_results else "Could not process any documents."
                    sources = all_sources
                    
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                ai_response = "I encountered an error while processing your question. Please try again."
                sources = []
        
        # Save AI message
        ai_message = Message(
            id=f"msg_{uuid.uuid4().hex}",
            chat_id=chat_id,
            content=ai_response,
            role="assistant",
            sources=sources
        )
        db.add(ai_message)
        chat.updated_at = ai_message.created_at
        db.commit()
        db.refresh(ai_message)
        
        return MessageResponse(
            user_message=ChatMessageResponse(
                id=user_message.id,
                role=user_message.role,
                content=user_message.content,
                sources=user_message.sources,
                created_at=user_message.created_at
            ),
            ai_message=ChatMessageResponse(
                id=ai_message.id,
                role=ai_message.role,
                content=ai_message.content,
                sources=ai_message.sources,
                created_at=ai_message.created_at
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Could not send message")

@router.post("/{chat_id}/stream")
async def stream_message(
    chat_id: str,
    message: str = Form(...),
    k: int = Form(4),
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Stream AI response for real-time chat experience"""
    try:
        # Verify chat ownership
        chat = db.query(Chat).filter(
            and_(Chat.id == chat_id, Chat.user_id == current_user.id)
        ).first()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Save user message
        user_message = Message(
            id=f"msg_{uuid.uuid4().hex}",
            chat_id=chat_id,
            content=message,
            role="user"
        )
        db.add(user_message)
        db.commit()
        
        # Get documents and history
        documents = db.query(Document).join(ChatDocument).filter(
            and_(ChatDocument.chat_id == chat_id, Document.is_active == True)
        ).all()
        
        chat_history = db.query(Message).filter(
            and_(Message.chat_id == chat_id, Message.id != user_message.id)
        ).order_by(Message.created_at.asc()).all()
        
        def generate_response():
            full_response = ""
            try:
                if not documents:
                    response = "I don't have any documents to reference. Please upload documents to this chat first."
                    yield response
                    full_response = response
                else:
                    # Use your existing stream_conversational_answer function
                    conversation_history = format_chat_messages_for_context(chat_history)
                    
                    # For streaming, use first document (multi-doc streaming is complex)
                    primary_doc = documents[0]
                    
                    for chunk in stream_conversational_answer(
                        message,
                        primary_doc.store_name,
                        conversation_history=conversation_history,
                        k=k
                    ):
                        full_response += chunk
                        yield chunk
                
                # Save complete AI response
                ai_message = Message(
                    id=f"msg_{uuid.uuid4().hex}",
                    chat_id=chat_id,
                    content=full_response,
                    role="assistant"
                )
                db.add(ai_message)
                chat.updated_at = ai_message.created_at
                db.commit()
                
            except Exception as e:
                error_msg = f"\n[ERROR] {str(e)}"
                yield error_msg
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-store"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming message: {e}")
        raise HTTPException(status_code=500, detail="Could not stream message")

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Soft delete a chat"""
    try:
        chat = db.query(Chat).filter(
            and_(Chat.id == chat_id, Chat.user_id == current_user.id)
        ).first()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat.is_active = False
        db.commit()
        
        return {"message": "Chat deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not delete chat")