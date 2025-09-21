from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from backend.app.db.session import get_db
from backend.app.db.models import User, OTPVerification
from backend.app.auth.jwt_handler import JwtHandler
from backend.app.auth.dependencies import get_current_user
from backend.app.auth.schemas import (
    EmailRequest, OTPVerifyRequest, TokenResponse, UserCreate, UserLogin, UserResponse,
    UserUpdate, PasswordChange, PasswordReset, PasswordResetConfirm
)
from backend.app.services.email_service import EmailService
from backend.app.auth.password_utils import hash_password, verify_password

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
        
        # create new user with hashed password
        new_user = User(
            email=user.email,
            full_name=user.full_name,
            phone=user.phone,
            hashed_password=hash_password(user.password),
            is_active=True,
            is_verified=False # require email verification
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Send verification email
        await send_verification_otp(new_user.email, new_user.id, "registration", db)
        
        return {
            "message": "User registered successfully. Please verify your email.",
            "user_id": new_user.id
        }
    
    except HTTPException as he:
        raise he    
    
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Registration failed.")
    
@router.post("/login")
async def login_user(user_login: UserLogin, db: Session = Depends(get_db)):
    """Login with email and password"""
    try:
        user = db.query(User).filter(User.email == user_login.email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
        if not verify_password(user_login.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
        # check if user is active
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
        # check if user is verified
        if not user.is_verified:
            await send_verification_otp(user.email, user.id, "login", db)
            return {
                "message": "Email not verified. A new OTP has been sent to your email.",
                "requires_verification": True,
                "user_id": user.id
            }
        
        token = JwtHandler.encode_token(user.id, user.email)
        
        return TokenResponse(access_token=token, token_type="bearer", expires_in=24 * 60, user=UserResponse.from_orm(user))
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Login failed.")
    
@router.post("/verify-email", response_model=TokenResponse)
async def verify_email(otp_request: OTPVerifyRequest, db: Session = Depends(get_db)):
    try:
        # find user
        user = db.query(User).filter(User.email == otp_request.email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        otp = db.query(OTPVerification).filter(
            OTPVerification.user_id == user.id,
            OTPVerification.email == otp_request.email,
            OTPVerification.otp_code == otp_request.otp_code,
            OTPVerification.purpose.in_(["registration", "login"]),
            OTPVerification.is_used == False,
            OTPVerification.expires_at > datetime.utcnow()
        ).first()
        
        if not otp:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OTP code")
        
        otp.is_used = True
        user.is_verified = True
        db.commit()
        
        access_token = JwtHandler.encode_token(user.id, user.email)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 60,
            user=UserResponse.from_orm(user)
        )
        
    except HTTPException as he:
        raise he
        
    except Exception as e:
        logger.error(f"Error verifying email: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Email verification failed.")
@router.post("/resend-verification")
async def resend_verification(email_request: EmailRequest, db: Session = Depends(get_db)):
    """Resend verification OTP to email"""
    try:
        user = db.query(User).filter(User.email == email_request.email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        if user.is_verified:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already verified")
        
        await send_verification_otp(user.email, user.id, "registration", db)
        
        return {"message": "Verification Code send successfully! Please check your email."}
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error resending verification OTP: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Could not resend OTP.") 
    
@router.post("/forgot-password")
async def forgot_password(request: PasswordReset, db: Session = Depends(get_db)):
    """Initiate password reset by sending OTP to email"""
    try:
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            return {"message": "If the email is registered, a password reset OTP has been sent."}
        
        await send_verification_otp(user.email, user.id, "password_reset", db)
        
        return {"message": "Password reset OTP sent successfully! Please check your email."}
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error initiating password reset: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Could not initiate password reset.")   
    
@router.post("/reset-password")
async def reset_password(reset_request: PasswordResetConfirm, db: Session = Depends(get_db)):
    """Reset password using OTP"""
    try:
        user = db.query(User).filter(User.email == reset_request.email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        otp = db.query(OTPVerification).filter(
            OTPVerification.user_id == user.id,
            OTPVerification.email == reset_request.email,
            OTPVerification.otp_code == reset_request.otp_code,
            OTPVerification.purpose == "password_reset",
            OTPVerification.is_used == False,
            OTPVerification.expires_at > datetime.utcnow()
        ).first()
        
        if not otp:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OTP code")
        
        otp.is_used = True
        user.hashed_password = hash_password(reset_request.new_password)
        db.commit()
        
        return {"message": "Password reset successfully. You can now log in with your new password."}
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Could not reset password.")
    
@router.post("/change-password")
async def change_password(request: PasswordChange, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Change password for authenticated user"""
    try:
        
        if not verify_password(request.current_password, current_user.hashed_password):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
        
        current_user.hashed_password = hash_password(request.new_password)
        db.commit()
        
        return {"message": "Password changed successfully"}
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Could not change password.")
    
@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user info"""
    return UserResponse.from_orm(current_user)

@router.put("/me", response_model=UserResponse)
async def update_current_user_info(update: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update current authenticated user info"""
    try:
        if update.full_name is not None:
            current_user.full_name = update.full_name
        if update.phone is not None:
            # check if phone is unique
            existing_user = db.query(User).filter(User.phone == update.phone, User.id != current_user.id).first()
            if existing_user:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone number already in use")
            current_user.phone = update.phone
        if update.profile_picture is not None:
            current_user.profile_picture = update.profile_picture
        
        db.commit()
        db.refresh(current_user)
        
        return UserResponse.from_orm(current_user)
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logger.error(f"Error updating user info: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error. Could not update user info.")
    
@router.post("/logout")
async def logout_user():
    """Logout user (handled client-side by deleting token)"""
    return {"message": "Logout successful. Please delete the token on client side."}

async def send_verification_otp(email: str, user_id: str, purpose: str, db: Session):
    """Send OTP email for verification or login"""
    try:
        # Deactivate existing unused OTPs for this user and purpose
        db.query(OTPVerification).filter(
            OTPVerification.user_id == user_id,
            OTPVerification.purpose == purpose,
            OTPVerification.is_used == False,
        ).update({"is_used": True})
        
        # Create new OTP
        otp = OTPVerification.create_otp(user_id=user_id, email=email, purpose=purpose)
        db.add(otp)
        db.commit()
        
        # Map purpose to EmailService format
        purpose_mapping = {
            "registration": "registration",      # Maps to EmailService "registration"
            "email_verification": "registration", # Alternative mapping
            "password_reset": "password_reset",
            "2fa": "login",
            "login": "login"
        }
        email_purpose = purpose_mapping.get(purpose, "login")
        
        # Send email using the EmailService instance (not creating new instance)
        from backend.app.services.email_service import email_service
        success = email_service.send_otp_email(
            to_email=email, 
            otp_code=otp.otp_code, 
            purpose=email_purpose
        )
        
        if not success:
            logger.error(f"Failed to send OTP email to {email}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Failed to send OTP email. Please try again later."
            )
        
        logger.info(f"OTP email sent successfully to {email}")
        return success
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in send_verification_otp: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process OTP request"
        )
