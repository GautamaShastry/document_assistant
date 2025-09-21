from fastapi import APIRouter
from .endpoints import router as core_router
from .auth import router as auth_router
from .chats import router as chats_router

router = APIRouter()

router.include_router(core_router, prefix="")
router.include_router(auth_router, prefix="")
router.include_router(chats_router, prefix="")