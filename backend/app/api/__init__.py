from fastapi import APIRouter
from .endpoints import router as core_router

router = APIRouter()
router.include_router(core_router, prefix="")