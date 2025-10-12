"""API Module"""
from fastapi import APIRouter
from backend.app.api.endpoints import router

__all__ = ["router"]