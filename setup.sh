#!/bin/bash

# setup.sh - Complete setup script for Intelligent Document Assistant

echo "🚀 Setting up Intelligent Document Assistant..."

# Create backend directory structure
echo "📁 Creating backend directory structure..."
mkdir -p backend/app/{api,auth,db,services,utils,data/{uploads,vectorstore}}

# Create all __init__.py files
echo "📝 Creating Python package files..."
touch backend/__init__.py
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/auth/__init__.py
touch backend/app/db/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file..."
    cat > .env << 'EOF'
# =================================================================
# INTELLIGENT DOCUMENT ASSISTANT - ENVIRONMENT CONFIGURATION
# =================================================================

# Database Configuration
DATABASE_URL=sqlite:///./backend/app/data/app.db

# JWT Authentication (CHANGE THIS!)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-make-it-very-long-and-random

# Ollama Configuration
OLLAMA_MODEL=qwen3:8b
OLLAMA_HOST=http://localhost:11434

# Document Processing Settings
CHUNK_SIZE=900
CHUNK_OVERLAP=100

# Embedding Model Settings
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBED_DEVICE=cpu
EMBED_NORMALIZE=false

# Storage Directories
VECTORSTORE_DIR=backend/app/data/vectorstore
UPLOAD_DIR=backend/app/data/uploads

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000

# Email Configuration (for OTP)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Intelligent Document Assistant

# Development Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file"
else
    echo "⚠️  .env file already exists"
fi

# Install dependencies
echo "📦 Installing backend dependencies..."
if command -v pip &> /dev/null; then
    pip install -r backend/requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  pip not found. Please install manually:"
    echo "   pip install -r backend/requirements.txt"
fi

# Check Ollama
echo "🤖 Checking Ollama..."
if command -v ollama &> /dev/null; then
    if pgrep -x "ollama" > /dev/null; then
        echo "✅ Ollama is running"
        if ollama list | grep -q "qwen3:8b"; then
            echo "✅ Model qwen3:8b is available"
        else
            echo "📥 Pulling model qwen3:8b..."
            ollama pull qwen3:8b
        fi
    else
        echo "⚠️  Ollama is installed but not running. Please run:"
        echo "   ollama serve"
        echo "   ollama pull qwen3:8b"
    fi
else
    echo "⚠️  Ollama not found. Please install from: https://ollama.ai"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Update .env with your credentials:"
echo "   - Change JWT_SECRET_KEY to something secure"
echo "   - Add your Gmail app password for OTP emails"
echo ""
echo "2. Start Ollama (if not running):"
echo "   ollama serve"
echo ""
echo "3. Start the development server:"
echo "   python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "4. Visit the API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "🔧 Backend Features:"
echo "   ✅ Password + OTP authentication"
echo "   ✅ Email verification"
echo "   ✅ Document upload & processing"
echo "   ✅ Conversational AI with context"
echo "   ✅ Real-time streaming responses"
echo "   ✅ User isolation & security"
echo ""
echo "📱 API Endpoints Ready:"
echo "   POST /api/auth/register       - Register user"
echo "   POST /api/auth/login          - Login user"
echo "   POST /api/auth/verify-email   - Verify OTP"
echo "   POST /api/upload              - Upload document"
echo "   POST /api/chats               - Create chat"
echo "   POST /api/chats/{id}/messages - Send message"
echo ""

# Make script executable
chmod +x setup.sh

echo "🚀 Ready to go!"