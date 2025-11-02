from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text, Table
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
import os, secrets, string, json
from collections import defaultdict

# ===============================================
# Load environment variables
# ===============================================
load_dotenv()

# ============= CONFIGURATION =============
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ============= DATABASE CONFIGURATION =============
# Use MySQL via PyMySQL on Render. Override with DATABASE_URL env var in Render.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://user:password@localhost:3306/face_attendance?charset=utf8mb4"
)

# ============= DATABASE SETUP =============
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create SQLAlchemy engine (no Cloud SQL connector)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=280
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ============= FACE CONFIGURATION =============
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.92"))
DUPLICATE_FACE_THRESHOLD = float(os.getenv("DUPLICATE_FACE_THRESHOLD", "0.90"))
NONCE_TTL_SECONDS = int(os.getenv("NONCE_TTL_SECONDS", "60"))
_nonces: dict[str, float] = {}

def _now() -> float:
    from time import time as _t
    return _t()

def issue_nonce() -> str:
    n = secrets.token_urlsafe(16)
    _nonces[n] = _now() + NONCE_TTL_SECONDS
    return n

def consume_nonce(nonce: Optional[str]) -> bool:
    if not nonce:
        return False
    exp = _nonces.pop(nonce, None)
    return bool(exp and exp > _now())

# --- Rate limit ---
_RATE_WINDOW = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_CHECKINS", "20"))
_user_hits = defaultdict(list)

def rate_limit(user_id: int):
    now = _now()
    hits = _user_hits[user_id]
    _user_hits[user_id] = [t for t in hits if now - t < _RATE_WINDOW]
    if len(_user_hits[user_id]) >= _RATE_LIMIT:
        raise HTTPException(429, detail="Too many check-ins. Please wait a minute.")
    _user_hits[user_id].append(now)

# ============= MODELS =============
enrollments = Table(
    'enrollments',
    Base.metadata,
    Column('student_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('course_id', Integer, ForeignKey('courses.id'), primary_key=True),
    Column('enrolled_at', DateTime, default=datetime.utcnow)
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    face_embedding = Column(Text, nullable=True)
    student_code = Column(String(20), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    courses_taught = relationship("Course", back_populates="professor")
    enrolled_courses = relationship("Course", secondary=enrollments, back_populates="students")
    attendance_records = relationship("Attendance", back_populates="student")

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    code = Column(String(20), unique=True, index=True, nullable=False)
    professor_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    professor = relationship("User", back_populates="courses_taught")
    students = relationship("User", secondary=enrollments, back_populates="enrolled_courses")
    sessions = relationship("Session", back_populates="course")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    late_after_minutes = Column(Integer, default=5)
    absent_after_minutes = Column(Integer, default=15)
    is_active = Column(Boolean, default=True)
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("Attendance", back_populates="session")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    session = relationship("Session", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# ✅ Create tables AFTER models are defined
Base.metadata.create_all(bind=engine)

# ============= FASTAPI APP =============
app = FastAPI(title="Face Attendance API")

# --- CORS setup (fixed commas) ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://attendoface.github.io",
    "https://attendo-backend.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()

# ============= DEPENDENCY =============
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= JWT HELPERS (python-jose) =============
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============= HELPER FUNCTIONS (UNCHANGED) =============
def generate_join_code(length=6):
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def l2_normalize(vec: List[float]) -> List[float]:
    import math
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec[:]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0

# ============= ROUTES (YOUR EXISTING ONES STAY AS-IS) =============
@app.get("/")
def root():
    return {"message": "Face Attendance API", "status": "running"}

# (All your other routes — register, login, courses, sessions, attendance, etc. — remain unchanged)
