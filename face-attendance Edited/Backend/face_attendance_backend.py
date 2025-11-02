# face_attendance_backend.py
from __future__ import annotations

import os
import json
import math
import secrets
import string
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

# ── Use PyJWT (only) ────────────────────────────────────────────────────────────
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

# ── SQLAlchemy (database-agnostic via DATABASE_URL) ────────────────────────────
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text, Table
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# ========================== CONFIG =============================================
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7)))

# Face thresholds
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.92"))
DUPLICATE_FACE_THRESHOLD = float(os.getenv("DUPLICATE_FACE_THRESHOLD", "0.90"))

# Rate limiting + nonce
NONCE_TTL_SECONDS = int(os.getenv("NONCE_TTL_SECONDS", "60"))
_RATE_WINDOW = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_CHECKINS", "20"))

# CORS
DEFAULT_ORIGINS = [
    "https://attendoface.github.io",
    "https://attendoface.github.io/attendo",  # project pages URL
]
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = DEFAULT_ORIGINS

# Database URL (works with Postgres/MySQL/SQLite)
# Examples:
#   postgresql+psycopg://user:pass@host:5432/dbname
#   mysql+pymysql://user:pass@host:3306/dbname
#   sqlite:///./attendo.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendo.db")

# ========================== UTILS ==============================================
def _now() -> float:
    from time import time as _t
    return _t()

_nonces: dict[str, float] = {}          # nonce -> expires_at (epoch)
_user_hits = defaultdict(list)          # user_id -> [timestamps]

def issue_nonce() -> str:
    n = secrets.token_urlsafe(16)
    _nonces[n] = _now() + NONCE_TTL_SECONDS
    return n

def consume_nonce(nonce: Optional[str]) -> bool:
    if not nonce:
        return False
    exp = _nonces.pop(nonce, None)
    return bool(exp and exp > _now())

def rate_limit(user_id: int):
    now = _now()
    hits = _user_hits[user_id]
    _user_hits[user_id] = [t for t in hits if now - t < _RATE_WINDOW]
    if len(_user_hits[user_id]) >= _RATE_LIMIT:
        raise HTTPException(429, detail="Too many check-ins. Please wait a minute.")
    _user_hits[user_id].append(now)

def generate_join_code(length=6) -> str:
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if not norm:
        return vec[:]
    return [v / norm for v in vec]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

# ========================== DB SETUP ===========================================
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create engine/session
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Assoc table
enrollments = Table(
    'enrollments', Base.metadata,
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
    role = Column(String(50), nullable=False)  # student/professor/admin
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
    status = Column(String(20), nullable=False)  # present/late/absent
    confidence = Column(Float, nullable=True)

    session = relationship("Session", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# Create tables on import
Base.metadata.create_all(bind=engine)

# ========================== SCHEMAS ============================================
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str
    face_embedding: Optional[List[float]] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class CourseCreate(BaseModel):
    name: str

class CourseEnroll(BaseModel):
    join_code: str

class SessionStart(BaseModel):
    course_id: int
    late_after_minutes: int = 5
    absent_after_minutes: int = 15

class CheckIn(BaseModel):
    course_id: int
    face_embedding: List[float]
    liveness: Optional[bool] = None
    nonce: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    name: str

# ========================== APP / CORS / SECURITY ==============================
app = FastAPI(title="Face Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()

# ========================== DEPENDENCIES =======================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ========================== ROUTES =============================================
@app.get("/")
def root():
    return {"message": "Face Attendance API", "status": "running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/checkin/nonce")
def get_checkin_nonce(token_data: dict = Depends(verify_token)):
    return {"nonce": issue_nonce(), "ttl": NONCE_TTL_SECONDS}

@app.post("/auth/register")
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user_data.role == "student" and (not user_data.face_embedding or len(user_data.face_embedding) == 0):
        raise HTTPException(status_code=400, detail="Face photo/embedding required for student accounts")

    hashed = pwd_context.hash(user_data.password[:72])

    face_str = None
    norm_emb = None
    if user_data.face_embedding:
        norm_emb = l2_normalize(user_data.face_embedding)
        existing_with_face = db.query(User).filter(User.face_embedding.isnot(None)).all()
        for u in existing_with_face:
            try:
                other = json.loads(u.face_embedding)
            except Exception:
                continue
            sim = cosine_similarity(norm_emb, l2_normalize(other))
            if sim >= DUPLICATE_FACE_THRESHOLD:
                raise HTTPException(status_code=409, detail=f"Face already registered to another account ({u.email}).")
        face_str = json.dumps(norm_emb)

    student_code = None
    if user_data.role == "student":
        last_student = db.query(User).filter(User.role == "student").order_by(User.id.desc()).first()
        next_id = 1 if not last_student else last_student.id + 1
        student_code = f"STU{next_id:04d}"

    user = User(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hashed,
        role=user_data.role,
        face_embedding=face_str,
        student_code=student_code
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"user_id": user.id, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "user_id": user.id,
        "student_code": student_code
    }

@app.post("/auth/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not pwd_context.verify(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"user_id": user.id, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "name": user.name, "user_id": user.id}

@app.post("/courses")
def create_course(course_data: CourseCreate, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can create courses")
    while True:
        code = generate_join_code()
        if not db.query(Course).filter(Course.code == code).first():
            break
    course = Course(name=course_data.name, code=code, professor_id=token_data["user_id"])
    db.add(course); db.commit(); db.refresh(course)
    return {"id": course.id, "name": course.name, "code": course.code}

@app.get("/courses/mine")
def get_my_courses(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") == "professor":
        courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    else:
        user = db.query(User).filter(User.id == token_data["user_id"]).first()
        courses = user.enrolled_courses
    return [{"id": c.id, "name": c.name, "code": c.code} for c in courses]

@app.post("/courses/enroll")
def enroll_course(enroll_data: CourseEnroll, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.code == enroll_data.join_code.upper()).first()
    if not course:
        raise HTTPException(status_code=404, detail="Invalid join code")
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if course in user.enrolled_courses:
        raise HTTPException(status_code=400, detail="Already enrolled")
    user.enrolled_courses.append(course); db.commit()
    return {"course_id": course.id, "message": "Enrolled successfully"}

@app.post("/sessions/start")
def start_session(session_data: SessionStart, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can start sessions")
    course = db.query(Course).filter(Course.id == session_data.course_id, Course.professor_id == token_data["user_id"]).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found or not owned by you")
    active = db.query(Session).filter(Session.course_id == session_data.course_id, Session.is_active == True).all()
    for s in active:
        s.is_active = False; s.end_time = datetime.utcnow()
    session = Session(course_id=session_data.course_id,
                      late_after_minutes=session_data.late_after_minutes,
                      absent_after_minutes=session_data.absent_after_minutes)
    db.add(session); db.commit(); db.refresh(session)
    return {
        "id": session.id, "course_id": session.course_id,
        "start_time": session.start_time.isoformat(),
        "late_after_minutes": session.late_after_minutes,
        "absent_after_minutes": session.absent_after_minutes
    }

@app.post("/sessions/{session_id}/end")
def end_session(session_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    course = db.query(Course).filter(Course.id == session.course_id).first()
    if not course or course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not allowed to end this session")
    session.is_active = False; session.end_time = datetime.utcnow(); db.commit()
    return {"message": "Session ended"}

@app.get("/sessions/active")
def check_active_session(course_id: int, db: Session = Depends(get_db)):
    session = db.query(Session).filter(Session.course_id == course_id, Session.is_active == True).first()
    return {"active": session is not None}

class CheckIn(BaseModel):
    course_id: int
    face_embedding: List[float]
    liveness: Optional[bool] = None
    nonce: Optional[str] = None

@app.post("/checkin")
def check_in(checkin_data: CheckIn, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    session = db.query(Session).filter(Session.course_id == checkin_data.course_id, Session.is_active == True).first()
    if not session:
        raise HTTPException(status_code=404, detail="No active session")

    rate_limit(token_data["user_id"])
    if not checkin_data.liveness:
        raise HTTPException(status_code=400, detail="Liveness not verified")
    if not consume_nonce(checkin_data.nonce):
        raise HTTPException(status_code=400, detail="Nonce missing/expired")

    student = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not student or not student.face_embedding:
        raise HTTPException(status_code=400, detail="No face profile found")

    stored_embedding = json.loads(student.face_embedding)
    if len(checkin_data.face_embedding) != len(stored_embedding):
        raise HTTPException(status_code=400,
            detail=f"Face embedding dimension mismatch: incoming={len(checkin_data.face_embedding)}, stored={len(stored_embedding)}")

    incoming = l2_normalize(checkin_data.face_embedding)
    stored_norm = l2_normalize(stored_embedding)
    similarity = cosine_similarity(incoming, stored_norm)

    if similarity < FACE_MATCH_THRESHOLD:
        raise HTTPException(status_code=401,
            detail=f"Face verification failed. Similarity: {similarity:.3f}, Required: {FACE_MATCH_THRESHOLD}")

    existing = db.query(Attendance).filter(
        Attendance.session_id == session.id, Attendance.student_id == student.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already checked in")

    elapsed = (datetime.utcnow() - session.start_time).total_seconds() / 60
    if elapsed <= session.late_after_minutes:
        status = "present"
    elif elapsed <= session.absent_after_minutes:
        status = "late"
    else:
        status = "absent"

    attendance = Attendance(session_id=session.id, student_id=student.id,
                            status=status, confidence=round(similarity, 4))
    db.add(attendance); db.commit()
    return {"status": status, "confidence": round(similarity, 4)}

@app.get("/attendance/session/{session_id}")
def get_session_attendance(session_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    records = db.query(Attendance).filter(Attendance.session_id == session_id).all()
    return [{
        "student_name": r.student.name,
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "confidence": r.confidence
    } for r in records]

@app.get("/attendance/history")
def get_attendance_history(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can view all attendance")
    professor_courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    course_ids = [c.id for c in professor_courses]
    records = db.query(Attendance).join(Session).filter(Session.course_id.in_(course_ids)).order_by(Attendance.timestamp.desc()).all()
    return [{
        "student_name": r.student.name,
        "student_code": r.student.student_code,
        "course_name": r.session.course.name,
        "course_id": r.session.course_id,
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "confidence": r.confidence
    } for r in records]

@app.get("/courses/enrolled")
def get_enrolled_courses(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return [{"id": c.id, "name": c.name, "code": c.code} for c in user.enrolled_courses]

@app.get("/attendance/my-history")
def get_my_attendance_history(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    records = db.query(Attendance).filter(Attendance.student_id == token_data["user_id"]).order_by(Attendance.timestamp.desc()).all()
    return [{
        "course_name": r.session.course.name,
        "course_id": r.session.course_id,
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "student_code": r.student.student_code,
        "confidence": r.confidence
    } for r in records]

@app.get("/admin/students")
def get_all_students(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    students = db.query(User).filter(User.role == "student").all()
    return [{"id": s.id, "name": s.name, "email": s.email} for s in students]

@app.delete("/admin/student/{student_id}/reset")
def reset_student_enrollment_and_face(student_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    if token_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    db.execute(enrollments.delete().where(enrollments.c.student_id == student_id))
    student.face_embedding = None
    db.commit()
    return {"message": f"Student '{student.name}' enrollments and face data cleared."}
