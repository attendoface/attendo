from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from jose import jwt, JWTError

# Load environment variables from .env file
load_dotenv()
import os
import secrets
import string
import json  # used in register/checkin
from collections import defaultdict

# JWT (PyJWT)
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

import pymysql

# ============= CONFIGURATION =============
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ============= DATABASE CONFIGURATION =============
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://user:password@localhost:3306/face_attendance?charset=utf8mb4"
)

# Face matching configuration
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.92"))  # stricter default
DUPLICATE_FACE_THRESHOLD = float(os.getenv("DUPLICATE_FACE_THRESHOLD", "0.90"))  # detect same face on signup

# --- Anti-replay nonce (simple in-memory; use Redis in production) ---
NONCE_TTL_SECONDS = int(os.getenv("NONCE_TTL_SECONDS", "60"))
_nonces: dict[str, float] = {}  # nonce -> expires_at (epoch seconds)

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

# --- Very small per-user rate limit (token bucket-ish) ---
_RATE_WINDOW = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
_RATE_LIMIT  = int(os.getenv("RATE_LIMIT_CHECKINS", "20"))  # X requests per window per user
_user_hits = defaultdict(list)  # user_id -> [timestamps]

def rate_limit(user_id: int):
    now = _now()
    hits = _user_hits[user_id]
    # drop old
    _user_hits[user_id] = [t for t in hits if now - t < _RATE_WINDOW]
    if len(_user_hits[user_id]) >= _RATE_LIMIT:
        raise HTTPException(429, detail="Too many check-ins. Please wait a minute.")
    _user_hits[user_id].append(now)

# ============= DATABASE SETUP =============
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize Cloud SQL Connector
connector = Connector()

def getconn() -> pymysql.connections.Connection:
    """Create database connection to Cloud SQL"""
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn

# Create SQLAlchemy engine using Cloud SQL connector
engine = create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ============= MODELS =============
# Association table for student enrollments
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
    role = Column(String(50), nullable=False)  # 'student', 'professor', 'admin'
    face_embedding = Column(Text, nullable=True)
    student_code = Column(String(20), unique=True, nullable=True)  # 👈 add this line
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
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
    
    # Relationships
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
    
    # Relationships
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("Attendance", back_populates="session")

class Attendance(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), nullable=False)  # 'present', 'late', 'absent'
    confidence = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# Create all tables
Base.metadata.create_all(bind=engine)

# ============= PYDANTIC SCHEMAS =============
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
    liveness: bool | None = None
    nonce: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    name: str
    # user_id is returned in responses (added for your frontend), but not required in schema

# ============= FASTAPI APP =============
app = FastAPI(title="Face Attendance API")

# CORS middleware (tighten ALLOWED_ORIGINS in prod)
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
# If wildcard, cannot set allow_credentials=True per browser rules
allow_creds = not (len(ALLOWED_ORIGINS) == 1 and ALLOWED_ORIGINS[0] == "*")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
    "https://attendoface.github.io"
    "https://attendo-backend.onrender.com"
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

# ============= HELPER FUNCTIONS =============
def generate_join_code(length=6):
    """Generate random alphanumeric join code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector to unit length"""
    import math
    norm = math.sqrt(sum(v * v for v in vec))
    if not norm:
        return vec[:]  # avoid division by zero
    return [v / norm for v in vec]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors (expects normalized vectors)"""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

# ============= ROUTES =============
@app.get("/")
def root():
    return {"message": "Face Attendance API", "status": "running"}

@app.get("/checkin/nonce")
def get_checkin_nonce(token_data: dict = Depends(verify_token)):
    """Issue a short-lived nonce to prevent replay of check-in payloads."""
    return {"nonce": issue_nonce(), "ttl": NONCE_TTL_SECONDS}

@app.post("/auth/register")
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user (prevents duplicate faces)"""
    # Check if email exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Guard: students must provide a face
    if user_data.role == "student" and (not user_data.face_embedding or len(user_data.face_embedding) == 0):
        raise HTTPException(status_code=400, detail="Face photo/embedding required for student accounts")

    # Hash password
    hashed = pwd_context.hash(user_data.password[:72])
    
    # Prepare face embedding (L2-normalized) and block duplicates
    face_str = None
    norm_emb = None
    if user_data.face_embedding:
        norm_emb = l2_normalize(user_data.face_embedding)

        # Duplicate-face check (compare against everyone with a stored face)
        existing_with_face = db.query(User).filter(User.face_embedding.isnot(None)).all()
        for u in existing_with_face:
            try:
                other = json.loads(u.face_embedding)
            except Exception:
                continue
            sim = cosine_similarity(norm_emb, l2_normalize(other))
            if sim >= DUPLICATE_FACE_THRESHOLD:
                raise HTTPException(
                    status_code=409,
                    detail=f"Face already registered to another account ({u.email})."
                )

        face_str = json.dumps(norm_emb)

    # === Generate Student Code ===
    student_code = None
    if user_data.role == "student":
        last_student = db.query(User).filter(User.role == "student").order_by(User.id.desc()).first()
        next_id = 1 if not last_student else last_student.id + 1
        student_code = f"STU{next_id:04d}"

    # === Create User ===
    user = User(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hashed,
        role=user_data.role,
        face_embedding=face_str,
        student_code=student_code  # 👈 attach it here
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # === Create Token ===
    token = create_access_token({"user_id": user.id, "role": user.role})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "user_id": user.id,
        "student_code": student_code  # 👈 optionally send to frontend
    }


@app.post("/auth/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not pwd_context.verify(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"user_id": user.id, "role": user.role})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "user_id": user.id
    }

@app.post("/courses")
def create_course(
    course_data: CourseCreate,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create new course (professors only)"""
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can create courses")
    
    # Generate unique join code
    while True:
        code = generate_join_code()
        existing = db.query(Course).filter(Course.code == code).first()
        if not existing:
            break
    
    course = Course(
        name=course_data.name,
        code=code,
        professor_id=token_data["user_id"]
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    
    return {"id": course.id, "name": course.name, "code": course.code}

@app.get("/courses/mine")
def get_my_courses(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get courses for current user"""
    if token_data.get("role") == "professor":
        courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    else:
        user = db.query(User).filter(User.id == token_data["user_id"]).first()
        courses = user.enrolled_courses
    
    return [{"id": c.id, "name": c.name, "code": c.code} for c in courses]

@app.post("/courses/enroll")
def enroll_course(
    enroll_data: CourseEnroll,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Enroll in course with join code"""
    course = db.query(Course).filter(Course.code == enroll_data.join_code.upper()).first()
    if not course:
        raise HTTPException(status_code=404, detail="Invalid join code")
    
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if course in user.enrolled_courses:
        raise HTTPException(status_code=400, detail="Already enrolled")
    
    user.enrolled_courses.append(course)
    db.commit()
    
    return {"course_id": course.id, "message": "Enrolled successfully"}

@app.post("/sessions/start")
def start_session(
    session_data: SessionStart,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Start attendance session"""
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can start sessions")

    # (Optional hardening) Ensure professor owns the course
    course = db.query(Course).filter(Course.id == session_data.course_id, Course.professor_id == token_data["user_id"]).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found or not owned by you")
    
    # End any active sessions for this course
    active = db.query(Session).filter(
        Session.course_id == session_data.course_id,
        Session.is_active == True
    ).all()
    for s in active:
        s.is_active = False
        s.end_time = datetime.utcnow()
    
    # Create new session
    session = Session(
        course_id=session_data.course_id,
        late_after_minutes=session_data.late_after_minutes,
        absent_after_minutes=session_data.absent_after_minutes
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {
        "id": session.id,
        "course_id": session.course_id,
        "start_time": session.start_time.isoformat(),
        "late_after_minutes": session.late_after_minutes,
        "absent_after_minutes": session.absent_after_minutes
    }

@app.post("/sessions/{session_id}/end")
def end_session(
    session_id: int,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """End attendance session"""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # (Optional hardening) Only professor who owns the course can end it
    course = db.query(Course).filter(Course.id == session.course_id).first()
    if not course or course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not allowed to end this session")

    session.is_active = False
    session.end_time = datetime.utcnow()
    db.commit()
    
    return {"message": "Session ended"} 

@app.get("/sessions/active")
def check_active_session(
    course_id: int,
    db: Session = Depends(get_db)
):
    """Check if course has active session"""
    session = db.query(Session).filter(
        Session.course_id == course_id,
        Session.is_active == True
    ).first()
    
    return {"active": session is not None}

@app.post("/checkin")
def check_in(
    checkin_data: CheckIn,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Student check-in with face recognition"""
    # Get active session
    session = db.query(Session).filter(
        Session.course_id == checkin_data.course_id,
        Session.is_active == True
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="No active session")

    # Anti-abuse gates (match your HTML flow)
    rate_limit(token_data["user_id"])
    if not checkin_data.liveness:
        raise HTTPException(status_code=400, detail="Liveness not verified")
    if not consume_nonce(checkin_data.nonce):
        raise HTTPException(status_code=400, detail="Nonce missing/expired")
    
    # Get student
    student = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not student or not student.face_embedding:
        raise HTTPException(status_code=400, detail="No face profile found")

    # Load stored embedding
    stored_embedding = json.loads(student.face_embedding)
    
    # Dimension sanity check to prevent cross-model mistakes
    if len(checkin_data.face_embedding) != len(stored_embedding):
        raise HTTPException(
            status_code=400,
            detail=f"Face embedding dimension mismatch: incoming={len(checkin_data.face_embedding)}, stored={len(stored_embedding)}"
        )

    # Normalize both embeddings for stable cosine similarity
    incoming = l2_normalize(checkin_data.face_embedding)
    stored_norm = l2_normalize(stored_embedding)

    similarity = cosine_similarity(incoming, stored_norm)
    print(f"[checkin] user={student.id} sim={similarity:.4f}")  # debug logging

    if similarity < FACE_MATCH_THRESHOLD:
        raise HTTPException(
            status_code=401,
            detail=f"Face verification failed. Similarity: {similarity:.3f}, Required: {FACE_MATCH_THRESHOLD}"
        )

    # Check if already checked in
    existing = db.query(Attendance).filter(
        Attendance.session_id == session.id,
        Attendance.student_id == student.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already checked in")
    
    # Calculate status based on time
    elapsed = (datetime.utcnow() - session.start_time).total_seconds() / 60
    if elapsed <= session.late_after_minutes:
        status = "present"
    elif elapsed <= session.absent_after_minutes:
        status = "late"
    else:
        status = "absent"
    
    # Record attendance
    attendance = Attendance(
        session_id=session.id,
        student_id=student.id,
        status=status,
        confidence=round(similarity, 4)
    )
    db.add(attendance)
    db.commit()
    
    return {"status": status, "confidence": round(similarity, 4)}

@app.get("/attendance/session/{session_id}")
def get_session_attendance(
    session_id: int,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get attendance records for session"""
    # (Optional hardening) ensure requester has rights
    records = db.query(Attendance).filter(Attendance.session_id == session_id).all()
    
    return [{
        "student_name": r.student.name,
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "confidence": r.confidence
    } for r in records]

@app.get("/attendance/history")
def get_attendance_history(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get all attendance history (professors only)"""
    if token_data.get("role") != "professor":
        raise HTTPException(status_code=403, detail="Only professors can view all attendance")
    
    professor_courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    course_ids = [c.id for c in professor_courses]
    
    records = db.query(Attendance).join(Session).filter(
        Session.course_id.in_(course_ids)
    ).order_by(Attendance.timestamp.desc()).all()
    
    return [{
    "student_name": r.student.name,
    "student_code": r.student.student_code,  # 👈 add this
    "course_name": r.session.course.name,
    "course_id": r.session.course_id,
    "timestamp": r.timestamp.isoformat(),
    "status": r.status,
    "confidence": r.confidence
} for r in records]

@app.get("/courses/enrolled")
def get_enrolled_courses(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get courses student is enrolled in"""
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return [{"id": c.id, "name": c.name, "code": c.code} for c in user.enrolled_courses]

@app.get("/attendance/my-history")
def get_my_attendance_history(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get student's personal attendance history"""
    records = db.query(Attendance).filter(
        Attendance.student_id == token_data["user_id"]
    ).order_by(Attendance.timestamp.desc()).all()
    
    return [{
        "course_name": r.session.course.name,
        "course_id": r.session.course_id,
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "student_code": r.student.student_code,
        "confidence": r.confidence
    } for r in records]

@app.get("/admin/students")
def get_all_students(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Admin: Get all student accounts."""
    if token_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    students = db.query(User).filter(User.role == "student").all()
    return [{"id": s.id, "name": s.name, "email": s.email} for s in students]

@app.delete("/admin/student/{student_id}/reset")
def reset_student_enrollment_and_face(
    student_id: int,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Admin: Remove a student's enrollments and face embedding, keeping the account."""
    if token_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Fetch student
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Remove enrollments (many-to-many)
    db.execute(enrollments.delete().where(enrollments.c.student_id == student_id))

    # Remove face embedding if it exists
    student.face_embedding = None

    db.commit()

    return {"message": f"Student '{student.name}' enrollments and face data cleared."}

# Cleanup connector on shutdown
@app.on_event("shutdown")
def shutdown():
    connector.close()

