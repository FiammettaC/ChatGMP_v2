from flask import Flask, render_template, request, jsonify, url_for, session, send_file, redirect
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
#import speech_recognition as sr
import io
from pydub import AudioSegment 
#from pydub.exceptions import CouldntDecodeError 
import logging
import re
import uuid
#import whisper
import subprocess
import platform
import time
import google.generativeai as genai
from mistralai import Mistral

# Load API config - try to import, fallback to environment variables
try:
    from api_config import gemini_key, lechat_key, flask_key
except (ImportError, ModuleNotFoundError):
    # Fallback to environment variables only - no hardcoded defaults
    import os
    def gemini_key():
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        return key
    def lechat_key():
        key = os.getenv("LECHAT_API_KEY")
        if not key:
            raise RuntimeError("LECHAT_API_KEY environment variable is not set")
        return key
    def flask_key():
        key = os.getenv("FLASK_SECRET_KEY")
        if not key:
            raise RuntimeError("FLASK_SECRET_KEY environment variable is not set")
        return key
import warnings
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

#from kokoro import KPipeline
#from IPython.display import display, Audio
#import soundfile as sf
import torchvision
torchvision.disable_beta_transforms_warning()

app = Flask(__name__)
app.secret_key = flask_key()
app.config['DEBUG'] = True
app.config['TESTING'] = False

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatgmp_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Database User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_self_signed_cert():
    """Generate self-signed SSL certificate if it doesn't exist"""
    cert_dir = 'certs'
    cert_file = os.path.join(cert_dir, 'cert.pem')
    key_file = os.path.join(cert_dir, 'key.pem')
    
    # Check if certificates already exist
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file
    
    # Create certs directory if it doesn't exist
    os.makedirs(cert_dir, exist_ok=True)
    
    try:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"34.61.233.89"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate to file
        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Write private key to file
        with open(key_file, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        app.logger.info(f"Self-signed certificate generated: {cert_file}")
        return cert_file, key_file
    except Exception as e:
        app.logger.error(f"Failed to generate self-signed certificate: {str(e)}")
        return None, None

# Initialize database and create default users if they don't exist
def init_db():
    """Initialize database with user from environment variables"""
    with app.app_context():
        db.create_all()
        
        username = os.getenv("DEFAULT_USERNAME")
        password = os.getenv("DEFAULT_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "DEFAULT_USERNAME and DEFAULT_PASSWORD environment variables must be set"
            )
        
        app.logger.info(f"[INIT_DB] Attempting to create user: {username}")
        
        # Delete existing user if it exists (fresh start)
        existing = User.query.filter_by(username=username).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()
            app.logger.info(f"[INIT_DB] Deleted existing user: {username}")
        
        # Create new user
        try:
            user = User(username=username, password_hash=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(user)
            db.session.commit()
            app.logger.info(f"[INIT_DB] Successfully created user: {username}")
        except Exception as e:
            app.logger.error(f"[INIT_DB] Error creating user: {str(e)}")
            db.session.rollback()

log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

gemini_model_name = "gemini-2.5-flash"
gemini_available = False
try:
    genai.configure(api_key=gemini_key())
    gemini_client = genai.GenerativeModel(gemini_model_name)
    gemini_available = True
except Exception as _gemini_init_err:
    gemini_client = None
    logging.warning(f"Gemini initialization failed — Gemini will be unavailable: {_gemini_init_err}")

mistral_model_name = "mistral-small-latest"
mistral_available = False
try:
    mistral_client = Mistral(api_key=lechat_key())
    mistral_available = True
except Exception as _mistral_init_err:
    mistral_client = None
    logging.warning(f"Mistral initialization failed — Mistral will be unavailable: {_mistral_init_err}")

DOCUMENT_MAPPINGS = {
    'management': 'Management Review No 3 minutes',
    'qa': 'Quality Assurance procedure',
    'qc': 'Quality Control procedure',
    'calibration': 'Calibration Temp Test Sheet filled out',
    'cleaning': 'Cleaning check list Ferm tank filled out',
    'audit': 'Internal Audit Program procedure',
    'maintenance': 'Maintenance procedure',
    'logistics': 'Packaging and Storage procedure',
    'packaging': 'Packaging and Storage procedure',
    'storage': 'Packaging and Storage procedure',
    'training': 'Training Program for Operators procedure',
    'training_filled': 'Training Plan Ferm operator filled out',
    'CAPA': 'CAPA Report No 23',
    'deviation': 'CAPA Report No 23',
}

# Distinctive terms that must appear in the AI output for a document open to be considered legitimate.
# Keys with generic terms (e.g. 'storage') need a more specific required term from the actual filename.
DOCUMENT_REQUIRED_TERMS = {
    'management': ['management'],
    'qa': ['quality assurance'],
    'qc': ['quality control'],
    'calibration': ['calibration'],
    'cleaning': ['cleaning'],
    'audit': ['audit'],
    'maintenance': ['maintenance'],
    'logistics': ['packaging'],
    'packaging': ['packaging'],
    'storage': ['packaging'],  # actual doc is "Packaging and Storage procedure"; 'storage' alone is too generic
    'training': ['training program', 'training procedure'],
    'training_filled': ['training plan', 'training record', 'ferm operator', 'filled out', 'filled in', 'filledout', 'training example'],
    'CAPA': ['capa', 'corrective'],
    'deviation': ['capa', 'corrective'],
}

available_docs = ", ".join([d for d in DOCUMENT_MAPPINGS.values()])
MAX_HISTORY = 6
SYSTEM_PROMPT_TEMPLATE = (
    "You are a GMP assistant representing a Pharmaceutical company, Pharma A/S, in an external audit. "
    "You are an employee of Pharma A/S and, more specifically, during the audit you represent the "
    "Plant Manager, if the questions are related to management, the logistics manager, if the questions are related to logistics, "
    "the maintenance manager, the quality control manager or quality assurance manager. "
    "You are being audited by NovoPharma A/S. "
    "GOAL: Your primary task is to answer questions directly and provide documents immediately upon request. "
    "Remember that all the documents (eg training records) are not related to your training, "
    "but to the employees of Pharma A/S. "
    "CRITICAL: When BACKGROUND INFORMATION is provided, it always takes priority over your general knowledge. "
    "Never contradict or ignore the provided context in favour of general GMP or pharmaceutical industry knowledge. "
    "You may supplement with general knowledge only when the context is silent on a topic, but if the context addresses it, follow the context exactly. "
    "You have access to the following documents: {available_docs}."
    
    "CRITICAL RULES: "
    "0. NEVER use the phrase 'I am sorry but I cannot show you that document as it is not available in my system' in response to a factual or procedural question (e.g. 'How are change control procedures implemented?', 'What is your deviation process?', 'How do you handle training?'). That refusal phrase is ONLY for when the user explicitly says 'show me', 'open', 'see', 'can I see' a specific document. For ALL other questions, answer from the BACKGROUND INFORMATION and your knowledge. "
    "1. If the auditor (user) asks to 'see', 'open', 'show', or 'check' a document, you MUST immediately confirm you are doing so,"
    "but only if the document the user is asking to see, is among the {available_docs} "
    "If you open a document, please add explanations of the documents in the response."
    "2. NEVER ask for permission or ask if the user wants to see it. Just state: 'I am opening the [document name] for you now.' and then add a description on the document and what it shows "
    "3. Do not use formatting like bolding or bullet points or **some text** with asterisks. "
    "4. CRITICAL: You can ONLY say 'I am opening' for documents whose topic or name closely matches a document in this list: {available_docs}. Use semantic matching — a user may say 'quality assurance document' to refer to 'Quality Assurance procedure'. If the user's request does not semantically match any document in this list, you MUST NOT say you are opening it. Instead say: 'I am sorry but I cannot show you that document as it is not available in my system.'"
    "Only show or suggest to show one document at a time, unless two documents are explicitly mentioned in the same question/request by the user. "
    "If the user asks to see multiple documents that you have, please open the most relevant one based on the question, or the one that is mentioned first in the question. "
    "If you cannot determine which one is more relevant, open the one that is mentioned first in the question."
    #"5. The only files that you can open are the following: {available_docs}"
    "5. If the user asks to see a 'batch record' or a 'certificate of analysis', please adhere to the responses suggested by the CONTEXT provided, e.g.'Sorry, I cannot show you a batch record, because it is proprietary information.'"
    "If the user asks to see a document that is not in the {available_docs} and is not a 'batch record' or a 'certificate of analysis', you should say: 'I am sorry but I cannot show you that document as it is not available in my system.'"
    "if the user asks to see 'this procedure' or 'this document' or 'it' or 'that' or 'yes' or 'sure' or 'ok' or 'okay' or 'please' or 'go ahead' or 'yep' or 'yeah' or 'absolutely do', you should determine which document they are referring to based on the conversation history and the context of the question, and if you can determine it, open it. "
    "6. Don't start the answers with 'At Pharma A/S' too many times."
    "We do NOT have a CIP (cleaning-in-place) procedure. ONLY if the user EXPLICITLY asks about CIP, cleaning-in-place, or a cleaning procedure, you MUST say: 'We do not have a CIP procedure, but I can show you our cleaning checklist.' Do NOT use this response for any other question. Do NOT describe or invent a CIP procedure. "
    "If the user asks for a deviation record, you can show the CAPA report"
    "Do not suggest for documents you do not have in your system to be provided after the audit. "
    "CRITICAL: You must NEVER ask the user if they want to see, or offer to show, any document that is NOT in the {available_docs} list. This includes documents like 'sampling procedure', 'LIMS', 'certificate of analysis', 'batch record', or any other document not explicitly listed in {available_docs}. If you cannot show a document, simply do not mention it."
    "CRITICAL: You MUST NEVER invent or mention the name of any procedure or document that is not in the {available_docs} list. "
    "For example, if the user asks about shipping, you MUST NOT say you have a 'Shipping and Transport Procedure' or any other invented document name. "
    "Instead, answer the topic as best you can based on your knowledge, WITHOUT naming or implying any specific undisclosed document. "
    "Do NOT say 'we have a procedure for that but I cannot show it' — this implies the document exists with that name, which may be false. "
    "IMPORTANT: Users often say 'document' when referring to what is listed as a 'procedure', and vice versa. Treat 'document' and 'procedure' as interchangeable words when matching user requests to {available_docs}. For example, 'quality assurance document' matches 'Quality Assurance procedure', and 'maintenance document' matches 'Maintenance procedure'. Use semantic/topic matching, not literal string matching, when identifying which document the user is asking about. "
    "IMPORTANT: If the user asks 'do you have a [document/procedure] for X?' or 'do you have a X document/procedure?', and X semantically matches a document in {available_docs}, you MUST confirm you have it AND immediately open it. Say: 'Yes, I have our [exact document name from {available_docs}]. I am opening it for you now.' Do NOT say it is unavailable or refuse to show it. "
    "CRITICAL: If the user asks 'do you have a procedure for X' or 'do you have a document for X', and X does not semantically match any document topic in the {available_docs} list, you MUST say you do NOT have a specific procedure for X. Never say 'Yes, we have a procedure for X' or imply that a procedure exists unless its topic is covered by a document in {available_docs}. "
    "Only ever name documents that are in the {available_docs} list."
    "CRITICAL: If the user asks for a list, index, or overview of all documents, procedures, or files available in your system, you MUST NOT provide any such list. Respond with exactly: 'I cannot provide the full list of documents. Please request a specific document, and I can provide it if allowed.' "
    "If the user asks about 'preventive maintenance', you can refer to the 'maintenance procedure'"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Suppress verbose sentence-transformers logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.models").setLevel(logging.ERROR)

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
try:
    import torch.cuda
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"CUDA error: {e}")

if platform.system() == "Windows":
    ######these lines are for Windows
    FFMPEG_BIN_DIR = 'C:/Tools/ffmpeg-2025-06-16-git-e6fb8f373e-full_build/bin'
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
    AudioSegment.converter = "C:/Tools/ffmpeg-2025-06-16-git-e6fb8f373e-full_build/bin/ffmpeg.exe"  # Update this path to your FFmpeg installation
    AudioSegment.ffprobe   = "C:/Tools/ffmpeg-2025-06-16-git-e6fb8f373e-full_build/bin/ffprobe.exe"
else:
    ######these lines are for Linux/Unix
    # For Docker containers, ffmpeg is installed via apt-get in /usr/bin
    AudioSegment.converter = "/usr/bin/ffmpeg"
    AudioSegment.ffprobe   = "/usr/bin/ffprobe"

#whisper_model = whisper.load_model("medium")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

# Initialize model variables as None for lazy loading
model = None
processor = None
pipe = None
model_search = None
knowledge_base = None
kb_questions = None
kb_answers = None
kb_embeddings = None

def load_whisper_model():
    """Lazy load Whisper model on first use"""
    global model, processor, pipe
    
    if pipe is not None:
        return pipe
    
    use_device = device
    use_dtype = torch_dtype

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=use_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(use_device)
        processor = AutoProcessor.from_pretrained(model_id)
        app.logger.info(f"Whisper model loaded successfully on {use_device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            app.logger.warning(f"CUDA OOM — falling back to CPU for Whisper: {e}")
            torch.cuda.empty_cache()
            use_device = torch.device("cpu")
            use_dtype = torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=use_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_id)
            app.logger.info("Whisper model loaded on CPU (fallback)")
        else:
            app.logger.error(f"Failed to load Whisper model: {str(e)}")
            pipe = None
            return pipe

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=use_dtype,
            device=use_device,
            chunk_length_s=30,
        )
        app.logger.info("ASR pipeline initialized successfully.")
    except Exception as e:
        app.logger.error(f"Failed to init ASR pipeline: {str(e)}")
        pipe = None
    
    return pipe

def load_search_model():
    """Lazy load SentenceTransformer model on first use"""
    global model_search, knowledge_base, kb_questions, kb_answers, kb_embeddings
    
    if model_search is not None:
        return model_search
    
    try:
        model_search = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        with open('data/data_gmp_03042025.json', encoding='utf-8-sig') as f:
            knowledge_base = json.load(f)
        
        kb_questions = list(knowledge_base.keys())
        kb_answers = list(knowledge_base.values())
        kb_embeddings = model_search.encode(kb_questions, convert_to_tensor=True)
        app.logger.info("Search model loaded successfully")
    except Exception as e:
        app.logger.error(f"Failed to load search model: {str(e)}")
        model_search = None
    
    return model_search

def semantic_search(asked_question):
    model = load_search_model()
    if model is None or kb_embeddings is None:
        return "We do not have a relevant procedure for this at the moment.", 0.0
    
    question_embedding = model.encode(asked_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_embedding, kb_embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = float(cosine_scores[best_match_idx])
    if best_score < 0.4:
        return "We do not have a relevant procedure for this at the moment.", best_score
    app.logger.info(f"Semantic search best match score: {best_score:.3f} — {kb_questions[best_match_idx][:80]}")
    return kb_answers[best_match_idx], best_score

def save_file(question, answer, dest_path):
    data = {}
    data[question] = answer
    with open(dest_path, 'a') as f:
        f.write(json.dumps([data]) + '\n')

def show_document(doc_name):
    """Log document request - in web app, documents are served via Flask routes"""
    app.logger.info(f"Document requested: {doc_name}")
    return None

def get_requested_document(user_question, ai_output, current_context, history):
    show_triggers = ['show', 'open', 'display', 'see', 'view', 'check', 'look at']
    ai_opening_phrases = ['i am opening', "i'm opening", 'opening the', 'here is the', 'here are the', 'i will open', 'i will show']
    refusal_phrases = ['cannot show', 'can\'t show', 'not available in my system', 'i am sorry', 'i\'m sorry', 'not able to show', 'unable to show', 'do not have']
    user_wants_to_see = any(trigger in user_question.lower() for trigger in show_triggers)
    ai_is_opening = any(phrase in ai_output.lower() for phrase in ai_opening_phrases)
    ai_refused = any(phrase in ai_output.lower() for phrase in refusal_phrases)

    app.logger.info(f"Checking for document request. User wants to see: {user_wants_to_see}, AI opening: {ai_is_opening}, AI refused: {ai_refused}")
    app.logger.info(f"User question: {user_question}")

    # Priority 0: user is confirming a document explicitly offered in the previous AI turn.
    # Only fires for short pure affirmatives (yes/ok/sure/please/go ahead etc.).
    # Explicit reference phrases like "show me this procedure" must go through normal
    # reference resolution so the correct document from conversation context is used.
    pending_key = session.get('pending_offered_doc_key')
    reference_words_p0 = ['this', 'it', 'that', 'the procedure', 'the document', 'a procedure', 'a document']
    user_is_referencing_p0 = any(
        re.search(r'\b' + re.escape(ref) + r'\b', user_question.lower()) for ref in reference_words_p0
    )
    if pending_key and pending_key in DOCUMENT_MAPPINGS and not user_is_referencing_p0:
        affirmatives = ['yes', 'sure', 'ok', 'okay', 'please', 'go ahead', 'yep', 'yeah', 'absolutely', 'please do']
        user_is_affirming = (
            len(user_question.split()) <= 4 and any(w in user_question.lower() for w in affirmatives)
        )
        if user_is_affirming:
            app.logger.info(f"Priority 0: user confirmed pending offered doc '{pending_key}'")
            show_document(DOCUMENT_MAPPINGS[pending_key])
            return [pending_key]

    # Implicit document requests: "do you have X records/documents?" in an audit context
    # is equivalent to asking to see them — check before the early-exit guard below.
    implicit_triggers = {
        'calibration': ['calibration record', 'calibration certificate', 'calibration data', 'calibrated'],
        'training_filled': ['training record', 'training filled', 'training plan filled',
                            'training plan filledout', 'training plan filled in', 'training example filled',
                            'training template filled', 'training document filled'],
        'CAPA': ['capa record', 'deviation record', 'deviation report'],
        'maintenance': ['maintenance record', 'maintenance log'],
        'cleaning': ['cleaning record', 'cleaning log'],
    }
    for doc_key, phrases in implicit_triggers.items():
        if any(p in user_question.lower() for p in phrases):
            app.logger.info(f"Implicit document trigger matched '{doc_key}'")
            show_document(DOCUMENT_MAPPINGS[doc_key])
            return [doc_key]

    # Only act when there is a real signal: user asked to see a doc or AI is opening one.
    # This prevents spurious matches on background topic words in informational questions.
    if not (user_wants_to_see or ai_is_opening):
        app.logger.info("No document signal — neither show trigger nor AI opening phrase")
        return []

    # Reference resolution: short follow-ups like "can I see it", "show me that" (≤10 words)
    reference_words = ['this', 'it', 'that', 'the procedure', 'the document']
    user_is_referencing = (
        len(user_question.split()) <= 10
        and any(re.search(r'\b' + re.escape(ref) + r'\b', user_question.lower()) for ref in reference_words)
    )
    if user_is_referencing:
        # First: use the explicitly tracked last-discussed document topic (most reliable)
        last_topic_key = session.get('last_topic_doc_key')
        if last_topic_key and last_topic_key in DOCUMENT_MAPPINGS:
            app.logger.info(f"Reference resolved via last_topic_doc_key='{last_topic_key}'")
            session.pop('last_topic_doc_key', None)
            show_document(DOCUMENT_MAPPINGS[last_topic_key])
            return [last_topic_key]
        # Fallback: scan backwards through history user messages for a document key mention.
        # This is reliable since history is passed directly (no session dependency).
        user_msgs = [msg.get('content', '') for msg in reversed(history) if msg.get('role') == 'user']
        for msg_content in user_msgs:
            msg_lower = msg_content.lower()
            for key in DOCUMENT_MAPPINGS:
                if re.search(r'\b' + re.escape(key.lower()) + r'\b', msg_lower):
                    app.logger.info(f"Reference resolved by backward history scan: '{key}' in '{msg_content[:60]}'")
                    show_document(DOCUMENT_MAPPINGS[key])
                    return [key]
        # Last resort: keyword frequency score across recent AI + user messages
        if len(history) >= 3:
            prev_ai_msg = history[-3].get('content', '').lower()
            prev_user_msg = history[-4].get('content', '').lower() if len(history) >= 4 else ''
            combined_context = prev_ai_msg + ' ' + prev_user_msg
            best_key, best_score = None, 0
            for key, value in DOCUMENT_MAPPINGS.items():
                key_count = len(re.findall(r'\b' + re.escape(key.lower()) + r'\b', combined_context))
                val_count = combined_context.count(value.lower())
                score = key_count + val_count
                if score > best_score:
                    best_score, best_key = score, key
            if best_key and best_score > 0:
                app.logger.info(f"Reference resolved by frequency scoring (score={best_score}): {best_key}")
                show_document(DOCUMENT_MAPPINGS[best_key])
                return [best_key]

    # Pass 1: keyword match in user question — strongest signal, user directly named the doc.
    # Only runs when user actually asked to see something (prevents topic-word false positives).
    if user_wants_to_see:
        for key, value in DOCUMENT_MAPPINGS.items():
            if key.lower() in user_question.lower():
                app.logger.info(f"Pass 1: user question contains doc key '{key}'")
                show_document(value)
                return [key]

    # Pass 2: fuzzy-match the document name from the AI's "I am opening …" phrase.
    # Uses significant-word overlap (≥60%) so variants like "Packaging & Storage procedure
    # PAS-LOG-001.EN" still match "Packaging and Storage procedure".
    if ai_is_opening:
        opening_match = re.search(
            r"(?:i am opening|i'm opening|opening the|here is the|here are the|i will open|i will show)"
            r"\s+(.+?)(?:\s+for you(?:\s+now)?|\s+now[,.]|[.!?\n]|$)",
            ai_output, re.IGNORECASE
        )
        if opening_match:
            claimed_name = opening_match.group(1).lower().strip()
            _STOP = {'a', 'an', 'and', 'for', 'the', 'of', 'to', 'in', 'is', 'it', 'no', 'or'}
            best_key, best_score = None, 0.0
            for key, value in DOCUMENT_MAPPINGS.items():
                sig = [w for w in re.findall(r'\b[a-z]+\b', value.lower()) if w not in _STOP and len(w) > 2]
                if not sig:
                    continue
                score = sum(1 for w in sig if w in claimed_name) / len(sig)
                if score > best_score:
                    best_score, best_key = score, key
            if best_key and best_score >= 0.6:
                app.logger.info(f"Pass 2 fuzzy match: '{claimed_name}' → '{best_key}' (score={best_score:.2f})")
                show_document(DOCUMENT_MAPPINGS[best_key])
                return [best_key]
            app.logger.info(f"Pass 2: no confident match for '{claimed_name}' (best score={best_score:.2f})")

    # Pass 3: AI refused AND user wanted to see a document — find an alternative by keywords.
    # Kept separate from the main flow so refusals on informational questions are not mishandled.
    if ai_refused and user_wants_to_see:
        for key, value in DOCUMENT_MAPPINGS.items():
            if key.lower() in user_question.lower():
                app.logger.info(f"Pass 3 refusal fallback: user question maps to '{key}'")
                show_document(value)
                return [key]
        for key, value in DOCUMENT_MAPPINGS.items():
            if value.lower() in ai_output.lower():
                app.logger.info(f"Pass 3 refusal fallback: AI output mentions '{value}'")
                show_document(value)
                return [key]

    app.logger.info("No document found")
    return []

def update_docs_list(doc_name, doc, dest_path):
    docs = {}
    docs[doc_name] = doc
    with open(dest_path, 'a') as f:
        f.write(json.dumps([docs]) + '\n')

def predict(question):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(available_docs=available_docs)
    # Load history WITHOUT the system prompt (which is ~3KB and causes session cookie overflow).
    # Always prepend a fresh system prompt each request.
    stored_history = session.get('chat_history', [])
    history = [{'role': 'system', 'content': system_prompt}] + stored_history
    if len(history) > MAX_HISTORY:
        history = [history[0]] + history[-MAX_HISTORY:]

    # If the user is referencing something from the previous turn, enrich the search query.
    # Only apply for short follow-up questions (≤10 words); longer questions contain
    # their own topic context and should not fall back to history.
    reference_words = ['this', 'it', 'that', 'the procedure', 'the document', 'a procedure', 'a document']
    user_is_referencing = (
        len(question.split()) <= 10
        and any(
            re.search(r'\b' + re.escape(ref) + r'\b', question.lower())
            for ref in reference_words
        )
    )
    search_query = question
    if user_is_referencing and len(history) >= 2:
        last_ai_msg = history[-1].get('content', '').lower()
        prev_user_msg = history[-2].get('content', '').lower() if len(history) >= 2 else ''
        combined_context = last_ai_msg + ' ' + prev_user_msg
        # Pick the most frequently mentioned document key
        best_key = None
        best_score = 0
        for key, value in DOCUMENT_MAPPINGS.items():
            score = combined_context.count(key.lower()) + combined_context.count(value.lower())
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score > 0:
            search_query = f"{question} {DOCUMENT_MAPPINGS[best_key]}"
            app.logger.info(f"Reference detected, enriched search query with '{best_key}' (score={best_score}): {search_query}")

    context, confidence = semantic_search(search_query)

    # High confidence (≥0.75): the KB has a direct answer — inject it as a hard fact.
    # The model may rephrase it more professionally but must not contradict it.
    # Lower confidence: treat as background context that informs the answer.
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        app.logger.info(f"High-confidence match ({confidence:.3f}) — injecting as direct fact")
        enriched_input = (
            f"DIRECT ANSWER FROM COMPANY DATA (confidence: {confidence:.2f}): {context}\n"
            f"INSTRUCTION: The above is the factual answer based on Pharma A/S company data. "
            f"You MUST use this as the basis of your response and must NOT contradict it, even if it differs from general industry practice. "
            f"You may rephrase it in a more professional and natural tone, and add brief relevant context if helpful, "
            f"but the core facts must remain exactly as stated above. "
            f"Do NOT speak from the perspective of the auditor.\n\n"
            f"USER QUESTION: {question}"
        )
    else:
        enriched_input = (
            f"BACKGROUND INFORMATION (this takes priority over any general knowledge — if it addresses the topic, follow it exactly; "
            f"you may use general knowledge only to supplement topics the context does not cover; "
            f"do NOT speak from the perspective of the auditor): {context}\n\n"
            f"USER QUESTION: {question}"
        )
    provider = session.get('provider', 'mistral')
    max_retries = 3
    retry_delay = 5  # seconds between retries on 429

    def _is_model_unavailable(err):
        msg = str(err).lower()
        return any(k in msg for k in ['no longer available', 'not found', 'deprecated', '404', 'model_not_found', 'does not exist'])

    def _is_rate_limited(err):
        return '429' in str(err)

    output = None
    if provider == 'mistral':
        if not mistral_available:
            return "Mistral is currently unavailable (API key missing or invalid). Please switch to Gemini using the AI selector below.", None
        mistral_messages = list(history) + [{"role": "user", "content": enriched_input}]
        for attempt in range(max_retries):
            try:
                response = mistral_client.chat.complete(
                    model=mistral_model_name,
                    messages=mistral_messages,
                    temperature=0.5,
                    max_tokens=1024
                )
                output = response.choices[0].message.content
                break
            except Exception as api_err:
                if _is_model_unavailable(api_err):
                    app.logger.error(f"Mistral model unavailable: {api_err}")
                    return "The Mistral AI model is temporarily unavailable. Please select Gemini using the AI selector below.", None
                if _is_rate_limited(api_err):
                    if attempt < max_retries - 1:
                        wait = retry_delay * (attempt + 1)
                        app.logger.warning(f"Mistral 429 rate limit on attempt {attempt + 1}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        app.logger.error(f"Mistral rate limit exceeded after {max_retries} attempts")
                        return "Mistral is currently rate-limited and could not respond. Please try again in a moment, or switch to Gemini using the AI selector below.", None
                else:
                    raise
    else:
        if not gemini_available:
            return "Gemini is currently unavailable (API key missing or invalid). Please switch to Mistral using the AI selector below.", None
        system_content = ""
        gemini_history = []
        for msg in history:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [msg["content"]]})
        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            system_instruction=system_content,
            generation_config=genai.types.GenerationConfig(temperature=0.5, max_output_tokens=1024)
        )
        for attempt in range(max_retries):
            try:
                chat_session = gemini_model.start_chat(history=gemini_history)
                chat_response = chat_session.send_message(enriched_input)
                output = chat_response.text
                break
            except Exception as api_err:
                if _is_model_unavailable(api_err):
                    app.logger.error(f"Gemini model unavailable: {api_err}")
                    return "The Gemini AI model is temporarily unavailable. Please select Mistral using the AI selector below.", None
                if _is_rate_limited(api_err):
                    if attempt < max_retries - 1:
                        wait = retry_delay * (attempt + 1)
                        app.logger.warning(f"Gemini 429 rate limit on attempt {attempt + 1}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        app.logger.error(f"Gemini rate limit exceeded after {max_retries} attempts")
                        return "Gemini is currently rate-limited and could not respond. Please try again in a moment, or switch to Mistral using the AI selector below.", None
                else:
                    raise

    # Detect if AI offered to show a document, so the next user affirmation can open it directly.
    offer_match = re.search(
        r"(?:i can show you|but i can show you|would you like to see|can show you our|show you our|show you the)"
        r"\s+(?:our\s+|the\s+|an?\s+)?(.+?)(?:[,.]|$)",
        output, re.IGNORECASE
    )
    if offer_match:
        offered_text = offer_match.group(1).lower().strip()
        _STOP = {'a', 'an', 'and', 'for', 'the', 'of', 'to', 'in', 'is', 'it', 'no', 'or'}
        best_offer_key, best_offer_score = None, 0.0
        for key, value in DOCUMENT_MAPPINGS.items():
            sig = [w for w in re.findall(r'\b[a-z]+\b', value.lower()) if w not in _STOP and len(w) > 2]
            if not sig:
                continue
            score = sum(1 for w in sig if w in offered_text) / len(sig)
            if score > best_offer_score:
                best_offer_score, best_offer_key = score, key
        if best_offer_key and best_offer_score >= 0.3:
            session['pending_offered_doc_key'] = best_offer_key
            app.logger.info(f"AI offered doc: stored pending key '{best_offer_key}' (score={best_offer_score:.2f})")

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": output})

    session['chat_history'] = history

    # Track which document topic was just discussed (for follow-up "show me this" requests).
    # Only update when the user was NOT already asking to see something — if they were,
    # the doc resolution logic below handles it. Use word-boundary matching on the question.
    show_triggers_local = ['show', 'open', 'display', 'see', 'view', 'check', 'look at']
    if not any(t in question.lower() for t in show_triggers_local):
        for key in DOCUMENT_MAPPINGS:
            if re.search(r'\b' + re.escape(key.lower()) + r'\b', question.lower()):
                session['last_topic_doc_key'] = key
                app.logger.info(f"Stored last_topic_doc_key='{key}' from user question")
                break

    # Save history WITHOUT the system prompt to keep session cookie small.
    session['chat_history'] = [msg for msg in history if msg.get('role') != 'system']

    docs = get_requested_document(question, output, context, history)
    print(docs)
    # Safeguard: validate and correct AI claims about opening documents
    show_triggers = ['show', 'open', 'display', 'see', 'view', 'check', 'look at']
    user_wants_to_see = any(trigger in question.lower() for trigger in show_triggers)
    ai_opening_phrases = ['i am opening', "i'm opening", 'opening the']
    ai_refused_phrases = ['cannot show', 'not available in my system', 'i am sorry', "i'm sorry"]
    ai_claimed_to_open = any(phrase in output.lower() for phrase in ai_opening_phrases)
    ai_refused = any(phrase in output.lower() for phrase in ai_refused_phrases)

    # If the AI wrongly fired the document-refusal phrase on a factual/procedural question
    # (user never asked to see, open, or show a document), replace with the KB answer directly.
    wrongful_refusal_phrases = ['not available in my system', 'cannot show you that document']
    if (not user_wants_to_see
            and not ai_claimed_to_open
            and any(phrase in output.lower() for phrase in wrongful_refusal_phrases)
            and confidence >= 0.4):
        app.logger.warning("AI wrongly fired document-refusal on factual question — replacing with KB answer")
        output = context
        ai_refused = False

    # Strips from "I am opening" to end of response (used when no doc can be shown)
    opening_tail_re = re.compile(r'\s*(?:i am opening|i\'m opening|opening the)\b.*$', re.IGNORECASE | re.DOTALL)
    # Replaces just the one opening sentence (leaves follow-up sentences intact).
    # The inner group allows a literal '.' only when NOT followed by whitespace or end-of-string,
    # so file-extension suffixes like ".EN" are consumed as part of the sentence rather than
    # being treated as the sentence-terminating period (which caused "for you now.EN for you now").
    opening_sentence_re = re.compile(r'(?:i am opening|i\'m opening|opening the)\b(?:[^.!?]|\.(?!\s|$))*[.!?]?', re.IGNORECASE)
    if ai_claimed_to_open and len(docs) == 0:
        if user_wants_to_see:
            # User asked for a doc we don't have — apologise
            app.logger.warning("AI claimed to open a non-existent document — replacing with apology")
            output = opening_tail_re.sub(
                ' I am sorry but I cannot show you that document as it is not available in my system.',
                output
            ).strip()
        else:
            # AI proactively offered to open a doc that doesn't exist — strip silently
            app.logger.warning("AI proactively claimed to open a non-existent document — stripping silently")
            stripped = opening_tail_re.sub('', output).strip()
            if stripped:
                output = stripped
            else:
                # The entire AI response was just the opening phrase — use the KB context instead
                app.logger.warning("Silent strip produced empty output — falling back to context answer")
                output = context
    elif ai_claimed_to_open and len(docs) > 0:
        if not user_wants_to_see:
            # User never asked to see a document — AI opened one unprompted.
            # Strip the opening silently and suppress the document.
            app.logger.warning("AI proactively opened a document the user did not request — stripping and suppressing")
            stripped = opening_tail_re.sub('', output).strip()
            if stripped:
                output = stripped
            else:
                output = "I am sorry, but I cannot show you that document as it is not available in my system."
            docs = []
        else:
            # Always normalise the opening sentence to the canonical document name.
            # This corrects fabricated variants like "Packaging & Storage procedure PAS-LOG-001.EN".
            actual_doc = DOCUMENT_MAPPINGS.get(docs[0], '')
            app.logger.info(f"Normalising opening sentence to canonical doc name: '{actual_doc}'")
            output = opening_sentence_re.sub(f'I am opening the {actual_doc} for you now.', output, count=1)
    elif ai_refused and len(docs) > 0:
        actual_doc = DOCUMENT_MAPPINGS.get(docs[0], '')
        app.logger.info(f"AI refused but alternative found: '{actual_doc}' — correcting response")
        output = f"We do not have that specific document, but I am opening the {actual_doc} for you now."
    try:
        from zoneinfo import ZoneInfo
        now_cph = datetime.datetime.now(tz=ZoneInfo("Europe/Copenhagen"))
    except Exception:
        now_cph = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=1)))
    today = now_cph.strftime("%Y-%m-%d")
    if 'session_id' not in session:
        ts = now_cph.strftime("%H%M%S")
        session['session_id'] = f"{current_user.username}_{today}_{ts}_{uuid.uuid4().hex[:6]}"
    output_path = f"data/audits/audit_{session['session_id']}.jsonl"
    os.makedirs("data/audits", exist_ok=True)
    save_file(question, output, output_path)
 
    document_to_show = None
    if len(docs) > 0:
        doc_name = docs[0]
        session.pop('pending_offered_doc_key', None)
        update_docs_list(doc_name, "shown", output_path)
        app.logger.info(f"Docs returned: {docs}")
        # Get the actual document filename from DOCUMENT_MAPPINGS
        for key, value in DOCUMENT_MAPPINGS.items():
            if key == doc_name:
                document_to_show = value
                app.logger.info(f"Document to show: {document_to_show}")
                break
    
    app.logger.info(f"Returning document_to_show: {document_to_show}")
    return output, document_to_show

#def text_to_speech(response):
#    pipeline = KPipeline(lang_code='a')
#    generator = pipeline(response, voice='af_heart')
#    for i, (gs, ps, audio) in enumerate(generator):
#        display(Audio(data=audio, rate=24000, autoplay=i==0))

@app.route("/")
@login_required
def home():    
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            session.pop('session_id', None)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/set_provider', methods=['POST'])
@login_required
def set_provider():
    data = request.get_json()
    provider = data.get('provider', 'gemini') if data else 'gemini'
    if provider not in ('gemini', 'mistral'):
        return jsonify({'error': 'Invalid provider'}), 400
    session['provider'] = provider
    return jsonify({'provider': provider})

@app.route('/health')
def health():
    """Health check endpoint for Cloud Run - no authentication required"""
    return jsonify({'status': 'ok'}), 200

@app.route('/transcribe_audio', methods=['POST'])
@login_required
def transcribe_audio():
    """
    Receives audio, preprocesses it, and transcribes using Whisper with optimizations.
    Returns JSON with transcription and chatbot response.
    """
    asr_pipe = load_whisper_model()
    if not asr_pipe: # Check if the pipeline was successfully initialized
        return jsonify({'error': 'ASR service not available. Model failed to load or pipeline failed to initialize.'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = audio_file.filename or 'recorded_audio.webm'
        fmt = 'mp4' if filename.endswith('.mp4') else 'webm'
        webm_bytes = io.BytesIO(audio_file.read())
        audio_segment = AudioSegment.from_file(webm_bytes, format=fmt)

        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
            app.logger.info(f"Resampled audio to 16kHz. Original: {audio_segment.frame_rate}")

        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
            app.logger.info("Converted audio to mono.")

        audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        audio_data /= 32768.0

        result = asr_pipe(audio_data) 
        print(f"Transcription result: {result['text']}")

        chatbot_answer, document = predict(result["text"])

        response_data = {
            'transcribed_text': result["text"],
            'chatbot_answer': chatbot_answer
        }
        if document:
            response_data['document'] = document
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route("/get")
@login_required
def get_bot_response():    
    userText = request.args.get('msg')  
    user_text_lower = userText.lower()
    try:
        if "show video describing if you prioritize training or audit procedure" in user_text_lower or "play video" in user_text_lower:
            # Replace with actual video URL
            # Option 1: Direct video file (e.g., hosted in your 'static' folder)
            video_url = url_for('static', filename='/videos/training_audit.mp4')
            # For testing, you can use a public test video:
            #video_url = "https://www.w3schools.com/html/mov_bbb.mp4" # Example .mp4
            return jsonify({"type": "video", "url": video_url})
        else:
            response, document = predict(userText)
            if response is None:
                return jsonify({"type": "text", "answer": "Sorry, I could not generate a response."}), 200
            result = {"type": "text", "answer": response}
            if document:
                result["document"] = document
            return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in /get: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/document/<document_name>')
@login_required
def get_document(document_name):
    """Serve PDF documents from the static/documents directory"""
    try:
        # Sanitize filename to prevent directory traversal attacks
        if '..' in document_name or '/' in document_name or '\\' in document_name:
            return jsonify({"error": "Invalid document name"}), 400
        
        doc_path = os.path.join('static', 'documents', f'{document_name}.pdf')
        
        if not os.path.exists(doc_path):
            app.logger.warning(f"Document not found: {doc_path}")
            return jsonify({"error": "Document not found"}), 404
        
        return send_file(doc_path, as_attachment=False, mimetype='application/pdf')
    except Exception as e:
        app.logger.error(f"Error retrieving document {document_name}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error retrieving document: {str(e)}"}), 500

if __name__ == '__main__':
    init_db()
    
    # Generate SSL certificates
    cert_file, key_file = generate_self_signed_cert()
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    # Run with SSL if certificates are available
    if cert_file and key_file:
        app.logger.info("Running with HTTPS (SSL/TLS enabled)")
        app.run(host='0.0.0.0', port=port, debug=debug_mode, ssl_context=(cert_file, key_file))
    else:
        app.logger.warning("Running without HTTPS - certificates not available")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    
#python -m flask --app main run
