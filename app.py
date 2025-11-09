import streamlit as st
import speech_recognition as sr
import language_tool_python
import os
import tempfile
from audio_recorder_streamlit import audio_recorder
import json
import psycopg2
from dotenv import load_dotenv
import google.generativeai as genai
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

st.set_page_config(page_title="Grammar Evaluator", page_icon="🎤", layout="wide")

# Header with word meaning finder
col1, col2 = st.columns([5, 1])
with col1:
    st.title("🎤 Voice-Based Grammar Evaluator")
    st.markdown("Evaluate your English grammar from voice inputs with AI-powered feedback")
with col2:
    st.markdown("")
    if st.button("📖 Word Meaning", type="secondary", key="word_finder_btn"):
        st.session_state.show_word_finder = not st.session_state.get('show_word_finder', False)

@st.cache_resource(ttl=300)
def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        # Try DATABASE_URL first, then individual parameters
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'voicegrammar'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD'),
                port=os.getenv('DB_PORT', '5432')
            )
        
        cursor = conn.cursor()
        
        # Create grammar_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grammar_sessions (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                score REAL NOT NULL CHECK (score >= 0 AND score <= 10),
                errors INTEGER NOT NULL DEFAULT 0,
                method VARCHAR(50) NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create grammar_errors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grammar_errors (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES grammar_sessions(id) ON DELETE CASCADE,
                error_message TEXT NOT NULL,
                error_context TEXT,
                error_category VARCHAR(100),
                rule_id VARCHAR(100),
                suggestions JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user_stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                id SERIAL PRIMARY KEY,
                total_sessions INTEGER DEFAULT 0,
                total_words INTEGER DEFAULT 0,
                total_errors INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0,
                best_score REAL DEFAULT 0,
                last_session_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_grammar_sessions_date ON grammar_sessions(date DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_grammar_errors_session_id ON grammar_errors(session_id)')
        
        # Insert initial user stats if not exists
        cursor.execute('INSERT INTO user_stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING')
        
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def save_session_to_db(text, score, errors, input_method):
    """Save session to PostgreSQL database with detailed error tracking"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        cur = conn.cursor()
        
        # Insert session record
        cur.execute(
            "INSERT INTO grammar_sessions (text, score, errors, method) VALUES (%s, %s, %s, %s) RETURNING id",
            (text or '', float(score), len(errors) if errors else 0, input_method or '')
        )
        session_id = cur.fetchone()[0]
        
        # Insert detailed errors if any
        if errors and session_id:
            for error in errors:
                cur.execute(
                    """INSERT INTO grammar_errors 
                       (session_id, error_message, error_context, error_category, rule_id, suggestions) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (session_id,
                     error.get('message', ''),
                     error.get('context', ''),
                     error.get('category', ''),
                     error.get('rule', ''),
                     json.dumps(error.get('suggestions', [])))
                )
        
        # Update user statistics
        word_count = len(text.split()) if text else 0
        cur.execute(
            """UPDATE user_stats SET 
               total_sessions = total_sessions + 1,
               total_words = total_words + %s,
               total_errors = total_errors + %s,
               average_score = (SELECT AVG(score) FROM grammar_sessions),
               best_score = GREATEST(best_score, %s),
               last_session_date = CURRENT_TIMESTAMP
               WHERE id = 1""",
            (word_count, len(errors) if errors else 0, float(score))
        )
        
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return False

def get_user_history():
    """Get recent sessions with detailed information"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor()
        cur.execute(
            """SELECT id, text, score, errors, method, date 
               FROM grammar_sessions 
               ORDER BY date DESC LIMIT 10"""
        )
        result = [{
            'id': r[0], 'text': r[1], 'score': r[2], 
            'errors': r[3], 'method': r[4], 'date': r[5]
        } for r in cur.fetchall()]
        cur.close()
        return result
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return []

def get_user_stats():
    """Get user statistics from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        cur = conn.cursor()
        cur.execute(
            """SELECT total_sessions, total_words, total_errors, 
                      average_score, best_score, last_session_date 
               FROM user_stats WHERE id = 1"""
        )
        row = cur.fetchone()
        if row:
            return {
                'total_sessions': row[0],
                'total_words': row[1],
                'total_errors': row[2],
                'average_score': round(row[3], 1) if row[3] else 0,
                'best_score': row[4] if row[4] else 0,
                'last_session_date': row[5]
            }
        cur.close()
        return None
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")
        return None

def get_session_errors(session_id):
    """Get detailed errors for a specific session"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor()
        cur.execute(
            """SELECT error_message, error_context, error_category, 
                      rule_id, suggestions 
               FROM grammar_errors 
               WHERE session_id = %s""",
            (session_id,)
        )
        result = [{
            'message': r[0],
            'context': r[1],
            'category': r[2],
            'rule': r[3],
            'suggestions': json.loads(r[4]) if r[4] else []
        } for r in cur.fetchall()]
        cur.close()
        return result
    except Exception as e:
        st.error(f"Error fetching session errors: {str(e)}")
        return []



@st.cache_resource(ttl=600)
def load_grammar_tool():
    """Load LanguageTool for grammar checking"""
    return language_tool_python.LanguageTool('en-US')

def transcribe_audio(audio_file_path):
    """Convert audio to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text, None
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please speak clearly."
    except sr.RequestError as e:
        return None, f"Could not request results; {e}"
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

def analyze_grammar(text):
    """Analyze grammar using LanguageTool - Grammar mistakes only"""
    tool = load_grammar_tool()
    matches = tool.check(text)
    
    # Filter only grammar-related errors
    grammar_categories = ['GRAMMAR', 'MORFOLOGIK_RULE', 'TYPOS', 'PUNCTUATION']
    
    errors = []
    for match in matches:
        # Only include grammar-related errors
        if any(cat in match.category.upper() for cat in grammar_categories):
            error = {
                'message': match.message,
                'context': match.context,
                'offset': match.offset,
                'error_length': match.errorLength,
                'suggestions': match.replacements[:3] if match.replacements else [],
                'rule': match.ruleId,
                'category': match.category
            }
            errors.append(error)
    
    return errors





def calculate_grammar_score(text, errors):
    """Calculate grammar score (0-10) based on errors and completeness"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # Check if sentence is complete (ends with proper punctuation)
    text_stripped = text.strip()
    is_complete = text_stripped.endswith(('.', '!', '?'))
    
    # Start with base score
    if len(errors) == 0 and is_complete:
        return 10.0  # Perfect score for no errors and complete sentence
    
    # Base score calculation
    base_score = 10.0
    
    # Reduce score for incomplete sentence
    if not is_complete:
        base_score -= 1.0
    
    # Reduce score based on number of errors
    word_count = len(text.split())
    if word_count > 0:
        error_penalty = (len(errors) / word_count) * 8  # Max 8 points penalty for errors
        base_score -= error_penalty
    
    # Ensure score is between 0 and 10
    score = max(0.0, min(10.0, base_score))
    
    return round(score, 1)

def get_score_color(score):
    """Return color based on score"""
    if score >= 8:
        return "green"
    elif score >= 6:
        return "orange"
    else:
        return "red"

def display_grammar_feedback(text, errors, score):
    """Display grammar analysis results"""
    
    # Display score with color
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Grammar Score", f"{score}/10", delta=None)
    
    with col2:
        st.metric("Total Errors", len(errors))
    
    with col3:
        st.metric("Word Count", len(text.split()))
    
    # Display transcribed text
    st.subheader("📝 Transcribed Text")
    st.text_area("Your Speech", text, height=150, disabled=True)
    
    # Display errors and suggestions
    if errors:
        st.subheader("❌ Grammar Issues Detected")
        
        for idx, error in enumerate(errors, 1):
            with st.expander(f"Issue #{idx}: {error['category']} - {error['message'][:50]}..."):
                st.write(f"**Error:** {error['message']}")
                st.write(f"**Context:** {error['context']}")
                
                if error['suggestions']:
                    st.write(f"**Suggestions:** {', '.join(error['suggestions'])}")
                else:
                    st.write("**Suggestions:** No suggestions available")
                
                st.write(f"**Rule ID:** {error['rule']}")
    else:
        st.success("✅ No grammar errors detected! Excellent work!")
    
    # Overall feedback
    st.subheader("📊 Overall Assessment")
    if score >= 8:
        st.success(f"🌟 Excellent! Your grammar is outstanding with a score of {score}/10.")
    elif score >= 6:
        st.warning(f"👍 Good job! Your grammar is decent with a score of {score}/10. Review the suggestions above to improve.")
    else:
        st.error(f"📚 Needs improvement. Your grammar score is {score}/10. Please review the errors and practice more.")

def save_audio_file(audio_bytes):
    """Save audio bytes to a temporary WAV file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name

def get_word_meaning(word):
    """Get word meaning using Gemini API for 5th grade level"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = f"Define '{word}' simply for a 5th grader. Give meaning in one sentence, then example in another sentence. Format: Meaning|Example"
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Split meaning and example
        if '|' in text:
            parts = text.split('|', 1)
            meaning = parts[0].strip()
            example = parts[1].strip()
        else:
            # Fallback parsing
            lines = text.split('\n')
            meaning = lines[0].strip() if lines else text
            example = lines[1].strip() if len(lines) > 1 else "Example not available"
        
        return meaning, example
    except Exception as e:
        return f"Sorry, couldn't find meaning for '{word}'", ""

# Word Meaning Finder - Top Right Only
if st.session_state.get('show_word_finder', False):
    # Create top-right positioned container
    _, _, col_right = st.columns([3, 1, 2])
    
    with col_right:
        with st.container(border=True):
            # Header with close button
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown("**📖 Word Finder**")
            with header_col2:
                if st.button("❌", key="close_finder", help="Close"):
                    st.session_state.show_word_finder = False
                    st.rerun()
            
            # Word input and search
            word_input = st.text_input("Word:", placeholder="Enter word...", key="word_input", label_visibility="collapsed")
            
            if st.button("🔍 Find Meaning", type="primary", key="get_meaning", use_container_width=True):
                if word_input.strip():
                    with st.spinner("Searching..."):
                        meaning, example = get_word_meaning(word_input.strip())
                        st.markdown(f"**{word_input.title()}:**")
                        st.write(f"**Meaning:** {meaning}")
                        if example:
                            st.write(f"**Example:** {example}")
                else:
                    st.warning("Enter a word first!")

# Main app layout
st.markdown("---")

# Initialize session state
if 'show_word_finder' not in st.session_state:
    st.session_state.show_word_finder = False

# Create tabs for different views
tab1, tab2 = st.tabs(["🎤 Analyze Grammar", "📊 History & Progress"])

with tab1:
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Audio File", "🎙️ Record Audio"],
        horizontal=True
    )

    if input_method == "📁 Upload Audio File":
        st.info("Upload an audio file (WAV, MP3, FLAC, etc.) with English speech.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("🔍 Analyze Grammar", type="primary"):
                with st.spinner("Processing audio and analyzing grammar..."):
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        uploaded_file_path = tmp_file.name
                    
                    try:
                        # Convert to WAV if needed
                        if not uploaded_file_path.endswith('.wav'):
                            from pydub import AudioSegment
                            audio = AudioSegment.from_file(uploaded_file_path)
                            wav_path = uploaded_file_path.replace(os.path.splitext(uploaded_file_path)[1], '.wav')
                            audio.export(wav_path, format='wav')
                            os.remove(uploaded_file_path)
                            uploaded_file_path = wav_path
                        
                        # Transcribe audio
                        text, error = transcribe_audio(uploaded_file_path)
                        
                        if error:
                            st.error(error)
                        elif text:
                            # Analyze grammar
                            errors = analyze_grammar(text)
                            score = calculate_grammar_score(text, errors)
                            
                            # Save to database
                            save_session_to_db(text, score, errors, "Upload Audio File")
                            
                            # Display results
                            st.markdown("---")
                            display_grammar_feedback(text, errors, score)
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(uploaded_file_path):
                            os.remove(uploaded_file_path)

    else:  # Record Audio
        st.info("Click the microphone button below to start recording. Speak clearly in English.")
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔍 Analyze Grammar", type="primary"):
                with st.spinner("Processing audio and analyzing grammar..."):
                    # Save audio to temporary file
                    audio_file_path = save_audio_file(audio_bytes)
                    
                    try:
                        # Transcribe audio
                        text, error = transcribe_audio(audio_file_path)
                        
                        if error:
                            st.error(error)
                        elif text:
                            # Analyze grammar
                            errors = analyze_grammar(text)
                            score = calculate_grammar_score(text, errors)
                            
                            # Save to database
                            save_session_to_db(text, score, errors, "Record Audio")
                            
                            # Display results
                            st.markdown("---")
                            display_grammar_feedback(text, errors, score)
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(audio_file_path):
                            os.remove(audio_file_path)

with tab2:
    st.header("📊 Your Grammar Progress History")
    st.markdown("Track your grammar improvement over time with detailed session history.")
    
    # Get user statistics
    stats = get_user_stats()
    sessions = get_user_history()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", stats['total_sessions'])
        with col2:
            st.metric("Average Score", f"{stats['average_score']}/10")
        with col3:
            st.metric("Total Words", stats['total_words'])
        with col4:
            st.metric("Best Score", f"{stats['best_score']}/10")
    
    if sessions:
        
        st.markdown("---")
        st.subheader("Recent Sessions")
        
        # Display each session
        for i, session in enumerate(sessions):
            with st.expander(f"📅 Score: {session['score']}/10 | Errors: {session['errors']}"):
                st.write(f"**Method:** {session['method']}")
                st.text_area("Text", session['text'], height=80, disabled=True, key=f"text_{i}")
                
                # Show detailed errors for this session
                session_errors = get_session_errors(session['id'])
                if session_errors:
                    st.write(f"**Grammar Errors ({len(session_errors)}):**")
                    for idx, error in enumerate(session_errors, 1):
                        with st.expander(f"Error {idx}: {error['category']}"):
                            st.write(f"**Message:** {error['message']}")
                            st.write(f"**Context:** {error['context']}")
                            if error['suggestions']:
                                st.write(f"**Suggestions:** {', '.join(error['suggestions'])}")
                else:
                    st.success("No errors in this session!")
    else:
        st.info("No analysis history yet. Start by analyzing some audio in the 'Analyze Grammar' tab!")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application evaluates English grammar from voice inputs using:
    
    - **Speech Recognition**: Converts speech to text
    - **LanguageTool**: Detects grammar errors
    - **ML Scoring**: Calculates grammar score (0-10)
    
    ### How to use:
    1. Choose your input method
    2. Record or upload audio
    3. Click "Analyze Grammar"
    4. Review your score and feedback
    
    ### Tips for best results:
    - Speak clearly and at a moderate pace
    - Use proper sentence structure
    - Ensure good audio quality
    - Minimize background noise
    """)
    
    st.markdown("---")
    st.markdown("**Scoring Guide:**")
    st.markdown("🟢 8-10: Excellent")
    st.markdown("🟡 6-7.9: Good")
    st.markdown("🔴 0-5.9: Needs Improvement")
    
    st.markdown("---")
    st.markdown("**📖 Word Finder:**")
    st.markdown("Click the button at top-right to find simple word meanings!")

    