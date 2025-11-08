import streamlit as st
import speech_recognition as sr
import language_tool_python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import tempfile
from audio_recorder_streamlit import audio_recorder
import re
import psycopg2
from datetime import datetime
import json
from textblob import TextBlob
import textstat

st.set_page_config(page_title="Grammar Evaluator", page_icon="🎤", layout="wide")

st.title("🎤 Voice-Based Grammar Evaluator")
st.markdown("Evaluate your English grammar from voice inputs with AI-powered feedback")

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(os.environ['DATABASE_URL'])

def save_session_to_db(text, score, errors, input_method):
    """Save analysis session to database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        word_count = len(text.split())
        error_count = len(errors)
        
        cur.execute(
            """INSERT INTO grammar_sessions 
               (transcribed_text, grammar_score, word_count, error_count, input_method) 
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (text, score, word_count, error_count, input_method)
        )
        session_id = cur.fetchone()[0]
        
        for error in errors:
            suggestions_json = json.dumps(error['suggestions'])
            cur.execute(
                """INSERT INTO grammar_errors 
                   (session_id, error_message, error_context, error_category, rule_id, suggestions) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (session_id, error['message'], error['context'], error['category'], 
                 error['rule'], suggestions_json)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return False

def get_user_history():
    """Retrieve user's analysis history"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """SELECT id, session_date, transcribed_text, grammar_score, 
                      word_count, error_count, input_method 
               FROM grammar_sessions 
               ORDER BY session_date DESC 
               LIMIT 20"""
        )
        
        sessions = []
        for row in cur.fetchall():
            sessions.append({
                'id': row[0],
                'date': row[1],
                'text': row[2],
                'score': row[3],
                'word_count': row[4],
                'error_count': row[5],
                'input_method': row[6]
            })
        
        cur.close()
        conn.close()
        return sessions
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return []

def get_session_errors(session_id):
    """Get errors for a specific session"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """SELECT error_message, error_context, error_category, rule_id, suggestions 
               FROM grammar_errors 
               WHERE session_id = %s""",
            (session_id,)
        )
        
        errors = []
        for row in cur.fetchall():
            errors.append({
                'message': row[0],
                'context': row[1],
                'category': row[2],
                'rule': row[3],
                'suggestions': json.loads(row[4]) if row[4] else []
            })
        
        cur.close()
        conn.close()
        return errors
    except Exception as e:
        st.error(f"Error fetching session errors: {str(e)}")
        return []

@st.cache_resource
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
    """Analyze grammar using LanguageTool"""
    tool = load_grammar_tool()
    matches = tool.check(text)
    
    errors = []
    for match in matches:
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

def analyze_style_and_tone(text):
    """Analyze style and tone using advanced NLP"""
    if not text or len(text.strip()) == 0:
        return None
    
    blob = TextBlob(text)
    
    # Sentiment analysis
    sentiment = blob.sentiment
    sentiment_label = "Neutral"
    if sentiment.polarity > 0.1:
        sentiment_label = "Positive"
    elif sentiment.polarity < -0.1:
        sentiment_label = "Negative"
    
    # Readability metrics
    reading_ease = textstat.flesch_reading_ease(text)
    reading_grade = textstat.flesch_kincaid_grade(text)
    
    # Readability interpretation
    if reading_ease >= 90:
        reading_level = "Very Easy"
    elif reading_ease >= 80:
        reading_level = "Easy"
    elif reading_ease >= 70:
        reading_level = "Fairly Easy"
    elif reading_ease >= 60:
        reading_level = "Standard"
    elif reading_ease >= 50:
        reading_level = "Fairly Difficult"
    elif reading_ease >= 30:
        reading_level = "Difficult"
    else:
        reading_level = "Very Difficult"
    
    # Formality analysis (heuristic-based)
    words = text.lower().split()
    informal_words = ['gonna', 'wanna', 'gotta', 'yeah', 'yep', 'nope', 'ok', 'okay', 
                      'kinda', 'sorta', 'dunno', 'ain\'t', 'y\'all', 'cause']
    formal_indicators = ['therefore', 'furthermore', 'consequently', 'nevertheless', 
                        'moreover', 'thus', 'hence', 'accordingly']
    
    informal_count = sum(1 for w in words if w in informal_words)
    formal_count = sum(1 for w in words if w in formal_indicators)
    
    if formal_count > informal_count:
        formality = "Formal"
    elif informal_count > formal_count:
        formality = "Informal"
    else:
        formality = "Neutral"
    
    # Word complexity
    avg_word_length = sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0
    
    # Passive voice detection (simple heuristic)
    passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']
    passive_count = sum(1 for word in words if word in passive_indicators)
    passive_percentage = (passive_count / len(words) * 100) if words else 0
    
    return {
        'sentiment_label': sentiment_label,
        'sentiment_polarity': round(sentiment.polarity, 2),
        'sentiment_subjectivity': round(sentiment.subjectivity, 2),
        'reading_ease': round(reading_ease, 1),
        'reading_level': reading_level,
        'reading_grade': round(reading_grade, 1),
        'formality': formality,
        'avg_word_length': round(avg_word_length, 1),
        'passive_voice_percentage': round(passive_percentage, 1)
    }

def extract_linguistic_features(text, errors):
    """Extract linguistic features for ML model"""
    if not text or len(text.strip()) == 0:
        return np.zeros(12)
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = [
        len(words),
        len(errors),
        len(errors) / len(words) if len(words) > 0 else 0,
        len(sentences),
        len(words) / len(sentences) if len(sentences) > 0 else 0,
        sum(1 for w in words if len(w) > 6) / len(words) if len(words) > 0 else 0,
        sum(1 for w in words if w[0].isupper()) / len(words) if len(words) > 0 else 0,
        len([c for c in text if c in '.,!?;:']) / len(words) if len(words) > 0 else 0,
        sum(1 for e in errors if 'spelling' in e['category'].lower()),
        sum(1 for e in errors if 'grammar' in e['category'].lower()),
        sum(1 for e in errors if 'punctuation' in e['category'].lower()),
        1 if any(e['suggestions'] for e in errors) else 0
    ]
    
    return np.array(features)

@st.cache_resource
def load_ml_model():
    """Load pre-trained ML model for grammar scoring"""
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    
    X_train = np.array([
        [20, 0, 0.0, 2, 10, 0.3, 0.1, 0.15, 0, 0, 0, 0],
        [15, 1, 0.067, 2, 7.5, 0.2, 0.13, 0.13, 1, 0, 0, 1],
        [25, 2, 0.08, 3, 8.3, 0.28, 0.12, 0.16, 0, 2, 0, 1],
        [30, 0, 0.0, 3, 10, 0.35, 0.1, 0.13, 0, 0, 0, 0],
        [10, 3, 0.3, 1, 10, 0.2, 0.2, 0.1, 1, 2, 0, 1],
        [18, 1, 0.056, 2, 9, 0.22, 0.11, 0.17, 0, 1, 0, 1],
        [12, 4, 0.33, 2, 6, 0.17, 0.08, 0.08, 2, 2, 0, 1],
        [22, 1, 0.045, 3, 7.3, 0.27, 0.14, 0.18, 0, 1, 0, 1],
        [8, 2, 0.25, 1, 8, 0.13, 0.13, 0.13, 1, 1, 0, 1],
        [35, 0, 0.0, 4, 8.75, 0.4, 0.11, 0.14, 0, 0, 0, 0],
        [14, 5, 0.36, 2, 7, 0.14, 0.07, 0.07, 2, 3, 0, 1],
        [28, 1, 0.036, 3, 9.3, 0.32, 0.11, 0.14, 0, 1, 0, 1],
        [16, 3, 0.19, 2, 8, 0.19, 0.13, 0.13, 1, 2, 0, 1],
        [40, 0, 0.0, 5, 8, 0.38, 0.1, 0.15, 0, 0, 0, 0],
        [11, 6, 0.55, 1, 11, 0.18, 0.09, 0.09, 3, 3, 0, 1],
        [24, 2, 0.083, 3, 8, 0.25, 0.13, 0.17, 1, 1, 0, 1],
        [19, 2, 0.11, 2, 9.5, 0.21, 0.11, 0.16, 0, 2, 0, 1],
        [13, 4, 0.31, 2, 6.5, 0.15, 0.08, 0.08, 2, 2, 0, 1],
        [32, 1, 0.031, 4, 8, 0.34, 0.13, 0.16, 0, 1, 0, 1],
        [9, 3, 0.33, 1, 9, 0.11, 0.11, 0.11, 2, 1, 0, 1]
    ])
    
    y_train = np.array([9.5, 8.2, 7.8, 9.8, 5.5, 8.5, 4.2, 8.7, 6.8, 10.0,
                        3.5, 9.0, 6.5, 10.0, 2.8, 7.5, 7.2, 4.8, 9.2, 6.0])
    
    model.fit(X_train, y_train)
    return model

def calculate_grammar_score(text, errors):
    """Calculate grammar score (0-10) using ML model"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    features = extract_linguistic_features(text, errors)
    model = load_ml_model()
    
    score = model.predict(features.reshape(1, -1))[0]
    
    score = max(0.0, min(10.0, score))
    
    return round(score, 1)

def get_score_color(score):
    """Return color based on score"""
    if score >= 8:
        return "green"
    elif score >= 6:
        return "orange"
    else:
        return "red"

def display_grammar_feedback(text, errors, score, style_analysis=None):
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
    
    # Display style and tone analysis
    if style_analysis:
        st.subheader("🎨 Style & Tone Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sentiment", style_analysis['sentiment_label'])
            st.caption(f"Polarity: {style_analysis['sentiment_polarity']}")
        
        with col2:
            st.metric("Formality", style_analysis['formality'])
            st.caption(f"Avg Word Length: {style_analysis['avg_word_length']}")
        
        with col3:
            st.metric("Reading Level", style_analysis['reading_level'])
            st.caption(f"Grade: {style_analysis['reading_grade']}")
        
        with col4:
            st.metric("Reading Ease", f"{style_analysis['reading_ease']}")
            st.caption(f"Passive Voice: {style_analysis['passive_voice_percentage']}%")
        
        # Additional insights
        with st.expander("📖 Detailed Style Insights"):
            st.write(f"**Sentiment Polarity:** {style_analysis['sentiment_polarity']} (-1 to 1, where -1 is very negative and 1 is very positive)")
            st.write(f"**Sentiment Subjectivity:** {style_analysis['sentiment_subjectivity']} (0 to 1, where 0 is objective and 1 is subjective)")
            st.write(f"**Flesch Reading Ease:** {style_analysis['reading_ease']} (Higher scores indicate easier readability)")
            st.write(f"**Flesch-Kincaid Grade:** {style_analysis['reading_grade']} (U.S. school grade level)")
            st.write(f"**Average Word Length:** {style_analysis['avg_word_length']} characters")
            st.write(f"**Passive Voice Usage:** {style_analysis['passive_voice_percentage']}% (Lower is generally better)")
            
            # Recommendations
            st.markdown("**Recommendations:**")
            if style_analysis['passive_voice_percentage'] > 20:
                st.info("Consider reducing passive voice usage for more direct and engaging writing.")
            if style_analysis['reading_ease'] < 60:
                st.info("Your text is fairly difficult to read. Consider using shorter sentences and simpler words.")
            if style_analysis['avg_word_length'] > 6:
                st.info("Try using shorter, more common words to improve clarity.")
    
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

# Main app layout
st.markdown("---")

# Create tabs for different views
tab1, tab2 = st.tabs(["🎤 Analyze Grammar", "📊 History & Progress"])

with tab1:
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🎙️ Record Audio", "📁 Upload Audio File"],
        horizontal=True
    )

    if input_method == "🎙️ Record Audio":
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
                            
                            # Analyze style and tone
                            style_analysis = analyze_style_and_tone(text)
                            
                            # Save to database
                            save_session_to_db(text, score, errors, "Record Audio")
                            
                            # Display results
                            st.markdown("---")
                            display_grammar_feedback(text, errors, score, style_analysis)
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(audio_file_path):
                            os.remove(audio_file_path)

    else:  # Upload Audio File
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
                            
                            # Analyze style and tone
                            style_analysis = analyze_style_and_tone(text)
                            
                            # Save to database
                            save_session_to_db(text, score, errors, "Upload Audio File")
                            
                            # Display results
                            st.markdown("---")
                            display_grammar_feedback(text, errors, score, style_analysis)
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(uploaded_file_path):
                            os.remove(uploaded_file_path)

with tab2:
    st.header("📊 Your Grammar Progress History")
    st.markdown("Track your grammar improvement over time with detailed session history.")
    
    sessions = get_user_history()
    
    if sessions:
        # Display summary statistics
        avg_score = sum(s['score'] for s in sessions) / len(sessions)
        total_words = sum(s['word_count'] for s in sessions)
        total_errors = sum(s['error_count'] for s in sessions)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", len(sessions))
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/10")
        with col3:
            st.metric("Total Words Analyzed", total_words)
        with col4:
            st.metric("Total Errors Found", total_errors)
        
        st.markdown("---")
        st.subheader("Recent Sessions")
        
        # Display each session
        for session in sessions:
            with st.expander(
                f"📅 {session['date'].strftime('%Y-%m-%d %H:%M')} - Score: {session['score']}/10 | "
                f"Errors: {session['error_count']} | Words: {session['word_count']}"
            ):
                st.write(f"**Input Method:** {session['input_method']}")
                st.write(f"**Transcribed Text:**")
                st.text_area(
                    "Text",
                    session['text'],
                    height=100,
                    disabled=True,
                    key=f"text_{session['id']}"
                )
                
                # Show errors for this session
                session_errors = get_session_errors(session['id'])
                if session_errors:
                    st.write(f"**Grammar Errors ({len(session_errors)}):**")
                    for idx, error in enumerate(session_errors, 1):
                        st.markdown(f"**{idx}.** {error['category']} - {error['message']}")
                        if error['suggestions']:
                            st.markdown(f"   *Suggestions:* {', '.join(error['suggestions'])}")
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
