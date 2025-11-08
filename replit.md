# Grammar Evaluator Application

## Overview

This is a comprehensive voice-based grammar evaluation application built with Streamlit. The application allows users to record live audio or upload audio files in multiple formats, transcribes speech to text using Google's Speech Recognition API, evaluates grammar quality using LanguageTool, performs ML-based scoring, and provides advanced style and tone analysis. The system includes user history tracking and provides detailed AI-powered feedback with improvement suggestions.

## Recent Changes (November 8, 2025)

- Added PostgreSQL database for session history tracking
- Implemented style and tone analysis (sentiment, formality, readability)
- Enhanced user interface with tabbed navigation (Analyze and History)
- Integrated advanced NLP features using TextBlob and TextStat
- All audio sessions are automatically saved to database
- Added comprehensive progress tracking with statistics

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Layout**: Wide layout configuration for better user experience
- **UI Components**: Custom audio recorder using `audio_recorder_streamlit` package
- **User Flow**: Record/upload audio → transcribe → analyze grammar → display results

**Rationale**: Streamlit was chosen for rapid development of data/ML applications with minimal frontend code. It provides built-in state management and easy integration with Python libraries.

### Speech Recognition
- **Library**: SpeechRecognition with Google Speech API
- **Audio Processing**: Handles audio file conversion and recognition
- **Error Handling**: Comprehensive error handling for unclear audio, API failures, and processing errors

**Rationale**: Google's Speech Recognition API provides good accuracy for English speech-to-text without requiring extensive setup or API keys for basic usage.

### Grammar Analysis
- **Engine**: LanguageTool Python library for English (US)
- **Caching**: Uses Streamlit's `@st.cache_resource` decorator to load grammar tool once
- **Error Detection**: Captures grammar errors with context, offset positions, and replacement suggestions
- **Error Metadata**: Tracks rule IDs, error length, and provides up to 3 suggestions per error

**Rationale**: LanguageTool is an open-source grammar checker that works offline and supports comprehensive rule-based grammar checking without external API dependencies.

### Machine Learning Component
- **Model**: RandomForestRegressor from scikit-learn
- **Purpose**: Likely used for scoring or predicting grammar quality metrics (implementation incomplete in provided code)

**Rationale**: Random Forest provides interpretable results and handles non-linear relationships well for scoring tasks.

### File Processing
- **Temporary Files**: Uses Python's `tempfile` module for handling uploaded audio files
- **Audio Formats**: Processes audio files compatible with SpeechRecognition library

## External Dependencies

### Third-Party Libraries
1. **Streamlit** - Web application framework
2. **speech_recognition** - Audio transcription via Google Speech API
3. **language_tool_python** - Grammar checking engine
4. **audio_recorder_streamlit** - Custom audio recording component
5. **scikit-learn** - Machine learning library (RandomForestRegressor)
6. **numpy** - Numerical computing support

### External APIs
- **Google Speech Recognition API** - Cloud-based speech-to-text service (used through SpeechRecognition library)

### Language Processing
- **LanguageTool** - Downloaded language model for English (US) grammar rules

**Note**: The application currently has no database dependency. All processing appears to be stateless and session-based through Streamlit's session state.