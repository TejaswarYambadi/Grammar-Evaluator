# 🎤 VoiceGrammarAI - AI-Powered Grammar Evaluator

A Streamlit web application that evaluates English grammar from voice inputs using Google's Gemini AI and provides real-time feedback with scoring.

## 📂 Repository

**GitHub**: [https://github.com/TejaswarYambadi/Grammar-Evaluator.git](https://github.com/TejaswarYambadi/Grammar-Evaluator.git)

```bash
# Clone the repository
git clone https://github.com/TejaswarYambadi/Grammar-Evaluator.git
cd Grammar-Evaluator
```

## ✨ Features

- **📁 Audio File Upload**: Support for WAV, MP3, FLAC, OGG, M4A formats
- **🎙️ Live Recording**: Record audio directly in the browser
- **🤖 AI Grammar Analysis**: Powered by Google Gemini AI and LanguageTool
- **📊 Smart Scoring**: Grammar scores from 0-10 based on errors and completeness
- **📖 Word Meaning Finder**: Get simple word definitions for 5th-grade level
- **📈 Progress Tracking**: PostgreSQL database stores analysis history
- **🎯 Detailed Feedback**: Error explanations with suggestions for improvement

## 🔧 Prerequisites

### Required Accounts
- **Google AI Studio**: Get API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

### System Requirements
- **Python 3.8+**
- **PostgreSQL 12+** (local or cloud)
- **Internet connection** (for speech recognition and AI analysis)

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/TejaswarYambadi/Grammar-Evaluator.git
cd Grammar-Evaluator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy example file
copy .env.example .env     # Windows
cp .env.example .env       # Linux/macOS

# Edit .env with your credentials
GEMINI_API_KEY=your_actual_api_key
DB_HOST=localhost
DB_NAME=voicegrammar
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432
```

### 4. Setup PostgreSQL Database
```sql
-- Create database
CREATE DATABASE voicegrammar;
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE voicegrammar TO your_user;
```

### 5. Run Application
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📱 How to Use

### Grammar Analysis
1. **Choose Input Method**: Upload audio file or record live
2. **Process Audio**: Click "🔍 Analyze Grammar"
3. **Review Results**: Get score, errors, and suggestions
4. **Track Progress**: View history in "📊 History & Progress" tab

### Word Meaning Finder
1. Click "📖 Word Meaning" button (top-right)
2. Enter any English word
3. Get simple definition and example sentence

## 🏗️ Technical Architecture

### Core Technologies
- **Streamlit**: Web interface
- **Google Gemini AI**: Grammar analysis and word definitions
- **LanguageTool**: Grammar error detection
- **SpeechRecognition**: Audio-to-text conversion
- **PostgreSQL**: Data persistence
- **PyDub**: Audio format conversion

### Database Schema
The app automatically creates these tables:
- `grammar_sessions`: Analysis results and scores
- `grammar_errors`: Detailed error information
- `user_stats`: Progress statistics

### Scoring Algorithm
- **Base Score**: 10 points
- **Incomplete Sentence**: -1 point
- **Grammar Errors**: Penalty based on error-to-word ratio
- **Final Range**: 0-10 with color coding (🟢 8-10, 🟡 6-7.9, 🔴 0-5.9)

## 📁 Project Structure

```
VoiceGrammarAI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (private)
├── .env.example       # Configuration template
├── .gitignore         # Git ignore rules
└── README.md          # This documentation
```

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Database connection failed"**
- Check PostgreSQL is running
- Verify credentials in `.env` file
- Ensure database exists

**"Microphone not working"**
- Allow browser microphone permissions
- Check system audio settings
- Try refreshing the page

**"Audio processing failed"**
- Install FFmpeg for audio conversion
- Check supported file formats
- Try converting to WAV format

### Performance Tips
- Use good quality microphone
- Speak clearly and at moderate pace
- Minimize background noise
- Ensure stable internet connection

## 🔒 Security

- `.env` file is in `.gitignore` (never commit API keys)
- Use `.env.example` as template
- Keep API keys secure and regenerate if compromised

## 📊 Dependencies

```
streamlit>=1.28.0          # Web framework
speechrecognition>=3.14.3  # Audio transcription
audio-recorder-streamlit   # Browser audio recording
language-tool-python       # Grammar checking
google-generativeai        # Gemini AI integration
psycopg2-binary           # PostgreSQL connector
pydub>=0.25.1             # Audio processing
python-dotenv>=1.0.0      # Environment variables
```

## 🎯 Usage Examples

### Scoring Examples
- **Score 10**: "Hello, how are you today?" (Perfect grammar, complete sentence)
- **Score 9**: "Hello how are you today" (Missing punctuation)
- **Score 7**: "Hello, how is you today?" (Grammar error with suggestion)
- **Score 5**: "Hello how is you" (Multiple errors, incomplete)

### Supported Audio Formats
- WAV, MP3, FLAC, OGG, M4A
- Automatic conversion to WAV for processing
- Maximum file size depends on available memory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Tejaswar Yambadi**
- GitHub: [@TejaswarYambadi](https://github.com/TejaswarYambadi)
- Repository: [Grammar-Evaluator](https://github.com/TejaswarYambadi/Grammar-Evaluator)

---

**🎉 Ready to improve your English grammar with AI!**

For support, check the troubleshooting section or verify your setup matches the prerequisites.