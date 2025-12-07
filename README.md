# AI Health Interview Assistant

An intelligent Persian-language health interview system that conducts sequential medical interviews, calculates blood pressure predictions, and generates personalized health visualizations.

## Features

- **Sequential Interview**: Conducts step-by-step health interviews in Persian
- **Blood Pressure Prediction**: Calculates BP changes based on salt intake scenarios
- **Risk Assessment**: Determines patient risk factors (K-value: 5 for low risk, 10 for high risk)
- **Visual Analytics**: Generates medical charts and digital twin visualizations
- **Multi-Agent System**: Uses specialized agents for interviewing, BP calculation, and image generation

## Architecture

### Agents
- **Interviewer Agent**: Conducts Persian health interviews with validation
- **K-Finder Agent**: Assesses cardiovascular risk based on patient data
- **BP Expert Agent**: Calculates immediate and predicted blood pressure values
- **Image Generator Agent**: Creates medical visualizations using Gemini AI

### Core Functions
- `calculate_blood_pressure()`: Computes BP for 3 salt scenarios (±10%, current)
- `predict_blood_pressure()`: Projects 30-day BP changes using exponential model
- `clamp_to_range()`: Ensures BP values stay within medical range (105-180 mmHg)

## Interview Questions (Persian)
1. Age and gender
2. Current blood pressure readings
3. Salt consumption patterns
4. Water intake and beverage habits
5. Exercise frequency
6. Height and weight
7. Medications and medical history
8. Family medical history
9. Stress levels and lifestyle habits

## Technical Stack
- **AI Framework**: OpenAI Agents SDK
- **UI**: Gradio with RTL support
- **Image Generation**: Google Gemini 2.5 Flash
- **Models**: GPT-4o for text processing
- **Data Validation**: Pydantic models

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

3. Run the application:
```bash
python app.py
```

## Usage

1. Start the interview by providing your age
2. Answer each question in Persian
3. Complete all 16 health-related questions
4. View generated BP prediction charts and digital twin visualization
5. Watch educational video content

## Output

- **Blood Pressure Charts**: 3-panel visualization showing BP trends for different salt scenarios
- **Digital Twin**: Personalized health risk visualization with colored risk indicators
- **Educational Content**: Embedded video for additional health information

## Medical Disclaimer

This tool is for educational purposes only and should not replace professional medical advice.
