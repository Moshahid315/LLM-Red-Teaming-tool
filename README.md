# LLM Red Teaming Tool

A comprehensive Streamlit application for testing and evaluating the robustness of open-source Large Language Models (LLMs) using proven red teaming techniques. This tool allows security researchers and AI practitioners to systematically assess model safety mechanisms and identify potential vulnerabilities.

## 🚀 Features

- **15 Red Teaming Techniques**: Comprehensive collection of attack methodologies including jailbreaking, prompt injection, and social engineering
- **Multiple LLM Support**: Compatible with popular open-source models like LLaMA, Mistral, Qwen, and more
- **Interactive Interface**: User-friendly Streamlit web application with real-time results
- **Technique Analysis**: Detailed descriptions and effectiveness ratings for each methodology
- **Response Evaluation**: Automatic analysis of model responses with success/failure detection
- **Local Deployment**: Runs entirely on your local machine for privacy and security

## 📋 Prerequisites

- **Python 3.8+**
- **8GB RAM minimum** (16GB recommended for larger models)
- **5-50GB storage** (depending on models used)
- **Windows 10/11, macOS, or Linux**

## 🛠️ Installation

### Step 1: Install Ollama

**Windows:**
1. Download the installer from [ollama.com](https://ollama.com)
2. Run `OllamaSetup.exe` and follow the installation wizard
3. Verify installation: `ollama --version`

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv red_team_env

# Activate virtual environment
# Windows:
red_team_env\Scripts\activate
# macOS/Linux:
source red_team_env/bin/activate

# Install dependencies
pip install streamlit pandas requests openpyxl
```

### Step 3: Download Models

```bash
# Start Ollama service
ollama serve

# Download recommended models (choose based on your system)
ollama pull llama3.1:8b      # 4.9GB - Recommended for most users
ollama pull mistral:7b       # 4.1GB - Fast and efficient
ollama pull qwen2.5:7b       # 4.3GB - Good multilingual support
ollama pull tinyllama:1.1b   # 638MB - Ultra-light for testing
```

## 📦 Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28.0 | Web application framework |
| `pandas` | ≥1.5.0 | Data manipulation and Excel file handling |
| `requests` | ≥2.28.0 | HTTP requests to Ollama API |
| `openpyxl` | ≥3.0.0 | Excel file reading for red teaming techniques |

### System Dependencies

- **Ollama**: Local LLM server and model management
- **Excel File**: `Most-effective-Red-teaming-techniques.xlsx` containing technique templates

## 🏃‍♂️ Running the Application

### Step 1: Start Ollama Service

```bash
# In terminal/command prompt
ollama serve
```

Keep this terminal open - Ollama needs to run continuously.

### Step 2: Launch Streamlit App

```bash
# In a new terminal, activate your virtual environment
red_team_env\Scripts\activate  # Windows
source red_team_env/bin/activate  # macOS/Linux

# Run the Streamlit application
streamlit run red_team_app.py
```

### Step 3: Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

## 🎯 Usage Guide

### Basic Workflow

1. **Enter Your Prompt**: Input the text you want to test in the prompt area
2. **Select Red Teaming Technique**: Choose from 15 available methodologies with effectiveness ratings
3. **Choose Target Model**: Select from available open-source LLMs
4. **Generate Attack**: Click "Generate Red Team Prompt" to apply the technique
5. **Analyze Results**: Review the modified prompt, model response, and success metrics

### Available Red Teaming Techniques

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Bury the Lead** | Normal prompt followed by violative request | High |
| **System Prompt Override** | Attempts to override system instructions | High |
| **Jailbreak Combo v1** | Developer mode with custom formatting | High |
| **Role Playing** | Character-based attack scenarios | Medium |
| **LeetSpeak** | Character substitution encoding | Medium |
| **Flattery** | Compliments before malicious requests | Medium |
| **Chain Attack** | Multiple combined techniques | High |
| **Misinformation** | False contextual information | Medium |
| **Leading Questions** | Suggestive questioning approach | Medium |
| **Stress Testing** | Overwhelming with nonsensical inputs | Low |
| **Bribery** | Theoretical rewards for compliance | Low |
| **Mirroring** | Emotional manipulation tactics | Medium |
| **Skeleton Key** | False disclaimers before requests | High |
| **Restatement** | Simple request restatement | Low |
| **STAN Mode** | Simulated unrestricted AI mode | High |

### Supported Models

| Model Family | Recommended Version | Size | Use Case |
|--------------|-------------------|------|----------|
| **LLaMA 3.1** | `llama3.1:8b` | 4.9GB | General testing |
| **Mistral** | `mistral:7b` | 4.1GB | Fast inference |
| **Qwen 2.5** | `qwen2.5:7b` | 4.3GB | Multilingual support |
| **DeepSeek-R1** | `deepseek-r1:8b` | 4.7GB | Reasoning tasks |
| **Falcon** | `falcon:7b` | 4.1GB | Lightweight deployment |
| **Code Llama** | `codellama:7b` | 3.8GB | Code generation |
| **TinyLlama** | `tinyllama:1.1b` | 638MB | Ultra-light testing |

## 🔧 Configuration

### Model Configuration

Edit the `OPEN_SOURCE_LLMS` dictionary in `red_team_app.py` to add or modify models:

```python
OPEN_SOURCE_LLMS = {
    "Custom Model": {
        "api_endpoint": "http://localhost:11434/api/generate",
        "model_name": "your-model-name"
    }
}
```

### Timeout Settings

Adjust request timeout for slower systems:

```python
response = requests.post(
    llm_config["api_endpoint"],
    json=payload,
    timeout=120  # Increase for slower systems
)
```

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error**
```
⚠️ Connection Error: Unable to connect to the LLM server
```
- **Solution**: Ensure `ollama serve` is running
- **Check**: Port 11434 is not blocked by firewall

**Model Not Found (HTTP 404)**
```
Error: HTTP 404 - model 'llama3.1:8b' not found
```
- **Solution**: Download the model with `ollama pull llama3.1:8b`
- **Verify**: Run `ollama list` to see available models

**Timeout Errors**
```
⚠️ Timeout Error: The request took too long
```
- **Solution**: Increase timeout duration in code
- **Alternative**: Use smaller models for testing
- **Tip**: Pre-warm models with simple queries

**Streamlit Won't Close (Windows)**
- **Solution**: Reopen browser tab, then press Ctrl+C
- **Alternative**: Use Task Manager to end Python processes
- **Prevention**: Keep browser tab open during development

### Performance Optimization

**For Limited Resources:**
- Use models under 2GB (TinyLlama, Qwen 2.5 0.5B)
- Reduce context window in model options
- Close unnecessary applications

**For Better Performance:**
- Use SSD storage for models
- Allocate more RAM to system
- Consider GPU acceleration if available

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Network**: Internet for initial model downloads

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **GPU**: 4GB+ VRAM (optional but beneficial)

## 🔒 Security Considerations

- **Local Deployment**: All processing happens on your machine
- **No Data Transmission**: Prompts and responses stay local
- **Model Safety**: Test responsibly and ethically
- **Output Review**: Always review generated content before use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Use responsibly and in accordance with your organization's AI safety policies.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Verify all dependencies are installed correctly
- Ensure Ollama service is running
- Test with smaller models first

**Note**: This tool is designed for legitimate security research and AI safety evaluation. Always use responsibly and in compliance with applicable laws and regulations.
