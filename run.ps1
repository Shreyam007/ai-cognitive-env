py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Ensure you have your GROQ_API_KEY or OPENAI_API_KEY set in your environment
.\.venv\Scripts\python.exe benchmark.py
Invoke-Item benchmark_LLMBaseline_hard.png
