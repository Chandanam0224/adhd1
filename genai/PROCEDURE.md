# Deployment & Local Run Procedure

1. Create a Python virtual environment:
   python -m venv .venv
   .\.venv\Scripts\activate    # Windows PowerShell

2. Install requirements:
   pip install -r requirements.txt

3. Update secrets:
   - Open `secrets.toml` and add your OpenAI API key:
     [openai]
     api_key = "YOUR_KEY_HERE"

4. Run locally:
   streamlit run app.py

5. Deploy to Streamlit Cloud:
   - Push this repo to GitHub.
   - On share.streamlit.io connect the repo and deploy.
   - Add `OPENAI_API_KEY` to the Streamlit Secrets (or use this repo's `secrets.toml` in local runs).

Security notes:
- Never commit real API keys.
- For production, implement stronger crisis detection, clinician escalation, and data privacy (encryption, consent).
