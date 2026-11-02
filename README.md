# AI-Powered Customer Support Chatbot

A beginner-friendly college project built with Python, Google Colab, TF-IDF FAQ matching, Hugging Face FLAN-T5 Small, and Gradio.

## Features
- Matches customer questions against a sample FAQ dataset.
- Uses FLAN-T5 Small to generate a fallback response when no FAQ is sufficiently similar.
- Provides a simple web chat interface using Gradio.

## Run in Google Colab
1. Upload `Customer_Support_Chatbot.ipynb` to Google Colab.
2. Run the installation cell first.
3. If Colab requests a runtime restart, restart the session and continue running the notebook cells from the top.
4. Run the remaining cells in order.
5. Open the Gradio link printed by the final cell.

The notebook removes `torchvision` because this text-only project does not need it; removing an incompatible optional torchvision installation can resolve the `operator torchvision::nms does not exist` import error.

## Files
- `Customer_Support_Chatbot.ipynb` — complete Google Colab notebook.
- `app.py` — optional local Gradio app.
- `requirements.txt` — Python dependencies.

## Customize
Edit the `faq_data` list in the notebook or `app.py` to add your own customer questions and approved answers.

## Limitations
This is an academic demonstration using sample FAQs. FLAN-T5 responses can be inaccurate; do not use it for real customer transactions or sensitive information without verification, privacy safeguards, and human support escalation.
