# Stool Image Review

Streamlit web application for `realpoop.py`.

This version uses sample images first, blurs images by default, and shows static Thai guidance for:

- สาเหตุที่เป็นไปได้
- ความเสี่ยงหรือโรคที่อาจเกี่ยวข้อง
- คำแนะนำเบื้องต้นและการดูแลตัวเอง

The UI supports Thai and English. Thai is the default language on first load.

The app also includes a support chat page for OpenAI API Q&A. If `OPENAI_API_KEY` is not configured, the chat page remains visible and users can type messages, but the assistant does not reply yet.

## OpenAI support chat

Add `OPENAI_API_KEY` to Streamlit secrets or your environment:

```toml
OPENAI_API_KEY = "your-api-key"
```

Optionally set `OPENAI_MODEL`; otherwise the app uses `gpt-5-nano`.

## Run

From the project root:

```powershell
streamlit run Prototype/app.py
```

The app reads the model and sample images from the parent project:

- `convnextv2_thev1_best_for_good.pkl`
- `Image/`
