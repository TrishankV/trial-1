import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Language code mapping for mBART
LANGUAGE_CODES = {
    "English": "en_XX",
    "French": "fr_XX",
    "Spanish": "es_XX",
    "German": "de_DE",
    "Hindi": "hi_IN",
    "Bengali": "bn_IN",
    "Gujarati": "gu_IN",
    "Marathi": "mr_IN",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Malayalam": "ml_IN",
    "Kannada": "kn_IN",
    "Punjabi": "pa_IN",
    "Oriya": "or_IN",
    # Add more languages as needed
}

# Streamlit app UI with emojis
st.title("🌐 Multi-Lingual Text Summarizer")
st.write("📝 Enter text in any language and get a concise summary with just one click!")

# Text input with emoji
input_text = st.text_area("✍️ Enter Text:", placeholder="Input your text here...")

# Language selection with emoji
source_lang = st.selectbox("🌍 Select input language", 
                           list(LANGUAGE_CODES.keys()))
target_lang = st.selectbox("🌐 Select summary language", 
                           ["Same as input"] + list(LANGUAGE_CODES.keys()))

# Summarize button with emoji
if st.button("✨ Summarize"):
    if input_text:
        try:
            # Set the source language for tokenization
            tokenizer.src_lang = LANGUAGE_CODES.get(source_lang, "en_XX")

            # Tokenize input text
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest")

            # Set the target language for generation
            if target_lang != "Same as input":
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES[target_lang])
            else:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES.get(source_lang, "en_XX"))

            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=150, 
                num_beams=4, 
                length_penalty=2.0, 
                forced_bos_token_id=forced_bos_token_id
            )

            # Decode summary with truncation
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)[:150]

            # Display summary with emoji
            st.write("📄 **Summary**")
            st.write(summary)

            # Download button for summary
            st.download_button("💾 Download Summary", summary, file_name="summary.txt")

        except Exception as e:
            st.error(f"⚠️ Error generating summary: {e}")
    else:
        st.warning("⚠️ Please enter some text to summarize.")
