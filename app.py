"""Streamlit app: paste news text, pick a model, see entities highlighted.

Run: streamlit run app.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from spacy import displacy

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import load_spacy
from rule_based import build_ruler_nlp

ROOT = Path(__file__).parent

MODELS = {
    "en_core_web_sm — small pre-trained model": "sm",
    "en_core_web_lg — large pre-trained model": "lg",
    "Rule-based (EntityRuler gazetteer + patterns)": "rule",
    "Trained demo model (CoNLL-2003 subset)": "trained",
}

DEFAULT_TEXT = (
    "Barack Obama visited Paris last week to meet officials from the "
    "European Commission and Microsoft Corp. The United Nations said the "
    "World Bank will lend India $ 500 million for schools in New Delhi."
)


@st.cache_resource
def load_model(name):
    if name == "rule":
        return build_ruler_nlp()
    if name == "trained":
        return load_spacy(str(ROOT / "models" / "ner_demo"))
    return load_spacy({"sm": "en_core_web_sm", "lg": "en_core_web_lg"}[name])


st.set_page_config(page_title="NER from News Articles", layout="wide")
st.title("Named Entity Recognition from News Articles")
st.write(
    "Paste news text, pick a model, and see the people, locations, "
    "organizations and miscellaneous entities it finds — highlighted in "
    "the text and listed by type."
)

model_label = st.selectbox("Model", list(MODELS))
text = st.text_area("News text", DEFAULT_TEXT, height=150)

if st.button("Extract entities"):
    nlp = load_model(MODELS[model_label])
    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=False)
    st.markdown(html, unsafe_allow_html=True)

    rows = [{"Entity": ent.text, "Type": ent.label_} for ent in doc.ents]
    if rows:
        df = pd.DataFrame(rows)
        st.subheader(f"Entities found: {len(rows)}")
        st.dataframe(df, use_container_width=True)
        counts = Counter(ent.label_ for ent in doc.ents)
        st.bar_chart(pd.DataFrame(
            {"count": counts.values()}, index=list(counts.keys())
        ))
    else:
        st.info("No named entities found in this text.")
