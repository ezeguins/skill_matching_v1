import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Skill Matcher Application")

# 0. User-configurable parameters in a collapsible sidebar expander
with st.sidebar.expander("Matching Parameters", expanded=True):
    THRESHOLD = st.slider(
        label="A: Base Similarity Threshold", min_value=0.8, max_value=1.0, value=0.96, step=0.01
    )
    WORD_STEP = st.slider(
        label="B: Threshold Step per Word", min_value=0.0, max_value=0.4, value=0.030, step=0.005, format="%.3f"
    )
    st.caption("Threshold = A – B × Number of Words")

# Initialize embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("ezequiel/similarity-search-v1")
    # return SentenceTransformer(r"C:\Users\ezequ\DOLPH\Skills Matching\SentenceEmbeddingsApproach\5-ModelInferences\Models\v1")

model = load_model()

# Load canonical skills dataframe
@st.cache_data
def load_canonical(path):
    with gzip.open(path, 'rb') as f:
        df = pickle.load(f)
    if isinstance(df.loc[0, 'embeddings'], str):
        df['embeddings'] = df['embeddings'].apply(lambda s: np.fromstring(s.strip('[]'), sep=' '))
    return df

df_canonical = load_canonical('flat_skills_agregated_cleaned_emb_v1.pkl')

# Load matched skills dataframe
@st.cache_data
def load_matched(path):
    with gzip.open(path, 'rb') as f:
        df = pickle.load(f)
    return df

df_matched = load_matched('matched_skills.pkl')

# 1. Ask user to input 1 to 5 skills
skills_input = []
st.write("Enter up to 5 professional skills:")
for i in range(5):
    skill = st.text_input(f"Skill {i+1}", key=f"skill_{i}")
    if skill:
        skills_input.append(skill)

# Display formula for user reference in main area
st.markdown(
    f"**Threshold Formula for Id Method :** Similarity > {THRESHOLD} - {WORD_STEP} × Number of Words"
)

# Match button triggers matching process
if st.button("Match Skills") and skills_input:
    inputs_lower = [s.lower() for s in skills_input]
    inputs_cleaned = [s.replace("/", "-") for s in inputs_lower]
    embeddings = model.encode(inputs_lower, max_length=30, truncation=True)

    rows = []
    id_sets = []
    df_b_list = []

    cans = np.vstack(df_canonical['embeddings'].values)
    for skill, emb in zip(inputs_lower, embeddings):
        sims = cosine_similarity([emb], cans)[0]
        best_idx = np.argmax(sims)
        cid = df_canonical.loc[best_idx, 'id']
        best_label = df_canonical.loc[best_idx, 'label']
        best_sim = float(sims[best_idx])
        skill_len = len(skill.split())

        dynamic_thr = THRESHOLD - WORD_STEP * skill_len
        if best_sim > dynamic_thr:
            method = 'Id method'
            df_b = df_matched[
                (df_matched['canonical_label_id'] == cid) &
                (df_matched['similarity'] > (THRESHOLD - WORD_STEP * df_matched['label_length']))
            ]
        else:
            method = 'Exact match method'
            df_b = df_matched[df_matched['label'].str.lower() == skill]

        ids = set(df_b['Identifier'])
        rows.append({
            'skill': skill,
            'label': best_label,
            'similarity': best_sim,
            'skill_length': skill_len,
            'method': method,
            'identifiers_count': len(ids)
        })
        id_sets.append(ids)
        df_b_list.append(df_b)

    df_a = pd.DataFrame(rows)[['skill', 'label', 'similarity', 'skill_length', 'method', 'identifiers_count']]
    st.subheader("DataFrame A: Identifiers count per skill")
    st.dataframe(df_a)

    if id_sets:
        common_ids = set.intersection(*id_sets)
        found_count = len(common_ids)
        st.write(f"Found {found_count} common Identifiers across all input skills.")
        if found_count:
            st.subheader("Common Identifiers (showing up to 20)")
            for cid in list(common_ids)[:20]:
                st.write(f"#### Identifier: {cid}")
                info = []
                for skill, df_b in zip(df_a['skill'], df_b_list):
                    matched_labels = df_b.loc[df_b['Identifier'] == cid, 'label'].unique().tolist()
                    info.append({
                        'input_skill': skill,
                        'matched_label': ', '.join(matched_labels) if matched_labels else 'N/A'
                    })
                df_common = pd.DataFrame(info)
                st.table(df_common)
        else:
            st.write("No common Identifiers found across all input skills.")
