import pandas as pd
import numpy as np
from tqdm import tqdm

def match_and_save(
    labels_csv: str = r"C:\Users\ezequ\DOLPH\Skills Matching\Embeddings Generation\filtered_skills_10M_emb_v1.csv",
    column_name_label: str = "SkillName",
    column_name_emb: str = "embeddings",
    identifier_col: str = "Identifier",                # <— name of your ID column
    canonical_csv: str = r"C:\Users\ezequ\DOLPH\Skills Matching\Esco taxonomy\flat_skills_agregated_cleaned_emb_v1.csv",
    canonical_id_col: str = "id",
    canonical_label: str = "label",
    canonical_emb: str = "embeddings",
    output_csv: str = r"C:\Users\ezequ\DOLPH\Skills Matching\POC HYBRID\matched_skills.csv",
    output_pkl: str = r"C:\Users\ezequ\DOLPH\Skills Matching\POC HYBRID\matched_skills.pkl"
):
    # 1) Load data
    df = pd.read_csv(labels_csv, encoding='utf-8')
    df_can = pd.read_csv(canonical_csv, encoding='utf-8')

    # 2) Parse embeddings
    def parse_emb(col):
        return col.apply(lambda s: np.fromstring(s.strip('[]'), sep=' ', dtype=np.float32))
    df['emb_array'] = parse_emb(df[column_name_emb])
    df_can['can_emb_array'] = parse_emb(df_can[canonical_emb])

    # 3) Canonical matrix
    can_matrix = np.vstack(df_can['can_emb_array'].values)

    # 4) Unique labels → best matches
    df['label_lower'] = df[column_name_label].astype(str).str.lower()
    unique = df[['label_lower', 'emb_array']].drop_duplicates('label_lower').reset_index(drop=True)
    best_map = {}
    for _, row in tqdm(unique.iterrows(), total=len(unique), desc="Matching embeddings"):
        emb = row['emb_array']
        sims = can_matrix.dot(emb)
        best_idx = int(np.argmax(sims))
        best_map[row['label_lower']] = {
            'canonical_id':    df_can.iloc[best_idx][canonical_id_col],
            'canonical_label': df_can.iloc[best_idx][canonical_label],
            'similarity':      float(sims[best_idx])
        }

    # 5) Build output, now including Identifier
    out = []
    for _, row in df.iterrows():
        m = best_map[row['label_lower']]
        out.append({
            'Identifier':           row[identifier_col],     # <— carry this through
            'label':                row[column_name_label],
            'canonical_label_id':   m['canonical_id'],
            'canonical_label':      m['canonical_label'],
            'similarity':           m['similarity'],
            'label_length':         len(str(row[column_name_label]).split())
        })

    # 6) Save
    out_df = pd.DataFrame(out)
    out_df.to_csv(output_csv, index=False, encoding='utf-8')
    out_df.to_pickle(output_pkl, compression='gzip')
    print(f"[✓] Matched {len(df)} rows; saved to '{output_csv}'")

if __name__ == '__main__':
    match_and_save()
