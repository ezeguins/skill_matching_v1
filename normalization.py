import pandas as pd
import numpy as np
from tqdm import tqdm

def match_and_save(
    identifiers_pkl: str = r"C:\Users\ezequ\DOLPH\Skills Matching\Embeddings Generation\filtered_skills_10M_v1_emb.pkl",
    column_name_label: str = "SkillName",
    column_name_emb: str = "embeddings",
    identifier_col: str = "Identifier",                # <— name of your ID column
    canonical_pkl: str = r"C:\Users\ezequ\DOLPH\Skills Matching\Canonical\canonical_v1_emb.pkl",
    canonical_id_col: str = "id",
    canonical_label: str = "label",
    canonical_emb: str = "embeddings",
    output_pkl: str = r"C:\Users\ezequ\DOLPH\Skills Matching\POC HYBRID\matched_skills_v1.pkl"
):
    # 1) Load data
    df = pd.read_pickle(identifiers_pkl,compression='gzip')
    df_can = pd.read_pickle(canonical_pkl,compression='gzip')

    # 2) Parse embeddings
    def parse_emb(col):
        # print("type:  ",type(col.iloc[0]))
        # print("shape: ",col.iloc[0].shape)
        # print(col.iloc[0])
        if isinstance(col, str):
            return np.fromstring(col.strip('[]'), sep=' ', dtype=np.float32)
        return col 
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
    out_df.to_pickle(output_pkl, compression='gzip')
    print(f"[✓] Matched {len(df)} rows; saved to '{output_pkl}'")

if __name__ == '__main__':
    match_and_save()
