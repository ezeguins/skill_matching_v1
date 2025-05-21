
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

def encode_csv():
    input_csv = r"C:\Users\ezequ\DOLPH\Skills Matching\Canonical\canonical_v1.csv"
    output_pkl = r"C:\Users\ezequ\DOLPH\Skills Matching\Canonical\canonical_v1_emb.pkl"
    model_name = r'C:\Users\ezequ\DOLPH\Skills Matching\SentenceEmbeddingsApproach\5-ModelInferences\Models\v1'
    label_name = "label"
    max_tokens = 30
    batch_size = 100
    device = "cpu"

    df = pd.read_csv(input_csv, encoding='utf-8', sep=',')
    print("loaded csv of length", len(df))
    if label_name not in df.columns:
        raise ValueError(f"Column {label_name} not found in input CSV.")

    model = SentenceTransformer(model_name, device=device)

    # Diccionario para cachear embeddings por texto
    cache = {}
    unique_lower_texts = []
    # Primero detectamos todas las etiquetas únicas (en minúsculas y str)
    for text in df[label_name].astype(str):
        key = text.lower()
        if key not in cache:
            cache[key] = None
            unique_lower_texts.append(key)

    # Codificamos solo los únicos en batches
    for start in tqdm(range(0, len(unique_lower_texts), batch_size), desc="Encoding unique batches"):
        batch_lowers  = unique_lower_texts[start : start + batch_size]
        emb = model.encode(
            batch_lowers ,
            show_progress_bar=False,
            batch_size=batch_size,
            max_length=max_tokens,
            truncation=True,
            convert_to_numpy=True,
        )
        emb = normalize(emb, norm='l2', axis=1)
        # Almacenamos en el cache
        for txt, vec in zip(batch_lowers , emb.astype(np.float32)):
            cache[txt] = vec

    # Ahora mapeamos cada fila original a su embedding cacheada
    embeddings = [cache[orig.lower()] for orig in df[label_name].astype(str)]

    df["embeddings"] = embeddings
    # df.to_csv(output_csv, index=False, encoding='utf-8')
    # save to pkl file
    df.to_pickle(output_pkl, compression='gzip')
    print(f"[✓] Processed {len(df)} rows; saved with embeddings to '{output_pkl}'")

if __name__ == "__main__":
    encode_csv()
