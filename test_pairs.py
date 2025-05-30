from sentence_transformers import SentenceTransformer, util
import time

# 1. Carga tu modelo fine-tuned
model_path = r"C:\Users\ezequ\DOLPH\Skills Matching\SentenceEmbeddingsApproach\5-ModelInferences\Models\v1"
model = SentenceTransformer(model_path)
model = model.half() 
model.eval()  

# first_mod = model._first_module()           
# first_mod.max_seq_length = 30              
# model.tokenizer.model_max_length = 30


start_time = time.perf_counter()
sentence_1 = "full stack software development"
sentence_2 = "full stack"
sentence_3 = "full stack software"
sentence_4 = "full stack development"
# sentence_2 = "adobe creative cloud software"
emb1 = model.encode(sentence_1, max_length=30, truncation=True)
emb2 = model.encode(sentence_2, max_length=30, truncation=True)
emb3 = model.encode(sentence_3, max_length=30, truncation=True)
emb4 = model.encode(sentence_4, max_length=30, truncation=True)
print("embeddings shape = ", emb2.shape) 

cos_scores_1 = util.cos_sim(emb1, emb2)
cos_scores_2 = util.cos_sim(emb1, emb3)
cos_scores_3 = util.cos_sim(emb1, emb4)

print("cos_scores 1 shape = ", cos_scores_1.shape)
print("cos_scores 1 = ", cos_scores_1)
print("cos_scores 2 = ", cos_scores_2)
print("cos_scores 3 = ", cos_scores_3)
similarity = cos_scores_1.diag().cpu().numpy()

end_time = time.perf_counter()
elapsed_time = end_time - start_time


