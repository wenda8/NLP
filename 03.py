from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

generator = pipeline('text-generation', model='gpt2')
base_sentence = "After your workout, remember to"
generated_texts = generator(base_sentence, max_length=30, num_return_sequences=20)
generated_sentences = [text['generated_text'].split('\n')[0] for text in generated_texts]

model = SentenceTransformer('all-MiniLM-L6-v2')
original_sentence_embedding = model.encode("After your workout, remember to focus on maintaining a good water balance.", convert_to_tensor=True)
similar_sentences = []

for sentence in generated_sentences:
    generated_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(original_sentence_embedding, generated_sentence_embedding)
    similar_sentences.append((sentence, similarity.item()))

similar_sentences = sorted(similar_sentences, key=lambda x: x[1], reverse=True)[:5]

for sentence, _ in similar_sentences:
    print(sentence)