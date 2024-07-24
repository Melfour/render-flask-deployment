import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker

nltk.download('punkt')

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def train_model(knowledge_base: dict):
    questions = [q['question'] for q in knowledge_base['questions']]
    labels = list(range(len(questions)))

    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    X = vectorizer.fit_transform(questions)

    model = LogisticRegression()
    model.fit(X, labels)
    
    return model, vectorizer

def correct_spelling(text: str) -> str:
    spell = SpellChecker()
    tokens = nltk.word_tokenize(text)
    corrected_tokens = []

    for token in tokens:
        candidates = spell.candidates(token)
        if candidates:
            corrected_token = spell.candidates(token).pop()
        else:
            corrected_token = token
        corrected_tokens.append(corrected_token)
    
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def get_best_response(user_query: str, vectorizer, model, knowledge_base: dict) -> str:
    corrected_query = correct_spelling(user_query)
    user_query_vec = vectorizer.transform([corrected_query])
    prediction = model.predict(user_query_vec)[0]

    distances = model.decision_function(user_query_vec)[0]
    top_indices = distances.argsort()[-3:][::-1]

    closest_questions = [knowledge_base['questions'][i]['question'] for i in top_indices]
    closest_answers = [knowledge_base['questions'][i]['answer'] for i in top_indices]

    if distances.max() < 0.5:
        return f"Sorry, it seems that I do not currently know an answer for your inquiry. The closest I can think of are for: {', '.join(closest_questions)}."

    return knowledge_base['questions'][prediction]['answer']

knowledge_base = load_knowledge_base('knowledge_base.json')
model, vectorizer = train_model(knowledge_base)
