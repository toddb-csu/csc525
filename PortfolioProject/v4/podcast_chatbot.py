# Todd Bartoszkiewicz
# CSC525: Introduction to Machine Learning
# Portfolio Project Option #2
#
# Podcast Recommendation Chatbot
#
import nltk
import requests
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from textblob import TextBlob
from spacy.cli import download

HEADERS = {"X-ListenAPI-Key": "488139ea5ae74edb8363b22eac961aa2"}

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not found. Downloading...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))
summarizer = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")

memory = {
    "tokens": set(),
    "previous_queries": [],
    "recommended_episodes": set()
}


def update_memory(user_input, tokens):
    memory["previous_queries"].append(user_input)
    for token in tokens:
        if token not in memory["tokens"]:
            memory["tokens"].add(token)


def search_podcasts(api_query):
    params = {"q": api_query, "type": "episode", "language": "English"}
    # print(f"{params}")
    api_response = requests.get("https://listen-api.listennotes.com/api/v2/search", headers=HEADERS, params=params)
    # api_response = requests.get("https://listen-api-test.listennotes.com/api/v2/search", params=params)
    return api_response.json().get("results", [])


def rank_results(text, results):
    user_embedding = model.encode(text, convert_to_tensor=True)
    ranked_results = []

    for r in results[:10]:
        text = r["title_original"] + " " + r.get("description_original", "")
        emb = model.encode(text, convert_to_tensor=True)
        score = util.cos_sim(user_embedding, emb)[0][0]
        if (score, r) not in ranked_results:
            ranked_results.append((score, r))

    ranked_results.sort(reverse=True, key=lambda x: x[0])
    return ranked_results[:3]


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    else:
        return "neutral"


def generate_response(recommendations):
    generated_response = ""
    i = 1
    for _, r in recommendations:
        message_history = [
            {
                "role": "user",
                "content": (f"Act as the podcast expert. Recommend the podcast title of '{r['title_original']}'"
                            f" that aired on the podcast: '{r['podcast']['title_original']}'"
                            f" and summarize the description: {r['description_original']} in less than 100 words."
                            f" Write a short friendly recommendation in less than 100 words.")
            }
        ]
        generated_response += f"{i}. Podcast Recommendation\n"
        generated_response += f"=========================\n"
        generated_response += summarizer(message_history)[0]["generated_text"][-1]["content"]
        generated_response += "\n"
        i += 1

    generated_response += "Would you like recommendations with a different genre, host, or topic?"
    return generated_response


if __name__ == "__main__":
    print("Welcome to the Podcast Recommendation Chatbot")
    print("Tell me what you are interested in (i.e. 'true crime', 'health tips', etc.)")
    print("Type 'quit' to exit.")
    print("")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Goodbye")
            break

        tone = analyze_sentiment(user_input)
        if tone == "positive":
            print("Chatbot: Great! Love it")
        elif tone == "negative":
            print("Chatbot: You don't sound happy. Let's try something else.")
            print("Chatbot: Could you elaborate a little more about what you like?")
            continue

        tokens = word_tokenize(user_input.lower())
        query = [w for w in tokens if w.isalpha() and w not in stop_words]
        api_results = search_podcasts(query)
        if not api_results:
            print("Chatbot: Could you elaborate a little more about what you like?")
            continue

        ranked = rank_results(user_input, api_results)
        response = generate_response(ranked)
        print("Chatbot: You might enjoy these podcasts:\n")
        print("Chatbot: ", response)
        if memory["tokens"]:
            for t in memory["tokens"]:
                query.append(t)
            api_results = search_podcasts(query)
            if api_results:
                print("\nChatbot: Based on your history, you might also enjoy these podcasts:\n")
                ranked = rank_results(user_input, api_results)
                response = generate_response(ranked)
        update_memory(user_input, tokens)
