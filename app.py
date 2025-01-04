import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the scraped data
DATA_FILE = "data/courses.json"


def load_data():
    """
    Load course data from the JSON file.
    """
    with open(DATA_FILE, "r") as f:
        return json.load(f)


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


@st.cache_resource
def generate_embedding(text):
    """
    Generate embeddings for the input text using BERT.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def search_courses(query, data):
    """
    Perform course search based on query similarity to course titles/descriptions.
    """
    query_embedding = generate_embedding(query)

    # Create a 2D array for course embeddings
    course_embeddings = np.array(
        [
            generate_embedding(course["title"] + " " + course.get("description", ""))
            for course in data
        ]
    )

    # Ensure all embeddings are 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()

    # Assign similarity scores to courses
    for i, course in enumerate(data):
        course["score"] = similarities[i]

    # Return top 10 results sorted by similarity score
    return sorted(data, key=lambda x: x["score"], reverse=True)[:10]


def main():
    """
    Main function to render the Streamlit app.
    """
    # Add custom CSS for new styling
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #121212;
        }
        .title {
            text-align: center;
            font-size: 3.5rem;
            color: #00bcd4;
            margin-top: 20px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: #a7ffeb;
            margin-bottom: 30px;
        }
        .search-box {
            text-align: center;
            margin: 20px auto;
        }
        .stTextInput input {
            background-color: #2c2c2c;
            color: #FFFFFF;
            border: 1px solid #00bcd4;
            border-radius: 5px;
            padding: 10px;
        }
        .result-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            background-color: #1e1e1e;
            color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s;
            width: 85%;  /* Adjusted to allow two cards per row */
            overflow: hidden;
        }
        .card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        }
        .card img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-bottom: 2px solid #00bcd4;
        }
        .card-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #00bcd4;
            margin: 20px 30px;
            text-decoration: none;
        }
        .card-title a {
            color: inherit;
            text-decoration: none;
        }
        .card-title a:hover {
            color: #a7ffeb;
            text-decoration: underline;
        }
        .card-desc {
            font-size: 0.9rem;
            color: #b0bec5;
            margin: 20px 30px;
        }
        .score {
            font-size: 1rem;
            color: #ffeb3b;
            margin: 10px 15px;
            font-weight: bold;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Page Title
    st.markdown(
        '<h1 class="title">Analytics Vidhya Smart Search</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Find your next learning adventure with relevant, free courses..</p>',
        unsafe_allow_html=True,
    )

    # Search Input
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    query = st.text_input(
        "Search for a course", placeholder="e.g., Machine Learning, Python"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if query:
        data = load_data()
        results = search_courses(query, data)

        if results:
            st.markdown(
                f"<h3>Showing results for: {query}</h3>", unsafe_allow_html=True
            )

            # Display results in a responsive grid (2 cards per row)
            for i in range(0, len(results), 2):  # 2 cards per row
                cols = st.columns(2)  # Create two columns for each row of cards
                for idx, result in enumerate(results[i : i + 2]):
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div class="card">
                                <img src="{result['image_url']}" alt="Course Image">
                                <div class="card-title">
                                    <a href="{result['course_link']}" target="_blank">{result['title']}</a>
                                </div>
                                <div class="card-desc">{result.get('description', 'No description available.')}</div>
                                <div class="score">Relevance Score: {round(result['score'] * 100, 2)}%</div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
        else:
            st.write("No courses found. Please try a different query.")


if __name__ == "__main__":
    main()
