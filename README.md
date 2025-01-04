
# Analytics Vidhya Smart Search

## Overview
This project creates a semantic search tool for free courses available on the Analytics Vidhya platform. Users can search using keywords or natural language queries.

## Setup
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit app using `streamlit run app/app.py`.

## Technical Choices
- **Embedding Model**: `all-MiniLM-L6-v2` from sentence-transformers.
- **Vector Database**: FAISS for efficient similarity search.
- **Frontend Framework**: Streamlit for a user-friendly interface.

## How It Works
1. Scrapes course details (titles, images, and links) from the Analytics Vidhya website.
2. Uses BERT embeddings to calculate the semantic similarity between user queries and course titles.
3. Displays results in a user-friendly Streamlit interface.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repo_url>
   cd <repo_directory>
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the scraper:
   ```
   python scripts/scrape_data.py
   ```
4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

![Interface](https://github.com/user-attachments/assets/53c3f90e-e519-49a5-8bd4-5b811f7539ec)
![image](https://github.com/user-attachments/assets/2d303e5e-ddf3-43c5-aecc-d78d5c7ac9eb)
![image](https://github.com/user-attachments/assets/6f312072-6621-4c61-b2e6-244db88402be)





