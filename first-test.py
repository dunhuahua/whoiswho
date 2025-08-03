import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def open_file():
    """
    Loads JSON data from 'pid_to_info_all.json' with UTF-8 encoding.

    Returns:
        dict or list: The loaded JSON data if successful, None otherwise.
    """
    try:
        # Open the file with UTF-8 encoding
        with open("pid_to_info_all.json", "r", encoding="utf-8") as file:
            x = json.load(file)
        print("JSON data loaded successfully!")
        return x # Return the loaded data
    except FileNotFoundError:
        print("Error: The file 'pid_to_info_all.json' was not found. Please check the file path and name.")
        return None # Return None on error
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. The file might be corrupted or not valid JSON.")
        return None
    except UnicodeDecodeError:
        print("Error: UnicodeDecodeError. The file might not be UTF-8 encoded, or contains invalid characters.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def run_author_prediction(data, X_values_to_test, num_test_papers=500):
    if not data or not isinstance(data, dict):
        print("Invalid data provided. Aborting.")
        return

    # 1. Adapt Data Structure: Convert the main dictionary to a list for consistent indexing.
    papers_list = list(data.values())
    # Ensure we don't test more papers than we have
    num_test_papers = min(num_test_papers, len(papers_list))
    if num_test_papers == 0:
        print("No papers to test.")
        return
    print(f"Total papers loaded: {len(papers_list)}")
    print(f"Running evaluation on the first {num_test_papers} papers...")

    print("Creating text vectors with TF-IDF...")
    corpus = [p.get('title', '') + ' ' + p.get('abstract', '') for p in papers_list]
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=3, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("Vectorization complete.")
    print("\n--- Model Performance ---")
    results = {}
    for X in X_values_to_test:
        hits = 0
        # Loop through 
        for i in range(num_test_papers):
            # Find the top X similar papers (excluding the paper itself)
            sim_scores = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
            top_x_indices = np.argsort(sim_scores)[::-1][1:X+1]
            # Gather candidate authors from the top X similar papers
            candidate_authors = set()
            for idx in top_x_indices:
                paper = papers_list[idx]
                # Correctly extract author names from the list of author dictionaries
                authors_of_paper = {author['name'] for author in paper.get('authors', [])}
                candidate_authors.update(authors_of_paper)
            
            # Get the actual authors of the target paper
            actual_authors = {author['name'] for author in papers_list[i].get('authors', [])}

            # Check for a "hit" 
            if actual_authors.intersection(candidate_authors):
                hits += 1

        hit_rate = (hits / num_test_papers) * 100
        results[X] = hit_rate
        print(f"Hit Rate @ {X}: {hit_rate:.2f}%")
        
    return results
def main():
    x = open_file()
    X_values = [10, 50, 100, 200]
    run_author_prediction(x, X_values)

main()