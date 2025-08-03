import json
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

def create(x):
    """
    Processes the loaded JSON data (assumed to be a dictionary of author PIDs to info).
    Prints the first author ID found and their column names.
    """
    if x is None:
        print("Error: No data provided to the 'create' function.")
        return
    if not isinstance(x, dict):
        print("Error: Expected the loaded JSON data to be a dictionary.")
        return
    if not x: # Check if the dictionary is empty
        print("Error: The provided JSON data is empty.")
        return

    try:
        # Get the first author ID
        first_author_id = list(x.keys())[0]

        # Get the dictionary for the first author
        first_author_data = x[first_author_id]
        if isinstance(first_author_data, dict):
            column_names = list(first_author_data.keys())
            print("The 'column names' for the first author's data are:", column_names)
        else:
            print(f"Warning: Data for '{first_author_id}' is not a dictionary. Cannot determine column names.")

        print(f"Total number of authors/entries in the JSON: {len(x)}")

    except IndexError:
        print("Error: The JSON data is empty or does not contain any top-level keys.")
    except Exception as e:
        print(f"An error occurred in the 'create' function: {e}")

def venuelist(x):
    venues = set()
    sample_data = []
    author_names = [
    author['name']
    for pub_data in x.values()
    if 'ICDE' in (pub_data.get('venue') or '').upper()
    for author in pub_data.get('authors', [])
    ]
    return author_names

def find_papers_by_author(dataset, author_name):

    target_name_lower = author_name.lower()

    return [
        # This is the only line that changed
        pub_data.get('title', 'Untitled Paper')
        for pub_data in dataset.values()
        if any(
            author.get('name', '').lower() == target_name_lower
            for author in pub_data.get('authors', [])
        )
    ]
def main():
    data = open_file() # Call open_file and store the returned data
    # print(venuelist(data))
    print(find_papers_by_author(data, 'huan lin'))


if __name__ == "__main__":
    main()