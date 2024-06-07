from collections import defaultdict
import os
import argparse
from dotenv import load_dotenv
from atlassian import Bitbucket

def get_all_file_content(bitbucket: Bitbucket, project_key: str, 
                         repository_slug: str, file_types: list = None) -> dict:
    """
    Fetches the content of all files in a given Bitbucket repository.

    Args:
        bitbucket (Bitbucket): An instance of the Bitbucket class.
        project_key (str): The key of the project in Bitbucket.
        repository_slug (str): The slug (URL-friendly version of the name) 
        of the repository in Bitbucket.

    Returns:
        dict: A dictionary where the keys are the filenames 
        and the values are the respective file contents.
    """
    all_repo_file_names = bitbucket.get_file_list(project_key, repository_slug)
    if file_types:
        all_repo_file_names = [filename for filename
                               in all_repo_file_names if filename.split('.')[-1] in file_types]
    all_repo_file_content = defaultdict()
    for filename in all_repo_file_names:
        file_content = bitbucket.get_content_of_file(project_key, repository_slug, filename)
        all_repo_file_content[filename] = file_content
    return all_repo_file_content


def main():
    """
    Main function to run the script.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Process some file types.")
    parser.add_argument("file_types", nargs="*", help="File types to process.")
    args = parser.parse_args()
    bitbucket = Bitbucket(
        url=os.getenv("BITBUCKET_URL"),
        token=os.getenv("BITBUCKET_API_TOKEN")
    )
    project_key = os.getenv("PROJECT_KEY")
    file_types = args.file_types if args.file_types else None
    all_repo_file_content = get_all_file_content(bitbucket, project_key, 
                                                 os.getenv("REPOSITORY_SLUG"), file_types)
    for filename, file_content in all_repo_file_content.items():
        print("Content of file: " + filename)
        print(file_content)
        print("\n\n")

if __name__ == "__main__":
    main()
