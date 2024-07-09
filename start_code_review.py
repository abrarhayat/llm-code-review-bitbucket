import os
from collections import defaultdict
import argparse
from dotenv import load_dotenv
from atlassian import Bitbucket
from bitbucket_api import get_all_file_content, get_files_from_pull_request
from ai_code_review import review, start_console_chat_with_llm


def main():
    """
    Main function to initiate the ai code review script.
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
    pull_request_id = os.getenv("PULL_REQUEST_ID")
    file_types = args.file_types if args.file_types else None
    all_file_content = defaultdict()
    if(pull_request_id):
        all_file_content = get_files_from_pull_request(bitbucket, project_key,
                                                      os.getenv("REPOSITORY_SLUG"),
                                                      pull_request_id, file_types)
    else:
        all_file_content = get_all_file_content(bitbucket, project_key,
                                                 os.getenv("REPOSITORY_SLUG"), file_types)
    code_review_model = os.getenv("AI_MODEL")
    code_review_max_tokens = int(os.getenv("AI_MAX_TOKENS"))
    for filename, file_content in all_file_content.items():
        print(f"Reviewing {filename}...")
        code_review = review(filename, file_content, code_review_model, 0.1, code_review_max_tokens, None)
        print(code_review)
    start_console_chat_with_llm()

if __name__ == "__main__":
    main()