from collections import defaultdict
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

def get_files_from_pull_request(bitbucket: Bitbucket, project_key: str,
                                repository_slug: str, pull_request_id: str,
                                file_types: list = None) -> dict:
    """
    Fetches the content of all files in a given pull request.

    Args:
        bitbucket (Bitbucket): An instance of the Bitbucket class.
        project_key (str): The key of the project in Bitbucket.
        repository_slug (str): The slug (URL-friendly version of the name) 
        of the repository in Bitbucket.
        pull_request_id (int): The ID of the pull request in Bitbucket.
        file_types (list): A list of file types to filter the files by.

    Returns:
        dict: A dictionary where the keys are the filenames 
        and the values are the respective file contents.
    """
    pull_request = bitbucket.get_pull_request(project_key, repository_slug, pull_request_id)
    latest_commit_id = pull_request['fromRef']['latestCommit']
    pr_changes = bitbucket.get_pull_requests_changes(project_key, repository_slug, pull_request_id)
    pr_files = defaultdict()
    for change in pr_changes:
        file_name = change['path']['toString']
        if(file_types and file_name.split('.')[-1] not in file_types):
            continue
        pr_files[file_name] = bitbucket.get_content_of_file(project_key, repository_slug, file_name, latest_commit_id) 
    return pr_files
