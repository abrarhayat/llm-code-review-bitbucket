# Environment Variables

This project uses environment variables for configuration. These variables should be stored in a file named `.env` at the root of the project.

## Setup

1. Copy the `.env.example` file and create a new file named `.env` in the same directory.

    ```bash
    cp .env.example .env
    ```

2. Open the `.env` file and replace the `XXXX` placeholders with your actual values.

    ```shell
    BITBUCKET_URL=your_bitbucket_url
    BITBUCKET_API_TOKEN=your_bitbucket_api_token
    PROJECT_KEY=your_project_key
    REPOSITORY_SLUG=your_repo_name
    ```

## Variables

- `BITBUCKET_URL`: The URL of your Bitbucket instance.
- `BITBUCKET_API_TOKEN`: Your Bitbucket API token.
- `PROJECT_KEY`: The key of your project in Bitbucket.
- `REPOSITORY_SLUG`: The repository name in Bitbucket.
