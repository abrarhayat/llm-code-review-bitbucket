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

3. Download and Install Ollama if you're planning to use Ollama AI models. You can find the installation instructions on the official Ollama repository [here](https://ollama.com/download).

   Alternatively, you can use OpenAI's APIs. If you choose this option, make sure to provide your OpenAI API key in the `.env` file.

4. If you're using Ollama, start the server by running the following command in your terminal before proceeding to the next step.

    ```bash
    ollama serve
    ```

5. Replace the remaining variables for LLM

    ```shell
    OPENAI_API_KEY=Your OpenAI API key. This is required if you're using OpenAI's APIs.
    OLLAMA_URL=The URL of your Ollama server. The default value is http://localhost:11434/api/chat. This is required if you're using Ollama as your AI model.
    AI_MODEL=The AI model to use. The default is mistral. This can be either mistral for using Ollama or openAI models for using OpenAI's APIs.
    AI_MAX_TOKENS=The maximum number of tokens to generate in a single API call. The default value is 10000. This is used only when AI_MODEL is set to openAI.
    ```

6. Install the required packages in your current Python environment or virtual environment

    ```python
    pip install -r requirements.txt
    ```

7. Run the following to initiate the code AI review process:

    ```python
    python start_code_review.py
    ```

## Variables

- `BITBUCKET_URL`: The URL of your Bitbucket instance.
- `BITBUCKET_API_TOKEN`: Your Bitbucket API token.
- `PROJECT_KEY`: The key of your project in Bitbucket.
- `REPOSITORY_SLUG`: The repository name in Bitbucket.
- `OPENAI_API_KEY`: Your OpenAI API key. This is required if you're using OpenAI's APIs.
- `OLLAMA_URL`: The URL of your Ollama server. The default value is http://localhost:11434/api/chat. This is required if you're using Ollama as your AI model.
- `AI_MODEL`: The AI model to use. The default value is mistral. This can be either mistral for using Ollama or openAI for using OpenAI's APIs.
- `AI_MAX_TOKENS`: The maximum number of tokens to generate in a single API call. The default value is 10000. This is used only when AI_MODEL is set to openAI.
