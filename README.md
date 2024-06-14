# Strategy-based Search Agent

We construct a search agent to improve the human-chatbot conversation quality, especially focus on long contextual queries.

## Installation

This project uses Python 3.9, which you can install using [Anaconda](https://www.anaconda.com/products/distribution). 

To create a new Anaconda environment with Python 3.9, use the following command:

```{bash}
conda create -n search-agent python==3.9
pip install -r requirements.txt
```

## Usage

First, fill in all the API required, but only OpenAI API is necessary.
Set up a FastAPI application that handles search queries using a search agent and rewrite agent, leveraging Langchain, Langserve, and various API keys for functionality.

```{bash}
conda activate search-agent
python server_search_agent.py
```


## Evaluation

First, establish the FastAPI application at local computer which enables API call for testing.

```{bash}
conda activate search-agent
python server_search_agent.py
```

Then, use inference.py to get the results of queries in the dataset.
```{bash}
cd evaluation
python inference.py
```

Finally, use checker.py to get the evaluation results.
```{bash}
cd evaluation
python checker.py
```

Here are the results of our experiments:
| Model                          | Avg Utility | Avg Reality |
|--------------------------------|-------------|-------------|
| Rewrite Agent                  | 3.991       | 1.428       |
| LLM with Search Tool           | 3.304       | 0.940       |
| Offline Model                  | 2.850       | 0.312       |
| (Ours) Search Agent            | 4.364       | 1.648       |
| (Ours) Search Agent without Strategy Hit | 3.929 | 1.357       |
| (Ours) Search Agent with Strategy Hit    | **4.405** | **1.673** |
