from langchain_core.runnables import RunnableLambda
from typing import Dict, List
# from langchain.memory import ConversationBufferMemory
from search_agent.search_agent import SearchAgent, SearchAgentOutput
from search_agent.rewrite_search import RewriteAgent, RewriteAgentOutput
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from langsmith import Client
from langserve import add_routes
import datetime
import os
os.environ["GOOGLE_CSE_ID"] = "GOOGLE_CSE_ID"
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'LANGCHAIN_API_KEY' 

os.environ["TAVILY_API_KEY"] = "tvly-TAVILY_API_KEY" 

os.environ["LLMONITOR_APP_ID"] = "LLMONITOR_APP_ID"
os.environ.setdefault('OPENAI_API_KEY', "OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = "SERPAPI_API_KEY"
os.environ['SUPABASE_URL'] = "https://zmcygznfjkuqvcppkaxz.supabase.co"
os.environ['SUPABASE_KEY'] = "SUPABASE_KEY"

client = Client()

RAW_CONTENT = False
MODEL = 'openai'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d')

os.environ['LANGCHAIN_PROJECT'] = f"search_agent_{MODEL}_raw-content_{RAW_CONTENT}_{CURRENT_TIME}"

app = FastAPI()
search_agent = SearchAgent(model = MODEL, raw_content = RAW_CONTENT)
rewrite_agent = RewriteAgent()

class Input(BaseModel):
    query: str
    conversation_id: int

# load for conversation memory
conversation_memories_dict: Dict[int, List[str]] = {}

def get_memory_for_conversation(conversation_memories: Dict[int, List[str]], conversation_id: int) -> List[str]:
    if conversation_id not in conversation_memories:
        conversation_memories[conversation_id] = []
    return conversation_memories[conversation_id]


async def run_search(input: Input):
    try:
        query = input['query']
        has_memory = False
        conversation_memory: List[str] = get_memory_for_conversation(conversation_memories_dict, input['conversation_id'])
        if len(conversation_memory) > 0:
            query = conversation_memory + ['User: {}'.format(input['query'])]
            query = '\n'.join(query)
            has_memory = True

        result: SearchAgentOutput = await search_agent._run(user_query=query, has_memory = has_memory) 

        if result.Action == "Further":
            conversation_memories_dict[input['conversation_id']].append('User: {}'.format(input['query']))
            conversation_memories_dict[input['conversation_id']].append('Assistant: {}'.format(result.Result))
            return result.Result
        else:
            return result.Result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_search_test_mode(input: Input) -> SearchAgentOutput:
    try:
        query = input['query']
        has_memory = False
        conversation_memory: List[str] = get_memory_for_conversation(conversation_memories_dict, input['conversation_id'])
        if len(conversation_memory) > 0:
            query = conversation_memory + ['User: {}'.format(input['query'])]
            query = '\n'.join(query)
            has_memory = True

        result: SearchAgentOutput = await search_agent._run(user_query=query, has_memory = has_memory)

        if result.Action == "Further":
            conversation_memories_dict[input['conversation_id']].append('User: {}'.format(input['query']))
            conversation_memories_dict[input['conversation_id']].append('Assistant: {}'.format(result.Result))
            return result
        else:
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def run_rewrite_search_test_mode(input: Input) -> RewriteAgentOutput:
    try:
        result: RewriteAgentOutput = await rewrite_agent._run(user_query=input['query'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

add_routes(
    app,
    RunnableLambda(run_search),
    path="/search"
)

add_routes(
    app,
    RunnableLambda(run_search_test_mode),
    path="/search_test_mode"
)

add_routes(
    app,
    RunnableLambda(run_rewrite_search_test_mode),
    path="/rewrite_search_test_mode"
)

if __name__ == "__main__":
    uvicorn.run("server_search_agent:app", host="0.0.0.0", port=8002, reload=True)
