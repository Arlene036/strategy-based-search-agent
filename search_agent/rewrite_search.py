from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from langchain.output_parsers import ListOutputParser
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from tools.tool_utils import TavilySearchAPIWrapper
from tools.tool_utils import create_react_agent_with_suggestions
import re
import json
from search_agent.parser import AskUserParser, StrategySuggestionParser, RephraseParser, GeneratedQuestionsSeparatedListOutputParser
from prompts.search_prompt import SEARCH_STRATEGY_CLASSIFY_PROMPT, SEARCH_PARALLEL_PROMPT, \
SEARCH_PLANNING_REACT_PROMPT, GENERATING_RESULT_PROMPT, SEARCH_DIRECT_REPHRASE_PROMPT, SEARCH_PLANNING_REACT_PROMPT_SUGGESTIONS, \
ASK_USER_PROMPT, REPHRASE_MEMORY_PROMPT

import asyncio
from fastapi import HTTPException
# from sentence_transformers import SentenceTransformer
import openai
import re

# Define a custom parser by extending the BaseParser class

def get_time():
    import datetime
    current_time = datetime.datetime.now()
    tomorrow_time = current_time + datetime.timedelta(days=1)

    current_time_str = current_time.strftime('%Y %b %d')
    tomorrow_time_str = tomorrow_time.strftime('%Y %b %d')

    return current_time_str, tomorrow_time_str

class RewriteAgentOutput(BaseModel):
    Result: Optional[str] # str or None
    Url: Optional[List[str]] # str or None
    Rerference: Optional[List[str]] # str or None

class RewriteAgent():
    """
    Input -> 
    understanding question and planning (chain of thoughts and few shot - for prompt choosing/generating) -> 
    [OPT]ask user or not -> 
    according to search strategy, generate questions tree (HyDE & Query Expansion), include domain recoginization: choose one of them{'general','news'} ->
    search and reranker -> 
    summarize each (each search result go for a llm call, finding most related snippet) -> 
    combine each ->
    [OPT]self-consistency check -> 
    for whole reference
    """
    tavily_search: any
    search_strategy_prompt: BasePromptTemplate
    strategy_parser: BaseOutputParser
    self_consistency_check_prompt: BasePromptTemplate
    generating_result_prompt: BasePromptTemplate
    side_llm: ChatOpenAI # for generating question tree
    side_llm_openai: ChatOpenAI
    rephrase_prompt: BasePromptTemplate
    rephrase_parser: BaseOutputParser

    def __init__(self, model = 'openai', raw_content = False):

        self.side_llm_openai = ChatOpenAI()

        if model == 'openai':
            self.llm = self.side_llm_openai

        # search parameter
        self.raw_content = raw_content
        self.tavily_search = TavilySearchAPIWrapper(context_str_limit=800)
        # self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2') # TODO？

        # prompt
        self.generating_result_prompt = ChatPromptTemplate.from_template(GENERATING_RESULT_PROMPT)
        self.rephrase_prompt = ChatPromptTemplate.from_template(SEARCH_DIRECT_REPHRASE_PROMPT)

        # parser
        self.rephrase_parser = RephraseParser()


    async def _run(self, user_query: str) -> str:

        refer_url = []

        rephrased_question = ''

        rephrase_chain = (
                self.rephrase_prompt | 
                self.llm |
                self.rephrase_parser
            )
        rephrased_question: str = await rephrase_chain.ainvoke({'input': user_query})
        
        direct_search_result = await self.tavily_search.results_async(rephrased_question, 
                                                                        max_results=5,
                                                                        include_raw_content=self.raw_content)

        direct_search_reference = 'User Query: ' + user_query + '\n'
        direct_search_reference += f"Rephrased Question: {rephrased_question}\n"

        refer_url = []
        refer_content = []
        for i, result in enumerate(direct_search_result, 1):
            direct_search_reference += f"\nResult {i}:\n"
            for key, value in result.items():
                if key == 'url':
                    refer_url.append(value)
                if key == 'content':
                    refer_content.append(value)
                direct_search_reference += f"{key}: {value}\n"

        ######## III.for whole reference ######################
        FINAL_REFERENCE = f"""User's original question is {user_query}.
        Some potential questions and answers for reference are as follow:\n{direct_search_reference}
        """
        rag_chain = (
            self.generating_result_prompt | 
            self.llm |
            StrOutputParser()
        )

        final_result = await rag_chain.ainvoke({'question': user_query,'context': FINAL_REFERENCE})
        return RewriteAgentOutput(Result=final_result, 
                                 Url=refer_url, Rerference=refer_content)

        




