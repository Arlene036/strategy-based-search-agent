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

class SearchAgentOutput(BaseModel):
    Strategy: Optional[str] # choose from ['Parallel', 'Planning', 'Direct', None]
    Action: str # choose from ['Further', 'Done']
    Result: Optional[str] # str or None
    Url: Optional[List[str]] # str or None
    Rerference: Optional[List[str]] # str or None

class SearchAgent():
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
    side_llm_openai: ChatOpenAI
    rephrase_prompt: BasePromptTemplate
    rephrase_parser: BaseOutputParser

    def __init__(self, model = 'base', raw_content = False):

        # model
        self.side_llm_openai = ChatOpenAI(model='gpt-4',temperature=0)

        if model == 'base':
            self.llm = self.side_llm
        elif model == 'openai':
            self.llm = self.side_llm_openai

        # search parameter
        self.raw_content = raw_content
        self.tavily_search = TavilySearchAPIWrapper(context_str_limit=800)
        # self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2') # TODO？

        # prompt
        self.ask_user_prompt = ChatPromptTemplate.from_template(ASK_USER_PROMPT)
        self.search_strategy_prompt = ChatPromptTemplate.from_template(SEARCH_STRATEGY_CLASSIFY_PROMPT)
        self.generating_result_prompt = ChatPromptTemplate.from_template(GENERATING_RESULT_PROMPT)
        self.rephrase_prompt = ChatPromptTemplate.from_template(SEARCH_DIRECT_REPHRASE_PROMPT)
        self.rephrase_from_memory_prompt = ChatPromptTemplate.from_template(REPHRASE_MEMORY_PROMPT)

        self.prompt_map = {
            'Parallel':SEARCH_PARALLEL_PROMPT,
            'Planning_suggestions': SEARCH_PLANNING_REACT_PROMPT_SUGGESTIONS,
            "Planning": SEARCH_PLANNING_REACT_PROMPT
        }

        # parser
        self.ask_user_parser = AskUserParser()
        self.strategy_parser = StrategySuggestionParser()
        self.parallel_question_generate_parser = GeneratedQuestionsSeparatedListOutputParser()
        self.rephrase_parser = RephraseParser()

    
    async def _run_parallel_search(self, query: str) -> Tuple[str, List[str], List[str]]:
        url_list = []
        refer_content = []
        parallel_prompt = ChatPromptTemplate.from_template(self.prompt_map['Parallel'])
        parallel_question_generate_chain = (
            parallel_prompt | 
            self.llm |
            self.parallel_question_generate_parser
        )
        generated_questions = await parallel_question_generate_chain.ainvoke({'input': query})

        if len(generated_questions) == 0:
            return "Direct", [], []
        else:
            tasks = [self.tavily_search.results_async(key_word, 
                                                      max_results=5, 
                                                      include_raw_content=self.raw_content) for key_word in generated_questions]
            results = await asyncio.gather(*tasks, return_exceptions=True) # including raw content
    
            final_reference = ""
            for key_word, result in zip(generated_questions, results):
                if self.raw_content: # TODO, this mode is not ready yet
                    most_related_content = await self._get_relevant_content(key_word, result)
                    final_reference += f"Question: {key_word}\nSearch Result:\n{most_related_content}\n"
                else:
                    final_reference += f"Question: {key_word}\nSearch Result:"
                    for i, res in enumerate(result, 1):
                        for key, value in res.items():
                            if key == 'url':
                                url_list.append(value)
                            if key == 'content':
                                refer_content.append(value)
                            final_reference += f"{key}: {value}\n"
            return final_reference, url_list, refer_content
        

    async def _run_planning_search(self, query: str, suggestions: str):
        if suggestions is not None:
            planing_react_prompt = ChatPromptTemplate.from_template(self.prompt_map['Planning_suggestions'])
        else:
            planing_react_prompt = ChatPromptTemplate.from_template(self.prompt_map['Planning'])

        search_tool = Tool(
            name='online_search',
            description='search',
            func=self.tavily_search.results
        )
        tools = [search_tool]

        planing_react_agent = create_react_agent_with_suggestions(llm=self.llm, 
                                                 prompt=planing_react_prompt, 
                                                 tools=tools,
                                                 suggestions=suggestions)
        
        planing_react_agent_executor = AgentExecutor(agent=planing_react_agent, 
                                                     tools=tools, 
                                                     max_iterations=5, 
                                                     verbose=True,
                                                     handle_parsing_errors=True,
                                                     return_intermediate_steps=True)
        
        result = await planing_react_agent_executor.ainvoke({'input':query})

        if 'intermediate_steps' in result:
            intermediate_steps = result['intermediate_steps']
        
            formatted_steps = []
            refer_url = []
            refer_content = []

            for step in intermediate_steps:
                action, observation = step
                formatted_step = f"Search Input: {action.tool_input}\nSearch Output: {observation}\n"
                print('>>>>>formatted_step>>>>>>>',formatted_step)
                print('>>>>>observation>>>>>>>',observation)
                formatted_steps.append(formatted_step)
                for o in observation:
                    if isinstance(o, dict):
                        for key, value in o.items():
                            if key == 'url':
                                refer_url.append(value)
                            if key == 'content':
                                refer_content.append(value)
                            formatted_steps = []
                    else:
                        refer_url.append('')
                        refer_content.append(o)
                    

            iteration_log = "\n\n".join(formatted_steps)
            return result['output'], iteration_log, refer_url, refer_content
        else:
            raise HTTPException(status_code=500, detail="Planning search failed")


    async def _run(self, user_query: str, has_memory: bool) -> str:

        refer_url = []

        ##################### -1.if has memory，user_query rephrase #####################
        if has_memory:
            rephrase_chain = (
                self.rephrase_from_memory_prompt | 
                self.llm |
                self.rephrase_parser
            )
            user_query = await rephrase_chain.ainvoke({'conversation': user_query})
        
        ##################### O.ask users or not ###########
        ask_user_chain = (
            self.ask_user_prompt |
            self.llm |
            self.ask_user_parser
        )
        ask_user_result = await ask_user_chain.ainvoke({'input': user_query})

        
        ##################### I.search strategy classification ##################### 
        search_strategy_chain = ( 
          self.search_strategy_prompt | 
          self.llm |
          self.strategy_parser
        )

        selected_strategy, suggestions = await search_strategy_chain.ainvoke({'input': user_query,
                                                                              'current_time': get_time()[0],
                                                                              'tomorrow_time': get_time()[1],}) 
        print('>>>>>suggestions>>>>>>>>>',suggestions)
        
        if ask_user_result['Answer'] == 'Unclear':
            return SearchAgentOutput(Strategy=selected_strategy, 
                                     Action='Further', 
                                     Result=ask_user_result['Question'], 
                                     Url=None, Rerference=None)
    
        #########  II.search strategy #################
        parallel_reference = ''
        planning_reference = ''
        direct_search_reference = ''

        # Paralle
        if selected_strategy == "Parallel":
            parallel_reference, refer_url, refer_content = await self._run_parallel_search(user_query)
            if parallel_reference == "Direct":
                selected_strategy = "Direct"
                parallel_reference = ''

        # Planning
        if selected_strategy == "Planning":
            planning_res, iteration_logs, refer_url, refer_content = await self._run_planning_search(user_query, suggestions)
            
            planning_reference = f"{iteration_logs}\n\n'Summarization:'{planning_res}\n"
            
            # return SearchAgentOutput(Strategy=selected_strategy, Action='Done', Result=planning_res, 
                                #  Url=refer_url, Rerference=refer_content)

        # Direct
        if selected_strategy == "Direct":

            rephrased_question = ''
            if suggestions is not None:
                parts = suggestions.split("'")
                if len(parts) > 1:
                    rephrased_question = parts[1]
            
            if rephrased_question == '':
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
        Search Strategy: {selected_strategy}
        Some potential questions and answers for reference are as follow:\n{parallel_reference}{planning_reference}{direct_search_reference}
        """
        rag_chain = (
            self.generating_result_prompt | 
            self.llm |
            StrOutputParser()
        )

        final_result = await rag_chain.ainvoke({'question': user_query,'context': FINAL_REFERENCE})
        return SearchAgentOutput(Strategy=selected_strategy, Action='Done', Result=final_result, 
                                 Url=refer_url, Rerference=refer_content)

        




