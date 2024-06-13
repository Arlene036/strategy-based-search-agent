SEARCH_Q_GEN_PROMPT="""
You are a Chinese detailed question generator. You will receive a question from a user. You need to determine which type of question generation strategy the question belongs to, and then generate multiple questions similar to the question to form a Question List based on the generation strategy.

The question generation strategy is as follows: When the user's question explicitly mentions multiple parallel concepts, split the parallel concepts and search them separately. When the user's question has a planning intention, you need to sort out the ideas and split the concepts first, and then search.

You need to strictly output in the format, which is as follows:
Ideas:...
Generated questions:
1. ...
2. ...

Example:
User question: I plan to return to Shanghai from Shenzhen. Which one is more cost-effective, airplane or high-speed rail?
Idea: There are two parallel concepts in the question, airplane and high-speed rail. You can separate them and search for airplanes or high-speed rail from Shenzhen to Shanghai.
Generated questions:
1. Flights and airfares from Shenzhen to Shanghai
2. High-speed rail and ticket prices from Shenzhen to Shanghai
User question: I plan to play in Shenzhen for 3 days. Please help me make a cost-effective Shenzhen travel guide.
Idea: Points to consider for tourism include accommodation, travel and scenic spot planning.
Generated questions:
1. Cost-effective hotels in Shenzhen
2. Recommended travel methods for Shenzhen tourism
3. Recommended attractions in Shenzhen
User question: What are the recent news about artificial intelligence?
Idea: AI is a parallel concept and can be split for search.
Generated questions:
1. What are the recent news related to artificial intelligence?
2. What are the recent news about natural language processing?
3. What are the recent news about deep learning?
User question: {input}
Idea:
"""

SERP_SEARCH_TOOL_PROMPT = """
online search for simple questions, like a concept searching (e.g. how is the weather, what is photosynthesis).
Input the question and output the context for reference.
Because you are a pre-trained model, the training data stays in historical time. When users ask about time-varying information such as dates, recent news and weather, you need to use the online search tool.
This online search tool is suitable for answering simple questions, such as conceptual questions (such as weather conditions, definition of photosynthesis).
When users input simple and direct questions such as "What is the weather today?", "What is photosynthesis?", "What is the highest mountain in the world?", etc., they can search for direct answers in one search and use this tool.
This tool will provide relevant contextual information for reference.
"""

HIERARCHY_SEARCH_PROMPT = """
online search for complex questions.
Because you are a pre-trained model, the training data stays in historical time. When users ask about time-varying information such as dates, recent news and weather, you need to use the online search tool.
This online search tool is specially designed to solve complex problems, such as making travel plans and understanding complex concepts.
When users enter complex and multi-faceted questions such as "How should I plan a trip to Europe?", "What is the difference between photosynthesis and respiration?", "How to prepare for a successful business meeting?", etc., use this advanced search tool.
This tool will provide relevant contextual information for reference.
"""

SEARCH_STRATEGY_CLASSIFY_PROMPT = """Given a user's query, your task is to determine which of the following three search strategies should be applied: Parallel, Planning, or Direct.
Parallel: The query explicitly mentions multiple parallel concepts.
Planning: The query requires a sequence of searches, where each step's inquiry depends on the information obtained from the previous search.
Direct: The query asks about a clear, singular concept, conduct a direct search.

Reasoning Method:
Step 1: Extract the key concepts from the user's query.
Step 2: Identify if the query necessitates a sequential approach where the outcome of one search dictates the direction of the next. This is indicative of a "Planning" strategy.
Step 3: Decide whether the key concepts are parallel and require separate searches, or if there is a single concept that can be directly searched.
Step 4: Give searching suggestions based on the determined strategy. If the query involves "current time", include the current time in the suggestions. 

Current time is {current_time}. 
If user query involves key words like 'today今天' or 'tomorrow明天', you should replace those words with true time.
For example, if the user query is "What is the weather like in New York tomorrow?", and today is 2024 Apr 2, you should replace 'tomorrow' with the 2024 Apr 3.

You should strictly follow the format as the example.
Examples:

Query: What are the benefits of yoga and meditation?
Strategy: Parallel
Reasoning: The key concepts are 'yoga' and 'meditation'. The query mentions two parallel concepts, 'yoga' and 'meditation'. They should be searched for separately to provide comprehensive information on each.
Suggestions: Search for the benefits of yoga and the benefits of meditation.

Query: What should I prepare for my hiking trip on LA next week?
Strategy: Planning
Reasoning: This involves a series of searches where each step depends on the previous one. The first step could be to check the current weather forecast for the hiking location. Depending on the forecast, the next step might involve searching for safety tips for hiking in those specific conditions, followed by a search for the necessary gear. Each step requires information obtained from the previous step to make informed decisions.
Suggestions: Current time is 2024 Apr 2. Check the weather forecast for LA from Apr 7 to Apr 13, then search for safety tips for hiking in those conditions, and finally, look for a list of essential gear for hiking.

Query: 在当前全球经济形势下，黄金价格走势如何？
Strategy: Direct
Reasoning: The key concept is '黄金价格走势'. The query asks about a clear, singular concept, '黄金价格走势'. A direct search will suffice to provide the required information.
Suggestions: Search for '黄金价格走势'.

Query: What is photosynthesis?
Strategy: Direct
Reasoning: The key concept is 'photosynthesis'. The query asks about a clear, singular concept, 'photosynthesis'. A direct search will suffice to provide the required information.
Suggestions: Search for 'photosynthesis'.

Query: What causes aurora borealis?
Strategy: Direct
Reasoning: The key concept is 'aurora borealis'. The query is specific, asking for the cause behind a single, well-defined phenomenon, 'aurora borealis'. A direct search will yield a concise explanation of the natural mechanisms that produce the northern lights.
Suggestions: Search for 'aurora borealis cause'.

Query: How is Shenzhen's weather tomorrow?
Strategy: Direct
Reasoning: This query is about a specific single question. It involves time and today is {current_time}, so tomorrow is {tomorrow_time}. Search for 'Shenzhen weather forecast {tomorrow_time}'.
Suggestion: Search for 'Shenzhen weather forecast {tomorrow_time}'.

Query: {input}
"""


SEARCH_PARALLEL_PROMPT = """
You are given a question user cares about. However, if this question is entered directly into the search engine, it may be difficult to find useful answers. 
Please help me generate a link using the search engine, that is, multiple questions that need to be entered into the search engine.
Your task is to break down the initial inquiry into multiple more specific questions that can be effectively used in a search engine to gather relevant information.
These questions should appear in the form of keywords or declarative sentences as much as possible, rather than questions with what, when, and how.

Generated questions should be in the same language as the query. If query is in Chinese, you should output Chinese. But \"Generated Questions:\" should be in English and serve as a header for the list of generated questions.

Example:
Query: Compare the health benefits of running and swimming.
Thoughts: The query mentions two parallel concepts: 'running' and 'swimming'. We need to split these concepts.
Generated Questions:
1. Health benefits running
2. Health benefits swimming

Query: Evaluate the nutritional values of apples versus oranges and their impact on digestion.
Thought: This query intertwines two major parallel concepts: 'apples' and 'oranges', asking their separate nutritional values and impact on digestion. The task is to separate these intertwined concepts.
Generated Questions:
1. Nutritional values apples
2. Nutritional values oranges
3. Apples digestion impact
4. Oranges digestion impact

Query: I have a spinal disease and want to buy a Simmons mattress for home use. Do you think a spring mattress is better or a latex mattress is better?
Thought: Directly entering the scenario question into the search engine will not get good results. Keywords should be extracted and rewritten into concise questions. The keyword here is "spinal disease", and there are two parallel keywords, spring mattress and latex mattress. We need to split these concepts.
Generated Questions:
1. Mattress selection for patients with spinal cord disease
2. The impact of spring mattresses on the spine
3. The impact of latex mattresses on the spine
4. Comparison between spring mattresses and latex mattresses

Now, follow the chain of thoughts method to identify the parallel concepts, generate separate search terms or statements for each, and structure your response accordingly. 
Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words.
You should strictly follow the format as the example. 
Query: {input}
"""

SEARCH_PLANNING_REACT_PROMPT_SUGGESTIONS = """
Your task is to conduct a series of sequential searches. 
After each search, assess the results to decide whether further information is needed or if the search can conclude.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, you should input Chinese.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: (Here you should output Chinese) the final answer to the original input question, this should be in very detailed and include the summarization of all observation above. 

Begin!
Suggestions for multiple searches: {suggestions}
Question: {input}
Thought: {agent_scratchpad}
"""

SEARCH_PLANNING_REACT_PROMPT = """
Your task is to conduct a series of sequential searches. 
After each search, assess the results to decide whether further information is needed or if the search can conclude.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, you should input Chinese.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: (Here you should output Chinese) the final answer to the original input question, this should be in very detailed and include the summarization of all observation above. 

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

SEARCH_DIRECT_PROMPT = """
You are given a query that asks about a clear, singular concept. Your task is to conduct a direct search to find the answer.
"""

SEARCH_DIRECT_REPHRASE_PROMPT = """Your task is to rephrase the complicated question into a better and concise question, but also contains all the key information, suitable for direct input into the search engine for query to obtain high-quality results.
Follow the format below:
Query: the original question
Rephrased Question: the rephrased question
Key Words: the key words are the topic, usually are broader concepts or higher-level concepts (in English)

Noting that the rephrased question should be in the same language as the original question.
Example:
Query: As a game enthusiast, a good monitor is essential. I want to change to a 34-inch monitor with a screen resolution of 3440x1440. You can find a suitable monitor in Dell according to my requirements?
Rephrased Question: 34-inch 3440x1440 monitor Dell
Key Words: monitor, game, electronic product
Query: My parents and I are traveling in Sichuan. We just finished breakfast and are going to take the subway. Do you have any recommended tourist attractions?
Rephrased Question: Tourist attractions near the subway in Sichuan when traveling with my family in the morning
Key Words: Travel, subway, recommendation
Query: I have a spinal disease and want to buy a Simmons mattress for home use. Do you think a spring mattress or a latex mattress is better?
Rephrased Question: Should people with spinal diseases use a spring mattress or a latex mattress?
Key Words: Shopping, health
Query: {input}
"""



GENERATING_RESULT_PROMPT = """
You are given a user question, and please write clean, concise and accurate answer to the question.
Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. 
Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Context for reference: {context} 

Now answer the question based on the reference context and your own knowledge. 
However, if context for reference is useless, then you should answer the question based on your own knowledge. DO NOT expose that the search results are false or useless.
The answer should be well-founded, detailed and has results and reasoning process. Your answer should be in the same language as the question.

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
{question} 
Answer:
"""


ASK_USER_PROMPT = """You are going to assist with online searches for user inquiries. 
If the user asks a particularly vague question and you need to ask the user further to know how to answer, return Unknown.
If the question is relatively clear, you can guess the user's situation and answer Clear.

If the question is unclear, please point out what you think is unclear to gather more information.
If the question is clear, you can directly answer it.

You will receive a user question, and you need to strictly follow the output format below:
Clear Score: ... (an integer between 0 and 10, 0 means completely unclear, 10 means completely clear)
Question: ... (If the question is unclear, ask a question to clarify the user's question; if the question is clear, leave it as None)
Your question must be in the same language as the user's question.

Here are some examples:
User question: What will the weather be like tomorrow?
Clear Score: 0
Question: Could you please specify which city's weather you are inquiring about?

User question: How can I win back my girlfriend's heart?
Clear Score: 1
Question: Could you please share what happened between you and your girlfriend?

User question: How is the XGIMI H6 projector? Which XGIMI projector is worth buying? Please recommend it to me.
Clear Score: 8
Question: What is your budget?

User question: Compared with the Changdi F40S1 and other ovens of the Changdi brand, which oven is easier to operate and more suitable for the elderly who are not good at using electrical equipment?
Clear Score: 9
Question: None

User question: What is photosynthesis?
Clear Score: 10
Question: None

User question: Planning a three-day and two-night trip to Xi'an.
Answer: 7
Question: What is your budget?

User question: {input}
"""


REPHRASE_MEMORY_PROMPT = """Here is a conversation dialog between user and assistant. User asks a question and the assistant finds it is unclear so asks more detailed questions to clarify the user's question.
Your task is to rephrase users' question according to the conversation dialog. You should rephrase the question in a more detailed and clear way.

For example:

User: What is the weather?
Assistant: Could you please specify which city's weather you are inquiring about?
User: New York
Rephrased Question: What is the weather in New York?
User: I plan to travel for 5 days, and my budget is 2000 RMB per day. Others are up to you
Assistant: Do you have any specific destinations or travel preferences?
User: I plan to travel to Xiamen for 5 days, where there are fewer people
Assistant: Do you need any suggestions on accommodation, transportation, food, or attractions?
User: All
Assistant: What is your budget for accommodation?
User: Whatever
Rephrased Question: I plan to travel to Xiamen for 5 days, where there are fewer people, and my budget is 2000 RMB per day. Please give me suggestions on accommodation, transportation, food, and attractions in Xiamen.

Now here is the conversation dialog:
{conversation}
"""