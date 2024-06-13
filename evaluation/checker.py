import os
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
import re
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import datetime
from langsmith import Client
import csv

os.environ.setdefault('OPENAI_API_KEY', "OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = 'LANGCHAIN_API_KEY'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d')
os.environ['LANGCHAIN_PROJECT'] = f"search_agent_autochecker_{CURRENT_TIME}"

########## SETUP ###########
t = '2024-04-18'
save_root = f'../checker/{t}'
if not os.path.exists(save_root):
    os.makedirs(save_root) 

TEST_L_O_REALITY = False
TEST_L_O_USEFUL = False
TEST_REWRITE_REALITY = True
TEST_REWRITE_USEFUL = False
TEST_S_A_REALITY = True
TEST_S_A_USEFUL = True

########## utility ##########

CHECKER_USEFULL_PROMPT = """You are to evaluate the usefullness of the results of search agent.
Evaluation Criteria:
Usefulness (1-5) - the degree to which the answer provides relevant, detailed, and enriching information to the query. 
This dimension assesses how well the answer addresses the critical aspects of the query, ensuring high relevance. 
It also evaluates the level of detail in the answer, favoring answers that delve deeply into the subject matter. 
Furthermore, usefulness includes the richness of information, emphasizing insights and details that are not commonly known, thereby significantly enhancing the overall value of the response to the reader.

Evaluation Steps:
1. Carefully read the query to identify the main topic and key points.
2. Review the answer in relation to the query. Ensure the answer is relevant by checking if it covers the main topic and key points.
3. Assess if the answer provides detailed and rich information, which means a good answer leaves no room for further questions and offers enlightening content. Conversely, a poor answer is ambiguous and prompts additional queries.
4. Assign a score for usefulness on a scale of 1 to 5, where 1 is the lowest and 5 is the highest, based on the Evaluation Criteria.

Format:
query: ...
answer: ...
thought: ...
score: ...

Examples:
query: Please analyze from the functional aspect whether the HUAWEI WATCH GT 4 smartwatch is suitable for people over 70 years old?
answer: From the functional aspect, the HUAWEI WATCH GT 4 smartwatch has a number of health functions, including heart rate monitoring, sleep tracking and GPS positioning. These functions may be very useful for people over 70 years old, especially the heart rate monitoring and sleep tracking functions, which can help them monitor their health. In addition, its long battery life also means that the elderly do not have to charge frequently. However, it should be noted that the operating system may have a certain learning curve for the elderly who are not familiar with technology products, so the user's technological proficiency should be considered before purchasing. In general, the HUAWEI WATCH GT 4 is full-featured and practical for the elderly over 70 years old, but it is necessary to decide whether it is suitable for purchase based on personal circumstances and preferences.
thought: 1. The topic of the query is whether the HUAWEI WATCH GT 4 smartwatch is suitable for the elderly. 2. The answer is very relevant: The answer answers the question of whether it is suitable for the elderly from the perspective of the watch's functions, mentions the problems that the elderly may encounter, gives suggestions, and finally has a relatively complete summary. 3. The answer is detailed and informative, covering the functions of the watch, the advantages and disadvantages of using it for the elderly, etc.
socre: 5

query: Help me analyze the mainstream milk powder on the market. I need to write a research report for the company's milk powder product procurement department. The report should include a detailed introduction to the product's price, taste, ingredients, and effects on children.
answer: According to market research, the price range of mainstream milk powder is wide, and different brands and ingredients will lead to price differences. In terms of taste, mainstream milk powder focuses on nutritional formulas, such as prebiotics and probiotics, while paying attention to brain and vision development-related ingredients required for children's development. Ingredient analysis shows that different brands of milk powder contain essential and non-essential ingredients, and the specific ingredients are slightly different. For children, mainstream milk powder is usually designed to meet their growth and development needs. Taking into account price, taste, ingredients, and effects on children, you can write a detailed research report for the company to help the procurement department make an informed decision.
thought: 1. The query requires writing a milk powder procurement research report covering price, taste, ingredients, and effects on children. 2. The answer is not highly relevant: Although the answer involves milk powder, price and taste, it does not specify the mainstream milk powder brands, nor does it provide the required brand comparison table. 3. The answer lacks detailed and rich information, and fails to provide a detailed research report or detailed information on specific milk powder.
socre: 1

query: I want to know why Zhou Hei Ya's gross profit margin is higher than Juewei Duck Neck in the past two years?
answer: The reason why Zhou Hei Ya's gross profit margin is higher than Juewei Duck Neck is mainly because Zhou Hei Ya has a higher net profit margin and better cost control in the product sales process. Although Juewei Duck Neck is slightly inferior in gross profit margin, its sales and management cost control is relatively weak, resulting in Zhou Hei Ya's stronger performance in comprehensive profitability. In addition, Zhou Hei Ya started by selling ducks, and has continuously increased its own stores with its taste advantage. Although its revenue growth is not fast, its net profit margin is high. In contrast, Juewei Duck Neck is mainly based on franchise stores, with fast revenue growth but lower profit margin than Zhou Hei Ya. Therefore, Zhou Hei Ya's ability to maintain a high gross profit margin is mainly due to its excellent cost control and net profit margin performance.
thought: 1. The keywords in the query are "in the past two years" and "Zhou Hei Ya's gross profit margin is higher than Juewei Duck Neck". 2. The answer is relevant: The answer explains why Zhou Hei Ya's gross profit margin is higher than Juewei Duck Neck by comparing the business models of Zhou Hei Ya and Juewei Duck Neck. 3. The answer is detailed and informative, covering the business models and gross profit margins of Zhou Hei Ya and Juewei Duck Neck, but does not involve the specific time information of "in the past two years".
score: 4

Now, strictly follow the format. use 'score:' to indicate the score you give to the answer.

query: {query}
answer: {answer}
thought: 
"""

CHECKER_REALITY_PROMPT = """You are to check whether the answer is realistic and feasible.
You will be given the 'query', 'answer' and 'reference', where the reference is the information from retrieval.
Evaluation Criteria:
Reality (0,1,2):
0 if the answer is against the reference, or unrealistic.
1 if the answer is unrelated to the reference but realistic.
2 if the answer contains the correct information from the reference and is realistic.

Evaluation Steps:
1. Read the query, answer, and reference carefully.
2. Compare the answer with the reference to determine if the answer is realistic.
3. Assign a score for reality based on the Evaluation Criteria.


Format:
query: ...
answer: ...
reference: ...
thought: ...
score: ...

Examples:
query: Why do people need to wear jackets instead of ordinary jackets when they go hiking?
answer: In hiking, people need to wear jackets instead of ordinary jackets, mainly because jackets have windproof and waterproof functions, which can effectively prevent the invasion of bad weather such as strong winds and rain, and keep the body dry and comfortable. In addition, jackets are usually made of lightweight and durable materials, suitable for outdoor activities such as hiking, and can provide better protection and comfort, so they are more suitable and practical in hiking. ]
thought: The answer mentions "windproof, waterproof and other functions", "keep the body dry and comfortable", "lightweight and durable", etc., which are consistent with the reference "1. Windproof and waterproof... Keep the body dry and comfortable. 2. Lightweight and durable". Therefore, the answer is consistent with the reference content, and the answer content is reasonable, with a score of 2.
score: 2

query: I want to give my bestie a bracelet as a birthday gift. I want a simple and high-end style. It is best to be a niche brand that is not easy to collide with, suitable for people with fair skin. Should I buy a pearl bracelet or a gold bracelet?
answer: For those who want to give a simple and high-end bracelet to a fair-skinned girlfriend, you can consider buying a pearl bracelet. Pearl bracelets usually show an elegant atmosphere and are suitable for matching simple and high-end styles. In addition, pearl bracelets can also show an elegant effect among people with fair skin. If you want to choose a niche brand, you can look for some less common pearl bracelet brands to ensure that it is not easy to collide with others. Although gold bracelets are also a high-end choice, pearl bracelets are more in line with gift needs and show unique taste and style. Therefore, it is recommended to choose a simple, high-end, niche brand pearl bracelet as a birthday gift.
reference: ['It is especially suitable for thin wrists to wear, and it is also very white. It makes up for the lack of delicacy in simplicity. It is especially good for wearing on the hands to enhance the temperament. It is a good gift for girlfriends. 5. Recommended for prices above 3,000 yuan (to be updated) Tiffany bracelet; Tiffany's prices are relatively high, mainly because of branded goods. I think the cost performance is not high. I recommend a few models for you to take a look. 4. JD Best Sellers', 'Little Queen Crown Bracelet. Every girl is the queen in her own mind, and she has dreamed of wearing a crown and being crowned as your queen one day. This Nordic AWNL Little Queen Crown Bracelet is exquisite and designed to bring a romantic feeling to every girl. The crown shape makes people love it as soon as they get it. It is luxurious and playful, and girls can feel it when they wear it...', '2. GLTEN. The chain is inlaid with Swarovski diamonds, which are dazzling. .]
thought: The answer shows that pearl bracelets are an elegant and unique gift, which is consistent with the reference material's emphasis on the elegance and skin-enhancing properties of various bracelets. Although the reference material does not specifically compare pearls and gold bracelets, it does describe similar high-quality options that meet the query requirements.
score: 2

Now, strictly follow the format. Continue with the following, and use 'score:' to indicate how many points you would like to give to your answer.

query: {query}
answer: {answer}
reference: {reference}
thought: 
"""

class UsefulnessParser(BaseOutputParser):
    def parse(self, text) -> Tuple[str,int]:
        _text = text.strip().lower()
        print(_text)

        try:
            pattern = re.compile(r'score: (\d)')
            score = pattern.findall(_text)[-1]
            return _text, int(score)
        except:
            try:
                pattern = re.compile(r'score:(\d)')
                score = pattern.findall(_text)[-1]
                return _text, int(score)
            except:
                return '', -1


def evaluate_usefulness(df, results_root=f'../checker/{t}', search_agent = False, l_o = False, rewrite = True):
    if search_agent:
        df = df[df['interaction_x']==0]
    
    test_id = []
    model_names = []
    usefulness_reasons = []
    usefulness_scores = []
    
    if l_o:
        results_root = os.path.join(results_root, 'l_o')
    elif search_agent:
        results_root = os.path.join(results_root, 'search_agent')
    elif rewrite:
        results_root = os.path.join(results_root, 'rewrite')
    

    result_file = os.path.join(results_root, 'usefulness_results.csv')
    with open(result_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(result_file).st_size == 0: 
            writer.writerow(['test_id', 'model_name', 'usefulness_reason', 'usefulness_score'])

        for index, row in df.iterrows():
            t_id = row['test_id']
            model_name = row['model']
            
            if search_agent:
                usefulness_reason, usefulness_score = usefulness_eval_chain.invoke({'query': row['query_x'], 'answer': row['result']})
            else:
                usefulness_reason, usefulness_score = usefulness_eval_chain.invoke({'query': row['query'], 'answer': row['result']})

            test_id.append(t_id)
            model_names.append(model_name)
            usefulness_reasons.append(usefulness_reason)
            usefulness_scores.append(usefulness_score)

            writer.writerow([t_id, model_name, 
                         usefulness_reason,
                         usefulness_score])
            
            
    result_df = pd.DataFrame({'test_id': test_id, 'model_name': model_names, 
                              'usefulness_reason': usefulness_reasons, 
                              'usefulness_score': usefulness_scores})
    

    result_file_full = os.path.join(results_root, 'usefulness_results_full.csv')
    result_df.to_csv(result_file_full, mode='a', header=False)
    
    return result_df

def evaluate_reality(df, results_root, search_agent = False, l_o = False, rewrite = True):
    if search_agent:
        df = df[df['interaction_x']==0]
    
    test_id = []
    model_names = []
    reality_reasons = []
    reality_scores = []
    
    if l_o:
        results_root = os.path.join(results_root, 'l_o')
    elif search_agent:
        results_root = os.path.join(results_root, 'search_agent')
    elif rewrite:
        results_root = os.path.join(results_root, 'rewrite')
    
    if not os.path.exists(results_root):
        os.makedirs(results_root) 

    result_file = os.path.join(results_root, 'reality_results.csv')
    with open(result_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(result_file).st_size == 0: 
            writer.writerow(['test_id', 'model_name', 'reality_reason', 'reality_score'])

        for index, row in df.iterrows():
            t_id = row['test_id']
            model_name = row['model']
            refer = row['reference']
            if search_agent:
                reason, score = reality_eval_chain.invoke({'query': row['query_x'], 'answer': row['result'], 'reference': refer})
            else:
                reason, score = reality_eval_chain.invoke({'query': row['query'], 'answer': row['result'], 'reference': refer})
            test_id.append(t_id)
            model_names.append(model_name)
            reality_reasons.append(reason)
            reality_scores.append(score)

            writer.writerow([t_id, model_name, 
                         reason,
                         score])

    result_df = pd.DataFrame({'test_id': test_id, 'model_name': model_names, 
                              'reality_reason': reality_reasons, 
                              'reality_score': reality_scores})
    
        
    result_file_full = os.path.join(results_root, 'reality_results_full.csv')
    result_df.to_csv(result_file_full, mode='a', header=False)
    
    return result_df


########## initialize models ##########
client = Client()
eval_llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),temperature=0)

usefulness_prompt = ChatPromptTemplate.from_template(CHECKER_USEFULL_PROMPT)
reality_prompt = ChatPromptTemplate.from_template(CHECKER_REALITY_PROMPT)

usefulness_eval_chain = (
    usefulness_prompt |
    eval_llm |
    UsefulnessParser()
)


reality_eval_chain = (
    reality_prompt |
    eval_llm |
    UsefulnessParser()
)


if __name__ == '__main__':
    pred_df = pd.read_csv(f'../results/search_agent/{t}/CQED.csv') # the results
    groud_truth_df = pd.read_csv('../dataset/CQED.csv') # the ground truth

    merged_df = pd.merge(pred_df, groud_truth_df, left_on='test_id', right_index=True)

    if TEST_S_A_REALITY:
        reality_eval_result = evaluate_reality(merged_df, save_root, search_agent=True, l_o=False, rewrite=False)

    if TEST_S_A_USEFUL:
        usefulness_eval_result = evaluate_usefulness(merged_df, save_root, True, False, False)

    if TEST_L_O_REALITY:
        reality_eval_result = evaluate_reality(pred_df, save_root, l_o = True, rewrite=False)

    if TEST_L_O_USEFUL:
        useful_eval_result = evaluate_usefulness(pred_df, save_root, l_o = True, rewrite=False)

    if TEST_REWRITE_REALITY:
        reality_eval_result = evaluate_reality(pred_df, save_root, search_agent=False,
                                            l_o = False, rewrite=True)

    if TEST_REWRITE_USEFUL:
        useful_eval_result = evaluate_usefulness(pred_df, save_root, search_agent=False,
                                            l_o = False, rewrite=True)
