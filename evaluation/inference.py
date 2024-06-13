import pandas as pd
import requests
import time
import os
import csv

# URLs
SEARCH_AGENT_URL = 'http://localhost:8002/search_test_mode/invoke'
REWRITE_AGENT_URL = 'http://localhost:8002/rewrite_search_test_mode/invoke'
DIRECT_SEARCH_MODEL_URL = 'http://localhost:7003/tavily-search/invoke'
OFFLINE_MODEL_URL = 'http://localhost:7003/offline/invoke'

data = pd.read_csv('CQED.csv')
today = time.strftime("%Y-%m-%d", time.localtime())

############### Utility Functions ##############
def fetch(url, payload=None):
    print(f"Sending request to {url} with payload {payload}")
    try:
        response = requests.post(url, json=payload)
        response_json = response.json()
        print(f"Response from {url}: {response_json}")
        return response_json
    except Exception as e:
        print(f"Error with request to {url}: {e}")
        raise

# Search Agent Functions
def test_search_agent(query, test_id):
    input_payload = {"input": {"query": query, "conversation_id": test_id}}
    output_json = fetch(SEARCH_AGENT_URL, payload=input_payload)
    return output_json['output']

# Rewrite Agent Functions
def test_rewrite_agent(query, test_id):
    input_payload = {"input": {"query": query, "conversation_id": test_id}}
    output_json = fetch(REWRITE_AGENT_URL, payload=input_payload)
    return output_json['output']

def test_question(test_id, question, count, writer): # Direct and Offline Model Functions
    input_payload = {
        "input": {
            "input": {"input": question, "agent_name": "a", "user_name": "b"},
            "config": {"agent_id": "1", "conversation_id": count, "user_id": "2"}
        }
    }

    # Direct Search Model
    try:
        direct_search_response = fetch(DIRECT_SEARCH_MODEL_URL, input_payload)
        reference = direct_search_response['output'].split('tavily_search_results_limited with output: [')[1].split(']')[0]
        writer.writerow(["Direct Search Model", test_id, question, direct_search_response['output'].split('agent with output:')[1], reference])
    except (KeyError, IndexError):
        print(f"Error with Direct Search Model for question: {question}, skipping this test case.")

    # Offline Model
    try:
        offline_response = fetch(OFFLINE_MODEL_URL, input_payload)
        writer.writerow(["Offline Model", test_id, question, offline_response['output'].split('agent with output:')[1], 'None'])
    except (KeyError, IndexError):
        print(f"Error with Offline Model for question: {question}, skipping this test case.")

    time.sleep(3)

def read_existing_questions(filepath):
    existing_questions = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                existing_questions.add(row[1])
    except FileNotFoundError:
        print("Results file not found, creating a new one.")
    return existing_questions

############### testing ##############

def batch_testing_search_agent(data, log_root, save_root):
    result_df = pd.DataFrame(columns=['test_id', 'model', 'query', 'strategy', 'result', 'interaction', 'reference', 'url'])

    for index, row in data.iterrows():
        try:
            test_id = index
            query = row['query']

            print(f'Start testing index: {index} ...')
            output = test_search_agent(query, test_id)

            with open(log_root, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([test_id, 'search_agent', query, output['Strategy'], output['Result'],
                                 1 if output['Action'] == 'Further' else 0, output['Rerference'], output['Url']])

            result_df = result_df.append({'test_id': test_id, 'model': 'search_agent', 'query': query,
                                          'result': output['Result'], 'strategy': output['Strategy'],
                                          'interaction': 1 if output['Action'] == 'Further' else 0,
                                          'reference': output['Rerference'], 'url': output['Url']}, ignore_index=True)
            time.sleep(3)
        except Exception as e:
            print(f'Error in index: {index}, {e}')
            continue

    result_df.to_csv(save_root, index=False)
    return result_df

def batch_testing_rewrite_agent(data, log_root, save_root):
    result_df = pd.DataFrame(columns=['test_id', 'model', 'query', 'strategy', 'result', 'interaction', 'reference', 'url'])

    for index, row in data.iterrows():
        try:
            test_id = index
            query = row['query']

            print(f'Start testing index: {index} ...')
            output = test_rewrite_agent(query, test_id)

            with open(log_root, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([test_id, 'rewrite_agent', query, 'None', output['Result'], 'None',
                                 output['Rerference'], output['Url']])

            time.sleep(3)
        except Exception as e:
            print(f'Error in index: {index}, {e}')
            continue

def inference_l_o(questions_df, results_file):
    existing_questions = set()

    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'test_id', 'query', 'result', 'reference'])
    else:
        existing_questions = read_existing_questions(results_file)

    count = 0
    with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for idx, row in questions_df.iterrows():
            question = row['query']
            test_id = idx
            if question not in existing_questions:
                test_question(test_id, question, count, writer)
                count += 1
                print(f'Test count: {count}')

############### Directory Setup ##############
def setup_directories(root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

# Search Agent Directories and Testing
search_log_root = f'../results/search_agent/{today}'
search_log_file = f'{search_log_root}/search_agent_results_log.txt'
search_save_root = f'../results/search_agent/{today}'
search_save_file = f'{search_save_root}/search_agent_results.csv'
setup_directories(search_log_root)
setup_directories(search_save_root)

# Rewrite Agent Directories and Testing
rewrite_log_root = f'../results/rewrite_agent/{today}'
rewrite_log_file = f'{rewrite_log_root}/rewrite_agent_results_log.txt'
rewrite_save_root = f'../results/rewrite_agent/{today}'
rewrite_save_file = f'{rewrite_save_root}/rewrite_agent_results.csv'
setup_directories(rewrite_log_root)
setup_directories(rewrite_save_root)

# Direct and Offline Model Directories and Testing
direct_offline_save_root = f'../results/llm_direct_offline/{today}'
direct_offline_save_file = f'{direct_offline_save_root}/llm_direct_offline_results.csv'
setup_directories(direct_offline_save_root)


############## main ##############
if __name__ == '__main__':
    if not os.path.exists(search_log_file):
        print('generate files')
        with open(search_log_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_id', 'model', 'query', 'strategy', 'result', 'interaction', 'reference', 'url'])

    print('>>>Start testing>>>')

    batch_testing_search_agent(data, search_log_file, search_save_file)
    batch_testing_rewrite_agent(data, rewrite_log_file, rewrite_save_file)
    inference_l_o(data, direct_offline_save_file)

    print('>>>Testing done>>>')

