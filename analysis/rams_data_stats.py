import argparse
import json
import pathlib
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Path to directory containing data files')

    args = parser.parse_args()
    return args

def print_stats(data, is_test):
    num_examples = len(data)
    num_docs = len(set(example['source_url'] for example in data))

    if not is_test:
        event_types = []
        for example in data:
            event_triggers = example['evt_triggers']
            assert len(event_triggers) == 1
            event_type = event_triggers[0][2][0][0]
            event_types.append(event_type)

        roles = []        
        for example in data:
            event_links = example['gold_evt_links']
            for event_link in event_links:
                # Get the part of the role string after the 'arg<#>' part
                role = re.split(r'(?:.*?)\d(.*?)', event_link[2])[-1]
                roles.append(role)

        num_event_types = len(set(event_types))
        num_roles = len(set(roles))
    else:
        num_event_types = '[redacted]'
        num_roles = '[redacted]'

    num_arguments = sum(len(example['gold_evt_links']) for example in data)
    
    print(f'Docs: {num_docs}')    
    print(f'Examples: {num_examples}')
    print(f'Event Types: {num_event_types}')
    print(f'Roles: {num_roles}')
    print(f'Arguments: {num_arguments}')

def load_data(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
    
def main():
    args = parse_args()
    train_data = load_data(pathlib.Path(args.dir, 'train.jsonlines'))
    print(f"{'='*10}TRAIN{'='*10}")
    print_stats(train_data, is_test=False)

    dev_data = load_data(pathlib.Path(args.dir, 'dev.jsonlines'))
    print(f"{'='*10}DEV{'='*10}")
    print_stats(dev_data, is_test=False)

    test_data = load_data(pathlib.Path(args.dir, 'test.jsonlines'))
    print(f"{'='*10}TEST{'='*10}")
    print_stats(test_data, is_test=True)

if __name__ == "__main__":
    main()
