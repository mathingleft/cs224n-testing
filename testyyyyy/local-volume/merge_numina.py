import json
import random

with open('Numina_200.json') as f:
    base = json.load(f)

with open('Numina_combined_1200_1_to_7.json') as f:
    extra = json.load(f)

existing_ids = {x['example_idx'] for x in base}
new_entries = [x for x in extra if x['example_idx'] not in existing_ids]

merged = base + new_entries

print(f'Base: {len(base)} entries')
print(f'New unique entries from 1200: {len(new_entries)}')
print(f'Total merged: {len(merged)} entries')

# Sample 35 for test set and remove them from merged
random.seed(42)
test_set = random.sample(merged, 35)
test_ids = {x['example_idx'] for x in test_set}
train_merged = [x for x in merged if x['example_idx'] not in test_ids]

output_path = 'Numina_merged.json'
with open(output_path, 'w') as f:
    json.dump(train_merged, f, indent=2)

print(f'Written {len(train_merged)} entries to {output_path}')

test_path = 'Numina_test35.json'
with open(test_path, 'w') as f:
    json.dump(test_set, f, indent=2)

print(f'Sampled 35 test prompts written to {test_path}')
