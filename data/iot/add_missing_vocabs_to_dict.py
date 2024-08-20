import json

# Load the current word dictionary
with open('./data/coco/word_dict_coco.json', 'r') as f:
    word_dict = json.load(f)

unique_missing_words = unique_entries = [
    'unlocks', 'briskly', 'disposes', 'dumpsters', 'conversed', 'shifting', 
    'apprehend', 'lightweight', 'attends', 'informally', 'patrolled', 
    'energetically', 'extracts', 'departed', 'scurried', 'rearranges', 
    'flashed', 'suspect', 'shovels', 'complexs', 'accumulate', 'organize', 
    'lawnmower', 'scans', 'reverses', 'sped', 'jogs', 'momentarily', 
    'sanitation', 'sidebyside', 'briefly', 'underconstruction', 'tidies', 
    'chores', 'repositioning', 'refilling', 'tidying', 'channels', 
    'separately', 'transaction', 'housework', 'wakes', 'cares', 'iot', 
    'ascended'
]

# Find the next available index in the word dictionary
next_index = max(word_dict.values()) + 1

# Add the missing words to the word dictionary
for word in unique_missing_words:
    if word not in word_dict:
        word_dict[word] = next_index
        next_index += 1

# Save the updated word dictionary back to a file
with open('./data/iot/word_dict_coco_iot.json', 'w') as f:
    json.dump(word_dict, f, indent=4)

print(f"Added {len(unique_missing_words)} new words to the word dictionary.")
