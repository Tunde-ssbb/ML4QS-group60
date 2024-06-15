import re
from collections import Counter

# Function to read the file, count word occurrences in quotations, and write the result to an output file
def count_quoted_words_in_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        text = file.read()
    
    # Use regex to find words in single quotations and convert to lowercase
    quoted_words = re.findall(r"'(.*?)'", text.lower())
    
    # Count occurrences of each unique quoted word
    word_counts = Counter(quoted_words)
    
    # Sort words by their counts (descending) and then alphabetically
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
    
 
    

    # Write the sorted word counts to the output file
    with open(output_file_path, 'w') as output_file:
        best = []
        for word, count in sorted_word_counts[:25]:
            best.append(word)
        output_file.write(f"25 best: {best}")
        for word, count in sorted_word_counts:
            output_file.write(f'{word}: {count}\n')
    
    # Print the 25 most occurring words
    print("25 Most Occurring Words:")
    for word, count in sorted_word_counts[:25]:
        print(f'{word}: {count}')

    

# Example usage:
input_file_path = './misc/observations.txt'  # Replace with the path to your input text file
output_file_path = './misc/feature_counts.txt'
count_quoted_words_in_file(input_file_path, output_file_path)
