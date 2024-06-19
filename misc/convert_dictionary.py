def process_line(line):
    return line.split(' ')[0] + '\n'

def clean_text_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            cleaned_line = process_line(line)
            outfile.write(cleaned_line)

# Example usage:
input_file = 'NWL2020.txt'
output_file = 'NWL2020_formatted.txt'
clean_text_file(input_file, output_file)
