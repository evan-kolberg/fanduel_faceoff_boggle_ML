from pyggle.lib.pyggle import Boggle
import io
import sys

# INPUT BOARD
# Separate columns by spaces
#
#  (0,0)  - x +
#  -   Y A E L 
#  y   N M K Y
#  +   O K E S
#      O O J I
#           (3,3)

board_str = "YNOO AMKO EKEJ LYSI"

# INPUT BONUS TILES
bonus_tiles = {
    'TL': [(1, 1), (3, 0), (0, 3), (3, 2)], 
    'DL': [],         
    'TW': [(2, 2)],         
    'DW': []  
}

# SCORING FROM https://www.fanduel.com/faceoff-training-guide-boggle

letter_points = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
}

# Convert the board string to a 2D list
board = [list(row) for row in board_str.split()]

# Create the Boggle object
boggle = Boggle(board_str)

# Capture the output of boggle.print_result()
captured_output = io.StringIO()
sys.stdout = captured_output
boggle.print_result()
captured_output.seek(0)  # Reset the pointer to the beginning of the StringIO object

# Get the captured output
output = captured_output.getvalue()

# Reset stdout to its original state
sys.stdout = sys.__stdout__

# Parse the captured output into a dictionary
words_with_coords = {}
for line in output.splitlines():
    if line.strip(): 
        word, coords_str = line.split(':')
        word = word.strip()
        coords = eval(coords_str.strip())
        words_with_coords[word] = coords

# Function to calculate the word score including bonus points for longer words
def calculate_word_score(word, coords, bonus_tiles, letter_points):
    word_score = 0
    word_multipliers = []
    
    for (x, y) in coords:
        letter = board[x][y]
        base_score = letter_points[letter.upper()]
        
        # Apply letter multipliers
        for bonus, positions in bonus_tiles.items():
            if (x, y) in positions:
                if bonus == 'TL':
                    base_score *= 3
                elif bonus == 'DL':
                    base_score *= 2
        
        word_score += base_score
        
        # Apply word multipliers
        for bonus, positions in bonus_tiles.items():
            if (x, y) in positions:
                if bonus == 'TW':
                    word_multipliers.append(3)
                elif bonus == 'DW':
                    word_multipliers.append(2)
    
    # Apply word multipliers
    for multiplier in word_multipliers:
        word_score *= multiplier
    
    # Add bonus points for words with five or more letters
    if len(word) >= 5:
        bonus_points = (len(word) - 4) * 5
        word_score += bonus_points
    
    return word_score

# Calculate scores for each word
word_scores = []
for word, coords in words_with_coords.items():
    score = calculate_word_score(word, coords, bonus_tiles, letter_points)
    word_scores.append((word, score))

# Sort words by score in descending order
word_scores.sort(key=lambda x: x[1], reverse=True)

# Print the words and their scores
for word, score in word_scores:
    print(f"{word}: {score}")



