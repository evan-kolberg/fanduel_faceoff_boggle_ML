import time
from pyggle.lib.pyggle import Boggle, boggle

#  (0,0)  - x +
#  -   Y A E L 
#  y   N M K Y
#  +   O K E S
#      O O J I
#           (3,3)

# INPUT BOARD
board = [['Y', 'A', 'E', 'L'],
         ['N', 'M', 'R', 'Y'],
         ['O', 'K', 'E', 'S'],
         ['O', 'O', 'J', 'I']
]

boggle = Boggle(board)

# INPUT BONUS TILES
bonus_tiles = {
    'TL': [(1, 1), (3, 0), (0, 3), (3, 2)],
    'DL': [],
    'TW': [(0, 3)],
    'DW': []
}


# SCORING FROM https://www.fanduel.com/faceoff-training-guide-boggle

letter_points = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
}

def calculate_word_score(word, coords, board, bonus_tiles, letter_points) -> int:
    word_score = 0
    word_multipliers = []
    
    for (x, y) in coords:
        letter = board[x][y]
        base_score = letter_points[letter]
        
        for bonus, positions in bonus_tiles.items():
            if (x, y) in positions:
                if bonus == 'TL':
                    base_score *= 3
                elif bonus == 'DL':
                    base_score *= 2
                elif bonus == 'TW':
                    word_multipliers.append(3)
                elif bonus == 'DW':
                    word_multipliers.append(2)
        
        word_score += base_score
    
    for multiplier in word_multipliers:
        word_score *= multiplier
    
    if len(word) >= 5:
        bonus_points = (len(word) - 4) * 5
        word_score += bonus_points

    return word_score


solved = boggle.solve()
time_taken = boggle.time_solve()
print(solved)
print(f"Compute Time: {time_taken}")

word_scores = []
for word, coords in solved.items():
    score = calculate_word_score(word, coords, board, bonus_tiles, letter_points)
    word_scores.append((word, score))

word_scores.sort(key=lambda x: x[1], reverse=True)

for word, score in word_scores:
    print(f"{word}: {score}")







