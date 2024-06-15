import random
import string

def generate_random_name(length=8):
    # Define the characters that can be used in the name
    characters = string.ascii_letters + string.digits
    # Generate a random name with the specified length
    random_name = ''.join(random.choice(characters) for i in range(length))
    return random_name

# Example usage:
print(generate_random_name(10))  # Adjust the number for different length
