from collections import Counter

def predict_words(input, segment_length):
    segments = [input[i:i+segment_length] for i in range(0, len(input), segment_length)]
    
    most_common_letters = ""
    
    for segment in segments:
        counter = Counter(segment)
        most_common_letter = max(counter, key=counter.get)
        most_common_letters += most_common_letter
    
    return most_common_letters

file_path = "Prueba1.txt"
with open(file_path, 'r') as file:
    input_string = file.read()

segment_length = 25
result = predict_words(input_string, segment_length)
print("INPUT: ", input_string)
print("OUTPUT: ", result)