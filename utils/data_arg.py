from itertools import permutations

def get_circular_permutations(letters:list)->map:
    # Generate all permutations
    all_permutations = permutations(letters)

    # Convert permutations to list for easier viewing
    permutation_list = list(all_permutations)

    # Filter out circular permutations
    circular_permutations = []
    for perm in permutation_list:
        if (perm not in circular_permutations and 
            tuple(perm[-1:] + perm[:-1]) not in circular_permutations and 
            tuple(perm[-2:] + perm[:-2]) not in circular_permutations and 
            tuple(perm[-3:] + perm[:-3]) not in circular_permutations and 
            tuple(perm[-4:] + perm[:-4]) not in circular_permutations):
            circular_permutations.append(perm)
    
    # Create the map
    circular_permutations_map = {}
    for i, perm in enumerate(circular_permutations, 1):
        key = f'order{i}'
        circular_permutations_map[key] = list(perm)
    
    return circular_permutations_map

# Given letters
