def split_data(input_file, train_file, test_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Keep header
    header = lines[0]
    
    # Initialize training and test sets, including header
    train_data = [header]
    test_data = [header]
    
    # Iterate through data lines excluding header
    for index in range(1, len(lines)):
        if (index - 1) % 5 == 0:
            test_data.append(lines[index])
        else:
            train_data.append(lines[index])
    
    # Write training set data
    with open(train_file, 'w') as file:
        file.writelines(train_data)
    
    # Write test set data
    with open(test_file, 'w') as file:
        file.writelines(test_data)

# Call function
split_data('out.txt', 'train_set.txt', 'test_set.txt')
