# Python program to standardize a dataset to fit model training
# May need to be modified depending on input

import csv

is_first = True
with open('../datasets/lightbulb/t_lightbulbs.csv', newline='') as input_file:
    with open('../datasets/lightbulb/t_lightbulbs_stan.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        reader = csv.reader(input_file)
        idNum = 0
        for row in reader:
            if is_first:
                row.insert(0, 'ID')
                row.insert(1, 'Joke')
                row.remove('Question')
                row.remove('Answer')
                row.remove('meta')
                is_first = False
            else:
                idNum += 1
                row.insert(0, idNum)

                # Grab Question and Answer and combine them
                # Then, add double quotes around combo
                joke = row[1]+" "+row[2]
                row.insert(1, joke)

                # Remove last three columns
                row.remove(row[2])
                row.remove(row[2])
                row.remove(row[2])
            writer.writerow(row)
