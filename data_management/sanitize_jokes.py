import pandas

# Establish list of innappropriate terms
bad_terms = ['marijuana', 'semen', 'strippers', 'circumcision', 'srascist', 'piss', 'pissed', 'slave', 'slaves', 'porn', 'condom', 'sex', 'shit', 'shits',
             'retarded', 'vagina', 'cunt', 'penis', 'ass', 'gays', 'gay', 'black', 'sex']

# Read csv into DataFrame
result = pandas.read_csv('data/short_jokes/shortjokes.csv')

print(result.shape)

# Parse through jokes in DataFrame
# Only copy ones with NO innappropriate terms
for row in result.itertuples():
    # Flag to represent if joke is bad or not
    bad_flag = 0

    # Extract joke using iterator
    joke = row[2]
    index = row[0]

    # Check each word for curse word
    for word in joke.split(" "):
        # print(word)
        # If bad word detected, exit loop and do not add joke to new document
        # TODO: .lower() could be used somehow to reduce variations of words needed in list
        if word in bad_terms:
            bad_flag = 1
            break

    # Mark bad jokes with additional column
    # TODO: Could potentially train an AI to determine what a bad joke is
    if bad_flag == 1:
        result.at[index, 'Bad'] = 'Y'
    else:
        result.at[index, 'Bad'] = 'N'


# Drop any rows which do not meet no-bad condition
result.drop(result[result['Bad'] != 'N'].index, inplace=True)

print(result.shape)
# Save remaining jokes to new file
result.to_csv('data/short_jokes_clean/shortjokesclean.csv')
