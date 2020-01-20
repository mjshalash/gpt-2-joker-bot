import pandas

# Read csv into DataFrame
result = pandas.read_csv('data/short_jokes/shortjokestest.csv')

# Parse through jokes in DataFrame
# Only copy ones without innappropriate terms
for(i, j) in result.iterrows():

    joke = j[1].split(" ")

    for word in joke:
        print(word)
