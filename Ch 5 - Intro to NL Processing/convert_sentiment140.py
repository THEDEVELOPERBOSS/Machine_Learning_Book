import csv

# Read the downloaded dataset
with open("training.1600000.processed.noemoticon.csv", encoding="utf-8") as infile:

    reader = csv.reader(infile)

    # Skip the header row
    next(reader)

    # Write the dataset in the format the book expects
    with open("binary.csv", "w", newline="", encoding="utf-8") as outfile:

        writer = csv.writer(outfile)

        count = 0

        for row in reader:

            # Stop around the same size as the book's dataset
            if count >= 35327:
                break

            # Tweet text is column 0
            sentence = row[0]

            # Sentiment is column 3
            label = int(row[3])

            writer.writerow([label, sentence])

            count += 1

print(f"Finished! Wrote {count} examples to binary.csv")