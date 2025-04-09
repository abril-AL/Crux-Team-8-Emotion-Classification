import csv

def txt_to_csv(txt_file_path, csv_file_path, delimiter=','):
    """
    Converts a text file to a CSV file.

    Args:
        txt_file_path (str): Path to the input .txt file.
        csv_file_path (str): Path to the output .csv file.
        delimiter (str, optional): Delimiter used in the text file. Defaults to ','.
    """
    try:
        with open(txt_file_path, 'r') as infile, open(csv_file_path, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(row)
        print(f"Successfully converted '{txt_file_path}' to '{csv_file_path}'")
    except FileNotFoundError:
        print(f"Error: File '{txt_file_path}' not found.")
    except Exception as e:
         print(f"An error occurred: {e}")

if __name__ == "__main__":
    txt_file = 'Data/Navya/navya_data.txt'
    csv_file = 'output.csv'
    txt_to_csv(txt_file, csv_file)