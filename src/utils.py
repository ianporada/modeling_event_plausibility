import csv


def read_tsv(fname):
    with open(fname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        return [row for row in csv_reader]
