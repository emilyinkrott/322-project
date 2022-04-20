"""
Programmer: Greeley Lindberg
Class: CPSC 322-02, Spring 2021
Programming Assignment #7
4/13/22
Description: This program implements the MyPyTable class
MyPyTable can import data from a .txt and store it in a table.
MyPyTable has several methods to perform table operations.
"""

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        num_cols = len(self.column_names)
        num_rows = 0
        # pylint: disable=unused-variable
        for row in self.data:
            num_rows += 1

        return num_rows, num_cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str):
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int) and 0 <= col_identifier < len(self.column_names):
            col_index = col_identifier
        else:
            raise ValueError

        values = []

        for row in self.data:
            if not include_missing_values and (row[col_index] == "NA" or row[col_index] == "N/A" \
               or row[col_index] == ""):
                continue
            values.append(row[col_index])

        return values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        nonnumeric_cols = []
        # pylint: disable=consider-using-enumerate
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if j in nonnumeric_cols:
                    continue
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    if self.data[i][j] != "NA" and self.data[i][j] != "N/A" \
                       and self.data[i][j] != "":
                        # column probably not numeric, skip in future
                        nonnumeric_cols.append(j)

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        sorted_indices = sorted(row_indexes_to_drop, reverse=True)
        for row_index in sorted_indices:
            try:
                del self.data[row_index]
            except IndexError:
                print(f"Invalid index. Cannot drop row at index {row_index}.")


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.column_names = next(csvreader)

            for row in csvreader:
                self.data.append(row)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.column_names)
            csvwriter.writerows(self.data)


    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        # make sure all keys are valid
        for key in key_column_names:
            if key not in self.column_names:
                raise ValueError

        duplicate_indices = []

        # makes a table of a just the keys for each row
        row_keys = []
        for i, row in enumerate(self.data):
            row_key = []
            for key in key_column_names:
                row_key.append(row[self.column_names.index(key)])
            if row_key in row_keys:
                duplicate_indices.append(i)
            else:
                row_keys.append(row_key)

        return duplicate_indices

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        missing_indices = []
        for i, row in enumerate(self.data):
            for value in row:
                if value in ('NA', 'N/A', ''):
                    missing_indices.append(i)

        self.drop_rows(missing_indices)


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        try:
            col_index = self.column_names.index(col_name)
        except ValueError:
            print(f"Invalid column name. Unkown column '${col_name}'.")

        # check if the values in column are continuous
        test_value = self.data[0][col_index]
        i = 1
        while test_value == "NA" or test_value == "N/A" or test_value == "" and i < len(self.data):
            test_value = self.data[i][col_index]
            i += 1
        if not isinstance(test_value, float):
            # value is not float, column is not continuous
            return

        # find the average
        col_sum = 0
        num_col = 0
        missing_value_indices = []
        for i, row in enumerate(self.data):
            if row[col_index] == "NA" or row[col_index] == "N/A" or row[col_index] == "":
                missing_value_indices.append(i)
            else:
                col_sum += row[col_index]
                num_col += 1

        avg = col_sum / num_col

        # replace missing values with the average
        for i in missing_value_indices:
            self.data[i][col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        header = ["attribute",  "min", "max", "mid", "avg", "median"]

        # if there is no data, return and empty list
        if len(self.data) == 0:
            return MyPyTable(header, [])

        # check to make sure each colum is valid and continious
        col_indices = []
        for col in col_names:
            try:
                col_index = self.column_names.index(col)
                col_indices.append(col_index)
            except ValueError:
                print(f"Invalid column name. Unknown column '{col}'.")

            # check if the values in column are continuous
            test_value = self.data[0][col_index]
            if not isinstance(test_value, float):
                # value is not float, column is not continuous
                print(f"Column '${col}' is not continuous.")
                raise ValueError

        # collect data from each column to be summarized
        cols_data = []
        for col in col_indices:
            cols_data.append(self.get_column(col, False))


        # summarize the data
        summary = []
        for i, row in enumerate(cols_data):
            if len(row) == 0:
                summary.append([])
                continue
            col_summary = []
            col_summary.append(col_names[i])    # attribute
            col_summary.append(min(row))        # min
            col_summary.append(max(row))        # max
            col_summary.append((col_summary[2] - abs(col_summary[1])) / 2)   # mid
            col_summary.append(sum(row) / len(row))       # average
            # median
            sorted_row = sorted(row)
            if len(row) % 2 != 0:
                col_summary.append(sorted_row[len(row) // 2])
            else:
                median = (sorted_row[len(row) // 2] + sorted_row[len(row) // 2 - 1]) / 2
                col_summary.append(median)
            summary.append(col_summary)


        return MyPyTable(header, summary)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # validate key columns
        for col_name in key_column_names:
            if col_name not in self.column_names or col_name not in other_table.column_names:
                raise ValueError

        # construct joined header
        header = self.column_names
        for col in other_table.column_names:
            if col not in key_column_names:
                header.append(col)

        # perform inner join
        joined_table = []
        for row_self in self.data:
            for row_other in other_table.data:
                is_match = False
                for col_name in key_column_names:
                    if row_self[self.column_names.index(col_name)] == \
                       row_other[other_table.column_names.index(col_name)]:
                        is_match = True
                    else:
                        is_match = False
                        break

                if is_match:
                    matching_data = copy.deepcopy(row_self)
                    for col_other in other_table.column_names:
                        if col_other not in key_column_names:
                            col_other_index = other_table.column_names.index(col_other)
                            matching_data.append(row_other[col_other_index])
                    joined_table.append(matching_data)

        return MyPyTable(header, joined_table)

    # pylint: disable=too-many-branches
    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # validate key columns
        for col_name in key_column_names:
            if col_name not in self.column_names or col_name not in other_table.column_names:
                raise ValueError

        # construct joined header
        header = self.column_names
        for col in other_table.column_names:
            if col not in key_column_names:
                header.append(col)

        joined_table = []

        # iterate through self and find matches in other_table
        matching_indices_in_other = []  # keeps track of rows in other table that have been matched
        for row_self in self.data:
            has_match = False       # True if row_self has any match. Prevents duplication of data.
            for index, row_other in enumerate(other_table.data):
                is_match = False    # True when row_self matches with row_other
                for col_name in key_column_names:
                    if row_self[self.column_names.index(col_name)] == \
                       row_other[other_table.column_names.index(col_name)]:
                        is_match = True
                    else:
                        is_match = False
                        break

                self_data = copy.deepcopy(row_self)
                if is_match:
                    has_match = True
                    matching_indices_in_other.append(index)
                    for col_other in other_table.column_names:
                        if col_other not in key_column_names:
                            col_other_index = other_table.column_names.index(col_other)
                            self_data.append(row_other[col_other_index])
                    joined_table.append(self_data)

            if not has_match:
                for col_other in other_table.column_names:
                    if col_other not in key_column_names:
                        self_data.append("NA")
                joined_table.append(self_data)

        # Iterate through other_table and add rows that didn't match
        for i, row_other in enumerate(other_table.data):
            if i not in matching_indices_in_other:
                other_data = ["NA"] * len(header)   # populate the row with NAs and fill in values
                for j, value in enumerate(row_other):
                    other_data[header.index(other_table.column_names[j])] = value
                joined_table.append(other_data)

        return MyPyTable(header, joined_table)
