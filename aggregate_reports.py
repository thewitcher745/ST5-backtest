import os
import pandas as pd


# Convert the Excel files in reports/ to pandas dataframes and aggregate them into one. Then sort the positions by exit time.
def aggregate_reports():
    # Get the list of files in the reports folder
    report_files = os.listdir("./reports")

    # Create an empty list to store the dataframes
    df_list = []

    # Iterate through the files and convert them to dataframes
    for report_file in report_files:
        if report_file.endswith(".xlsx") and report_file != "all_positions.xlsx":
            df = pd.read_excel(f"./reports/{report_file}")
            df_list.append(df)

    # Concatenate the dataframes into one
    all_positions = pd.concat(df_list)

    # Sort the positions by exit time
    all_positions.sort_values(by='Exit time', inplace=True)

    # Save the aggregated positions to a new Excel file
    all_positions.to_excel("./reports/all_positions.xlsx", index=False)

    print("Aggregated positions saved to all_positions.xlsx")


if __name__ == "__main__":
    aggregate_reports()
