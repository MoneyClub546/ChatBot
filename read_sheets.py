import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import pandas as pd

SHEET_NAME = "Copy of Revival Call Disposition"
CREDENTIALS_FILE = "trusty-bearing-443508-h4-c846c63e1580.json"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
spreadsheet = client.open(SHEET_NAME)


def get_data(sheet_name="Sheet1"):
    data=[]
    worksheet = spreadsheet.worksheet(sheet_name)
    rows = worksheet.get_all_values()

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:]

    header_map = {header: idx for idx, header in enumerate(headers)}

    required_columns = ["Name", "Phone number", "Common Call Disposition"]
    for col in required_columns:
        if col not in header_map:
            raise ValueError(f"Missing column: {col}")

    extracted_data = []
    for row in data_rows:
        if len(row) < len(headers):
            row += [''] * (len(headers) - len(row))
        extracted_row = {
            "Name": row[header_map["Name"]],
            "Phone number": row[header_map["Phone number"]],
            "Common Call Disposition": row[header_map["Common Call Disposition"]],
        }
        extracted_data.append(extracted_row)

    return extracted_data

data = get_data()
