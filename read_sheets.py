import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import pandas as pd

SHEET_NAME = "Revival Inactive leads"
CREDENTIALS_FILE = "healthy-gasket-467014-v4-86372daa8200.json"

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

    required_columns = ["User Name", "User number", "Disposition"]
    for col in required_columns:
        if col not in header_map:
            raise ValueError(f"Missing column: {col}")

    extracted_data = []
    for row in data_rows:
        if len(row) < len(headers):
            row += [''] * (len(headers) - len(row))
        extracted_row = {
            "User Name": row[header_map["User Name"]],
            "User number": row[header_map["User number"]],
            "Disposition": row[header_map["Disposition"]],
        }
        extracted_data.append(extracted_row)

    return extracted_data


