# Embeddings, attendance, etc.

import numpy as np
import pandas as pd
from datetime import datetime
import os

def load_embeddings(embed_file, names_file):
    return np.load(embed_file), np.load(names_file)

def mark_attendance(name, attendance_file):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        if ((df['Name'] == name) & (df['Date'] == date_str)).any():
            return
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df = pd.concat([df, pd.DataFrame([{'Name': name, 'Date': date_str, 'Time': time_str}])], ignore_index=True)
    df.to_csv(attendance_file, index=False)

# from utils import load_embeddings, mark_attendance

# embeddings, names = load_embeddings("../notebooks/saved_embeddings/face_embeddings.npy", "../notebooks/saved_embeddings/face_names.npy")
# print(embeddings.shape, names.shape)

# mark_attendance("Papa", "../attendance_log.csv")
