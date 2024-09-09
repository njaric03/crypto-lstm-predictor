import pandas as pd

# Specify the output file
output_file = '../data/combined.csv'

# Initialize the output file
open(output_file, 'w').close()

for y in range(17, 25):
    for m in range(1, 13):
        if y == 24 and m == 3:
            break
        if m < 10:
            m = f'0{m}'
        DATA_CSV = f'data/BTCUSDT-1s-20{y}-{m}.csv'
        try:
            chunksize = 10 ** 6  # adjust this value depending on your available memory
            for chunk in pd.read_csv(DATA_CSV, chunksize=chunksize):
                # downcast float64 columns to float32 to save memory
                chunk = chunk.astype({col: 'float32' for col in chunk.select_dtypes('float64').columns})
                # process the data as needed
                chunk = chunk.iloc[:, [0, 1, 2, 3, 4, 5]]
                chunk.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
                chunk['date'] = pd.to_datetime(chunk['date'], unit='ms')
                # from float to integer
                chunk[chunk.columns[1:]] = chunk.iloc[:, 1:].fillna(0).astype(float).astype(int)
                # Append the processed chunk to the output file
                chunk.to_csv(output_file, mode='a', index=False)
        except FileNotFoundError:
            continue
        except pd.errors.EmptyDataError:
            print(f"No data in file: {DATA_CSV}")
            continue
