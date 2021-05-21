# Import the libraries
import json
from matplotlib import pyplot as plt
import pandas as pd

with open('setting.json') as f:
    setting = json.load(f)

dataset = pd.read_csv("./dataset/{0}.csv".format(setting["file"]))

arr = []

runtime = setting['runtime']
for i in range(runtime):

    if setting["file"] != "corn_3m":
        path = "./result/{0}/{1}-{2}-{3}.csv".format(
            setting["file"], setting["training_rate"], setting["epochs"], i)
    else:
        path = path = "./result/{0}/{1}-{2}/{1}-{2}-{3}.csv".format(
            setting["file"], setting["training_rate"], setting["epochs"], i)
    df = pd.read_csv(path, encoding='UTF-8', low_memory=False)
    arr.append(df)


close = arr[0]["Close"]
lstm = arr[0]["LSTM"]
rnn = arr[0]["RNN"]
slr = arr[0]["SLR"]
for i in range(1, runtime):
    lstm += arr[i]["LSTM"]
    rnn += arr[i]["RNN"]
    slr += arr[i]["SLR"]


start = len(dataset)-len(close)

# process the data
export = pd.DataFrame()
pd.options.display.float_format = "{:.2f}".format
export["Date"] = dataset[start:]["Date"].values
export["Close"] = close
export["LSTM"] = lstm/runtime
export["RNN"] = rnn/runtime
export["SLR"] = slr/runtime
print(export)
# plot the figure
plt.figure(figsize=(16, 8))
plt.title('result')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
# plt.plot(train['Close'])
plt.plot(export['Date'], export[['Close', 'LSTM', 'RNN', 'SLR']])
plt.legend(['Actually data', 'Predictions(LSTM)',
            'Predictions(RNN)', 'Predictions(SLR)'], loc='lower right')
# plt.show()

# save as csv
log_path = "./result/{0}/avg_predict.csv".format(setting["file"])
export.to_csv(log_path, float_format='%.2f')
