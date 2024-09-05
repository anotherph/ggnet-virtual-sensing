import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

# datamodule load
dm = torch.load('datamodules/clmHourly-8-0.0-0.0-va0.1-te0.2-s1.pth')

# pkl 파일 경로
# pkl_file_path = 'logs/virtual_sensing/test--2024-09-03--15-34-38/clmHourly/ggnet/seed1/df_hat.pkl'
# pkl_file_path = 'logs/virtual_sensing/test--2024-09-03--17-08-35/clmHourly/ggnet/seed1/df_hat.pkl' # epoch 8
pkl_file_path = 'logs/virtual_sensing/test--2024-09-03--18-48-26/clmHourly/ggnet/seed1/df_hat.pkl' # epoch 14
# pkl_file_path = 'logs/virtual_sensing/test--2024-08-07--19-09-31/clmHourly/ggnet/seed1/df_hat.pkl' # epoch 30

# pkl (predicted data) 파일 로드
with open(pkl_file_path, 'rb') as file:  # 'rb' 모드는 파일을 읽기 모드로 엽니다 (바이너리).
    df_hat = pickle.load(file)

# mask configuration 
check_col = []
eval_col = []
train_col = []

for col in dm.dataframe().columns:
    y_mask = df_hat['df_hat']['eval_mask'][col].values-df_hat['df_hat']['mask'][col].values
    y_evalmask = df_hat['df_hat']['eval_mask'][col].values
    y_trainmask = df_hat['df_hat']['mask'][col].values
    # check if train mask equals to eval_mask
    if not np.all(np.abs(y_mask) == 1):
        check_col.append(col)
    # Save the indices of the columns in eval_mask where the values are 1
    if np.all(y_evalmask == 1):
        if not np.all(dm.dataframe()[col].values == 0): # excluding the columns with no data
            # y= dm.dataframe()[col].values
            # print(np.sum(y))
            # print(np.all(dm.dataframe()[col].values == 0))
            # print(dm.dataframe()[col].values)
            eval_col.append(col)
    # Save the indices of the columns in train_mask where the values are 1
    if np.all(y_trainmask == 1):
        train_col.append(col)

# calculate MRE (mean relative error = |measured-real|/real x 100% )

MRE =[]

for col in eval_col:
    y_tr=dm.dataframe()[col].values
    y_pred=df_hat['df_hat']['med'][col].values

    # mre = np.mean(np.abs(y_tr-y_pred)/y_tr*100)

    # non_zero_mask = y_tr != 0
    # y_tr_non_zero = y_tr[non_zero_mask]
    # y_pred_non_zero = y_pred[non_zero_mask]
    # mre = np.mean(np.abs(y_tr_non_zero - y_pred_non_zero) / y_tr_non_zero * 100)

    # y_tr의 최대값 계산
    max_y_tr = np.max(y_tr)

    # 최대값의 10% 계산
    threshold = 0.1 * max_y_tr

    # threshold 보다 큰 값만 선택
    mask = y_tr >= threshold
    y_tr_filtered = y_tr[mask]
    y_pred_filtered = y_pred[mask]

    # 필터링된 값이 없을 경우 처리
    if len(y_tr_filtered) == 0:
        print("No values meet the threshold criteria.")
    else:
        # MRE 계산
        print(col)
        mre = np.mean(np.abs(y_tr_filtered - y_pred_filtered) / y_tr_filtered * 100)
        print("Mean Relative Error (MRE):", mre)

    # print(y_tr)
    # print(y_pred)

    MRE.append(np.abs(mre)) 

# mean relative error
plt.figure(figsize=(8, 6))
plt.plot(eval_col, MRE, label='MRE', color = 'limegreen', )  # 선 그래프에 점을 추가
# plt.plot(x, y_pred, label='prediction',color = 'dodgerblue')  # 선 그래프에 점을 추가
plt.title('Plot of First Column vs Second Column')
plt.xlabel('First Column')
plt.ylabel('Second Column')
plt.grid(True)  # 격자 추가
plt.legend()
plt.savefig('plot_mre.png', format='png')  # 파일 이름과 포맷 설정

# to draw out as an example
# col = eval_col[10]
# col = train_col[10]
col = (233,3)
x = dm.dataframe().index.to_numpy()
y_tr=dm.dataframe()[col].values
y_pred=df_hat['df_hat']['med'][col].values

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x, y_tr, label='true', color = 'limegreen', )  # 선 그래프에 점을 추가
plt.plot(x, y_pred, label='prediction',color = 'dodgerblue')  # 선 그래프에 점을 추가
plt.title('Plot of First Column vs Second Column')
plt.xlabel('First Column')
plt.ylabel('Second Column')
plt.grid(True)  # 격자 추가
plt.legend()
plt.savefig('plot_8.png', format='png')  # 파일 이름과 포맷 설정

# # mask 그래프 출력
# plt.figure(figsize=(5,10))
# plt.imshow(df_hat['df_hat']['mask'].values, cmap='gray', interpolation='none', origin='upper')
# plt.colorbar(label='Mask Value')
# plt.title('Mask Visualization')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.savefig('mask.png', format='png')  # 파일 이름과 포맷 설정

####

# print(f'Contains any non-1 values: {not np.all(np.abs(y_mask) == 1)}')

# print(y_tr)
# print(y_pred)
# print(df_hat['df_hat']['eval_mask'].values)
# print(df_hat['df_hat']['mask'][col].values)

# print('------')
# print(dm.dataframe().dtypes)
# print(df_hat['df_hat']['true'].dtypes)
# print('------')
# print(dm.dataframe().index)
# print(df_hat['df_hat']['true'].index)
# print(dm.dataframe().columns)
# print(df_hat['df_hat']['true'].columns)

# print(df_hat['df_hat']['true'].shape)
# print(type(df_hat['df_hat']['true']))

# print('--------------------------')
# print(dm.dataframe()[(0, 0)])
# print('--------------------------')
# print(df_hat['df_hat']['true'][(0, 0)])
# print(dm.dataframe().columns)

# print(dm.dataframe().shape)
# print(type(dm.dataframe()))

