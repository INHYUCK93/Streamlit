import streamlit as st
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import rainflow as rf
import math
import os
import sqlite3
import hashlib

conn = sqlite3.connect('database.db')
c = conn.cursor()

def createDirectory(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print("Error")

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_user():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_user(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def main():
	menu = ["홈", "로그인", "가입"]
	choice = st.sidebar.selectbox("메뉴",menu)

	if choice == "홈":
		st.subheader("홈화면")

	elif choice == "로그인":

		username = st.sidebar.text_input("ID")
		password = st.sidebar.text_input("PASSWORD",type='password')

		if st.sidebar.checkbox("LOGIN"):
			create_user()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))

			if result:
				st.sidebar.success("LOGIN SUCESS")
				menu_2 = ["1.1 데이터 분리(SCADAS)", "1.2 데이터 분리(LSTA)", "2.1 데이터 분석- 최대값", "2.2 데이터 분석- 등가토크",
						  "3.1 데이터 확인"]
				choice_2 = st.sidebar.selectbox("선택", menu_2)
				# if choice_2 == "0. Intro":
				# 	st.subheader("Intro")
				if choice_2 == "1.1 데이터 분리(SCADAS)":
					st.markdown("#### ● SCADAS DAQ 데이터 분리")
					st.markdown("###### 4축 or 5축(앞 좌측, 앞 우측, 뒤 좌측, 뒤 우측, PTO) 데이터를 축별로, 별개의 파일로 분리하는 작업")
					st.markdown("###### * 사용방법 *")
					st.markdown("  1) 계측한 작업기 개수를 설정하고, 각 작업기 이름을 설정한다..")
					st.markdown("  2) 각 작업기에 맞게 파일을 Drag & Drop이나 불러오기로 파일을 업로드한다.")
					st.markdown("  3) 해당 작업기가 PTO를 사용/계측했다면, PTO 사용에 체크한다.")
					st.markdown("  4) 데이터 분리시작 버튼을 누르면 축별로 파일이 분리된다." )
					st.markdown("  5) 분리된 데이터는 C:/Test 폴더에 저장된다. (폴더 자동 생성)")

					number = st.number_input("- 작업기 개수를 입력", step = 1)
					for i in range(1, number+1):
						st.markdown("###### ● {}번 작업기".format(i))
						col_1, col_2, col_3 = st.columns(3)
						with col_1:
							work_name = st.text_input("- {}번 작업기 이름 설정".format(i))
						with col_2:
							option_2 = st.number_input("{} 데이터 시작 지점 입력".format(work_name), step=1, value=48)
						with col_3:
							option_1 = st.radio("- {} 작업기 PTO 사용여부".format(work_name), ("사용", "미사용"))

						upload_files_1 = st.file_uploader("- {} 데이터 파일 업로드".format(work_name), accept_multiple_files=True)

						col_4, col_5 = st.columns([3,1.4])
						with col_4:
							st.empty()
						with col_5:
							if st.button("{} 데이터 분리시작 ".format(work_name)):
								for upload_file in upload_files_1:
									data = pd.read_csv(upload_file, sep="\t", encoding="CP949")
									st.write("filename : ", upload_file.name)
									if option_1 == "사용":
										createDirectory('C:/Test/Front/{}'.format(work_name))
										createDirectory('C:/Test/Rear/{}'.format(work_name))
										createDirectory('C:/Test/PTO/{}'.format(work_name))

										FR, FL = data.iloc[:,[1]][option_2:].astype(float), data.iloc[:,[5]][option_2:].astype(float)
										RR, RL = data.iloc[:,[9]][option_2:].astype(float), data.iloc[:,[13]][option_2:].astype(float)
										PTO = data.iloc[:,[17]][option_2:].astype(float)

										FL.to_csv("C:/Test/Front/{}/{}_FL.txt".format(work_name, upload_file.name), sep="\t", index=False,header=None)
										FR.to_csv("C:/Test/Front/{}/{}_FR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RL.to_csv("C:/Test/Rear/{}/{}_RL.txt".format(work_name, upload_file.name), sep="\t", index=False,header=None)
										RR.to_csv("C:/Test/Rear/{}/{}_RR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										PTO.to_csv("C:/Test/PTO/{}/{}_PTO.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)

									elif option_1 == "미사용":
										createDirectory('C:/Test/Front/{}'.format(work_name))
										createDirectory('C:/Test/Rear/{}'.format(work_name))

										FR, FL = data.iloc[:, [1]][option_2:].astype(float), data.iloc[:, [5]][option_2:].astype(float)
										RR, RL = data.iloc[:, [9]][option_2:].astype(float), data.iloc[:, [13]][option_2:].astype(float)

										FL.to_csv("C:/Test/Front/{}/{}_FL.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										FR.to_csv("C:/Test/Front/{}/{}_FR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RL.to_csv("C:/Test/Rear/{}/{}_RL.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RR.to_csv("C:/Test/Rear/{}/{}_RR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)

				elif choice_2 == "1.2 데이터 분리(LSTA)":
					st.markdown("#### ● LSTA용 DAQ 데이터 분리")
					st.markdown("###### 4축 or 5축 (앞 좌측, 앞 우측, 뒤 좌측, 뒤 우측, PTO) 데이터를 축별로, 별개의 파일로 분리하는 작업")
					st.markdown("###### * 사용방법 *")
					st.markdown("  1) 계측한 작업기 개수를 설정하고, 각 작업기 이름을 설정한다..")
					st.markdown("  2) 각 작업기에 맞게 파일을 Drag & Drop이나 불러오기로 파일을 업로드한다.")
					st.markdown("  3) 해당 작업기가 PTO를 사용/계측했다면, PTO 사용에 체크한다.")
					st.markdown("  4) 데이터 분리시작 버튼을 누르면 축별로 파일이 분리된다.")
					st.markdown("  5) 분리된 데이터는 C:/Test 폴더에 저장된다. (폴더 자동 생성)")

					number = st.number_input("- 작업기 개수를 입력", step = 1)
					for i in range(1, number+1):
						st.markdown("###### ● {}번 작업기".format(i))
						col_1, col_2 = st.columns(2)
						with col_1:
							work_name = st.text_input("- {}번 작업기 이름 설정".format(i))
						with col_2:
							option_1 = st.radio("- {} 작업기 PTO 사용여부".format(work_name), ("사용", "미사용"), horizontal=True)

						upload_files = st.file_uploader("- {} 데이터 파일 업로드".format(work_name), accept_multiple_files=True)
						col_3, col_4 = st.columns([3,1.4])
						with col_3:
							st.empty()
						with col_4:
							if st.button("{} 데이터 분리시작 ".format(work_name)):
								for upload_file in upload_files_1:
									data = pd.read_csv(upload_file, sep="\t", encoding="UTF-8")
									st.write("filename : ", upload_file.name)
									if option_1 == "사용":
										createDirectory('C:/Test/Front/{}'.format(work_name))
										createDirectory('C:/Test/Rear/{}'.format(work_name))
										createDirectory('C:/Test/PTO/{}'.format(work_name))

										FR, FL = data.iloc[:,[0]], data.iloc[:,[1]]
										RR, RL = data.iloc[:,[2]], data.iloc[:,[3]]
										PTO = data.iloc[:,[5]]

										FL.to_csv("C:/Test/Front/{}/{}_FL.txt".format(work_name, upload_file.name), sep="\t", index=False,header=None)
										FR.to_csv("C:/Test/Front/{}/{}_FR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RL.to_csv("C:/Test/Rear/{}/{}_RL.txt".format(work_name, upload_file.name), sep="\t", index=False,header=None)
										RR.to_csv("C:/Test/Rear/{}/{}_RR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										PTO.to_csv("C:/Test/PTO/{}/{}_PTO.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)

									elif option_1 == "미사용":
										createDirectory('C:/Test/Front/{}'.format(work_name))
										createDirectory('C:/Test/Rear/{}'.format(work_name))

										FR, FL = data.iloc[:, [0]], data.iloc[:, [1]]
										RR, RL = data.iloc[:, [2]], data.iloc[:, [3]]

										FL.to_csv("C:/Test/Front/{}/{}_FL.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										FR.to_csv("C:/Test/Front/{}/{}_FR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RL.to_csv("C:/Test/Rear/{}/{}_RL.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)
										RR.to_csv("C:/Test/Rear/{}/{}_RR.txt".format(work_name, upload_file.name), sep="\t", index=False, header=None)

				elif choice_2 == "2.1 데이터 분석- 최대값":
					st.markdown("#### ● 데이터 분석 - 최대값 확인")
					Hz = st.number_input("Sampling Hz 입력", step=1)

					if st.button("최대값 분석 시작"):
						createDirectory('C:/Result')

						path1 = "C:/Test"
						file_list1 = os.listdir(path1)
						result1 = []
						for name1 in file_list1:
							path2 = path1 + '/' + name1
							file_list2 = os.listdir(path2)
							for name2 in file_list2:  # 작업기
								path3 = path2 + '/' + name2
								file_list3 = os.listdir(path3)
								for name3 in file_list3:  # 데이터파일
									name4 = path3 + "/" + name3
									data1 = pd.read_csv("{}".format(name4), sep="\t", encoding="CP949", header=None)
									data1.columns = ["name"]
									data1 = np.array(data1["name"].tolist())
									rfc1 = []  # 레인 플로우 카운팅
									for rng, mean, count, i_start, i_end in rf.extract_cycles(data1):
										rfc1.append([rng, mean, count])
									swt1 = []  # 스미스 완슨 토퍼식 적용
									for i in range(0, len(rfc1)):
										swt0 = math.sqrt((abs(rfc1[i][0]) + abs(rfc1[i][1])) * abs(rfc1[i][0]))
										swt1.append([swt0, rfc1[i][2]])
									# Max/Min 값 확인
									swt2 = []
									for i in range(0, len(swt1)):
										if swt1[i][1] == 1:
											swt2.append([swt1[i][0], swt1[i][1]])
									max1, min1 = round(max(swt2)[0], 0), round(min(swt2)[0], 0)
									result1.append([name1, name2, name3, max1, min1, round(len(data1) / Hz, 1)])
						result1 = DataFrame(result1)
						result1.columns = ["F/R/P", "Implement", "File Name", "Max(N-m)", "Min(N-m)", "Time(sec)"]
						result1[["Max(N-m)", "Min(N-m)", "Time(sec)"]] = result1[["Max(N-m)", "Min(N-m)", "Time(sec)"]].astype(int)

						for i in result1["F/R/P"].value_counts().index:
							Max_0 = result1["Max(N-m)"][result1["F/R/P"] == i].max()
							st.text("{} 의 최대값 : {} N-m".format(i, Max_0))
							# for j in result1["Implement"].value_counts().index:
							# 	Max_1 = result1["Max"][(result1["F/R/P"] == i) & (result1["Implement"] == j)].max()
							# 	st.text("{}-{}의 최대값 : {} N-m".format(i, j, Max_1))

						result_F = result1[result1["F/R/P"] == "Front"]
						result_R = result1[result1["F/R/P"] == "Rear"]
						result_P = result1[result1["F/R/P"] == "PTO"]
						st.markdown("#### ● Front Min-Max 결과")
						index_1 = result_F[result_F["Max(N-m)"] == result_F["Max(N-m)"].max()].index[0]
						st.text("Front 최대값 : {} - {} N-m".format(result_F.loc[index_1][1], result_F.loc[index_1][3]))
						st.dataframe(result_F.style.highlight_max("Max(N-m)")) #, use_container_width=True)

						st.markdown("#### ● Rear Min-Max 결과")
						index_2 = result_R[result_R["Max(N-m)"] == result_R["Max(N-m)"].max()].index[0]
						st.text("Rear 최대값 : {} - {} N-m".format(result_R.loc[index_2][1], result_R.loc[index_2][3]))
						st.dataframe(result_R.style.highlight_max("Max(N-m)"))  # , use_container_width=True)

						st.markdown("#### ● PTO Min-Max 결과")
						index_3 = result_P[result_P["Max(N-m)"] == result_P["Max(N-m)"].max()].index[0]
						st.text("PTO 최대값 : {} - {} N-m".format(result_P.loc[index_3][1], result_P.loc[index_3][3]))
						st.dataframe(result_P.style.highlight_max("Max(N-m)"))  # , use_container_width=True)

				elif choice_2 == "2.2 데이터 분석- 등가토크":

					st.markdown("#### ● 데이터 분석 - 등가토크 확인")
					Hz = st.number_input("Sampling Hz 입력", step=1)
					st.markdown("###### - 전륜(Front) 설정")
					col_11, col_12, col_13, col_14 = st.columns(4)
					with col_11:
						F_RE = st.number_input("- 전륜 Range", step = 1)
					with col_12:
						F_Time1 = st.number_input("- 전륜 Time", step = 1)
					with col_13:
						F_slip_torque = st.number_input("- Front Slip Torque", step = 1)
					with col_14:
						Range_out = st.number_input("- Front %미만 제거 ", step = 1)

					st.markdown("###### - 작업기 비율 설정")

					path11 = "C:/Test/Front"
					file_list1 = os.listdir(path11)
					ratio1 =[]
					for i in file_list1:
						ratio2 = st.number_input("Front - {} 비율 입력 [Ex) 50% → 50]".format(i), step = 1)
						ratio1.append(ratio2)

					if st.checkbox("앞차축 등가토크 분석 시작"):
						path1 = "C:/Test"
						file_list1 = os.listdir(path1)
						for name1 in file_list1:  # Front/Rear/PTO
							path2 = path1 + '/' + name1  #c//test/front
							file_list2 = os.listdir(path2)
							if name1 == "Front":
								Re, Time1, slip_torque = F_RE, F_Time1, F_slip_torque
							elif name1 == "Rear":
								continue
							elif name1 == "PTO":
								continue
							sum_1 = []  # 앞차축 합계용
							sum_3 = []
							EQ1 = []
							for name2 in file_list2:  # 작업기
								path3 = path2 + '/' + name2
								file_list3 = os.listdir(path3)
								sum_2 = []  # 작업기별 합계용
								for name3 in file_list3:  # 데이터파일
									name4 = path3 + "/" + name3
									data1 = pd.read_csv("{}".format(name4), sep="\t", encoding="CP949", header=None)
									data1.columns = ["name"]
									data1 = np.array(data1["name"].tolist())
									rfc1 = []  # 레인 플로우 카운팅
									for rng, mean, count, i_start, i_end in rf.extract_cycles(data1):
										rfc1.append([rng, mean, count])
									swt1 = []  # 스미스 완슨 토퍼식 적용
									for i in range(0, len(rfc1)):
										swt0 = math.sqrt((abs(rfc1[i][0]) + abs(rfc1[i][1])) * abs(rfc1[i][0]))
										swt1.append([swt0, rfc1[i][2]])
									swt2 = []
									for i in range(0, len(swt1)):
										if swt1[i][1] == 1:
											swt2.append([swt1[i][0], swt1[i][1]])

									z, Time2 = Re / 32, len(data1) / Hz
									Range = []
									count1 = []
									for j in range(0, 32):  # 구간별 카운팅
										c = 0
										for i in range(1, len(swt2) - 1):
											if swt2[i][0] > (j * z):
												if swt2[i][1] == 1:
													c += swt2[i][1]
										Range.append(round(z / 2 + z * j, 2))
										count1.append(round(c * Time1 / Time2, 2))
									count1.insert(0, name1)
									count1.insert(1, name2)
									count1.insert(2, name3)
									count1.insert(3, Time1)
									sum_2.append(count1)  # 파일별 시간 동기화
									sum_1.append(count1)

								a2 = []
								for i in range(4, 36):
									a = 0
									for j in range(0, len(file_list3)):
										a += sum_2[j][i]
									a2.append(round(a / len(file_list3), 2))
								a2.insert(0, name1)
								a2.insert(1, name2)
								a2.insert(2, "SUM")
								a2.insert(3, Time1)
								sum_1.append(a2)
								sum_3.append(a2)

								metrial = 7
								slip1 = 0
								slip11 = 0
								for i in range(4, 36):
									if Range[i - 4] >= slip_torque * Range_out/100:
										slip1 += a2[i] * ((Range[i - 4]) ** metrial)
										slip11 += a2[i]
								eq_torque = round((slip1 / slip11) ** (1 / metrial), 2)
								ka = round(eq_torque / slip_torque, 3)
								EQ1.append(["", name2, "Equivalent Torque=", eq_torque, "Ka=", ka])

							a4 = []
							for i in range(4, 36):
								a3 = 0
								for j in range(0, len(file_list2)):
									a3 += round(sum_3[j][i] * (ratio1[j] / 100), 2)
								a4.append(a3)
							a4.insert(0, name1)
							a4.insert(1, name1)
							a4.insert(2, "SUM")
							a4.insert(3, Time1)

							slip2 = 0
							slip22 = 0
							for i in range(4, 36):
								if Range[i - 4] >= slip_torque * Range_out/100:
									slip2 += a4[i] * ((Range[i - 4]) ** metrial)
									slip22 += a4[i]
							eq_torque = round((slip2 / slip22) ** (1 / metrial), 2)
							ka = round(eq_torque / slip_torque, 3)
							EQ1.append(["", name1, "Equivalent Torque=", eq_torque, "Ka=", ka])
							EQ1.append(["", name1, "Slip Torque=", slip_torque])
							EQ1 = pd.DataFrame(EQ1)
							Range.insert(0, "Range")
							Range.insert(1, "Implement")
							Range.insert(2, "File name")
							Range.insert(3, "Time")
							Range = pd.DataFrame(Range)
							sum_1.append(a4)
							sum_1 = np.transpose(pd.DataFrame(sum_1))
							sum_1 = pd.concat([Range, sum_1, EQ1], axis=1)
							sum_1.to_csv("C:/Result/{}.txt".format(name1), encoding = "CP949", sep='\t', index=False, header=None)
							st.markdown("#### - {} 분석결과".format(name1))
							st.dataframe(EQ1)

					st.markdown("###### - 후륜(Rear) 설정")
					col_21, col_22, col_23, col_24 = st.columns(4)
					with col_21:
						R_RE = st.number_input("- 후륜 Range", step = 1)
					with col_22:
						R_Time1 = st.number_input("- 후륜 Time", step = 1)
					with col_23:
						R_slip_torque = st.number_input("- Rear Slip Torque", step = 1)
					with col_24:
						Range_out_R = st.number_input("- Rear % 미만 제거 ", step = 1)

					st.markdown("###### - 작업기 비율 설정")

					path11 = "C:/Test/Rear"
					file_list1 = os.listdir(path11)
					ratio1 =[]
					for i in file_list1:
						ratio2 = st.number_input("Rear - {} 비율 입력 [Ex) 50% → 50]".format(i), step = 1)
						ratio1.append(ratio2)

					if st.checkbox("뒷차축 등가토크 분석 시작"):
						path1 = "C:/Test"
						file_list1 = os.listdir(path1)
						for name1 in file_list1:  # Front/Rear/PTO
							path2 = path1 + '/' + name1  #c//test/front
							file_list2 = os.listdir(path2)
							if name1 == "Front":
								continue
							elif name1 == "Rear":
								Re, Time1, slip_torque = R_RE, R_Time1, R_slip_torque
							elif name1 == "PTO":
								continue
							sum_1 = []  # 앞차축 합계용
							sum_3 = []
							EQ1 = []
							for name2 in file_list2:  # 작업기
								path3 = path2 + '/' + name2
								file_list3 = os.listdir(path3)
								sum_2 = []  # 작업기별 합계용
								for name3 in file_list3:  # 데이터파일
									name4 = path3 + "/" + name3
									data1 = pd.read_csv("{}".format(name4), sep="\t", encoding="CP949", header=None)
									data1.columns = ["name"]
									data1 = np.array(data1["name"].tolist())
									rfc1 = []  # 레인 플로우 카운팅
									for rng, mean, count, i_start, i_end in rf.extract_cycles(data1):
										rfc1.append([rng, mean, count])
									swt1 = []  # 스미스 완슨 토퍼식 적용
									for i in range(0, len(rfc1)):
										swt0 = math.sqrt((abs(rfc1[i][0]) + abs(rfc1[i][1])) * abs(rfc1[i][0]))
										swt1.append([swt0, rfc1[i][2]])
									swt2 = []
									for i in range(0, len(swt1)):
										if swt1[i][1] == 1:
											swt2.append([swt1[i][0], swt1[i][1]])

									z, Time2 = Re / 32, len(data1) / Hz
									Range = []
									count1 = []
									for j in range(0, 32):  # 구간별 카운팅
										c = 0
										for i in range(1, len(swt2) - 1):
											if swt2[i][0] > (j * z):
												if swt2[i][1] == 1:
													c += swt2[i][1]
										Range.append(round(z / 2 + z * j, 2))
										count1.append(round(c * Time1 / Time2, 2))
									count1.insert(0, name1)
									count1.insert(1, name2)
									count1.insert(2, name3)
									count1.insert(3, Time1)
									sum_2.append(count1)  # 파일별 시간 동기화
									sum_1.append(count1)

								a2 = []
								for i in range(4, 36):
									a = 0
									for j in range(0, len(file_list3)):
										a += sum_2[j][i]
									a2.append(round(a / len(file_list3), 2))
								a2.insert(0, name1)
								a2.insert(1, name2)
								a2.insert(2, "SUM")
								a2.insert(3, Time1)
								sum_1.append(a2)
								sum_3.append(a2)

								metrial = 7
								slip1 = 0
								slip11 = 0
								for i in range(4, 36):
									if Range[i - 4] >= slip_torque * Range_out_R/100:
										slip1 += a2[i] * ((Range[i - 4]) ** metrial)
										slip11 += a2[i]
								eq_torque = round((slip1 / slip11) ** (1 / metrial), 2)
								ka = round(eq_torque / slip_torque, 3)
								EQ1.append(["", name2, "Equivalent Torque=", eq_torque, "Ka=", ka])

							a4 = []
							for i in range(4, 36):
								a3 = 0
								for j in range(0, len(file_list2)):
									a3 += round(sum_3[j][i] * (ratio1[j] / 100), 2)
								a4.append(a3)
							a4.insert(0, name1)
							a4.insert(1, name1)
							a4.insert(2, "SUM")
							a4.insert(3, Time1)

							slip2 = 0
							slip22 = 0
							for i in range(4, 36):
								if Range[i - 4] >= slip_torque * Range_out_R/100:
									slip2 += a4[i] * ((Range[i - 4]) ** metrial)
									slip22 += a4[i]
							eq_torque = round((slip2 / slip22) ** (1 / metrial), 2)
							ka = round(eq_torque / slip_torque, 3)
							EQ1.append(["", name1, "Equivalent Torque=", eq_torque, "Ka=", ka])
							EQ1.append(["", name1, "Slip Torque=", slip_torque])
							EQ1 = pd.DataFrame(EQ1)
							Range.insert(0, "Range")
							Range.insert(1, "Implement")
							Range.insert(2, "File name")
							Range.insert(3, "Time")
							Range = pd.DataFrame(Range)
							sum_1.append(a4)
							sum_1 = np.transpose(pd.DataFrame(sum_1))
							sum_1 = pd.concat([Range, sum_1, EQ1], axis=1)
							sum_1.to_csv("C:/Result/{}.txt".format(name1), encoding="cp949", sep='\t', index=False, header=None)
							st.markdown("#### - {} 분석결과".format(name1))
							st.dataframe(EQ1)

				elif choice_2 == "3.1 데이터 확인" :
					st.markdown("#### ● 파일별 그래프 확인")
					st.markdown("###### ● 전륜")
					path1 = "C:/Test/Front"
					file_list1 = os.listdir(path1)
					for name1 in file_list1:  # 작업기
						path2 = path1 + '/' + name1
						file_list2 = os.listdir(path2)
						if st.checkbox("front-{}".format(name1)):
							for name2 in file_list2:
								name3 = path2 + "/" + name2
								if st.checkbox("--- {}-files".format(name3)):
									data1 = pd.read_csv("{}".format(name3), sep="\t", encoding="CP949", header=None)
									st.line_chart(data1)

					st.markdown("###### ● 후륜")
					path1 = "C:/Test/Rear"
					file_list1 = os.listdir(path1)
					for name1 in file_list1:  # 작업기
						path2 = path1 + '/' + name1
						file_list2 = os.listdir(path2)
						if st.checkbox("rear-{}".format(name1)):
							for name2 in file_list2:
								name3 = path2 + "/" + name2
								if st.checkbox(" --- {}-files".format(name3)):
									data1 = pd.read_csv("{}".format(name3), sep="\t", encoding="CP949", header=None)
									st.line_chart(data1)

					st.markdown("###### ● PTO")
					path1 = "C:/Test/PTO"
					file_list1 = os.listdir(path1)
					for name1 in file_list1:  # 작업기
						path2 = path1 + '/' + name1
						file_list2 = os.listdir(path2)
						if st.checkbox("{}".format(name1)):
							for name2 in file_list2:
								name3 = path2 + "/" + name2
								if st.checkbox("--- PTO-{}-files".format(name3)):
									data1 = pd.read_csv("{}".format(name3), sep="\t", encoding="CP949", header=None)
									st.line_chart(data1)
			else:
				st.warning("user Name or Password is wrong")

	elif choice == "가입":
		st.subheader("Create New account")
		new_user = st.text_input("Insert User Name")
		new_password = st.text_input("Insert Password",type='password')

		if st.button("Join"):
			create_user()
			add_user(new_user,make_hashes(new_password))
			st.success("success Create New Account")
			st.info("로그인 화면에서 로그인 하세요.")

if __name__ == '__main__':
	main()
