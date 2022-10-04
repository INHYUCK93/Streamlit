import streamlit as st
import pandas as pd
import os
import sqlite3
import hashlib

conn = sqlite3.connect('database.db')
c = conn.cursor()

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

	st.title("로그인 기능 테스트")

	menu = ["홈","로그인","가입"]
	choice = st.sidebar.selectbox("메뉴",menu)

	if choice == "홈":
		st.subheader("홈화면 입니다.")

	elif choice == "로그인":
		st.subheader("로그인 화면입니다.")

		username = st.sidebar.text_input("사용자 이름을 입력하세요.")
		password = st.sidebar.text_input("비밀번호를 입력하세요.",type='password')
		if st.sidebar.checkbox("로그인"):
			create_user()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("{}님으로 로그인 했습니다.".format(username))

			else:
				st.warning("사용자 이름 또는 비밀번호가 잘못되었습니다.")

	elif choice == "가입":
		st.subheader("새 계정을 만듭니다.")
		new_user = st.text_input("사용자 이름을 입력하세요.")
		new_password = st.text_input("비밀번호를 입력하세요.",type='password')

		if st.button("가입"):
			create_user()
			add_user(new_user,make_hashes(new_password))
			st.success("계정 생성에 성공했습니다.")
			st.info("로그인 화면에서 로그인하세요.")
if __name__ == '__main__':
	main()