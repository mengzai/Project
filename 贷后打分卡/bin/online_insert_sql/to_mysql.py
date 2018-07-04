#coding=utf-8
import MySQLdb



def load_file(filename):
    if filename[-3:]=="txt":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\n').split('\t')
                data.append(line)
            return data
    if filename[-3:] == "csv":
        with open(filename, 'r') as data_file:
            data = []
            for line in data_file.readlines():
                line = line.strip('\r\n').split(',')
                data.append(line)
            return data
    else:
        print "此输入文件非txt 及csv,请以 txt or csv  结尾"

def main():
	conn= MySQLdb.connect(
			host='121.42.136.239',
			port = 3306,
			user='cs',
			passwd='cs1234@#$%',
			db ='creditscore',
			)
	cur = conn.cursor()

	# 创建数据表
	aa = cur.execute("show databases;")
	info = cur.fetchmany(aa)
	for ii in info:
		print ii

	# cur.execute("DELETE FROM  clone_cs_score")

	score_data=load_file('final_scorecsv.txt')
	for line in score_data:
		cur.execute(
			"INSERT INTO cs_score (id, idx, group_idx, score, dtime)  values('%s','%s','%s','%s','%s')"%(str(line[0]),str(line[1]),str(line[2]),str(line[3]),str(line[4])))

	# 删除表
	# cur.execute("drop table  clone_cs_score")

	# cur.execute(
	# 	"INSERT INTO clone_cs_score (id, idx, group_idx, score, dtime)  values('511322198304154618','1','0','0.0','2016-11-30')")

	#创建数据表
	# cur.execute("create table student(id int ,name varchar(20),class varchar(30),age varchar(10))")


	#插入一条数据
	# cur.execute("insert into student values('2','Tom','3 year 2 class','9')")


	#修改查询条件的数据
	#cur.execute("update student set class='3 year 1 class' where name = 'Tom'")

	#删除查询条件的数据
	#cur.execute("delete from student where age='9'")

	cur.close()
	conn.commit()
	conn.close()



if __name__ == "__main__":
    main()