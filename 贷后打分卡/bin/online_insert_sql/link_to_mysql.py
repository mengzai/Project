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
			host='10.130.98.99',
			port = 3308,
			user='cjgwap',
			passwd='3IuB4KHlQmtnSXB8ECpq',
			db ='cjgwaptest',
			)

	cur = conn.cursor()

	# 创建数据表
	aa = cur.execute("show databases;")
	info = cur.fetchmany(aa)
	for ii in info:
		print ii

	# cur.execute("drop table cs_score")
	# cur.execute("CREATE TABLE `cs_score` ("
	# 			"`sid` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '流水ID',"
	# 			"`id` varchar(50) COLLATE utf8_bin DEFAULT NULL COMMENT '身份证id',"
	# 			"`idx` bigint(20) DEFAULT NULL COMMENT '全局排名',"
	# 			"`group_idx` bigint(20) DEFAULT NULL COMMENT '组内排名',"
	# 			"`score` double NOT NULL DEFAULT '0' COMMENT '贷后得分',"
	# 			"`dtime` datetime DEFAULT NULL COMMENT '数据截止时间',"
	# 			"`deletestatus` int(11) NOT NULL DEFAULT '1' COMMENT '删除状态.0：删除;1：正常',"
	# 			"`createtime` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',"
	# 			"PRIMARY KEY (`sid`)"
	# 			") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;")

	cur.execute("truncate clon_cs_score")

	score_data=load_file('dh_score_1.txt')
	i = 0
	L = []
	for newline in score_data:

		i = i + 1
		L.append(newline[:5])

		if i % 10000 == 0:  # 满1000插入数据库
			try:# 当插入的字段并非table的全部字段时，要指定插入的字段及顺序与value内的值一致
				cur.executemany("insert into cs_score (id, idx, group_idx, score, dtime)  values(%s,%s,%s,%s,%s)", L)
				conn.commit()  # 没有提交的话，无法完成插入
				L = []
				print i
			except:
				conn.rollback()
				print 'No.:' + str(i)

	try:#将剩余插入数据库
		print len(L)
		cur.executemany("insert into cs_score (id, idx, group_idx, score, dtime)  values(%s,%s,%s,%s,%s)", L)
		conn.commit()
	except:
		conn.rollback()
		conn.close()


if __name__ == "__main__":
    main()