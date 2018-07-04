sqoop eval  --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"  -e "delete FROM cs_score" 
sqoop export -D mapred.child.java.opts="-Djava.security.egd=file:/dev/../dev/urandom"   --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"   --input-null-string '\\N'  --input-null-non-string '\\N'   --table cs_score  --columns "id,idx,group_idx,score,dtime"  --export-dir /user/hive/warehouse/db/test.db/zhxd_dh_final_score   --input-fields-terminated-by '\001';





sqoop eval --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest" --username 'tk_cjgwaptest_rw' --password 'qEVQh7a7T9wUMKpt967n' \
--query "show tables";

/db/test.db/zhxd_dh_final_score

mysql -h10.130.98.99 -utk_cjgwaptest_rw -p --port3308

密码：qEVQh7a7T9wUMKpt967n


sqoop eval  --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"  -e "describe cs_score" 


sqoop eval --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest" --username 'tk_cjgwaptest_rw' --password 'qEVQh7a7T9wUMKpt967n' \
--query "select * from  cs_score where  score=0";

