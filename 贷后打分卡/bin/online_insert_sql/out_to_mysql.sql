sqoop eval  --connect "jdbc:mysql://10.130.32.42:3307/outerassets?useUnicode=true&characterEncoding=utf8"   --username "tk_outera_rw"   --password "qEVQh7a7T9wUMKpt967n"  -e "show create table intf_customer_score "

sqoop export -D mapred.child.java.opts="-Djava.security.egd=file:/dev/../dev/urandom" \
--connect "jdbc:mysql://10.130.32.42:3307/outerassets?useUnicode=true&characterEncoding=utf8"   --username "tk_outera_rw"   --password "qEVQh7a7T9wUMKpt967n" \
--input-null-string '\\N'  --input-null-non-string '\\N'   \
--table intf_customer_score  --columns "idno,idx,group_idx,score,dtime"  \
--export-dir /user/hive/warehouse/ml/zhxd_dh_final_score   \
--input-fields-terminated-by '\001';


connection.url=jdbc:mysql://10.130.98.99:3308/outerassets_uat 
connection.username=outerassets 
connection.password=JuyftkAYlRagkcpK1lUP


sqoop eval  --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"  -e "delete FROM cs_score"
sqoop export -D mapred.child.java.opts="-Djava.security.egd=file:/dev/../dev/urandom"   --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"   --input-null-string '\\N'  --input-null-non-string '\\N'   --table cs_score  --columns "id,idx,group_idx,score,dtime"  --export-dir /user/hive/warehouse/db/test.db/zhxd_dh_final_score   --input-fields-terminated-by '\001';



