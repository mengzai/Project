package cn.creditease.invest.test
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

object describe {
  def fieldProcessing(hiveCtx:HiveContext,df:DataFrame) = {
    val name: Array[String] = df.columns
    val data: RDD[Row] = df.map(r=>r.toSeq.map(x=>
      if(x == null) null else if (x.toString.trim() =="") null else x.toString.trim()))
      .map(r=>Row(r:_*))
    val schema = StructType(name.map(fieldName => StructField(fieldName, StringType, true)))
    hiveCtx.createDataFrame(data, schema)
  }

  def fieldDescribe(sc: SparkContext,hiveCtx: HiveContext,rawdbName:String,hivedbName:String,tableName:String,
                    insertTableName:String) = {
    val df_raw = hiveCtx.sql(s"select * from $hivedbName.$tableName")
    val df_data = fieldProcessing(hiveCtx,df_raw)
    val dfSize: Double = df_data.count().toDouble
    //assert(dfSize>0)
    val columnName: Array[String] = df_data.columns
    val rawDescr: SchemaRDD = df_data.describe(columnName.toSeq:_*)

    // save DataFrame
    val fieldName: Array[String] = rawDescr.columns.drop(1)
    val dataTranspose: Array[Row] = fieldName
      .map(x =>(Array(rawdbName,hivedbName,tableName,x) ++
        rawDescr.select(x).map(x => x.toString.replaceAll("\\[|\\]", "")).collect()))
      .map(x=>Row(x:_*))

    // Trans to DataFrame
    val dfName: Array[String] =
      Array("rawdb_name","hivedb_name","table_name","field_name","count","mean","stddev","min","max")
    val data: RDD[Row] = sc.parallelize(dataTranspose)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescr: SchemaRDD = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)

    //  join data type Info
    val dfFieldType: SchemaRDD = hiveCtx.sql(s"desc $hivedbName.$tableName")
    val dfOutPut = dfDescr.join(dfFieldType,dfDescr("field_name") === dfFieldType("col_name") ,"left_outer")
      .select("rawdb_name","hivedb_name","table_name","field_name","data_type","comment","coverage_rate","count",
        "mean","stddev","min","max")
    dfOutPut.show()

    // insert into hive
    dfOutPut.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    //hiveCtx.sql(s"insert into table $insertTableName select * from tmp_table")
  }

  def file_name()={
    val all_file_name=
      List(
        "bs_ce_city                         ",
        "bs_tm_city                         ",
        "bs_tm_dictionary                   ",
        "bs_tsm_apply_contact_info          ",
        "bs_tsm_apply_house_info            ",
        "bs_tsm_apply_info                  ",
        "bs_tsm_area                        ",
        "bs_tsm_borrower_info               ",
        "bs_tsm_enterprise_product          ",
        "bs_tsm_inquiry_report              ",
        "bs_tsm_user                        ",
        "nplm_nplm_loan_contract            ",
        "nplm_nplm_repayment_plan           ",
        "nplm_nplm_repayment_record         ",
        "nplm_nplm_repayment_detail         ",
        "nplm_nplm_repayment_operate        ",
        "nplm_nplm_repayment_reduce         ",
        "nplm_nplm_reserve_repayment        ",
        "nplm_nplm_offline_repayment        ",
        "nplm_nplm_Lender_Account           ",
        "nplm_nplm_borrower_info            ",
        "nplm_nplm_offline_repay_confirm    ",
        "nplm_nplm_collection_client        ",
        "nplm_nplm_collection_account       ",
        "nplm_nplm_relation_person          ",
        "nplm_nplm_contract_lender          ",
        "nplm_nplm_repay_plan_orig          ",
        "nplm_nplm_warrantor                ",
        "nplm_nplm_contract_attribute       ",
        "nplm_nplm_OVERDUE_CONTRACT         ",
        "nplm_nplm_OVERDUE_INSTAL           ",
        "nplm_nplm_TERM_FEE_DETAIL          ",
        "nplm_nplm_BUSINESS_CHANGE          ",
        "nplm_nplm_ACCOUNT_CHANGE           ",
        "nplm_nplm_REFUND                   ",
        "nplm_nplm_REFUND_DETAIL            ",
        "nplm_nplm_DERATE_SETTLE            ",
        "nplm_nplm_SYSTEM_DICTIONARY        ",
        "nplm_nplm_SYSTEM_GLOBALTYPE        ",
        "nplm_nplm_OTHER_FEE_DETAIL         ",
        "nplm_nplm_DEPOSIT_DEAL             ",
        "nplm_nplm_DEPOSIT_DETAIL           ",
        "nplm_nplm_CONNECT_RECORD           ",
        "nplm_nplm_CONTRACT_BUSINESS        ",
        "nplm_nplm_BUSINESS_CDETAIL         ",
        "nplm_nplm_out_collection           ",
        "nplm_nplm_CHANGER_OTHER_FEE        ",
        "nplm_nplm_COMPENSATE               ",
        "nplm_nplm_COST_GPS                 ",
        "nplm_nplm_REFUND_GPS               ",
        "nplm_nplm_DEALERMARGIN_COST        ",
        "tcsv_lnscontractinfo               ",
        "tcsv_lnscoborrowerinfo             ",
        "tcsv_lnsacctrpyplan                ",
        "tcsv_lnsbankloanreqinfo            ",
        "tcsv_lnsinsureinfo                 ",
        "tcsv_lnstrustplaninfo              ",
        "tcsv_lnsacctrpyplanbak             ",
        "tcsv_pubproductclass               ",
        "tcsv_lnsacctinfo                   ",
        "tcsv_lnsacctdyninfo                ",
        "tcsv_lnsacctlist                   ",
        "tcsv_lnsacctfeeinfo                ",
        "tcsv_lnsacctcdlist                 ",
        "tcsv_lnsloanlist                   ",
        "tcsv_lnsrepaylist                  ",
        "tcsv_lnsrepaysublist               ",
        "icp_ICP_BORROWER_BASIC_INFO        ",
        "icp_ICP_BORROWER_CAR               ",
        "icp_ICP_BORROWER_COMPANY           ",
        "icp_ICP_BORROWER_CONTACT           ",
        "icp_ICP_BORROWER_CONTACT_WAY       ",
        "icp_ICP_BORROWER_DIC               ",
        "icp_ICP_BORROWER_EDUCATION         ",
        "icp_ICP_BORROWER_FINANCE           ",
        "icp_ICP_BORROWER_HOUSE             ",
        "icp_ICP_BORROWER_INFO              ",
        "icp_ICP_BORROWER_JOB               ",
        "icp_ICP_BORROWER_LIVEINFO          ",
        "icp_ICP_BORROWER_MARKET            ",
        "icp_ICP_BORROWER_TRANSACTION       ",
        "icp_ICP_BORROW_CONTRACT            ",
        "icp_ICP_BORROW_CONTRACT_HISTORY    ",
        "icp_ICP_CONTRACT_AUDIT_INFO        ",
        "icp_ICP_CONTRACT_PHASE_REPAYMENT   ",
        "icp_ICP_CREDIT_DEP                 ",
        "icp_ICP_CREDIT_DICT                ",
        "icp_ICP_DATA_DIC                   ",
        "icp_ICP_DEPARTMENT                 ",
        "icp_ICP_DIC_MAPPING                ",
        "icp_ICP_EMPLOYEE                   ",
        "icp_ICP_FINES_REDUCE               ",
        "icp_ICP_INTO_PIECES                ",
        "icp_ICP_LOAN_MATCHING              ",
        "icp_ICP_LOAN_RECEIPT               ",
        "icp_ICP_MAPPING_RELATION           ",
        "icp_ICP_MAPPING_RELATION_SYSTEM    ",
        "icp_ICP_OLDCONTRACT_INFO           ",
        "icp_ICP_ORGANIZATION_RELATION      ",
        "icp_ICP_REPAYMENT_BOOKING          ",
        "icp_ICP_REPAYMENT_SCHEME           ",
        "icp_ICP_RULE_DETAIL_INFO           ",
        "icp_ICP_RULE_INFO                  ",
        "icp_ICP_RULE_USER_MAPPING          ",
        "icp_ICP_SPECIAL_SITUATION_APPLY    ",
        "icp_ICP_STANDARD_DISTRICT          ",
        "icp_ICP_TM_CITY                    ",
        "icp_ICP_TRADE_RECORD               ",
        "icp_ICP_VERIFY_MATERIAL            ",
        "icp_TC_BS_TRANSPORT                ",
        "icp_TC_MORTGAGE_REPAYMENT          ",
        "icp_TC_MORTGAGOR                   ",
        "icp_TC_MORTGAGOR_CONTACT           ",
        "icp_TC_MORTGAGOR_FIXED_ASSETS      ",
        "icp_TC_MORTGAGOR_INFO_EXT          ",
        "icp_TC_MORTGAGOR_SALARY            ",
        "icp_T_BEE_CUSTOMER                 ",
        "icp_T_BEE_TRANSPORT                ",
        "icp_T_BEE_MORTGAGOR_CONTACT        ",
        "icp_T_BEE_FIXED_ASSETS             ",
        "icp_T_BEE_REVIEW                   ",
        "icp_T_BEE_RISK_INFO                ",
        "icp_T_BEE_NETOUT_CITY              ",
        "icp_T_BEE_PAID_REPORT              ",
        "icp_T_BEE_ACCOUNT_INFO             ",
        "icp_T_BEE_ADDED_INFO               ",
        "icp_T_BEE_MORTGAGOR                ",
        "icp_T_BEE_DATA_DIC                 ",
        "icp_T_BEE_ACCOUNT_REMARK           ",
        "icp_T_BEE_AUTH                     ",
        "icp_T_BEE_BANK_CARD                ",
        "icp_T_BEE_BANK_CARD_WATER          ",
        "icp_T_BEE_BANK_INFO                ",
        "icp_T_BEE_BANK_TRADE               ",
        "icp_T_BEE_SCHEDULE_INFO            ",
        "icp_T_BEE_PBOC_INFO                ",
        "icp_t_bee_transport_sfd            ",
        "icp_t_bee_car_info                 ",
        "icp_T_BEE_CARDRIVE_INFO            ",
        "icp_t_bee_transport_yxd            ",
        "clic_TC_ANTI_FRAUD                 ",
        "clic_TC_ANTI_SCORE                 ",
        "clic_TC_BS_TRANSPORT               ",
        "clic_TC_BUSI_DECISION              ",
        "clic_TC_BUSI_DECISION_EXT          ",
        "clic_TC_BUSI_DECISION_FAULT        ",
        "clic_TC_BUSI_DECISION_REFUSE       ",
        "clic_TC_CREDIT_GRADE               ",
        "clic_TC_CREDIT_GRADE_DETAIL        ",
        "clic_TC_CUSTOMER                   ",
        "clic_TC_HISTORYGUIHU_DETAIL        ",
        "clic_TC_IDENTIFI_WELFARE           ",
        "clic_TC_LCSP_APPLY_DETAILS         ",
        "clic_TC_LCSP_BLACK_DATA            ",
        "clic_TC_LOAN_PURPOSE               ",
        "clic_TC_MORTGAGOR                  ",
        "clic_TC_MORTGAGOR_APPLY_CORP       ",
        "clic_TC_MORTGAGOR_APPLY_HOUSE      ",
        "clic_TC_MORTGAGOR_CONTACT          ",
        "clic_TC_MORTGAGOR_CORPORATION      ",
        "clic_TC_MORTGAGOR_FIXED_ASSETS     ",
        "clic_TC_MORTGAGOR_PERSONALTY       ",
        "clic_TC_MORTGAGOR_RELATIVES        ",
        "clic_TC_MORTGAGOR_SALARY           ",
        "clic_TC_PBOC_ACCOUNT_OPEN          ",
        "clic_TC_PBOC_CAREER                ",
        "clic_TC_PBOC_INFO                  ",
        "clic_TC_PBOC_LOAN_DETAIL           ",
        "clic_TC_PBOC_LOAN_SUMMARY          ",
        "clic_TC_PBOC_STATISTICS_INFO       ",
        "clic_TC_PBOC_WELFARE               ",
        "clic_TC_PROOF_CAREER               ",
        "clic_TC_PROOF_EDUCATION            ",
        "clic_TC_PROOF_MARRIAGE             ",
        "clic_TC_THIRD_VERIFY               ",
        "clic_TC_VERIFYING_INSPECTION_DETAIL",
        "clic_TC_VERIFYING_INSPECTION_RESULT",
        "clic_TC_YRD_APPLY_INFO_EXT         ")
    all_file_name
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("fieldDescribe")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")

    val tableList = file_name()
    hiveCtx.sql(s"use test")
    val tableDescribe = fieldDescribe(sc,hiveCtx,_:String,"ods",_:String,"table_state_describe_v2")
    for (tableName <- tableList){
      val rawdbName = tableName.split("_")(0)
      tableDescribe(rawdbName,tableName)
    }
  }
}