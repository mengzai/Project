import find_best_p as fp

def main():
    # result_data=fp.load_file('train_multidim2.txt')
    # names_value = fp.find_best_p(result_data)
    # fp.output_to_csv(names_value, 'yijitezhengjieguo.csv')

    result_data=fp.load_file('train_p.txt')
    pre_result = fp.load_file('second_feature.csv')
    first_feature = fp.load_file('yijitezhengjieguo.csv')
    names_value=fp.find_best_p(result_data,pre_result,first_feature)
    fp.output_data(names_value, 'find_best_p.txt')




if __name__ == '__main__':
    main()