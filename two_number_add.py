def find_two_numbers(nums, target):
    # 创建一个字典来存储数值和对应的索引
    num_dict = {}
    
    # 遍历数组
    for index, num in enumerate(nums):
        # 计算需要的补数
        complement = target - num
        
        # 检查补数是否在字典中
        if complement in num_dict:
            return (complement, num)
        
        # 将当前数值和索引存入字典
        num_dict[num] = index
    
    # 如果没有找到符合条件的两个数，返回None
    return None

# 主程序
def main():
    nums = [2, 7, 11, 15]
    target = 9
    result = find_two_numbers(nums, target)
    
    if result:
        print(f"找到两个数: {result[0]} 和 {result[1]}，它们的和等于 {target}")
    else:
        print(f"没有找到两个数，它们的和等于 {target}")

if __name__ == "__main__":
    main()