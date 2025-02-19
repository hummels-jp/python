# 定义加法函数
def add(x, y):
    return x + y

# 定义减法函数
def subtract(x, y):
    return x - y

# 定义乘法函数
def multiply(x, y):
    return x * y

# 定义除法函数
def divide(x, y):
    if y == 0:
        return "Error! Division by zero."
    return x / y

# 主程序
def main():
    print("选择运算：")
    print("1. 加")
    print("2. 减")
    print("3. 乘")
    print("4. 除")

    choice = input("输入你的选择(1/2/3/4): ")

    if choice in ['1', '2', '3', '4']:
        num1 = float(input("输入第一个数字: "))
        num2 = float(input("输入第二个数字: "))

        if choice == '1':
            print(f"{num1} + {num2} = {add(num1, num2)}")

        elif choice == '2':
            print(f"{num1} - {num2} = {subtract(num1, num2)}")

        elif choice == '3':
            print(f"{num1} * {num2} = {multiply(num1, num2)}")

        elif choice == '4':
            result = divide(num1, num2)
            print(f"{num1} / {num2} = {result}")
    else:
        print("无效的输入")

if __name__ == "__main__":
    main()