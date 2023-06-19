def geometric_series_sum(a, r, n):
    """
    计算等比数列的求和
    
    参数：
    a: float，数列的首项
    r: float，公比
    n: int，要计算的项数
    
    返回值：
    float，等比数列的求和结果
    """
    if r == 1:
        return a * n
    else:
        return a * (1 - r ** n) / (1 - r)

# 示例用法
a = 2  # 首项
r = 3  # 公比
n = 5  # 项数

sum_result = geometric_series_sum(a, r, n)
print("等比数列的和为:", sum_result)