def get_factor_count(num:int):
    if num == 0:
        return 0
    elif num == 1:
        return 1
    else:
        ans = 0
        for i in range(2, num + 1):
            if num % i == 0:
                ans += get_factor_count(int(num / i))
        return ans


if __name__ == "__main__":
    print(get_factor_count(840))