#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

const int MOD = 998244353;

long long pow_mod(long long x, long long n, int mod)
{
    long long res = 1;
    x %= mod;
    while (n > 0)
    {
        if (n % 2 == 1)
        {
            res = (res * x) % mod;
        }
        x = (x * x) % mod;
        n /= 2;
    }
    return res;
}

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        vector<int> c(26);
        for (int i = 0; i < 26; ++i)
        {
            cin >> c[i];
        }

        long long sum_c = 0;
        for (int ci : c)
        {
            sum_c += ci;
        }

        long long k0 = (sum_c + 1) / 2;
        long long m0 = sum_c / 2;

        int max_fact = sum_c;
        vector<long long> fact(max_fact + 1);
        fact[0] = 1;
        for (int i = 1; i <= max_fact; ++i)
        {
            fact[i] = fact[i - 1] * i % MOD;
        }

        vector<long long> inv_fact(max_fact + 1);
        inv_fact[max_fact] = pow_mod(fact[max_fact], MOD - 2, MOD);
        for (int i = max_fact - 1; i >= 0; --i)
        {
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD;
        }

        long long product_inv = 1;
        for (int ci : c)
        {
            if (ci > 0)
            {
                product_inv = product_inv * inv_fact[ci] % MOD;
            }
        }

        unordered_map<long long, long long> dp;
        dp[0] = 1;
        for (int ci : c)
        {
            if (ci == 0)
                continue;
            unordered_map<long long, long long> new_dp;
            for (auto &entry : dp)
            {
                long long a = entry.first;
                long long cnt = entry.second;

                long long new_a = a + ci;
                new_dp[new_a] = (new_dp[new_a] + cnt) % MOD;

                new_dp[a] = (new_dp[a] + cnt) % MOD;
            }
            dp.swap(new_dp);
        }

        long long d = (dp.count(k0) ? dp[k0] : 0) % MOD;
        if (d == 0)
        {
            cout << 0 << endl;
            continue;
        }

        long long numerator = (fact[k0] * fact[m0]) % MOD;
        long long res = (numerator * product_inv) % MOD;
        res = (res * d) % MOD;
        cout << res << endl;
    }
    return 0;
}