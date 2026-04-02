#include <bits/stdc++.h>

using namespace std;
using i64 = long long;


i64 qpow(i64 a, i64 b)
{
    i64 ans = 1;
    while (b)
    {
        if (b & 1)
            ans = ans * a;
        a = a * a;
        b >>= 1;
    }
    return ans;
}

const int N = 2e5 + 10;
i64 b[N];
i64 dp[N];

void go()
{
    i64 n, m;
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
        cin >> b[i];
    for (int i = 1; i <= m; i++)
        dp[i] = LLONG_MAX / 2;

    dp[0] = 0;
    for (int i = 1; i <= m; i++)
        for (int j = i - 1; j >= max(0, i - 50); j--)
            dp[i] = min(dp[i], dp[j] + b[i] + qpow(i - j, 4));

    cout << dp[m] << '\n';
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    // local_go_m(go);
    go();
    return 0;
}