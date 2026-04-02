#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
void Murasame()
{
    int n, m;
    cin >> n >> m;
    vc<pii> g[n + 1];
    vi get(n + 1);
    ff(i, 1, n) cin >> get[i];
    ff(i, 1, m)
    {
        int u, v, w;
        cin >> u >> v >> w;
        g[u].pb({v, w});
    }
    auto check = [&](int mid) -> bool
    {
        vi dp(n + 1, 0);
        ff(i, 1, n)
        {
            if (i > 1 && dp[i] == 0) // 已经到不了i点了，直接跳过
                continue;
            dp[i] += get[i];
            dp[i] = min(dp[i], mid); // 不能超过mid
            for (auto j : g[i])
            {
                if (j.se <= dp[i]) // 可以到达
                {
                    dp[j.fi] = max(dp[j.fi], dp[i]);
                }
            }
        }
        return dp[n] > 0; // 能否到达终点
    };
    if (!check(INF))
    {
        cout << -1 << endl;
        return;
    }
    int l = 0, r = INF;
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid))
            r = mid - 1;
        else
            l = mid;
    }
    cout << l + 1 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}