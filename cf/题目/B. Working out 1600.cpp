#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int dp[1005][1005][5];
void solve()
{
    int n, m;
    cin >> n >> m;
    int maps[n + 5][m + 5];
    ff(i, 1, n)
            ff(j, 1, m)
                cin >> maps[i][j];
    ff(i, 1, n)
        ff(j, 1, m)
            dp[i][j][0] = maps[i][j] + max(dp[i - 1][j][0], dp[i][j - 1][0]); // 左上到右下
    ffg(i, n, 1)
        ffg(j, m, 1)
            dp[i][j][1] = maps[i][j] + max(dp[i + 1][j][1], dp[i][j + 1][1]); // 右上到左下
    ffg(i, n, 1)
        ff(j, 1, m)
            dp[i][j][2] = maps[i][j] + max(dp[i + 1][j][2], dp[i][j - 1][2]); // 左下到右上
    ff(i, 1, n)
        ffg(j, m, 1)
            dp[i][j][3] = maps[i][j] + max(dp[i - 1][j][3], dp[i][j + 1][3]); // 右下到左上
    int ans = -inf;
    ff(i, 2, n - 1)
        ff(j, 2, m - 1)
    {
        ans = max(ans, dp[i][j - 1][0] + dp[i][j + 1][1] + dp[i + 1][j][2] + dp[i - 1][j][3]);
        ans = max(ans, dp[i - 1][j][0] + dp[i + 1][j][1] + dp[i][j - 1][2] + dp[i][j + 1][3]);
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}