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
int dp[2][60][60];
void pre()
{
    ff(i, 0, 1) ff(j, 0, 59) ff(k, 0, 59) dp[i][j][k] = 4e18;
    dp[0][0][0] = 0;
    ff(i, 1, 59) ff(j, 0, 59) ff(k, 0, 59)
    {
        int temp = i & 1;
        if (j >= i)
            dp[temp][j][k] = min(dp[temp][j][k], dp[temp ^ 1][j - i][k] + (1LL << i));
        if (k >= i)
            dp[temp][j][k] = min(dp[temp][j][k], dp[temp ^ 1][j][k - i] + (1LL << i));
        dp[temp][j][k] = min(dp[temp][j][k], dp[temp ^ 1][j][k]);
    }
}
void solve()
{
    int x, y;
    cin >> x >> y;
    int ans = 1LL << 60;
    ff(i, 0, 59) ff(j, 0, 59) if (x >> i == y >> j)
        ans = min(ans, dp[0][i][j]);
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    pre();
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}