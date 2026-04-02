#include <bits/stdc++.h>
// #define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
void solve()
{
    int n, k;
    cin >> n >> k;
    vc<vc<int>> a(n + 1, vc<int>(k + 1)), dp(n + 1, vc<int>(k + 1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= k; j++)
            cin >> a[i][j];
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= k; j++)
            for (int l = 1; l <= j; l++)
                dp[i][j] = max(dp[i][j], dp[i - 1][l] + a[i][j]);
    int ans = -1;
    for (int i = 1; i <= k; i++)
        ans = max(ans, dp[n][i]);
    cout << ans << endl;
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
        solve();
    }
    return 0;
}