#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
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
#define endl '\n'
using namespace std;
void solve()
{
    int n, k;
    cin >> n >> k;
    int mod = 1e9 + 7;
    vector<vector<int>> dp(k + 1, vector<int>(n + 1));
    for (int i = 1; i <= n; i++)
    {
        dp[1][i] = 1;
        // dp[i][0]=1;
    }
    for (int i = 1; i <= k; i++)
    {
        dp[i][0] = 1;
    }
    for (int i = 1; i <= k; i++)
        for (int j = 1; j <= n; j++)
        {
            dp[i][j] = (dp[i][j - 1] + dp[i - 1][n - j]) % mod;
        }
    cout << dp[k][n] % mod << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}