#include <bits/stdc++.h>
#define int long long
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
const int mod = 1e9 + 7;
void solve()
{
    int n, m, v;
    cin >> n >> m >> v;
    vector<int> a(n + 1);
    ff(i, 1, n)
        cin >> a[i];
    vc<vc<int>> dp(m + 1, vc<int>(v + 1, 0));
    dp[0][0] = 1;
    ff(i, 1, n) ff(j, 1, m) ff(k, a[i], v)
        dp[j][k] = (dp[j][k] + dp[j - 1][k - a[i]]) % mod;
    int ans = 0;
    ff(i, 0, v)
        ans = (ans + dp[m][i]) % mod;
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