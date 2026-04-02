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
    int n, p, k;
    cin >> n >> p >> k;
    vector<int> a(n + 1);
    vector<int> dp(n + 1, INF);
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    sort(a.begin() + 1, a.end());
    dp[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        if (i < k)
            dp[i] = min(dp[i], dp[i - 1] + a[i]);
        else
            dp[i] = min(dp[i], dp[i - k] + a[i]);
    }
    int ans = 0;
    for (int i = 1; i <= n; i++)
        if (dp[i] <= p)
            ans = i;
    cout << ans << endl;
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