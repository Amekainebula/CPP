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
int a[100005], arr[100005];
int dp[100005][2];
void solve()
{
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    for (int i = 1; i < n; i++)
    {
        arr[i] = abs(a[i] - a[i + 1]);
    }
    for (int i = 1; i < n; i++)
    {
        dp[i][0] = max(dp[i - 1][1] + arr[i], arr[i]);
        dp[i][1] = max(dp[i - 1][0] - arr[i], 0LL);
    }
    int ans = 0;
    for (int i = 1; i < n; i++)
    {
        ans = max(ans, max(dp[i][0], dp[i][1]));
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