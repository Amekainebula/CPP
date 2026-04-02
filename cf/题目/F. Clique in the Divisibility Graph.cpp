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
int check[1000006];
void solve()
{
    int n;
    cin >> n;
    int ans = 0;
    // vector<vector<int>>dp(n);
    for (int i = 0; i < n; i++)
    {
        int x;
        cin >> x;
        check[x]++;
        for (int i = 2; i * x <= 1e6; i++)
        {
            check[i * x] = max(check[i * x], check[x]);
        }
        ans = max(ans, check[x]);
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