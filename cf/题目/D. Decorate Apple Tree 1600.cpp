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
    int n;
    cin >> n;
    vector<vector<int>> g(n + 1);
    vector<int> ans(n + 1);
    for (int i = 2; i <= n; i++)
    {
        int x;
        cin >> x;
        g[x].pb(i);
    }
    auto dfs = [&](this auto &&dfs, int u) -> void
    {
        if (sz(g[u]) == 0)
        {
            ans[u] = 1;
            return;
        }
        for (auto v : g[u])
        {
            dfs(v);
            ans[u] += ans[v];
        }
    };
    dfs(1);
    sort(ans.begin() + 1, ans.end());
    for (int i = 1; i <= n; i++)
        cout << ans[i] << " ";
    cout << endl;
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